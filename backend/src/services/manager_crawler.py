"""Manager crawler service for indexing FPL managers"""

import time
import threading
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import signal
import sys
from queue import Queue, Empty
import json
from pathlib import Path

from backend.src.services.fpl_api_client import FPLAPIClient
from backend.src.core.postgres_db import PostgresManagerDB
from backend.src.core.config import get_settings

settings = get_settings()
CONTROL_FILE = settings.DATA_DIR / "crawler_control.json"
logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple thread-safe rate limiter"""

    def __init__(self, interval_seconds: float):
        self.interval = max(interval_seconds, 0.05)
        self.lock = threading.Lock()
        self.last_request_time = 0.0

    def wait(self):
        """Block until we're allowed to make the next request"""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            wait_time = self.interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_request_time = time.time()


class ManagerCrawler:
    """Crawler service for indexing FPL managers"""
    
    def __init__(self):
        self.fpl_client = FPLAPIClient()
        self.db = PostgresManagerDB()
        self.running = False
        self.paused = False
        self.current_progress_id = None
        self.current_crawl_type: Optional[str] = None
        self.worker_threads: List[threading.Thread] = []
        self.lock = threading.Lock()
        self.stop_requested = False
        self.control_file: Path = CONTROL_FILE
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal, stopping crawler...")
        self.stop()
        sys.exit(0)
    
    def _read_control_command(self) -> Optional[Dict[str, Any]]:
        """Read control command from file if it exists"""
        if not self.control_file.exists():
            return None
        try:
            with self.control_file.open("r", encoding="utf-8") as file:
                data = json.load(file)
                return data
        except Exception as exc:
            logger.warning(f"Failed to read control command: {exc}")
            return None
    
    def _clear_control_command(self):
        try:
            if self.control_file.exists():
                self.control_file.unlink()
        except Exception as exc:
            logger.warning(f"Failed to clear control command file: {exc}")
    
    def _handle_control_signal(self, scope: str):
        """Handle external control signals (pause/resume/stop)"""
        data = self._read_control_command()
        if not data:
            return
        
        command_scope = data.get('scope', 'auto')
        if command_scope not in ('auto', scope):
            return
        
        action = data.get('action')
        if action == 'pause':
            self.paused = True
            logger.info(f"Pause requested for {scope} crawl via control signal")
        elif action == 'resume':
            self.paused = False
            self.running = True
            logger.info(f"Resume requested for {scope} crawl via control signal")
        elif action == 'stop':
            self.running = False
            self.stop_requested = True
            logger.info(f"Stop requested for {scope} crawl via control signal")
        else:
            logger.warning(f"Unknown control action: {action}")
        
        self._clear_control_command()
    
    @staticmethod
    def emit_control_signal(action: str, scope: str = 'auto'):
        """Emit a control signal for running crawler processes"""
        payload = {
            'action': action,
            'scope': scope,
            'issued_at': datetime.utcnow().isoformat()
        }
        try:
            CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
            with CONTROL_FILE.open("w", encoding="utf-8") as file:
                json.dump(payload, file)
        except Exception as exc:
            logger.error(f"Failed to write control signal: {exc}")
    
    def _extract_leagues(self, entry_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract league memberships from entry response"""
        memberships: List[Dict[str, Any]] = []
        leagues = entry_data.get('leagues') or {}
        for league in leagues.get('classic', []):
            memberships.append({
                'id': league.get('id'),
                'name': league.get('name'),
                'type': 'classic',
                'rank': league.get('entry_rank'),
                'total_points': league.get('entry_total')
            })
        for league in leagues.get('h2h', []):
            memberships.append({
                'id': league.get('id'),
                'name': league.get('name'),
                'type': 'h2h',
                'rank': league.get('entry_rank'),
                'total_points': league.get('entry_total')
            })
        return memberships
    
    def _parse_manager_entry(self, entry_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse FPL entry data into database format"""
        if not entry_data:
            return None
        
        player_first_name = entry_data.get('player_first_name', '')
        player_last_name = entry_data.get('player_last_name', '')
        player_name = f"{player_first_name} {player_last_name}".strip()
        
        return {
            'manager_id': entry_data.get('id'),
            'player_first_name': player_first_name,
            'player_last_name': player_last_name,
            'player_name': player_name,
            'team_name': entry_data.get('name', ''),
            'overall_rank': entry_data.get('summary_overall_rank'),
            'total_points': entry_data.get('summary_overall_points'),
            'region': entry_data.get('player_region_name'),
            'started_event': entry_data.get('started_event'),
            'favourite_team': entry_data.get('favourite_team'),
            'player_region_id': entry_data.get('player_region_id'),
            'summary_overall_points': entry_data.get('summary_overall_points'),
            'summary_overall_rank': entry_data.get('summary_overall_rank'),
            'summary_event_points': entry_data.get('summary_event_points'),
            'current_event': entry_data.get('current_event'),
            'leagues': self._extract_leagues(entry_data),
            'last_deadline_bank': entry_data.get('last_deadline_bank'),
            'last_deadline_value': entry_data.get('last_deadline_value'),
            'last_deadline_total_transfers': entry_data.get('last_deadline_total_transfers'),
            'is_active': True,
            'crawl_status': 'indexed'
        }
    
    def _fetch_manager_data(self, manager_id: int) -> Optional[Dict[str, Any]]:
        """Fetch manager data from FPL API"""
        try:
            entry_data = self.fpl_client.get_manager_entry(str(manager_id))
            if entry_data:
                return self._parse_manager_entry(entry_data)
            return None
        except Exception as e:
            logger.error(f"Error fetching manager {manager_id}: {str(e)}")
            return None

    def _parse_history_to_gameweeks(
        self, manager_id: int, history_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Parse manager history payload into manager_gameweeks rows."""
        if not history_data:
            return []

        current = history_data.get("current") or []
        chips = history_data.get("chips") or []

        chip_by_event: Dict[int, str] = {}
        chip_name_map = {
            "wildcard": "Wildcard",
            "freehit": "Free Hit",
            "3xc": "Triple Captain",
            "bboost": "Bench Boost",
        }
        for chip in chips:
            gw = chip.get("event")
            if gw is None:
                continue
            raw = chip.get("name")
            chip_by_event[gw] = chip_name_map.get(raw, raw)

        rows: List[Dict[str, Any]] = []
        for row in current:
            event = row.get("event")
            if event is None:
                continue
            rows.append(
                {
                    "manager_id": manager_id,
                    "event": event,
                    "gw_points": row.get("points"),
                    "total_points": row.get("total_points"),
                    "overall_rank": row.get("overall_rank"),
                    "event_rank": row.get("rank"),
                    "bank": row.get("bank"),
                    "team_value": row.get("value"),
                    "event_transfers": row.get("event_transfers"),
                    "event_transfers_cost": row.get("event_transfers_cost"),
                    "chip_played": chip_by_event.get(event),
                    "captain_id": None,
                    "vice_captain_id": None,
                    "points_on_bench": row.get("points_on_bench"),
                }
            )
        return rows

    def _parse_history_to_transfers(
        self, manager_id: int, history_data: Dict[str, Any]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Parse manager history payload into transfers grouped by event."""
        if not history_data:
            return {}

        transfers = history_data.get("transfers") or []
        by_event: Dict[int, List[Dict[str, Any]]] = {}
        for tr in transfers:
            event = tr.get("event")
            if event is None:
                continue
            rec = {
                "manager_id": manager_id,
                "event": event,
                "element_in": tr.get("element_in"),
                "element_out": tr.get("element_out"),
                "purchase_price": tr.get("element_in_cost"),
                "selling_price": tr.get("element_out_cost"),
            }
            by_event.setdefault(event, []).append(rec)
        return by_event

    def _sync_manager_event_picks(self, manager_id: int, event: int) -> None:
        """Fetch and persist picks for a single manager/gameweek."""
        data = self.fpl_client.get_manager_event_picks(str(manager_id), event)
        if not data:
            return

        picks = data.get("picks") or []
        pick_rows: List[Dict[str, Any]] = []
        captain_id = None
        vice_captain_id = None
        for pick in picks:
            element_id = pick.get("element")
            if element_id is None:
                continue
            is_captain = bool(pick.get("is_captain"))
            is_vice = bool(pick.get("is_vice_captain"))
            if is_captain:
                captain_id = element_id
            if is_vice:
                vice_captain_id = element_id
            pick_rows.append(
                {
                    "manager_id": manager_id,
                    "event": event,
                    "element_id": element_id,
                    "position": pick.get("position"),
                    "is_captain": is_captain,
                    "is_vice_captain": is_vice,
                    "multiplier": pick.get("multiplier"),
                }
            )

        self.db.replace_manager_picks(manager_id, event, pick_rows)

        # Optionally enrich manager_gameweeks with captain/vice from this call
        if captain_id is not None or vice_captain_id is not None:
            self.db.upsert_manager_gameweeks(
                [
                    {
                        "manager_id": manager_id,
                        "event": event,
                        "gw_points": None,
                        "total_points": None,
                        "overall_rank": None,
                        "event_rank": None,
                        "bank": None,
                        "team_value": None,
                        "event_transfers": None,
                        "event_transfers_cost": None,
                        "chip_played": None,
                        "captain_id": captain_id,
                        "vice_captain_id": vice_captain_id,
                        "points_on_bench": None,
                    }
                ]
            )

    def sync_manager_history(
        self, manager_id: int, max_event: Optional[int] = None
    ) -> None:
        """Sync manager history, transfers and picks into Postgres."""
        history = self.fpl_client.get_manager_history(str(manager_id))
        if not history:
            logger.warning(f"No history found for manager {manager_id}")
            return

        gw_rows = self._parse_history_to_gameweeks(manager_id, history)
        if max_event is not None:
            gw_rows = [row for row in gw_rows if row["event"] <= max_event]
        if gw_rows:
            self.db.upsert_manager_gameweeks(gw_rows)

        transfers_by_event = self._parse_history_to_transfers(manager_id, history)
        for event, rows in transfers_by_event.items():
            if max_event is not None and event > max_event:
                continue
            self.db.replace_manager_transfers(manager_id, event, rows)

        # Sync picks for each known event
        events = sorted({row["event"] for row in gw_rows})
        for ev in events:
            self._sync_manager_event_picks(manager_id, ev)
    
    def crawl_manager(self, manager_id: int, league_context: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Crawl a single manager and return result"""
        result = {
            'manager_id': manager_id,
            'success': False,
            'deleted': False,
            'error': None
        }
        
        try:
            manager_data = self._fetch_manager_data(manager_id)
            
            if manager_data:
                if league_context:
                    existing_ids = {league.get('id') for league in manager_data.get('leagues', []) if league}
                    merged = manager_data.get('leagues') or []
                    for league in league_context:
                        if not league:
                            continue
                        if league.get('id') not in existing_ids:
                            merged.append(league)
                    manager_data['leagues'] = merged
                
                success = self.db.upsert_manager(manager_data)
                if success:
                    self.db.upsert_manager_leagues(manager_id, manager_data.get('leagues'))
                result['success'] = success
            else:
                # Manager doesn't exist (404 or deleted)
                result['deleted'] = True
                self.db.mark_manager_deleted(manager_id)
                self.db.log_crawl_error({
                    'manager_id': manager_id,
                    'error_type': 'not_found',
                    'error_message': 'Manager not found or deleted',
                    'http_status_code': 404,
                    'retry_count': 0
                })
        except Exception as e:
            result['error'] = str(e)
            self.db.log_crawl_error({
                'manager_id': manager_id,
                'error_type': 'exception',
                'error_message': str(e),
                'http_status_code': None,
                'retry_count': 0
            })
        
        return result
    
    def crawl_range(
        self,
        start_id: int,
        end_id: Optional[int] = None,
        batch_size: int = None,
        progress_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Crawl a range of manager IDs (supports multi-threading)"""
        batch_size = batch_size or settings.CRAWLER_BATCH_SIZE
        self.running = True
        self.paused = False
        self.stop_requested = False
        self.current_crawl_type = 'range'

        worker_count = max(1, settings.CRAWLER_MAX_WORKERS)
        rate_limit = max(settings.CRAWLER_RATE_LIMIT, 0.05)
        max_consecutive_failures = 200

        next_manager_id = start_id
        id_lock = threading.Lock()
        stats_lock = threading.Lock()
        stats = {
            'current_id': start_id - 1,
            'indexed': 0,
            'failed': 0,
            'deleted': 0,
            'consecutive_failures': 0
        }

        # Create or use existing progress record
        if not progress_id:
            progress_data = {
                'crawl_type': 'range',
                'start_manager_id': start_id,
                'end_manager_id': end_id,
                'current_manager_id': start_id,
                'total_discovered': 0,
                'total_indexed': 0,
                'total_failed': 0,
                'total_deleted': 0,
                'status': 'running',
                'metadata': {
                    'workers': worker_count,
                    'rate_limit_seconds': rate_limit
                }
            }
            progress_id = self.db.create_crawl_progress(progress_data)
            self.current_progress_id = progress_id

        logger.info(
            f"Starting crawl from ID {start_id} to {end_id or 'end'} "
            f"with {worker_count} worker(s) at {1/rate_limit:.2f} req/sec each"
        )

        def get_next_id() -> Optional[int]:
            nonlocal next_manager_id
            with id_lock:
                if not self.running:
                    return None
                if end_id is not None and next_manager_id > end_id:
                    return None
                manager_id = next_manager_id
                next_manager_id += 1
                return manager_id

        def worker_loop(worker_idx: int):
            rate_limiter = RateLimiter(rate_limit)
            while self.running:
                self._handle_control_signal('range')
                if self.stop_requested:
                    break
                if self.paused:
                    time.sleep(0.5)
                    continue

                manager_id = get_next_id()
                if manager_id is None:
                    break

                rate_limiter.wait()
                result = self.crawl_manager(manager_id)

                if result['error'] and any(
                    token in (result['error'] or '').lower()
                    for token in ['429', 'too many requests', 'rate limit', '503']
                ):
                    logger.warning("Rate limit warning received, backing off for 5 seconds")
                    time.sleep(5)

                with stats_lock:
                    stats['current_id'] = manager_id
                    if result['success']:
                        stats['indexed'] += 1
                        stats['consecutive_failures'] = 0
                    elif result['deleted']:
                        stats['deleted'] += 1
                        stats['consecutive_failures'] += 1
                    else:
                        stats['failed'] += 1
                        stats['consecutive_failures'] += 1

                    if stats['consecutive_failures'] >= max_consecutive_failures:
                        logger.warning(
                            "Too many consecutive failures detected, stopping crawl to avoid ban"
                        )
                        self.running = False
                        break

        # Spin up worker threads
        self.worker_threads = []
        for idx in range(worker_count):
            thread = threading.Thread(target=worker_loop, args=(idx,), daemon=True)
            self.worker_threads.append(thread)
            thread.start()

        last_stats_flush = time.time()
        last_heartbeat = time.time()
        last_indexed_snapshot = 0
        heartbeat_interval = settings.CRAWLER_HEARTBEAT_INTERVAL
        update_interval = max(5, batch_size // max(worker_count, 1))

        try:
            while any(thread.is_alive() for thread in self.worker_threads):
                if not self.running:
                    break
                
                self._handle_control_signal('range')
                if self.stop_requested:
                    break

                now = time.time()
                stats_changed = stats['indexed'] - last_indexed_snapshot >= batch_size
                if (now - last_stats_flush) >= update_interval or stats_changed:
                    self._update_progress(
                        progress_id,
                        stats['current_id'],
                        stats['indexed'],
                        stats['failed'],
                        stats['deleted']
                    )
                    last_stats_flush = now
                    last_indexed_snapshot = stats['indexed']

                if (now - last_heartbeat) >= heartbeat_interval:
                    self._update_progress(
                        progress_id,
                        stats['current_id'],
                        stats['indexed'],
                        stats['failed'],
                        stats['deleted'],
                        heartbeat=True
                    )
                    last_heartbeat = now

                time.sleep(0.5)

            # Ensure all threads exit cleanly
            for thread in self.worker_threads:
                thread.join(timeout=1)

            final_status = 'completed' if self.running else 'stopped'
            self._update_progress(
                progress_id,
                stats['current_id'],
                stats['indexed'],
                stats['failed'],
                stats['deleted'],
                completed=(final_status == 'completed'),
                status=final_status
            )

            logger.info(
                "Crawl finished - Indexed: %s, Failed: %s, Deleted: %s",
                stats['indexed'],
                stats['failed'],
                stats['deleted']
            )

            success = (final_status == 'completed')
            self.running = False
            self.paused = False
            
            # Refresh materialized view for player meta features (performance optimization)
            if success and stats['indexed'] > 0:
                logger.info("Refreshing player meta features materialized view...")
                try:
                    self.db.refresh_player_meta_features(incremental=True)
                    logger.info("✅ Player meta features refreshed successfully")
                except Exception as exc:
                    logger.warning(f"Failed to refresh player meta features: {exc}")

            return {
                'success': success,
                'total_indexed': stats['indexed'],
                'total_failed': stats['failed'],
                'total_deleted': stats['deleted'],
                'last_id': stats['current_id']
            }

        except Exception as e:
            logger.error(f"Crawl error: {str(e)}", exc_info=True)
            self._update_progress(
                progress_id,
                stats['current_id'],
                stats['indexed'],
                stats['failed'],
                stats['deleted'],
                error=str(e),
                status='failed'
            )
            return {
                'success': False,
                'error': str(e),
                'total_indexed': stats['indexed'],
                'total_failed': stats['failed'],
                'total_deleted': stats['deleted']
            }
    
    def _update_progress(
        self,
        progress_id: int,
        current_id: int,
        total_indexed: int,
        total_failed: int,
        total_deleted: int,
        heartbeat: bool = False,
        completed: bool = False,
        error: Optional[str] = None,
        status: Optional[str] = None
    ):
        """Update crawl progress in database"""
        updates = {
            'current_manager_id': current_id,
            'total_indexed': total_indexed,
            'total_failed': total_failed,
            'total_deleted': total_deleted
        }
        
        if completed:
            updates['status'] = 'completed'
            updates['completed_at'] = True  # Will be converted to NOW() in SQL
        elif error:
            updates['status'] = status or 'failed'
            updates['error_message'] = error
        elif status:
            updates['status'] = status
        elif self.paused:
            updates['status'] = 'paused'
        else:
            updates['status'] = 'running'
        
        self.db.update_crawl_progress(progress_id, updates)
        
        if heartbeat:
            logger.info(f"Progress: ID {current_id}, Indexed: {total_indexed}, Failed: {total_failed}, Deleted: {total_deleted}")
    
    def resume_crawl(self) -> Optional[Dict[str, Any]]:
        """Resume interrupted crawl"""
        progress = self.db.get_crawl_progress('range')
        if not progress:
            logger.warning("No crawl progress found to resume")
            return None
        
        if progress['status'] not in ['running', 'paused']:
            logger.warning(f"Cannot resume crawl with status: {progress['status']}")
            return None
        
        start_id = progress['current_manager_id'] or progress['start_manager_id']
        end_id = progress['end_manager_id']
        
        self.current_progress_id = progress['id']
        self.running = True
        self.paused = False
        
        return self.crawl_range(
            start_id=start_id,
            end_id=end_id,
            progress_id=progress['id']
        )
    
    def resume_league_crawl(self, league_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Resume a paused league crawl"""
        progress = None
        if league_id:
            progress = self.db.get_league_progress(league_id)
        if not progress:
            progress = self.db.get_crawl_progress('league')
        if not progress:
            logger.warning("No league crawl progress found")
            return None
        
        metadata = progress.get('metadata') or {}
        target_league_id = metadata.get('league_id')
        if not target_league_id:
            logger.warning("League ID missing from progress metadata")
            return None
        
        start_page = int(metadata.get('current_page', 1))
        max_pages = metadata.get('max_pages')
        max_pages = int(max_pages) if max_pages is not None else None
        
        logger.info(f"Resuming league {target_league_id} from page {start_page} (max_pages: {max_pages or 'unlimited'})")
        
        self.current_progress_id = progress['id']
        self.running = True
        self.paused = False
        
        return self.crawl_league(
            league_id=int(target_league_id),
            max_pages=max_pages,
            start_page=start_page,
            progress_id=progress['id']
        )
    
    def crawl_seed_leagues(self, league_ids: Optional[List[int]] = None, max_pages: Optional[int] = None):
        """Sequentially crawl configured seed leagues"""
        targets = league_ids or settings.MANAGER_SEED_LEAGUES
        if not targets:
            logger.info("No league IDs provided for seed crawl")
            return
        
        for league_id in targets:
            logger.info(f"Starting league crawl for {league_id}")
            self.crawl_league(league_id=league_id, max_pages=max_pages or None)
            if self.stop_requested:
                logger.info("Seed league crawl stopped due to stop request")
                self.stop_requested = False
                break
    
    def pause(self):
        """Pause the crawler"""
        with self.lock:
            self.paused = True
            if self.current_progress_id:
                self.db.update_crawl_progress(self.current_progress_id, {'status': 'paused'})
            logger.info("Crawler paused")
    
    def resume(self):
        """Resume the crawler"""
        with self.lock:
            self.paused = False
            if self.current_progress_id:
                self.db.update_crawl_progress(self.current_progress_id, {'status': 'running'})
            logger.info("Crawler resumed")
    
    def stop(self):
        """Stop the crawler"""
        with self.lock:
            self.running = False
            self.stop_requested = True
            if self.current_progress_id:
                self.db.update_crawl_progress(
                    self.current_progress_id,
                    {'status': 'paused' if self.paused else 'stopped'}
                )
            logger.info("Crawler stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current crawler status"""
        progress = self.db.get_crawl_progress('range')
        if not progress:
            progress = self.db.get_crawl_progress('league')
        
        return {
            'running': self.running,
            'paused': self.paused,
            'progress': progress,
            'manager_count': self.db.get_manager_count()
        }

    def sync_league_top_history(
        self,
        league_id: int,
        rank_limit: int = 10000,
        max_event: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Sync history for managers in a league up to a given rank limit.

        Intended for cohorts like top 10k overall.
        """
        managers = self.db.get_top_managers_in_league(league_id, rank_limit)
        if not managers:
            logger.info(
                "No managers found in league %s up to rank %s", league_id, rank_limit
            )
            return {"success": False, "count": 0}

        logger.info(
            "Syncing history for %s managers in league %s (rank <= %s)",
            len(managers),
            league_id,
            rank_limit,
        )

        synced = 0
        errors = 0
        # Use a simple rate limiter aligned with API client interval
        limiter = RateLimiter(settings.CRAWLER_RATE_LIMIT)

        # For now we do not update crawl_progress; this is a separate maintenance job.
        for row in managers:
            manager_id = row.get("manager_id")
            league_rank = row.get("league_rank")
            if not manager_id:
                continue
            try:
                limiter.wait()
                self.sync_manager_history(manager_id, max_event=max_event)
                # Tag cohort membership (e.g., top10k_overall)
                current_event = None
                # We don't know current GW from Postgres here; leave event_from as 0
                self.db.upsert_manager_cohort(
                    manager_id=manager_id,
                    cohort_name="top10k_overall",
                    event_from=current_event or 0,
                    rank_at_entry=league_rank,
                    event_to=None,
                    meta={"league_id": league_id},
                )
                synced += 1
            except Exception as exc:
                logger.error(
                    "Error syncing history for manager %s: %s", manager_id, exc
                )
                errors += 1

        logger.info(
            "History sync complete for league %s: synced=%s, errors=%s",
            league_id,
            synced,
            errors,
        )
        
        # Refresh materialized view for player meta features (performance optimization)
        if synced > 0:
            logger.info("Refreshing player meta features materialized view...")
            try:
                self.db.refresh_player_meta_features(incremental=True)
                logger.info("✅ Player meta features refreshed successfully")
            except Exception as exc:
                logger.warning(f"Failed to refresh player meta features: {exc}")
        
        return {"success": errors == 0, "count": synced, "errors": errors}

    def crawl_league(
        self,
        league_id: int,
        max_pages: Optional[int] = None,
        start_page: int = 1,
        progress_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Crawl managers from a classic league"""
        batch_size = settings.CRAWLER_BATCH_SIZE
        worker_count = max(1, settings.CRAWLER_MAX_WORKERS)
        rate_limit = max(settings.CRAWLER_RATE_LIMIT, 0.05)
        
        self.running = True
        self.paused = False
        self.stop_requested = False
        self.current_crawl_type = 'league'
        
        manager_queue: Queue = Queue(maxsize=max(batch_size * 5, 500))
        stats_lock = threading.Lock()
        stats = {
            'current_page': start_page,
            'indexed': 0,
            'failed': 0,
            'deleted': 0,
            'queued_managers': 0
        }
        league_name = None
        league_type = 'classic'
        seen_ids = set()
        
        if not progress_id:
            progress_data = {
                'crawl_type': 'league',
                'start_manager_id': None,
                'end_manager_id': None,
                'current_manager_id': None,
                'total_discovered': 0,
                'total_indexed': 0,
                'total_failed': 0,
                'total_deleted': 0,
                'status': 'running',
                'metadata': {
                    'league_id': str(league_id),
                    'current_page': start_page,
                    'max_pages': max_pages
                }
            }
            progress_id = self.db.create_crawl_progress(progress_data)
            self.current_progress_id = progress_id
        else:
            self.current_progress_id = progress_id
            # Immediately update status to 'running' when resuming
            self.db.update_crawl_progress(progress_id, {'status': 'running'})
        
        def producer():
            nonlocal league_name, league_type
            page = start_page
            pages_processed = 0
            while self.running:
                self._handle_control_signal('league')
                if self.stop_requested:
                    break
                if self.paused:
                    time.sleep(0.5)
                    continue
                
                data = self.fpl_client.get_classic_league_page(league_id, page)
                if not data:
                    break
                
                league_meta = data.get('league') or {}
                if league_meta:
                    league_name = league_meta.get('name') or league_name
                    league_type = league_meta.get('league_type') or league_type
                
                results = data.get('standings', {}).get('results', [])
                if not results:
                    break
                
                for row in results:
                    manager_id = row.get('entry')
                    if not manager_id or manager_id in seen_ids:
                        continue
                    seen_ids.add(manager_id)
                    manager_queue.put({
                        'manager_id': manager_id,
                        'league_snapshot': {
                            'id': league_id,
                            'name': league_name,
                            'type': league_type,
                            'rank': row.get('rank'),
                            'total_points': row.get('total')
                        }
                    })
                    with stats_lock:
                        stats['queued_managers'] += 1
                
                page += 1
                pages_processed += 1
                
                with stats_lock:
                    stats['current_page'] = page
                    self.db.update_crawl_progress(
                        progress_id,
                        {
                            'metadata': {
                                'league_id': str(league_id),
                                'current_page': page,
                                'max_pages': max_pages,
                                'league_name': league_name,
                                'queued_managers': stats['queued_managers']
                            }
                        }
                    )
                
                if max_pages and pages_processed >= max_pages:
                    break
            
            if self.running:
                for _ in range(worker_count):
                    manager_queue.put(None)
            else:
                try:
                    while True:
                        manager_queue.get_nowait()
                except Empty:
                    pass
        
        def worker_loop(worker_idx: int):
            rate_limiter = RateLimiter(rate_limit)
            while self.running:
                self._handle_control_signal('league')
                if self.stop_requested:
                    break
                try:
                    item = manager_queue.get(timeout=1)
                except Empty:
                    if not any(thread.is_alive() for thread in producer_threads):
                        break
                    continue
                
                if item is None:
                    break
                
                while self.paused and self.running:
                    time.sleep(0.5)
                
                manager_id = item['manager_id']
                rate_limiter.wait()
                result = self.crawl_manager(manager_id, league_context=[item['league_snapshot']])
                
                with stats_lock:
                    if result['success']:
                        stats['indexed'] += 1
                    elif result['deleted']:
                        stats['deleted'] += 1
                    else:
                        stats['failed'] += 1
                    stats['queued_managers'] = max(stats['queued_managers'] - 1, 0)
                    self.db.update_crawl_progress(
                        progress_id,
                        {
                            'current_manager_id': manager_id,
                            'total_indexed': stats['indexed'],
                            'total_failed': stats['failed'],
                            'total_deleted': stats['deleted'],
                            'metadata': {
                                'league_id': str(league_id),
                                'current_page': stats['current_page'],
                                'league_name': league_name,
                                'queued_managers': stats['queued_managers']
                            }
                        }
                    )
        
        producer_threads: List[threading.Thread] = []
        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_threads.append(producer_thread)
        producer_thread.start()
        
        self.worker_threads = []
        for idx in range(worker_count):
            thread = threading.Thread(target=worker_loop, args=(idx,), daemon=True)
            self.worker_threads.append(thread)
            thread.start()
        
        try:
            producer_thread.join()
            for thread in self.worker_threads:
                thread.join()
            
            self.db.update_crawl_progress(
                progress_id,
                {
                    'status': 'completed' if self.running else 'stopped',
                    'completed_at': self.running,
                    'metadata': {
                        'league_id': str(league_id),
                        'league_name': league_name,
                        'current_page': stats['current_page'],
                        'queued_managers': stats['queued_managers']
                    }
                }
            )
            self.running = False
            self.paused = False
            return {
                'success': True,
                'total_indexed': stats['indexed'],
                'total_failed': stats['failed'],
                'total_deleted': stats['deleted']
            }
        except Exception as exc:
            logger.error(f"League crawl error: {exc}", exc_info=True)
            self.db.update_crawl_progress(
                progress_id,
                {
                    'status': 'failed',
                    'error_message': str(exc)
                }
            )
            return {
                'success': False,
                'error': str(exc),
                'total_indexed': stats['indexed'],
                'total_failed': stats['failed'],
                'total_deleted': stats['deleted']
            }

