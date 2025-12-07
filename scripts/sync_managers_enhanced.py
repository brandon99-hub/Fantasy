"""
Enhanced Manager Sync with Resume Capability and Faster Processing

Features:
- Resume from last checkpoint
- Parallel processing with multiple workers
- Progress tracking with ETA
- Automatic checkpointing
- Graceful shutdown handling
"""

import logging
import time
import json
import signal
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from backend.src.services.manager_crawler import ManagerCrawler, RateLimiter
from backend.src.core.postgres_db import PostgresManagerDB
from backend.src.core.config import get_settings

settings = get_settings()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Checkpoint file
CHECKPOINT_FILE = Path(__file__).parent.parent / "backend" / "data" / "manager_sync_checkpoint.json"


class EnhancedManagerSync:
    """Enhanced manager sync with resume capability"""
    
    def __init__(self):
        self.crawler = ManagerCrawler()
        self.db = PostgresManagerDB()
        self.checkpoint = self._load_checkpoint()
        self.should_stop = False
        
        # Statistics
        self.stats = {
            'synced': 0,
            'errors': 0,
            'skipped': 0,
            'start_time': time.time()
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.warning("\n\n⚠️  Shutdown signal received. Saving checkpoint...")
        self.should_stop = True
        self._save_checkpoint()
        logger.info("✅ Checkpoint saved. You can resume later.")
        sys.exit(0)
    
    def _load_checkpoint(self):
        """Load checkpoint from file"""
        if CHECKPOINT_FILE.exists():
            try:
                with open(CHECKPOINT_FILE, 'r') as f:
                    checkpoint = json.load(f)
                logger.info(f"📂 Loaded checkpoint: {checkpoint['processed']}/{checkpoint['total']} managers processed")
                return checkpoint
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        
        return {
            'processed': 0,
            'total': 0,
            'synced_ids': [],
            'last_updated': None
        }
    
    def _save_checkpoint(self):
        """Save checkpoint to file"""
        try:
            CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.checkpoint['last_updated'] = datetime.now().isoformat()
            
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(self.checkpoint, f, indent=2)
            
            logger.debug(f"Checkpoint saved: {self.checkpoint['processed']}/{self.checkpoint['total']}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _sync_single_manager(self, manager_data):
        """Sync a single manager (for parallel processing)"""
        manager_id = manager_data.get('manager_id')
        
        if not manager_id:
            return {'status': 'skipped', 'manager_id': None}
        
        # Skip if already synced
        if manager_id in self.checkpoint.get('synced_ids', []):
            return {'status': 'skipped', 'manager_id': manager_id}
        
        try:
            # Sync manager history
            self.crawler.sync_manager_history(manager_id, max_event=None)
            
            # Tag cohort membership
            league_rank = manager_data.get('league_rank')
            self.db.upsert_manager_cohort(
                manager_id=manager_id,
                cohort_name="top10k_overall",
                event_from=0,
                rank_at_entry=league_rank,
                event_to=None,
                meta={"league_id": 314},
            )
            
            return {'status': 'success', 'manager_id': manager_id}
            
        except Exception as exc:
            logger.error(f"Error syncing manager {manager_id}: {exc}")
            return {'status': 'error', 'manager_id': manager_id, 'error': str(exc)}
    
    def sync_managers(self, max_workers=None):
        """
        Sync managers with parallel processing
        
        Args:
            max_workers: Number of parallel workers (default: from settings)
        """
        # Get managers to sync
        logger.info("📥 Fetching managers to sync...")
        managers = self.db.get_top_managers_in_league(league_id=314, rank_limit=10000)
        
        if not managers:
            logger.error("❌ No managers found! Run league crawl first.")
            return
        
        total_managers = len(managers)
        self.checkpoint['total'] = total_managers
        
        # Filter out already synced
        remaining_managers = [
            m for m in managers 
            if m.get('manager_id') not in self.checkpoint.get('synced_ids', [])
        ]
        
        logger.info(f"📊 Total managers: {total_managers}")
        logger.info(f"✅ Already synced: {len(self.checkpoint.get('synced_ids', []))}")
        logger.info(f"⏳ Remaining: {len(remaining_managers)}")
        
        if not remaining_managers:
            logger.info("🎉 All managers already synced!")
            return
        
        # Confirm start
        input(f"\n▶️  Press Enter to start syncing {len(remaining_managers)} managers...")
        
        # Determine workers
        if max_workers is None:
            max_workers = min(settings.CRAWLER_MAX_WORKERS, 12)  # Cap at 12
        
        logger.info(f"\n🚀 Starting parallel sync with {max_workers} workers")
        logger.info(f"⚡ Rate limit: {1/settings.CRAWLER_RATE_LIMIT:.1f} req/sec per worker")
        logger.info(f"📈 Total throughput: ~{max_workers * (1/settings.CRAWLER_RATE_LIMIT):.1f} req/sec\n")
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_manager = {
                executor.submit(self._sync_single_manager, manager): manager
                for manager in remaining_managers
            }
            
            # Process results as they complete
            for i, future in enumerate(as_completed(future_to_manager)):
                if self.should_stop:
                    logger.warning("Stopping due to shutdown signal...")
                    break
                
                result = future.result()
                
                # Update statistics
                if result['status'] == 'success':
                    self.stats['synced'] += 1
                    self.checkpoint['synced_ids'].append(result['manager_id'])
                elif result['status'] == 'error':
                    self.stats['errors'] += 1
                else:
                    self.stats['skipped'] += 1
                
                self.checkpoint['processed'] = len(self.checkpoint['synced_ids'])
                
                # Save checkpoint every 50 managers
                if (i + 1) % 50 == 0:
                    self._save_checkpoint()
                
                # Show progress every 10 managers
                if (i + 1) % 10 == 0:
                    self._print_progress(i + 1, len(remaining_managers))
        
        # Final checkpoint
        self._save_checkpoint()
        
        # Print summary
        self._print_summary()
    
    def _print_progress(self, current, total):
        """Print progress update"""
        elapsed = time.time() - self.stats['start_time']
        rate = self.stats['synced'] / elapsed if elapsed > 0 else 0
        remaining = total - current
        eta_seconds = remaining / rate if rate > 0 else 0
        eta_minutes = eta_seconds / 60
        
        logger.info(
            f"📊 Progress: {current}/{total} ({current/total*100:.1f}%) | "
            f"✅ Synced: {self.stats['synced']} | "
            f"❌ Errors: {self.stats['errors']} | "
            f"⚡ Rate: {rate:.1f}/sec | "
            f"⏱️  ETA: {eta_minutes:.1f} min"
        )
    
    def _print_summary(self):
        """Print final summary"""
        elapsed_total = time.time() - self.stats['start_time']
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🎉 Sync Complete!")
        logger.info(f"{'='*70}")
        logger.info(f"⏱️  Total time: {elapsed_total/60:.1f} minutes ({elapsed_total/3600:.2f} hours)")
        logger.info(f"✅ Synced: {self.stats['synced']}")
        logger.info(f"❌ Errors: {self.stats['errors']}")
        logger.info(f"⏭️  Skipped: {self.stats['skipped']}")
        logger.info(f"⚡ Average rate: {self.stats['synced']/elapsed_total if elapsed_total > 0 else 0:.1f} managers/sec")
        logger.info(f"{'='*70}\n")
        
        # Clear checkpoint if complete
        if self.checkpoint['processed'] >= self.checkpoint['total']:
            logger.info("🗑️  Clearing checkpoint (sync complete)")
            if CHECKPOINT_FILE.exists():
                CHECKPOINT_FILE.unlink()


def main():
    """Main entry point"""
    sync = EnhancedManagerSync()
    
    # Use configured max workers (or override here)
    sync.sync_managers(max_workers=None)  # None = use settings


if __name__ == "__main__":
    main()
