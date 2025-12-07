"""CLI tool for managing FPL manager index crawler"""

import argparse
import sys
import logging
from typing import Optional, List

from backend.src.core.postgres_db import PostgresManagerDB
from backend.src.services.manager_crawler import ManagerCrawler
from backend.src.services.fpl_api_client import FPLAPIClient
from backend.src.core.config import get_settings

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CONTROL_SCOPES = ('auto', 'range', 'league')


def send_control_command(action: str, scope: str = 'auto'):
    """Issue a control command that active crawlers can pick up"""
    if scope not in CONTROL_SCOPES:
        raise ValueError(f"Invalid scope: {scope}")
    
    ManagerCrawler.emit_control_signal(action, scope)
    logger.info(f"Issued '{action}' command for scope '{scope}'")


def initialize_database():
    """Initialize PostgreSQL database schema"""
    logger.info("Initializing database schema...")
    db = PostgresManagerDB()
    
    if not db.test_connection():
        logger.error("Failed to connect to PostgreSQL database")
        logger.error(f"Connection string: {settings.POSTGRES_CONNECTION_STRING}")
        return False
    
    if db.initialize_schema():
        logger.info("Database schema initialized successfully")
        return True
    else:
        logger.error("Failed to initialize database schema")
        return False


def start_crawl(start_id: int, end_id: Optional[int] = None, batch_size: Optional[int] = None):
    """Start a new crawl"""
    logger.info(f"Starting crawl from ID {start_id} to {end_id or 'end'}")
    
    crawler = ManagerCrawler()
    result = crawler.crawl_range(
        start_id=start_id,
        end_id=end_id,
        batch_size=batch_size
    )
    
    if result['success']:
        logger.info(f"Crawl completed successfully!")
        logger.info(f"  Indexed: {result['total_indexed']}")
        logger.info(f"  Failed: {result['total_failed']}")
        logger.info(f"  Deleted: {result['total_deleted']}")
        logger.info(f"  Last ID: {result['last_id']}")
        return True
    else:
        logger.error(f"Crawl failed: {result.get('error', 'Unknown error')}")
        return False


def crawl_league(league_id: int, max_pages: Optional[int] = None, start_page: int = 1):
    """Crawl managers from a classic league"""
    logger.info(f"Crawling league {league_id} (pages: {max_pages or 'all'})")
    
    crawler = ManagerCrawler()
    result = crawler.crawl_league(
        league_id=league_id,
        max_pages=max_pages,
        start_page=start_page
    )
    
    if result.get('success'):
        logger.info(
            "League crawl complete | Indexed: %s | Failed: %s | Deleted: %s",
            result['total_indexed'],
            result['total_failed'],
            result['total_deleted']
        )
        return True
    else:
        logger.error(f"League crawl failed: {result.get('error', 'Unknown error')}")
        return False


def seed_leagues(max_pages: Optional[int] = None, league_ids: Optional[List[int]] = None):
    """Crawl configured seed leagues sequentially"""
    crawler = ManagerCrawler()
    crawler.crawl_seed_leagues(league_ids=league_ids, max_pages=max_pages)
    logger.info("Seed league crawl completed")


def resume_league(league_id: Optional[int] = None):
    """Resume a paused league crawl"""
    logger.info("Resuming league crawl...")
    crawler = ManagerCrawler()
    result = crawler.resume_league_crawl(league_id)
    if result:
        if result.get('success'):
            logger.info("League crawl resumed and completed!")
            logger.info(f"  Indexed: {result['total_indexed']}")
            logger.info(f"  Failed: {result['total_failed']}")
            logger.info(f"  Deleted: {result['total_deleted']}")
        else:
            logger.error(f"League crawl failed: {result.get('error', 'Unknown error')}")
    else:
        logger.warning("Unable to resume league crawl - no progress found. Start a new crawl first.")


def resume_crawl():
    """Resume interrupted crawl"""
    logger.info("Resuming crawl...")
    
    crawler = ManagerCrawler()
    result = crawler.resume_crawl()
    
    if result:
        if result.get('success'):
            logger.info(f"Crawl resumed and completed!")
            logger.info(f"  Indexed: {result['total_indexed']}")
            logger.info(f"  Failed: {result['total_failed']}")
            logger.info(f"  Deleted: {result['total_deleted']}")
            return True
        else:
            logger.error(f"Crawl failed: {result.get('error', 'Unknown error')}")
            return False
    else:
        logger.warning("No crawl progress found to resume")
        return False


def get_status(scope: str = 'auto'):
    """Get crawler status"""
    db = PostgresManagerDB()
    
    if scope not in CONTROL_SCOPES:
        raise ValueError(f"Invalid status scope: {scope}")
    
    if scope in ('auto', 'range'):
        progress = db.get_crawl_progress('range')
    else:
        progress = None
    
    if scope in ('league',) or (scope == 'auto' and progress is None):
        progress = db.get_crawl_progress('league')
    
    manager_count = db.get_manager_count()
    
    print("\n=== Crawler Status ===")
    if progress:
        running = progress.get('status') == 'running'
        paused = progress.get('status') == 'paused'
    else:
        running = paused = False
    
    print(f"Running: {running}")
    print(f"Paused: {paused}")
    print(f"Total Managers Indexed: {manager_count}")
    
    if progress:
        print(f"\n=== Current Progress ===")
        print(f"Type: {progress.get('crawl_type', 'N/A')}")
        print(f"Status: {progress.get('status', 'N/A')}")
        if progress.get('crawl_type') == 'league':
            metadata = progress.get('metadata') or {}
            print(f"League ID: {metadata.get('league_id', 'N/A')}")
            print(f"League Name: {metadata.get('league_name', 'N/A')}")
            print(f"Current Page: {metadata.get('current_page', 'N/A')}")
            print(f"Queued Managers: {metadata.get('queued_managers', '0')}")
        else:
            print(f"Start ID: {progress.get('start_manager_id', 'N/A')}")
            print(f"End ID: {progress.get('end_manager_id', 'N/A')}")
            print(f"Current ID: {progress.get('current_manager_id', 'N/A')}")
        print(f"Indexed: {progress.get('total_indexed', 0)}")
        print(f"Failed: {progress.get('total_failed', 0)}")
        print(f"Deleted: {progress.get('total_deleted', 0)}")
        print(f"Started: {progress.get('started_at', 'N/A')}")
        print(f"Last Heartbeat: {progress.get('last_heartbeat', 'N/A')}")
    else:
        print("No active or recent crawl found for requested scope.")
    
    print()


def show_league_summary(league_id: Optional[int] = None):
    """Show which leagues have been crawled and coverage per league."""
    db = PostgresManagerDB()
    
    if league_id is None:
        summaries = db.get_league_summaries()
        if not summaries:
            print("\nNo league memberships indexed yet.\n")
            return
        
        print("\n=== League Index Summary ===")
        for row in summaries:
            progress = db.get_league_progress(row["league_id"])
            status = progress["status"] if progress else "n/a"
            print(
                f"League {row['league_id']}: {row.get('league_name') or 'N/A'} | "
                f"Managers indexed: {row['manager_count']} | "
                f"Last crawl status: {status}"
            )
        print()
    else:
        summary = db.get_league_summary(league_id)
        manager_count = summary["manager_count"] if summary else 0
        league_name = (summary.get("league_name") if summary else None) or "N/A"
        
        progress = db.get_league_progress(league_id)
        status = progress["status"] if progress else "n/a"
        
        # Fetch total entries from FPL API for coverage, if possible
        client = FPLAPIClient()
        api_data = client.get_classic_league_page(league_id, page=1)
        total_entries = None
        if api_data:
            standings = api_data.get("standings") or {}
            total_entries = (
                standings.get("total")
                or standings.get("total_entries")
                or None
            )
        
        print("\n=== League Detail ===")
        print(f"League ID: {league_id}")
        print(f"League Name (DB): {league_name}")
        print(f"Managers indexed in DB: {manager_count}")
        print(f"Last crawl status: {status}")
        
        if total_entries is not None and total_entries > 0:
            coverage = (manager_count / total_entries) * 100.0
            print(f"Total entries (FPL API): {total_entries}")
            print(f"Coverage: {coverage:.2f}% of league managers indexed")
        else:
            print("Total entries (FPL API): unknown (could not fetch)")
        print()


def fix_league_status_cmd(league_id: int, status: str):
    """Fix a league's crawl status row in crawl_progress."""
    db = PostgresManagerDB()
    updated = db.fix_league_status(league_id, status)
    if updated > 0:
        logger.info(
            f"Updated crawl_progress status for league {league_id} "
            f"to '{status}' ({updated} row(s) affected)"
        )
    else:
        logger.warning(
            f"No crawl_progress rows updated for league {league_id}. "
            f"Check that a league crawl exists for this ID."
        )


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="FPL Manager Index Crawler CLI")
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Initialize command
    init_parser = subparsers.add_parser('init', help='Initialize database schema')
    
    # Start crawl command
    start_parser = subparsers.add_parser('start', help='Start a new crawl')
    start_parser.add_argument('--start-id', type=int, default=1, help='Starting manager ID (default: 1)')
    start_parser.add_argument('--end-id', type=int, default=None, help='Ending manager ID (default: unlimited)')
    start_parser.add_argument('--batch-size', type=int, default=None, help='Batch size (default: from config)')
    
    # Resume (control) command
    resume_control_parser = subparsers.add_parser('resume', help='Resume a paused crawl via control signal')
    resume_control_parser.add_argument('--scope', choices=CONTROL_SCOPES, default='auto', help='Which crawl scope to resume (default: auto)')
    
    # Resume range crawl command (legacy)
    resume_range_parser = subparsers.add_parser('resume-range', help='Resume interrupted range crawl')
    
    # Pause command
    pause_parser = subparsers.add_parser('pause', help='Pause active crawl')
    pause_parser.add_argument('--scope', choices=CONTROL_SCOPES, default='auto', help='Which crawl scope to pause (default: auto)')
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop active crawl')
    stop_parser.add_argument('--scope', choices=CONTROL_SCOPES, default='auto', help='Which crawl scope to stop (default: auto)')
    
    # League crawl command
    league_parser = subparsers.add_parser('league', help='Crawl a specific classic league')
    league_parser.add_argument('--league-id', type=int, required=True, help='League ID to crawl')
    league_parser.add_argument('--max-pages', type=int, default=None, help='Max pages to crawl')
    league_parser.add_argument('--start-page', type=int, default=1, help='Page to start from')
    
    # Seed leagues command
    seed_parser = subparsers.add_parser('seed', help='Crawl configured seed leagues')
    seed_parser.add_argument('--max-pages', type=int, default=None, help='Max pages per league')
    seed_parser.add_argument('--league-ids', type=str, default=None, help='Comma-separated league IDs to crawl (default: all configured leagues)')
    
    # League summary command
    league_summary_parser = subparsers.add_parser('league-summary', help='Show league index coverage')
    league_summary_parser.add_argument('--league-id', type=int, default=None, help='Optional league ID to inspect in detail')
    
    # Fix league status command
    fix_status_parser = subparsers.add_parser('fix-league-status', help='Fix crawl status for a given league')
    fix_status_parser.add_argument('--league-id', type=int, required=True, help='League ID to update')
    fix_status_parser.add_argument(
        '--status',
        type=str,
        default='completed',
        choices=['running', 'paused', 'completed', 'stopped', 'failed'],
        help="New status value (default: 'completed')",
    )
    
    # Sync league history command
    sync_history_parser = subparsers.add_parser(
        'sync-league-history', help='Sync manager history/picks for a league cohort'
    )
    sync_history_parser.add_argument(
        '--league-id',
        type=int,
        default=314,
        help='Classic league ID to sync from (default: 314 overall league)',
    )
    sync_history_parser.add_argument(
        '--rank-limit',
        type=int,
        default=10000,
        help='Maximum league rank to include (default: 10000)',
    )
    sync_history_parser.add_argument(
        '--max-event',
        type=int,
        default=None,
        help='Optional maximum gameweek to sync (default: all in history)',
    )
    # Resume league command
    resume_league_parser = subparsers.add_parser('resume-league', help='Resume last league crawl')
    resume_league_parser.add_argument('--league-id', type=int, default=None, help='League ID to resume (optional)')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Get crawler status')
    status_parser.add_argument('--scope', choices=CONTROL_SCOPES, default='auto', help='Scope to inspect (default: auto)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'init':
            success = initialize_database()
            sys.exit(0 if success else 1)
        
        elif args.command == 'start':
            success = start_crawl(
                start_id=args.start_id,
                end_id=args.end_id,
                batch_size=args.batch_size
            )
            sys.exit(0 if success else 1)
        
        elif args.command == 'resume':
            send_control_command('resume', args.scope)
            sys.exit(0)
        
        elif args.command == 'resume-range':
            success = resume_crawl()
            sys.exit(0 if success else 1)
        
        elif args.command == 'pause':
            send_control_command('pause', args.scope)
            sys.exit(0)
        
        elif args.command == 'stop':
            send_control_command('stop', args.scope)
            sys.exit(0)
        
        elif args.command == 'league':
            success = crawl_league(
                league_id=args.league_id,
                max_pages=args.max_pages,
                start_page=args.start_page
            )
            sys.exit(0 if success else 1)
        
        elif args.command == 'seed':
            league_ids = None
            if args.league_ids:
                league_ids = [int(id.strip()) for id in args.league_ids.split(',') if id.strip().isdigit()]
            seed_leagues(max_pages=args.max_pages, league_ids=league_ids)
            sys.exit(0)
        
        elif args.command == 'resume-league':
            resume_league(league_id=args.league_id)
            sys.exit(0)
        
        elif args.command == 'status':
            get_status(scope=args.scope)
            sys.exit(0)
        
        elif args.command == 'league-summary':
            show_league_summary(league_id=args.league_id)
            sys.exit(0)
        
        elif args.command == 'fix-league-status':
            fix_league_status_cmd(league_id=args.league_id, status=args.status)
            sys.exit(0)
        
        elif args.command == 'sync-league-history':
            crawler = ManagerCrawler()
            result = crawler.sync_league_top_history(
                league_id=args.league_id,
                rank_limit=args.rank_limit,
                max_event=args.max_event,
            )
            success = result.get('success', False)
            if success:
                logger.info(
                    "Synced history for %s managers (errors=%s)",
                    result.get('count', 0),
                    result.get('errors', 0),
                )
            else:
                logger.error(
                    "History sync finished with errors (synced=%s, errors=%s)",
                    result.get('count', 0),
                    result.get('errors', 0),
                )
            sys.exit(0 if success else 1)
        
        else:
            parser.print_help()
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

