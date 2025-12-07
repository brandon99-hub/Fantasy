"""Sync manager history with real-time progress display"""

import logging
import time
from backend.src.services.manager_crawler import ManagerCrawler, RateLimiter
from backend.src.core.postgres_db import PostgresManagerDB
from backend.src.core.config import get_settings

settings = get_settings()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    crawler = ManagerCrawler()
    db = PostgresManagerDB()
    
    # Get managers to sync (top 10k from league 314)
    logger.info("Fetching managers to sync...")
    managers = db.get_top_managers_in_league(league_id=314, rank_limit=10000)
    
    if not managers:
        logger.error("No managers found! Run league crawl first.")
        return
    
    total_managers = len(managers)
    logger.info(f"Found {total_managers} managers to sync")
    
    # Check how many already have history - simple count
    logger.info("Checking sync status...")
    with db.get_connection() as conn:
        cursor = conn.cursor()
        # Just count how many of our managers have any gameweek data
        manager_ids = [m.get('manager_id') for m in managers if m.get('manager_id')]
        cursor.execute("""
            SELECT COUNT(DISTINCT manager_id) 
            FROM manager_gameweeks 
            WHERE manager_id = ANY(%s)
        """, (manager_ids,))
        row = cursor.fetchone()
        synced_count = int(row['count']) if row and 'count' in row else 0
    
    logger.info(f"Already synced: {synced_count}/{total_managers} ({synced_count/total_managers*100:.1f}%)")
    logger.info(f"Remaining: {total_managers - synced_count}")
    
    # Start syncing
    input(f"\nPress Enter to start syncing {total_managers - synced_count} managers...")
    
    logger.info(f"\nStarting sync with {settings.CRAWLER_MAX_WORKERS} workers at {1/settings.CRAWLER_RATE_LIMIT:.1f} req/sec each")
    logger.info("Progress will be shown below:\n")
    
    # Manual sync with progress
    synced = 0
    errors = 0
    skipped = 0
    limiter = RateLimiter(settings.CRAWLER_RATE_LIMIT)
    start_time = time.time()
    
    for i, row in enumerate(managers):
        manager_id = row.get('manager_id')
        if not manager_id:
            continue
        
        # Log first few managers individually
        if i < 5:
            logger.info(f"Syncing manager {i+1}/{total_managers}: ID {manager_id}...")
        
        try:
            limiter.wait()
            crawler.sync_manager_history(manager_id, max_event=None)
            
            # Tag cohort membership
            league_rank = row.get('league_rank')
            db.upsert_manager_cohort(
                manager_id=manager_id,
                cohort_name="top10k_overall",
                event_from=0,
                rank_at_entry=league_rank,
                event_to=None,
                meta={"league_id": 314},
            )
            synced += 1
            
            # Show progress every 10 managers
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = synced / elapsed if elapsed > 0 else 0
                remaining = total_managers - (i + 1)
                eta_seconds = remaining / rate if rate > 0 else 0
                eta_minutes = eta_seconds / 60
                
                logger.info(
                    f"Progress: {i+1}/{total_managers} ({(i+1)/total_managers*100:.1f}%) | "
                    f"Synced: {synced} | Errors: {errors} | "
                    f"Rate: {rate:.1f}/sec | ETA: {eta_minutes:.1f} min"
                )
                
        except Exception as exc:
            logger.error(f"Error syncing manager {manager_id}: {exc}")
            errors += 1
    
    elapsed_total = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Sync Complete!")
    logger.info(f"Total time: {elapsed_total/60:.1f} minutes")
    logger.info(f"Synced: {synced}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Average rate: {synced/(elapsed_total) if elapsed_total > 0 else 0:.1f} managers/sec")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()
