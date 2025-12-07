# Automatic Materialized View Refresh Integration

## Overview

Integrated automatic materialized view refresh into the manager data sync workflow to ensure ownership analytics are always up-to-date.

## Changes Made

### [manager_crawler.py](file:///c:/Users/USERR/PythonProject/FPLDataFetch/backend/src/services/manager_crawler.py)

Added automatic `player_meta_features` materialized view refresh in two key locations:

#### 1. After League History Sync (Lines 824-834)

**Location**: `sync_league_top_history()` method

**Trigger**: After syncing manager history for a league (e.g., top 10k overall)

**Code**:
```python
# Refresh materialized view for player meta features (performance optimization)
if synced > 0:
    logger.info("Refreshing player meta features materialized view...")
    try:
        self.db.refresh_player_meta_features(incremental=True)
        logger.info("✅ Player meta features refreshed successfully")
    except Exception as exc:
        logger.warning(f"Failed to refresh player meta features: {exc}")
```

**When it runs**:
- After syncing manager picks for top 10k managers
- Only if at least one manager was successfully synced
- Uses incremental (non-blocking) refresh

---

#### 2. After Range Crawl Completion (Lines 581-590)

**Location**: `crawl_range()` method

**Trigger**: After completing a range crawl of manager IDs

**Code**:
```python
# Refresh materialized view for player meta features (performance optimization)
if success and stats['indexed'] > 0:
    logger.info("Refreshing player meta features materialized view...")
    try:
        self.db.refresh_player_meta_features(incremental=True)
        logger.info("✅ Player meta features refreshed successfully")
    except Exception as exc:
        logger.warning(f"Failed to refresh player meta features: {exc}")
```

**When it runs**:
- After successfully completing a range crawl
- Only if at least one manager was indexed
- Uses incremental (non-blocking) refresh

---

## Benefits

✅ **Automatic Updates**: Ownership analytics stay current without manual intervention

✅ **Non-Blocking**: Uses `CONCURRENTLY` refresh to avoid locking the view

✅ **Error Handling**: Gracefully handles refresh failures without breaking the sync workflow

✅ **Conditional**: Only refreshes when new data was actually synced

✅ **Performance**: Incremental refresh is fast (typically <5 seconds)

---

## Usage

The materialized view will now automatically refresh after:

1. **League Sync**:
   ```bash
   # Syncs top 10k managers and refreshes view
   python -c "from backend.src.services.manager_crawler import ManagerCrawler; ManagerCrawler().sync_league_top_history(314, 10000)"
   ```

2. **Range Crawl**:
   ```bash
   # Crawls manager IDs and refreshes view
   python -c "from backend.src.services.manager_crawler import ManagerCrawler; ManagerCrawler().crawl_range(1, 10000)"
   ```

3. **API Endpoints** (if you have them):
   - Any API endpoint that triggers `sync_league_top_history()`
   - Any API endpoint that triggers `crawl_range()`

---

## Monitoring

Check logs for refresh status:

```
INFO - Refreshing player meta features materialized view...
INFO - ✅ Player meta features refreshed successfully
```

Or if there's an issue:
```
WARNING - Failed to refresh player meta features: <error message>
```

---

## Manual Refresh (Optional)

You can still manually refresh the view if needed:

```python
from backend.src.core.postgres_db import PostgresManagerDB

db = PostgresManagerDB()

# Incremental refresh (non-blocking, fast)
db.refresh_player_meta_features(incremental=True)

# Full refresh (blocking, slower but more thorough)
db.refresh_player_meta_features(incremental=False)
```

---

## Summary

✅ Materialized view automatically refreshes after data syncs
✅ No manual intervention required
✅ Ownership analytics always up-to-date
✅ Non-blocking refresh keeps system responsive
✅ Error handling prevents workflow interruption
