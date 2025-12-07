"""Benchmark script for manager data aggregation performance"""

import time
import logging
import pandas as pd
from backend.src.core.manager_training_data import ManagerTrainingDataBuilder
from backend.src.core.postgres_db import PostgresManagerDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark_aggregation():
    """Benchmark player meta features aggregation"""
    
    db = PostgresManagerDB()
    builder = ManagerTrainingDataBuilder(db)
    
    # Test 1: SQL-based aggregation (no materialized view)
    logger.info("=" * 60)
    logger.info("Test 1: SQL-based aggregation")
    logger.info("=" * 60)
    start = time.time()
    features_sql = builder.build_player_meta_features(use_materialized_view=False)
    elapsed_sql = time.time() - start
    
    logger.info(f"Time: {elapsed_sql:.2f}s")
    logger.info(f"Records: {len(features_sql):,}")
    logger.info(f"Speedup vs baseline (300s): {300/elapsed_sql:.1f}x")
    
    # Test 2: Materialized view (first time - with refresh)
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Materialized view (with refresh)")
    logger.info("=" * 60)
    start = time.time()
    features_mv_refresh = builder.build_player_meta_features(force_refresh=True)
    elapsed_mv_refresh = time.time() - start
    
    logger.info(f"Time: {elapsed_mv_refresh:.2f}s")
    logger.info(f"Records: {len(features_mv_refresh):,}")
    logger.info(f"Speedup vs baseline (300s): {300/elapsed_mv_refresh:.1f}x")
    
    # Test 3: Materialized view (cached)
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Materialized view (cached)")
    logger.info("=" * 60)
    start = time.time()
    features_mv_cached = builder.build_player_meta_features(force_refresh=False)
    elapsed_mv_cached = time.time() - start
    
    logger.info(f"Time: {elapsed_mv_cached:.2f}s")
    logger.info(f"Records: {len(features_mv_cached):,}")
    logger.info(f"Speedup vs baseline (300s): {300/elapsed_mv_cached:.1f}x")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"SQL Aggregation:             {elapsed_sql:.2f}s ({300/elapsed_sql:.1f}x speedup)")
    logger.info(f"Materialized View (refresh): {elapsed_mv_refresh:.2f}s ({300/elapsed_mv_refresh:.1f}x speedup)")
    logger.info(f"Materialized View (cached):  {elapsed_mv_cached:.2f}s ({300/elapsed_mv_cached:.1f}x speedup)")
    
    # Data accuracy check
    logger.info("\n" + "=" * 60)
    logger.info("DATA ACCURACY CHECK")
    logger.info("=" * 60)
    logger.info(f"SQL records: {len(features_sql):,}")
    logger.info(f"MV records:  {len(features_mv_cached):,}")
    logger.info(f"Match: {len(features_sql) == len(features_mv_cached)}")
    
    # Sample comparison if both have data
    if not features_sql.empty and not features_mv_cached.empty:
        merged = features_sql.merge(
            features_mv_cached, 
            on=['player_id', 'event'], 
            suffixes=('_sql', '_mv')
        )
        if not merged.empty and len(merged) > 0:
            # Ensure numeric types for correlation
            try:
                sql_col = pd.to_numeric(merged['top_cohort_ownership_pct_sql'], errors='coerce')
                mv_col = pd.to_numeric(merged['top_cohort_ownership_pct_mv'], errors='coerce')
                ownership_corr = sql_col.corr(mv_col)
                logger.info(f"Ownership correlation: {ownership_corr:.6f}")
                logger.info(f"✅ Data accuracy verified" if ownership_corr > 0.999 else "❌ Data mismatch detected")
            except Exception as e:
                logger.warning(f"Could not calculate correlation: {e}")

if __name__ == "__main__":
    benchmark_aggregation()
