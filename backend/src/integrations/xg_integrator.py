"""
xG Integrator - Main integration logic for xG/xA data

Coordinates data collection from multiple sources (FBRef, Fotmob, estimation)
with fallback logic and database persistence.
"""

import logging
import asyncio
from typing import Optional, Dict
from datetime import datetime, timedelta
import pandas as pd

from backend.src.integrations.fbref_scraper import FBRefScraper
from backend.src.integrations.xg_estimator import XGEstimator
from backend.src.core.async_db import AsyncDatabaseWrapper
from backend.src.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class XGIntegrator:
    """
    Integrate xG/xA data from multiple sources with intelligent fallback
    
    Priority order:
    1. FBRef (most accurate, free)
    2. Fotmob (backup, real-time)
    3. Estimation (fallback)
    """
    
    def __init__(self, db: AsyncDatabaseWrapper):
        self.db = db
        self.fbref_scraper = None
        self.estimator = XGEstimator()
        self.last_update = None
        logger.info("xG Integrator initialized")
    
    async def enrich_player_data(
        self, 
        players_df: pd.DataFrame,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Add xG/xA data to player dataframe
        
        Args:
            players_df: FPL players DataFrame
            force_refresh: Force refresh even if recently updated
        
        Returns:
            DataFrame with xG/xA columns added
        """
        # Check if we need to refresh
        if not force_refresh and self._is_cache_valid():
            logger.info("Using cached xG data")
            return await self._get_cached_xg_data(players_df)
        
        logger.info("Refreshing xG data from sources...")
        
        # Try FBRef first
        xg_data = await self._fetch_from_fbref(players_df)
        
        # If FBRef failed or incomplete, try Fotmob
        if xg_data is None or len(xg_data) < len(players_df) * 0.5:
            logger.warning("FBRef data insufficient, trying Fotmob...")
            fotmob_data = await self._fetch_from_fotmob(players_df)
            if fotmob_data is not None:
                xg_data = self._merge_xg_sources(xg_data, fotmob_data)
        
        # Fill remaining with estimates
        enriched_df = self._apply_xg_data(players_df, xg_data)
        enriched_df = self._fill_missing_with_estimates(enriched_df)
        
        # Save to database
        await self._save_xg_data(enriched_df)
        
        self.last_update = datetime.now()
        logger.info(f"xG enrichment complete for {len(enriched_df)} players")
        
        return enriched_df
    
    async def _fetch_from_fbref(self, players_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Fetch xG data from FBRef"""
        try:
            if not getattr(settings, 'FBREF_ENABLED', True):
                logger.info("FBRef disabled in settings")
                return None
            
            async with FBRefScraper() as scraper:
                # Scrape data
                fbref_df = await scraper.scrape_player_xg()
                
                if fbref_df.empty:
                    logger.warning("FBRef scraping returned no data")
                    return None
                
                # Match to FPL players
                matched_df = scraper.match_player_names(fbref_df, players_df)
                
                logger.info(f"Fetched xG data for {len(matched_df)} players from FBRef")
                return matched_df
        
        except Exception as e:
            logger.error(f"Error fetching from FBRef: {e}")
            return None
    
    async def _fetch_from_fotmob(self, players_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Fetch xG data from Fotmob (placeholder for future implementation)"""
        # TODO: Implement Fotmob client
        logger.info("Fotmob integration not yet implemented")
        return None
    
    def _merge_xg_sources(
        self, 
        primary: Optional[pd.DataFrame], 
        secondary: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Merge xG data from multiple sources, preferring primary"""
        if primary is None:
            return secondary if secondary is not None else pd.DataFrame()
        if secondary is None:
            return primary
        
        # Combine, preferring primary source
        merged = primary.copy()
        
        # Add players from secondary that aren't in primary
        secondary_only = secondary[~secondary['player_id'].isin(primary['player_id'])]
        merged = pd.concat([merged, secondary_only], ignore_index=True)
        
        logger.info(f"Merged xG data: {len(merged)} total players")
        return merged
    
    def _apply_xg_data(
        self, 
        players_df: pd.DataFrame, 
        xg_data: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Apply xG data to players dataframe"""
        df = players_df.copy()
        
        # Initialize xG columns
        df['xg'] = 0.0
        df['xa'] = 0.0
        df['xg_per_90'] = 0.0
        df['xa_per_90'] = 0.0
        df['xg_source'] = 'none'
        
        if xg_data is None or xg_data.empty:
            return df
        
        # Merge xG data
        xg_cols = ['player_id', 'xg', 'xa', 'xg_per_90', 'xa_per_90', 'source']
        xg_subset = xg_data[xg_cols].copy()
        xg_subset = xg_subset.rename(columns={'source': 'xg_source'})
        
        df = df.merge(xg_subset, on='player_id', how='left', suffixes=('', '_xg'))
        
        # Fill NaN with 0
        df['xg'] = df['xg'].fillna(0)
        df['xa'] = df['xa'].fillna(0)
        df['xg_per_90'] = df['xg_per_90'].fillna(0)
        df['xa_per_90'] = df['xa_per_90'].fillna(0)
        df['xg_source'] = df['xg_source'].fillna('none')
        
        return df
    
    def _fill_missing_with_estimates(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing xG data with estimates"""
        df = players_df.copy()
        
        # Find players without xG data
        missing_xg = df['xg_source'] == 'none'
        
        if missing_xg.sum() == 0:
            logger.info("No missing xG data to estimate")
            return df
        
        logger.info(f"Estimating xG for {missing_xg.sum()} players...")
        
        # Estimate for missing players
        missing_df = df[missing_xg].copy()
        estimated_df = self.estimator.estimate_xg(missing_df)
        
        # Update main dataframe
        df.loc[missing_xg, 'xg'] = estimated_df['estimated_xg'].values
        df.loc[missing_xg, 'xa'] = estimated_df['estimated_xa'].values
        df.loc[missing_xg, 'xg_per_90'] = estimated_df['estimated_xg_per_90'].values
        df.loc[missing_xg, 'xa_per_90'] = estimated_df['estimated_xa_per_90'].values
        df.loc[missing_xg, 'xg_source'] = 'estimated'
        
        return df
    
    async def _save_xg_data(self, players_df: pd.DataFrame):
        """Save xG data to database"""
        try:
            # Update players table with xG data
            xg_cols = ['id', 'xg', 'xa', 'xg_per_90', 'xa_per_90', 'xg_source']
            xg_df = players_df[xg_cols].copy()
            xg_df = xg_df.rename(columns={'id': 'player_id'})
            
            # Use upsert to update existing players
            await self.db.upsert_players(xg_df)
            
            logger.info(f"Saved xG data for {len(xg_df)} players to database")
        
        except Exception as e:
            logger.error(f"Error saving xG data: {e}")
    
    async def _get_cached_xg_data(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Get cached xG data from database"""
        # Players already have xG data from database
        return players_df
    
    def _is_cache_valid(self, max_age_hours: int = 24) -> bool:
        """Check if cached xG data is still valid"""
        if self.last_update is None:
            return False
        
        age = datetime.now() - self.last_update
        return age < timedelta(hours=max_age_hours)
    
    async def update_xg_data_background(self):
        """Background task to periodically update xG data"""
        while True:
            try:
                logger.info("Starting background xG update...")
                
                # Get all players
                players_df = await self.db.get_players_with_stats()
                
                # Enrich with xG data
                await self.enrich_player_data(players_df, force_refresh=True)
                
                logger.info("Background xG update complete")
                
                # Wait 24 hours before next update
                await asyncio.sleep(24 * 3600)
            
            except Exception as e:
                logger.error(f"Error in background xG update: {e}")
                # Wait 1 hour before retry
                await asyncio.sleep(3600)


# Example usage
async def main():
    """Test the integrator"""
    from backend.src.core.db_factory import get_db
    
    db = get_db()
    integrator = XGIntegrator(db)
    
    # Get players
    players_df = await db.get_players_with_stats()
    
    # Enrich with xG
    enriched_df = await integrator.enrich_player_data(players_df)
    
    # Show results
    print(f"\nxG Data Summary:")
    print(enriched_df['xg_source'].value_counts())
    print(f"\nSample players with xG:")
    print(enriched_df[['web_name', 'position', 'xg', 'xa', 'xg_per_90', 'xg_source']].head(10))


if __name__ == "__main__":
    asyncio.run(main())
