"""
Cache warming module for FPL Optimizer

Pre-computes and caches expensive operations before they're needed,
especially useful before gameweek deadlines.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from backend.src.core.cache import get_cache, CACHE_TTL_STRATEGY
from backend.src.core.db_factory import get_db

logger = logging.getLogger(__name__)


class CacheWarmer:
    """Handles cache warming operations"""
    
    def __init__(self):
        self.cache = get_cache()
        self.db = get_db()
        logger.info("Cache warmer initialized")
    
    async def warm_players_cache(self) -> int:
        """Warm cache with player data"""
        try:
            logger.info("🔥 Warming players cache...")
            
            # Get all players with stats
            players = self.db.get_players_with_stats()
            
            if not players:
                logger.warning("No players found to cache")
                return 0
            
            # Cache all players
            cache_key = "players:all"
            self.cache.set(cache_key, players)
            
            # Cache individual players
            count = 0
            player_mapping = {}
            for player in players:
                player_id = player.get('id')
                if player_id:
                    player_mapping[f"players:{player_id}"] = player
                    count += 1
            
            # Batch set
            self.cache.set_many(player_mapping)
            
            logger.info(f"✅ Warmed {count} players in cache")
            return count
            
        except Exception as e:
            logger.error(f"Error warming players cache: {e}")
            return 0
    
    async def warm_fixtures_cache(self, gameweeks: int = 5) -> int:
        """Warm cache with upcoming fixtures"""
        try:
            logger.info(f"🔥 Warming fixtures cache (next {gameweeks} GWs)...")
            
            # Get current gameweek
            current_gw = self.db.get_current_gameweek()
            if not current_gw:
                logger.warning("No current gameweek found")
                return 0
            
            # Get fixtures for next N gameweeks
            with self.db.get_connection() as conn:
                fixtures_df = conn.execute(f"""
                    SELECT * FROM fixtures
                    WHERE event >= {current_gw}
                    AND event <= {current_gw + gameweeks}
                    ORDER BY event, kickoff_time
                """).fetchdf()
            
            if fixtures_df is None or len(fixtures_df) == 0:
                logger.warning("No fixtures found to cache")
                return 0
            
            # Cache fixtures by gameweek
            fixtures_by_gw = {}
            for gw in range(current_gw, current_gw + gameweeks + 1):
                gw_fixtures = fixtures_df[fixtures_df['event'] == gw].to_dict('records')
                if gw_fixtures:
                    fixtures_by_gw[f"fixtures:gw:{gw}"] = gw_fixtures
            
            # Batch set
            count = self.cache.set_many(fixtures_by_gw)
            
            # Also cache all upcoming fixtures
            self.cache.set("fixtures:upcoming", fixtures_df.to_dict('records'))
            
            logger.info(f"✅ Warmed {count} gameweeks of fixtures in cache")
            return count
            
        except Exception as e:
            logger.error(f"Error warming fixtures cache: {e}")
            return 0
    
    async def warm_predictions_cache(self, gameweek: Optional[int] = None) -> int:
        """Warm cache with player predictions"""
        try:
            if gameweek is None:
                gameweek = self.db.get_current_gameweek()
            
            if not gameweek:
                logger.warning("No gameweek specified for predictions")
                return 0
            
            logger.info(f"🔥 Warming predictions cache for GW{gameweek}...")
            
            # Get predictions from database
            with self.db.get_connection() as conn:
                predictions_df = conn.execute(f"""
                    SELECT * FROM predictions
                    WHERE gameweek = {gameweek}
                    ORDER BY predicted_points DESC
                """).fetchdf()
            
            if predictions_df is None or len(predictions_df) == 0:
                logger.warning(f"No predictions found for GW{gameweek}")
                return 0
            
            # Cache all predictions
            cache_key = f"predictions:gw:{gameweek}"
            self.cache.set(cache_key, predictions_df.to_dict('records'))
            
            # Cache top predictions
            top_predictions = predictions_df.head(50).to_dict('records')
            self.cache.set(f"predictions:gw:{gameweek}:top", top_predictions)
            
            logger.info(f"✅ Warmed {len(predictions_df)} predictions for GW{gameweek}")
            return len(predictions_df)
            
        except Exception as e:
            logger.error(f"Error warming predictions cache: {e}")
            return 0
    
    async def warm_teams_cache(self) -> int:
        """Warm cache with team data"""
        try:
            logger.info("🔥 Warming teams cache...")
            
            # Get all teams
            teams = self.db.get_teams()
            
            if teams is None or len(teams) == 0:
                logger.warning("No teams found to cache")
                return 0
            
            # Cache all teams
            teams_list = teams.to_dict('records')
            self.cache.set("teams:all", teams_list)
            
            # Cache individual teams
            team_mapping = {}
            for team in teams_list:
                team_id = team.get('id')
                if team_id:
                    team_mapping[f"teams:{team_id}"] = team
            
            # Batch set
            count = self.cache.set_many(team_mapping)
            
            logger.info(f"✅ Warmed {count} teams in cache")
            return count
            
        except Exception as e:
            logger.error(f"Error warming teams cache: {e}")
            return 0
    
    async def warm_gameweeks_cache(self) -> int:
        """Warm cache with gameweek data"""
        try:
            logger.info("🔥 Warming gameweeks cache...")
            
            # Get all gameweeks
            with self.db.get_connection() as conn:
                gameweeks_df = conn.execute("SELECT * FROM gameweeks ORDER BY id").fetchdf()
            
            if gameweeks_df is None or len(gameweeks_df) == 0:
                logger.warning("No gameweeks found to cache")
                return 0
            
            # Cache all gameweeks
            gameweeks_list = gameweeks_df.to_dict('records')
            self.cache.set("gameweeks:all", gameweeks_list)
            
            # Cache current gameweek
            current_gw = gameweeks_df[gameweeks_df['is_current'] == True]
            if not current_gw.empty:
                self.cache.set("gameweeks:current", current_gw.iloc[0].to_dict())
            
            logger.info(f"✅ Warmed {len(gameweeks_df)} gameweeks in cache")
            return len(gameweeks_df)
            
        except Exception as e:
            logger.error(f"Error warming gameweeks cache: {e}")
            return 0
    
    async def warm_all_caches(self, gameweek: Optional[int] = None) -> Dict[str, int]:
        """
        Warm all caches
        
        Args:
            gameweek: Specific gameweek for predictions (uses current if None)
        
        Returns:
            Dictionary with counts for each cache type
        """
        logger.info("🔥🔥🔥 Starting full cache warming...")
        
        results = {}
        
        # Run all warming operations concurrently
        tasks = [
            self.warm_teams_cache(),
            self.warm_gameweeks_cache(),
            self.warm_players_cache(),
            self.warm_fixtures_cache(),
            self.warm_predictions_cache(gameweek),
        ]
        
        counts = await asyncio.gather(*tasks, return_exceptions=True)
        
        results['teams'] = counts[0] if not isinstance(counts[0], Exception) else 0
        results['gameweeks'] = counts[1] if not isinstance(counts[1], Exception) else 0
        results['players'] = counts[2] if not isinstance(counts[2], Exception) else 0
        results['fixtures'] = counts[3] if not isinstance(counts[3], Exception) else 0
        results['predictions'] = counts[4] if not isinstance(counts[4], Exception) else 0
        
        total = sum(results.values())
        logger.info(f"✅ Cache warming complete! Total items cached: {total}")
        logger.info(f"   Teams: {results['teams']}, Gameweeks: {results['gameweeks']}, "
                   f"Players: {results['players']}, Fixtures: {results['fixtures']}, "
                   f"Predictions: {results['predictions']}")
        
        return results
    
    def warm_before_deadline(self, hours_before: int = 2) -> Dict[str, int]:
        """
        Warm caches before gameweek deadline
        
        Args:
            hours_before: How many hours before deadline to warm cache
        
        Returns:
            Dictionary with counts for each cache type
        """
        try:
            # Get current gameweek and deadline
            current_gw = self.db.get_current_gameweek()
            if not current_gw:
                logger.warning("No current gameweek found")
                return {}
            
            with self.db.get_connection() as conn:
                gw_data = conn.execute(f"""
                    SELECT deadline_time FROM gameweeks WHERE id = {current_gw}
                """).fetchone()
            
            if not gw_data or not gw_data['deadline_time']:
                logger.warning(f"No deadline found for GW{current_gw}")
                return {}
            
            deadline = gw_data['deadline_time']
            time_until_deadline = deadline - datetime.now()
            
            if time_until_deadline.total_seconds() < hours_before * 3600:
                logger.info(f"⏰ Deadline approaching in {time_until_deadline}. Warming caches...")
                return asyncio.run(self.warm_all_caches(current_gw))
            else:
                logger.info(f"Deadline is {time_until_deadline} away. No warming needed yet.")
                return {}
                
        except Exception as e:
            logger.error(f"Error in deadline-based warming: {e}")
            return {}


# Global warmer instance
_warmer_instance = None


def get_cache_warmer() -> CacheWarmer:
    """Get global cache warmer instance"""
    global _warmer_instance
    if _warmer_instance is None:
        _warmer_instance = CacheWarmer()
    return _warmer_instance


async def warm_all_caches(gameweek: Optional[int] = None) -> Dict[str, int]:
    """Convenience function to warm all caches"""
    warmer = get_cache_warmer()
    return await warmer.warm_all_caches(gameweek)
