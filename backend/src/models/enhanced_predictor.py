"""
Enhanced Model Integration

Integrates advanced features (xG, time-weighted, opponent-adjusted, etc.)
into existing prediction models.
"""

import logging
import pandas as pd
from typing import Optional

from backend.src.features.advanced_features import AdvancedFeatureEngineer
from backend.src.integrations.xg_integrator import XGIntegrator
from backend.src.core.async_db import AsyncDatabaseWrapper

logger = logging.getLogger(__name__)


class EnhancedModelPredictor:
    """
    Enhanced predictor that combines:
    - xG/xA data integration
    - Advanced feature engineering
    - Existing ML models
    """
    
    def __init__(self, db: AsyncDatabaseWrapper):
        self.db = db
        self.xg_integrator = XGIntegrator(db)
        self.feature_engineer = AdvancedFeatureEngineer()
        logger.info("Enhanced Model Predictor initialized")
    
    async def prepare_prediction_data(
        self,
        players_df: pd.DataFrame,
        enrich_xg: bool = True,
        create_advanced_features: bool = True
    ) -> pd.DataFrame:
        """
        Prepare player data with all enhancements for prediction
        
        Args:
            players_df: Base player data
            enrich_xg: Whether to add xG/xA data
            create_advanced_features: Whether to create advanced features
        
        Returns:
            Enhanced DataFrame ready for model prediction
        """
        logger.info(f"Preparing prediction data for {len(players_df)} players...")
        
        enriched_df = players_df.copy()
        
        # Step 1: Enrich with xG data
        if enrich_xg:
            logger.info("Enriching with xG/xA data...")
            enriched_df = await self.xg_integrator.enrich_player_data(enriched_df)
        
        # Step 2: Create advanced features
        if create_advanced_features:
            logger.info("Creating advanced features...")
            
            # Get required data
            history_df = await self.db.get_player_history_data()
            fixtures_df = await self.db.get_fixtures()
            teams_df = await self.db.get_teams()
            
            # Create features
            enriched_df = self.feature_engineer.create_features(
                enriched_df,
                history_df,
                fixtures_df,
                teams_df
            )
        
        logger.info(f"Prepared {len(enriched_df.columns)} features for prediction")
        
        return enriched_df
    
    def get_enhanced_feature_list(self) -> list:
        """
        Get list of all enhanced features for model training
        
        Returns:
            List of feature column names
        """
        base_features = [
            # Original FPL features
            'now_cost', 'total_points', 'points_per_game', 'form',
            'selected_by_percent', 'minutes', 'goals_scored', 'assists',
            'clean_sheets', 'bonus', 'bps', 'influence', 'creativity',
            'threat', 'ict_index',
            
            # xG features
            'expected_goals', 'expected_assists', 'expected_goals_per_90',
            'expected_assists_per_90',
            
            # Time-weighted features
            'tw_points', 'tw_minutes', 'tw_xg', 'tw_xa', 'tw_bonus', 'tw_bps',
            
            # Opponent-adjusted features
            'opp_adj_points', 'opp_adj_xg', 'opp_difficulty',
            
            # Fixture congestion
            'cong_fixtures_14d', 'cong_recent_minutes', 'cong_rest_days', 'cong_score',
            
            # Home/away splits
            'ha_home_points', 'ha_away_points', 'ha_advantage',
            
            # Momentum indicators
            'mom_form_trend', 'mom_points_streak', 'mom_consistency'
        ]
        
        return base_features
    
    def get_feature_importance_groups(self) -> dict:
        """
        Group features by type for analysis
        
        Returns:
            Dictionary mapping feature groups to column names
        """
        return {
            'base_stats': [
                'now_cost', 'total_points', 'points_per_game', 'form',
                'minutes', 'goals_scored', 'assists', 'clean_sheets'
            ],
            'ict_metrics': [
                'influence', 'creativity', 'threat', 'ict_index', 'bonus', 'bps'
            ],
            'xg_metrics': [
                'expected_goals', 'expected_assists', 
                'expected_goals_per_90', 'expected_assists_per_90'
            ],
            'time_weighted': [
                'tw_points', 'tw_minutes', 'tw_xg', 'tw_xa', 'tw_bonus', 'tw_bps'
            ],
            'opponent_adjusted': [
                'opp_adj_points', 'opp_adj_xg', 'opp_difficulty'
            ],
            'fixture_context': [
                'cong_fixtures_14d', 'cong_recent_minutes', 
                'cong_rest_days', 'cong_score'
            ],
            'home_away': [
                'ha_home_points', 'ha_away_points', 'ha_advantage'
            ],
            'momentum': [
                'mom_form_trend', 'mom_points_streak', 'mom_consistency'
            ]
        }


# Integration helper functions

async def get_enhanced_prediction_data(
    db: AsyncDatabaseWrapper,
    player_ids: Optional[list] = None
) -> pd.DataFrame:
    """
    Get player data with all enhancements for prediction
    
    Args:
        db: Database wrapper
        player_ids: Optional list of player IDs to filter
    
    Returns:
        Enhanced player DataFrame
    """
    predictor = EnhancedModelPredictor(db)
    
    # Get base player data
    players_df = await db.get_players_with_stats()
    
    if player_ids:
        players_df = players_df[players_df['id'].isin(player_ids)]
    
    # Enhance with all features
    enhanced_df = await predictor.prepare_prediction_data(players_df)
    
    return enhanced_df


def create_model_training_config(include_advanced_features: bool = True) -> dict:
    """
    Create configuration for model training with enhanced features
    
    Args:
        include_advanced_features: Whether to include advanced features
    
    Returns:
        Training configuration dictionary
    """
    predictor = EnhancedModelPredictor(None)  # No DB needed for config
    
    if include_advanced_features:
        features = predictor.get_enhanced_feature_list()
    else:
        # Just base features
        features = [
            'now_cost', 'total_points', 'points_per_game', 'form',
            'minutes', 'goals_scored', 'assists', 'clean_sheets',
            'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index'
        ]
    
    return {
        'features': features,
        'feature_groups': predictor.get_feature_importance_groups(),
        'model_params': {
            'n_estimators': 200,  # Increased for more complex features
            'max_depth': 6,       # Deeper trees for feature interactions
            'learning_rate': 0.05,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'subsample': 0.8,
            'random_state': 42
        }
    }


# Example usage
if __name__ == "__main__":
    import asyncio
    from backend.src.core.db_factory import get_db
    
    async def test_enhanced_predictor():
        """Test the enhanced predictor"""
        db = get_db()
        predictor = EnhancedModelPredictor(db)
        
        # Get players
        players_df = await db.get_players_with_stats()
        players_df = players_df.head(10)  # Test with 10 players
        
        # Prepare enhanced data
        enhanced_df = await predictor.prepare_prediction_data(players_df)
        
        # Show features
        feature_cols = [col for col in enhanced_df.columns 
                       if col.startswith(('tw_', 'opp_', 'cong_', 'ha_', 'mom_', 'expected_'))]
        
        print(f"\nEnhanced Features ({len(feature_cols)} total):")
        print(enhanced_df[['web_name'] + feature_cols[:10]].head())
        
        # Show feature groups
        groups = predictor.get_feature_importance_groups()
        print(f"\nFeature Groups:")
        for group, features in groups.items():
            print(f"  {group}: {len(features)} features")
    
    asyncio.run(test_enhanced_predictor())
