"""
Dynamic Fixture Difficulty Rating (FDR) Calculator

Calculates fixture difficulty based on:
- Recent team form (last 5 games)
- Home/away performance
- Defensive/offensive strength
- Injury impact
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DynamicFDRCalculator:
    """Calculate dynamic fixture difficulty ratings"""
    
    def __init__(self):
        self.cache = {}
        self.cache_expiry = timedelta(hours=6)
        self.last_calculation = None
        logger.info("Dynamic FDR Calculator initialized")
    
    async def calculate_fdr_for_gameweek(
        self,
        gameweek: int,
        fixtures_df: pd.DataFrame,
        teams_df: pd.DataFrame,
        recent_results_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate FDR for all fixtures in a gameweek
        
        Args:
            gameweek: Gameweek number
            fixtures_df: All fixtures data
            teams_df: Team strength data
            recent_results_df: Recent match results for form calculation
        
        Returns:
            DataFrame with dynamic FDR added
        """
        logger.info(f"Calculating dynamic FDR for gameweek {gameweek}...")
        
        # Get fixtures for this gameweek
        gw_fixtures = fixtures_df[fixtures_df['event'] == gameweek].copy()
        
        if gw_fixtures.empty:
            logger.warning(f"No fixtures found for gameweek {gameweek}")
            return pd.DataFrame()
        
        # Calculate FDR for each fixture
        fdr_results = []
        
        for _, fixture in gw_fixtures.iterrows():
            home_team_id = fixture['team_h']
            away_team_id = fixture['team_a']
            
            # Calculate FDR for home team (facing away team)
            home_fdr = self._calculate_fdr(
                team_id=home_team_id,
                opponent_id=away_team_id,
                is_home=True,
                teams_df=teams_df,
                recent_results_df=recent_results_df
            )
            
            # Calculate FDR for away team (facing home team)
            away_fdr = self._calculate_fdr(
                team_id=away_team_id,
                opponent_id=home_team_id,
                is_home=False,
                teams_df=teams_df,
                recent_results_df=recent_results_df
            )
            
            fdr_results.append({
                'fixture_id': fixture['id'],
                'event': gameweek,
                'team_h': home_team_id,
                'team_a': away_team_id,
                'dynamic_fdr_home': home_fdr,
                'dynamic_fdr_away': away_fdr,
                'static_fdr_home': fixture.get('team_h_difficulty', 3),
                'static_fdr_away': fixture.get('team_a_difficulty', 3),
                'calculated_at': datetime.now()
            })
        
        result_df = pd.DataFrame(fdr_results)
        logger.info(f"Calculated FDR for {len(result_df)} fixtures")
        
        return result_df
    
    def _calculate_fdr(
        self,
        team_id: int,
        opponent_id: int,
        is_home: bool,
        teams_df: pd.DataFrame,
        recent_results_df: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Calculate dynamic FDR for a specific matchup
        
        FDR Scale: 1 (very easy) to 5 (very hard)
        
        Args:
            team_id: Team for which to calculate difficulty
            opponent_id: Opponent team
            is_home: Whether team is playing at home
            teams_df: Team strength data
            recent_results_df: Recent results for form
        
        Returns:
            Dynamic FDR score (1-5)
        """
        # Get opponent team data
        opponent = teams_df[teams_df['id'] == opponent_id]
        
        if opponent.empty:
            logger.warning(f"Opponent team {opponent_id} not found")
            return 3.0  # Neutral difficulty
        
        opponent = opponent.iloc[0]
        
        # 1. Base FDR from opponent strength
        if is_home:
            # Facing opponent's away strength
            base_strength = opponent.get('strength_attack_away', 100)
        else:
            # Facing opponent's home strength
            base_strength = opponent.get('strength_attack_home', 100)
        
        # Normalize to 1-5 scale (100 = average = 3)
        base_fdr = (base_strength / 100) * 3
        base_fdr = np.clip(base_fdr, 1, 5)
        
        # 2. Form adjustment (last 5 games)
        form_adjustment = self._calculate_form_adjustment(
            opponent_id, recent_results_df
        )
        
        # 3. Home/away adjustment
        ha_adjustment = 0.3 if is_home else -0.3  # Easier at home, harder away
        
        # 4. Defensive strength adjustment
        if is_home:
            def_strength = opponent.get('strength_defence_away', 100)
        else:
            def_strength = opponent.get('strength_defence_home', 100)
        
        def_adjustment = (def_strength - 100) / 100  # -1 to +1
        
        # Combine all factors
        dynamic_fdr = (
            base_fdr * 0.4 +          # 40% base strength
            form_adjustment * 0.3 +    # 30% recent form
            ha_adjustment * 0.2 +      # 20% home advantage
            def_adjustment * 0.1       # 10% defensive strength
        )
        
        # Clip to valid range
        dynamic_fdr = np.clip(dynamic_fdr, 1, 5)
        
        return round(dynamic_fdr, 2)
    
    def _calculate_form_adjustment(
        self,
        team_id: int,
        recent_results_df: Optional[pd.DataFrame]
    ) -> float:
        """
        Calculate form-based adjustment to FDR
        
        Args:
            team_id: Team ID
            recent_results_df: Recent match results
        
        Returns:
            Form adjustment (-2 to +2)
        """
        if recent_results_df is None or recent_results_df.empty:
            return 0.0  # No adjustment if no data
        
        # Get team's recent results (last 5 games)
        team_results = recent_results_df[
            (recent_results_df['team_h'] == team_id) |
            (recent_results_df['team_a'] == team_id)
        ].sort_values('kickoff_time', ascending=False).head(5)
        
        if team_results.empty:
            return 0.0
        
        # Calculate points from results
        points = []
        for _, result in team_results.iterrows():
            if not result.get('finished', False):
                continue
            
            is_home = result['team_h'] == team_id
            home_score = result.get('team_h_score', 0)
            away_score = result.get('team_a_score', 0)
            
            if is_home:
                if home_score > away_score:
                    points.append(3)  # Win
                elif home_score == away_score:
                    points.append(1)  # Draw
                else:
                    points.append(0)  # Loss
            else:
                if away_score > home_score:
                    points.append(3)  # Win
                elif away_score == home_score:
                    points.append(1)  # Draw
                else:
                    points.append(0)  # Loss
        
        if not points:
            return 0.0
        
        # Average points per game
        avg_points = np.mean(points)
        
        # Convert to adjustment (-2 to +2)
        # 3 points/game (all wins) = -2 (much easier)
        # 1.5 points/game (average) = 0 (neutral)
        # 0 points/game (all losses) = +2 (much harder)
        adjustment = (1.5 - avg_points) * (2 / 1.5)
        
        return np.clip(adjustment, -2, 2)
    
    def get_fdr_for_player_fixtures(
        self,
        player_id: int,
        team_id: int,
        fixtures_df: pd.DataFrame,
        fdr_df: pd.DataFrame,
        num_fixtures: int = 5
    ) -> Dict:
        """
        Get FDR summary for a player's upcoming fixtures
        
        Args:
            player_id: Player ID
            team_id: Player's team ID
            fixtures_df: All fixtures
            fdr_df: Calculated FDR data
            num_fixtures: Number of upcoming fixtures to analyze
        
        Returns:
            Dictionary with FDR summary
        """
        # Get team's upcoming fixtures
        upcoming = fixtures_df[
            ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)) &
            (fixtures_df['finished'] == False)
        ].sort_values('kickoff_time').head(num_fixtures)
        
        if upcoming.empty:
            return {
                'player_id': player_id,
                'avg_fdr': 3.0,
                'min_fdr': 3.0,
                'max_fdr': 3.0,
                'num_fixtures': 0,
                'fixtures': []
            }
        
        fixture_details = []
        fdr_values = []
        
        for _, fixture in upcoming.iterrows():
            is_home = fixture['team_h'] == team_id
            
            # Get FDR from calculated data
            fdr_row = fdr_df[fdr_df['fixture_id'] == fixture['id']]
            
            if not fdr_row.empty:
                fdr = fdr_row.iloc[0]['dynamic_fdr_home' if is_home else 'dynamic_fdr_away']
            else:
                fdr = fixture.get('team_h_difficulty' if is_home else 'team_a_difficulty', 3)
            
            fdr_values.append(fdr)
            
            fixture_details.append({
                'gameweek': fixture['event'],
                'is_home': is_home,
                'fdr': fdr,
                'opponent_id': fixture['team_a'] if is_home else fixture['team_h']
            })
        
        return {
            'player_id': player_id,
            'avg_fdr': round(np.mean(fdr_values), 2) if fdr_values else 3.0,
            'min_fdr': round(min(fdr_values), 2) if fdr_values else 3.0,
            'max_fdr': round(max(fdr_values), 2) if fdr_values else 3.0,
            'num_fixtures': len(fixture_details),
            'fixtures': fixture_details
        }


# Example usage
if __name__ == "__main__":
    # Test with sample data
    calculator = DynamicFDRCalculator()
    
    # Sample teams
    teams_df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Arsenal', 'Liverpool', 'Man City'],
        'strength_attack_home': [120, 115, 125],
        'strength_attack_away': [110, 105, 115],
        'strength_defence_home': [95, 100, 90],
        'strength_defence_away': [100, 105, 95]
    })
    
    # Sample fixtures
    fixtures_df = pd.DataFrame({
        'id': [1, 2],
        'event': [15, 15],
        'team_h': [1, 2],
        'team_a': [2, 3],
        'team_h_difficulty': [4, 5],
        'team_a_difficulty': [3, 4],
        'finished': [False, False],
        'kickoff_time': [pd.Timestamp.now() + pd.Timedelta(days=3)] * 2
    })
    
    # Sample recent results
    recent_results_df = pd.DataFrame({
        'team_h': [1, 2, 3, 1, 2],
        'team_a': [3, 1, 2, 2, 3],
        'team_h_score': [2, 1, 3, 1, 2],
        'team_a_score': [1, 1, 2, 2, 1],
        'finished': [True] * 5,
        'kickoff_time': [pd.Timestamp.now() - pd.Timedelta(days=i*7) for i in range(5)]
    })
    
    # Calculate FDR
    import asyncio
    
    async def test():
        fdr_df = await calculator.calculate_fdr_for_gameweek(
            15, fixtures_df, teams_df, recent_results_df
        )
        print("\nDynamic FDR Results:")
        print(fdr_df[['fixture_id', 'dynamic_fdr_home', 'dynamic_fdr_away', 
                      'static_fdr_home', 'static_fdr_away']])
    
    asyncio.run(test())
