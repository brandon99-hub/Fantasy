import requests
import pandas as pd
import sqlite3
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json

from backend.src.integrations.odds_api import OddsAPIClient, normalize_team_name

class FPLDataCollector:
    """Handles data collection from the FPL API"""
    
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds between requests
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _rate_limit(self):
        """Implement rate limiting between API calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, retries: int = 3) -> Optional[Dict]:
        """Make API request with error handling and retries"""
        self._rate_limit()
        
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {str(e)}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"All retries failed for URL: {url}")
                    return None
        return None
    
    def get_bootstrap_static(self) -> Optional[Dict]:
        """Get general game data including players, teams, and events"""
        url = f"{self.base_url}/bootstrap-static/"
        return self._make_request(url)
    
    def get_fixtures(self) -> Optional[List[Dict]]:
        """Get all fixtures data"""
        url = f"{self.base_url}/fixtures/"
        return self._make_request(url)
    
    def get_gameweek_live(self, gameweek: int) -> Optional[Dict]:
        """Get live scores for a specific gameweek"""
        url = f"{self.base_url}/event/{gameweek}/live/"
        return self._make_request(url)
    
    def get_player_history(self, player_id: int) -> Optional[Dict]:
        """Get detailed history for a specific player"""
        url = f"{self.base_url}/element-summary/{player_id}/"
        return self._make_request(url)
    
    def update_bootstrap_data(self) -> bool:
        """Update players, teams, and gameweeks data"""
        try:
            self.logger.info("Updating bootstrap data...")
            bootstrap = self.get_bootstrap_static()
            
            if not bootstrap:
                self.logger.error("Failed to fetch bootstrap data")
                return False
            
            # Connect to database
            from backend.src.core.db_factory import get_db
            db = get_db()
            
            # Update teams
            teams_df = pd.DataFrame(bootstrap['teams'])
            db.update_teams(teams_df)
            
            # Update players
            players_df = pd.DataFrame(bootstrap['elements'])
            # Add position names
            positions = {pos['id']: pos['singular_name'] for pos in bootstrap['element_types']}
            players_df['position'] = players_df['element_type'].map(positions)
            
            # Add team names
            teams_map = {team['id']: team['name'] for team in bootstrap['teams']}
            players_df['team_name'] = players_df['team'].map(teams_map)
            
            db.update_players(players_df)
            
            # Update gameweeks
            events_df = pd.DataFrame(bootstrap['events'])
            db.update_gameweeks(events_df)
            
            self.logger.info("Bootstrap data updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating bootstrap data: {str(e)}")
            return False
    
    def update_fixtures_data(self) -> bool:
        """Update fixtures data"""
        try:
            self.logger.info("Updating fixtures data...")
            fixtures = self.get_fixtures()
            
            if not fixtures:
                self.logger.error("Failed to fetch fixtures data")
                return False
            
            from backend.src.core.db_factory import get_db
            db = get_db()
            
            fixtures_df = pd.DataFrame(fixtures)
            db.update_fixtures(fixtures_df)
            
            self.logger.info("Fixtures data updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating fixtures data: {str(e)}")
            return False
    
    def _build_team_lookup(self, teams_df: pd.DataFrame) -> Dict[str, int]:
        lookup = {}
        if teams_df.empty:
            return lookup
        for _, row in teams_df.iterrows():
            variants = {
                row.get('name', ''),
                row.get('short_name', ''),
                row.get('name', '').replace("FC", "").strip(),
            }
            for name in variants:
                key = normalize_team_name(name)
                if key:
                    lookup[key] = row['id']
        # Manual aliases
        aliases = {
            "man united": "man united",
            "man utd": "man united",
            "man city": "man city",
            "tottenham": "tottenham",
            "wolves": "wolves",
            "brighton": "brighton",
        }
        for alias, target in aliases.items():
            if target in lookup:
                lookup[alias] = lookup[target]
        return lookup
    
    def _match_fixture_row(self, fixtures_df: pd.DataFrame, home_team_id: int, away_team_id: int, commence_time: Optional[str]) -> Optional[pd.Series]:
        if fixtures_df.empty:
            return None
        candidates = fixtures_df[
            (fixtures_df['team_h'] == home_team_id) &
            (fixtures_df['team_a'] == away_team_id)
        ]
        if candidates.empty:
            return None
        unfinished = candidates[candidates.get('finished', 1) == 0]
        if not unfinished.empty:
            candidates = unfinished
        if commence_time:
            try:
                target_time = pd.to_datetime(commence_time, utc=True)
                candidates = candidates.copy()
                candidates['kickoff_dt'] = pd.to_datetime(candidates['kickoff_time'], utc=True, errors='coerce')
                candidates['time_diff'] = (candidates['kickoff_dt'] - target_time).abs()
                candidates = candidates.sort_values('time_diff')
            except Exception:
                candidates = candidates.sort_values('kickoff_time')
        else:
            candidates = candidates.sort_values('kickoff_time')
        return candidates.iloc[0]
    
    def _extract_implied_goals(self, odds_event: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
        bookmakers = odds_event.get('bookmakers') or []
        for bookmaker in bookmakers:
            markets = bookmaker.get('markets') or []
            team_totals = next((m for m in markets if m.get('key') == 'team_totals'), None)
            home_line = None
            away_line = None
            market_key = 'team_totals'
            if team_totals:
                for outcome in team_totals.get('outcomes', []):
                    outcome_name = normalize_team_name(outcome.get('name') or outcome.get('team'))
                    description = (outcome.get('description') or "").lower()
                    point = outcome.get('point')
                    if outcome_name == normalize_team_name(odds_event.get('home_team')) and description == 'over':
                        home_line = point
                    if outcome_name == normalize_team_name(odds_event.get('away_team')) and description == 'over':
                        away_line = point
            if home_line is not None and away_line is not None:
                return home_line, away_line, bookmaker.get('title') or bookmaker.get('key'), market_key
            totals_market = next((m for m in markets if m.get('key') == 'totals'), None)
            if totals_market:
                outcomes = totals_market.get('outcomes') or []
                if outcomes:
                    point = outcomes[0].get('point')
                    if point:
                        market_key = 'totals'
                        return point / 2, point / 2, bookmaker.get('title') or bookmaker.get('key'), market_key
        return None, None, None, None
    
    def update_fixture_odds(self) -> bool:
        """Fetch bookmaker odds and store implied goals."""
        try:
            self.logger.info("Updating fixture odds from TheOddsAPI...")
            client = OddsAPIClient()
            odds_events = client.fetch_fixture_odds()
            if not odds_events:
                self.logger.warning("No odds data retrieved.")
                return False
            
            from backend.src.core.db_factory import get_db
            db = get_db()
            fixtures_df = db.get_fixtures()
            teams_df = db.get_teams()
            if fixtures_df.empty or teams_df.empty:
                self.logger.warning("Fixtures or teams data unavailable; cannot map odds.")
                return False
            
            team_lookup = self._build_team_lookup(teams_df)
            odds_records = []
            
            for event in odds_events:
                home_name_raw = event.get('home_team')
                away_name_raw = event.get('away_team')
                home_id = team_lookup.get(normalize_team_name(home_name_raw))
                away_id = team_lookup.get(normalize_team_name(away_name_raw))
                if not home_id or not away_id:
                    continue
                
                fixture_row = self._match_fixture_row(fixtures_df, home_id, away_id, event.get('commence_time'))
                if fixture_row is None:
                    continue
                
                home_goals, away_goals, bookmaker, market = self._extract_implied_goals(event)
                if home_goals is None or away_goals is None:
                    continue
                
                odds_records.append({
                    'fixture_id': int(fixture_row['id']),
                    'event': int(fixture_row.get('event') or 0),
                    'home_team': home_id,
                    'away_team': away_id,
                    'bookmaker': bookmaker,
                    'market': market,
                    'home_implied_goals': float(home_goals),
                    'away_implied_goals': float(away_goals),
                    'raw_odds': json.dumps(event),
                })
            
            if not odds_records:
                self.logger.warning("No odds records matched to fixtures.")
                return False
            
            success = db.upsert_fixture_odds(odds_records)
            if success:
                self.logger.info(f"Updated odds for {len(odds_records)} fixtures.")
            return success
        except Exception as e:
            self.logger.error(f"Error updating fixture odds: {str(e)}")
            return False
    
    def update_gameweek_live_data(self, gameweek: int) -> bool:
        """Update live data for a specific gameweek"""
        try:
            self.logger.info(f"Updating live data for GW {gameweek}...")
            live_data = self.get_gameweek_live(gameweek)
            
            if not live_data:
                self.logger.error(f"Failed to fetch live data for GW {gameweek}")
                return False
            
            from backend.src.core.db_factory import get_db
            db = get_db()
            
            # Process player stats
            player_stats = []
            for element in live_data['elements']:
                stats = element['stats']
                stats['element'] = element['id']
                stats['gameweek'] = gameweek
                player_stats.append(stats)
            
            if player_stats:
                stats_df = pd.DataFrame(player_stats)
                db.update_player_gameweek_stats(stats_df)
            
            self.logger.info(f"Live data for GW {gameweek} updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating live data for GW {gameweek}: {str(e)}")
            return False
    
    def update_player_histories(self, player_ids: List[int] = None, limit: int = 50) -> bool:
        """Update detailed history for players (rate limited)"""
        try:
            from backend.src.core.db_factory import get_db
            db = get_db()
            
            if player_ids is None:
                # Get all players that should have trackable history (includes injured/suspended)
                player_ids = db.get_trackable_player_ids()
            
            # Limit to prevent excessive API calls
            if limit and len(player_ids) > limit:
                player_ids = player_ids[:limit]
                self.logger.info(f"Limited to {limit} players to avoid rate limiting")
            
            self.logger.info(f"Updating history for {len(player_ids)} players...")
            
            success_count = 0
            for i, player_id in enumerate(player_ids):
                if i % 10 == 0:
                    self.logger.info(f"Processing player {i+1}/{len(player_ids)}")
                
                history_data = self.get_player_history(player_id)
                if history_data:
                    # Update history
                    if history_data.get('history'):
                        history_df = pd.DataFrame(history_data['history'])
                        history_df['element'] = player_id
                        db.update_player_history(history_df)
                    
                    # Update fixtures
                    if history_data.get('fixtures'):
                        fixtures_df = pd.DataFrame(history_data['fixtures'])
                        fixtures_df['element'] = player_id
                        db.update_player_fixtures(fixtures_df)
                    
                    success_count += 1
            
            self.logger.info(f"Updated history for {success_count}/{len(player_ids)} players")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Error updating player histories: {str(e)}")
            return False
    
    def update_all_data(self, include_history: bool = True) -> bool:
        """Update all FPL data"""
        try:
            self.logger.info("Starting full data update...")
            
            success_count = 0
            
            # Update bootstrap data
            if self.update_bootstrap_data():
                success_count += 1
            
            # Update fixtures
            if self.update_fixtures_data():
                success_count += 1
            
            # Update bookmaker odds
            if self.update_fixture_odds():
                success_count += 1
            
            # Update current gameweek live data
            from backend.src.core.db_factory import get_db
            db = get_db()
            current_gw = db.get_current_gameweek()
            
            if current_gw and current_gw > 1:
                # Update previous gameweek data
                if self.update_gameweek_live_data(current_gw - 1):
                    success_count += 1
            
            # Optionally update player histories (rate limited)
            if include_history:
                if self.update_player_histories():
                    success_count += 1
            
            # Mark update time
            db.mark_last_update()
            
            self.logger.info(f"Data update completed. {success_count} operations successful.")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Error in full data update: {str(e)}")
            return False
    
    def get_upcoming_fixtures(self, team_id: int = None, gameweeks: int = 5) -> pd.DataFrame:
        """Get upcoming fixtures for analysis"""
        try:
            fixtures = self.get_fixtures()
            if not fixtures:
                return pd.DataFrame()
            
            fixtures_df = pd.DataFrame(fixtures)
            
            # Filter for upcoming fixtures
            upcoming = fixtures_df[fixtures_df['finished'] == False]
            
            if team_id:
                upcoming = upcoming[
                    (upcoming['team_h'] == team_id) | 
                    (upcoming['team_a'] == team_id)
                ]
            
            # Limit to next few gameweeks
            if gameweeks:
                current_gw = upcoming['event'].min() if not upcoming.empty else 1
                upcoming = upcoming[upcoming['event'] <= current_gw + gameweeks - 1]
            
            return upcoming.head(50)  # Reasonable limit
            
        except Exception as e:
            self.logger.error(f"Error getting upcoming fixtures: {str(e)}")
            return pd.DataFrame()
