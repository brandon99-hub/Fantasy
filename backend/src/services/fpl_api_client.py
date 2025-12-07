"""
Real FPL API client for manager search and team fetching
"""

import requests
import time
from typing import List, Dict, Optional
import logging

class FPLAPIClient:
    """Client for interacting with the official FPL API"""
    
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, retries: int = 3) -> Optional[Dict]:
        """Make API request with error handling"""
        self._rate_limit()
        
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {str(e)}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    self.logger.error(f"All retries failed for URL: {url}")
                    return None
        return None
    
    def search_managers(self, name: str) -> List[Dict]:
        """
        Search for managers by name using multiple FPL search methods
        """
        try:
            managers = []
            
            # Method 1: Search through FPL's global league (everyone is in this)
            try:
                # The global league has all managers - we can search through pages
                global_league_url = "https://fantasy.premierleague.com/api/leagues-classic/314/standings/"
                
                # Try searching multiple pages of the global league
                # Sample different rank ranges to find players at all levels
                search_pages = list(range(1, 21)) + list(range(100, 110)) + list(range(500, 510)) + list(range(1000, 1005))
                
                for page in search_pages:
                    response = self.session.get(
                        global_league_url,
                        params={'page_standings': page},
                        timeout=15
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if 'standings' in data and 'results' in data['standings']:
                            for result in data['standings']['results']:
                                player_name = result.get('player_name', '').lower()
                                entry_name = result.get('entry_name', '').lower()
                                search_name = name.lower()
                                
                                # Check if name matches (search both player name and team name)
                                if (search_name in player_name or 
                                    search_name in entry_name or
                                    player_name in search_name or
                                    entry_name in search_name):
                                    
                                    managers.append({
                                        'id': str(result.get('entry', '')),
                                        'player_name': result.get('player_name', ''),
                                        'team_name': result.get('entry_name', ''),
                                        'overall_rank': result.get('rank', 0),
                                        'total_points': result.get('total', 0)
                                    })
                                    
                                    # Stop if we found some matches
                                    if len(managers) >= 3:
                                        break
                    
                    # If we found managers, stop searching
                    if managers:
                        break
                        
                    # Small delay between requests
                    time.sleep(0.5)
                        
            except Exception as e:
                self.logger.error(f"Error searching global league: {str(e)}")
            
            # Method 2: Try searching other popular leagues
            if not managers:
                try:
                    # Some popular league IDs to search through
                    popular_leagues = [314, 633, 1441, 2351, 3567]  # These are real FPL leagues
                    
                    for league_id in popular_leagues:
                        league_url = f"https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/"
                        
                        response = self.session.get(league_url, timeout=10)
                        
                        if response.status_code == 200:
                            data = response.json()
                            if 'standings' in data and 'results' in data['standings']:
                                for result in data['standings']['results']:
                                    player_name = result.get('player_name', '').lower()
                                    entry_name = result.get('entry_name', '').lower()
                                    search_name = name.lower()
                                    
                                    if (search_name in player_name or 
                                        search_name in entry_name):
                                        
                                        managers.append({
                                            'id': str(result.get('entry', '')),
                                            'player_name': result.get('player_name', ''),
                                            'team_name': result.get('entry_name', ''),
                                            'overall_rank': result.get('rank', 0),
                                            'total_points': result.get('total', 0)
                                        })
                                        
                                        if len(managers) >= 3:
                                            break
                        
                        if managers:
                            break
                            
                        time.sleep(0.3)  # Rate limiting
                        
                except Exception as e:
                    self.logger.error(f"Error searching popular leagues: {str(e)}")
            
            self.logger.info(f"Found {len(managers)} managers for '{name}'")
            return managers
            
        except Exception as e:
            self.logger.error(f"Error searching managers: {str(e)}")
            return []
    
    def get_manager_team(self, manager_id: str) -> Optional[Dict]:
        """
        Get manager's current team by ID
        Uses: https://fantasy.premierleague.com/api/entry/{manager_id}/
        """
        try:
            # Get manager entry data
            entry_url = f"{self.base_url}/entry/{manager_id}/"
            entry_data = self._make_request(entry_url)
            
            if not entry_data:
                return None
            
            # Get current gameweek
            bootstrap_url = f"{self.base_url}/bootstrap-static/"
            bootstrap_data = self._make_request(bootstrap_url)
            
            if not bootstrap_data:
                return None
            
            current_gw = None
            for event in bootstrap_data.get('events', []):
                if event.get('is_current', False):
                    current_gw = event['id']
                    break
            
            if not current_gw:
                # Get most recent finished gameweek
                for event in reversed(bootstrap_data.get('events', [])):
                    if event.get('finished', False):
                        current_gw = event['id']
                        break
            
            if not current_gw:
                current_gw = 1
            
            # Get manager history (chips usage, etc.)
            history_url = f"{self.base_url}/entry/{manager_id}/history/"
            history_data = self._make_request(history_url) or {}
            
            # Get manager's team for current gameweek
            team_url = f"{self.base_url}/entry/{manager_id}/event/{current_gw}/picks/"
            team_data = self._make_request(team_url)
            
            if not team_data:
                return None
            
            # Get player details
            players_map = {p['id']: p for p in bootstrap_data.get('elements', [])}
            teams_map = {t['id']: t['name'] for t in bootstrap_data.get('teams', [])}
            positions_map = {pt['id']: pt['singular_name'] for pt in bootstrap_data.get('element_types', [])}
            
            picks = team_data.get('picks', [])
            team_players = []
            for pick in picks:
                player_id = pick['element']
                player_data = players_map.get(player_id)
                
                if player_data:
                    team_players.append({
                        'id': player_data['id'],
                        'web_name': player_data['web_name'],
                        'first_name': player_data['first_name'],
                        'second_name': player_data['second_name'],
                        'position': positions_map.get(player_data['element_type'], 'MID'),
                        'team': player_data.get('team'),
                        'team_name': teams_map.get(player_data['team'], 'Unknown'),
                        'now_cost': player_data['now_cost'],
                        'total_points': player_data['total_points'],
                        'form': float(player_data.get('form', 0)),
                        'selected_by_percent': float(player_data.get('selected_by_percent', 0)),
                        'points_per_game': float(player_data.get('points_per_game', 0)),
                        'goals_scored': player_data.get('goals_scored', 0),
                        'assists': player_data.get('assists', 0),
                        'clean_sheets': player_data.get('clean_sheets', 0),
                        'saves': player_data.get('saves', 0),
                        'bonus': player_data.get('bonus', 0),
                        'ict_index': float(player_data.get('ict_index', 0)),
                        'influence': float(player_data.get('influence', 0)),
                        'creativity': float(player_data.get('creativity', 0)),
                        'threat': float(player_data.get('threat', 0)),
                        'status': player_data.get('status', 'a'),
                        'chance_of_playing_this_round': player_data.get('chance_of_playing_this_round'),
                        'chance_of_playing_next_round': player_data.get('chance_of_playing_next_round'),
                        'news': player_data.get('news', ''),
                        'pick_position': pick.get('position'),
                        'multiplier': pick.get('multiplier'),
                        'is_captain': pick.get('is_captain'),
                        'is_vice_captain': pick.get('is_vice_captain')
                    })
            
            entry_history = team_data.get('entry_history', {}) or {}
            bank_raw = entry_history.get('bank')
            bank_value = round(bank_raw / 10, 1) if isinstance(bank_raw, (int, float)) else 0.0
            
            squad_value_raw = entry_history.get('value')
            squad_value = round(squad_value_raw / 10, 1) if isinstance(squad_value_raw, (int, float)) else None
            
            starting_xi = [pick['element'] for pick in picks if pick.get('position', 0) <= 11]
            
            chip_name_map = {
                'wildcard': 'Wildcard',
                'freehit': 'Free Hit',
                '3xc': 'Triple Captain',
                'bboost': 'Bench Boost'
            }
            used_chips = set()
            for chip in history_data.get('chips', []):
                chip_name = chip.get('name')
                mapped = chip_name_map.get(chip_name)
                if mapped:
                    used_chips.add(mapped)
            
            default_chips = ['Wildcard', 'Bench Boost', 'Triple Captain', 'Free Hit']
            chips_available = [chip for chip in default_chips if chip not in used_chips]
            
            active_chip_raw = team_data.get('active_chip')
            active_chip = chip_name_map.get(active_chip_raw, active_chip_raw)
            
            free_transfers = entry_data.get('last_deadline_total_transfers')
            if free_transfers is None:
                free_transfers = entry_history.get('event_transfers')  # fallback
            if free_transfers is None:
                free_transfers = 1
            
            return {
                "manager_id": manager_id,
                "manager_name": entry_data.get('player_first_name', '') + ' ' + entry_data.get('player_last_name', ''),
                "team_name": entry_data.get('name', ''),
                "overall_rank": entry_data.get('summary_overall_rank'),
                "total_points": entry_data.get('summary_overall_points'),
                "players": team_players,
                "starting_xi": starting_xi,
                "bank": bank_value,
                "squad_value": squad_value,
                "free_transfers": free_transfers,
                "chips_available": chips_available,
                "active_chip": active_chip
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching team for manager {manager_id}: {str(e)}")
            return None
    
    def get_manager_entry(self, manager_id: str) -> Optional[Dict]:
        """
        Get manager entry data by ID
        Uses: https://fantasy.premierleague.com/api/entry/{manager_id}/
        """
        try:
            entry_url = f"{self.base_url}/entry/{manager_id}/"
            entry_data = self._make_request(entry_url)
            return entry_data
        except Exception as e:
            self.logger.error(f"Error fetching manager entry {manager_id}: {str(e)}")
            return None

    def get_manager_history(self, manager_id: str) -> Optional[Dict]:
        """
        Get manager season history, chips and transfers.
        Uses: https://fantasy.premierleague.com/api/entry/{manager_id}/history/
        """
        try:
            history_url = f"{self.base_url}/entry/{manager_id}/history/"
            return self._make_request(history_url)
        except Exception as e:
            self.logger.error(f"Error fetching manager history {manager_id}: {str(e)}")
            return None

    def get_manager_event_picks(self, manager_id: str, event: int) -> Optional[Dict]:
        """
        Get manager picks for a specific gameweek.
        Uses: https://fantasy.premierleague.com/api/entry/{manager_id}/event/{event}/picks/
        """
        try:
            picks_url = f"{self.base_url}/entry/{manager_id}/event/{event}/picks/"
            return self._make_request(picks_url)
        except Exception as e:
            self.logger.error(
                f"Error fetching manager {manager_id} picks for event {event}: {str(e)}"
            )
            return None
    
    def get_manager_by_team_name(self, team_name: str) -> Optional[str]:
        """
        Try to find manager ID by team name
        This is very limited without proper search API
        """
        # This would require scraping or unofficial APIs
        # For now, return None
        return None
    
    def get_classic_league_page(self, league_id: int, page: int = 1) -> Optional[Dict]:
        """Fetch a single classic league standings page"""
        try:
            url = f"{self.base_url}/leagues-classic/{league_id}/standings/"
            params = {'page_standings': page}
            self._rate_limit()
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            self.logger.error(f"Error fetching league {league_id} page {page}: {exc}")
            return None

