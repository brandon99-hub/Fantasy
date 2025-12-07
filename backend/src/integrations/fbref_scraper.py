"""
FBRef web scraper for xG/xA data

Scrapes player-level expected goals and assists data from FBRef.com
with rate limiting and player name matching.
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import httpx
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter for web scraping"""
    
    def __init__(self, requests_per_minute: int = 20):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request = 0
    
    async def wait(self):
        """Wait if necessary to respect rate limit"""
        now = asyncio.get_event_loop().time()
        time_since_last = now - self.last_request
        
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request = asyncio.get_event_loop().time()


class FBRefScraper:
    """Scrape xG/xA data from FBRef"""
    
    def __init__(self, requests_per_minute: int = 20):
        self.base_url = "https://fbref.com"
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.session = None
        logger.info("FBRef scraper initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = httpx.AsyncClient(
            timeout=30.0,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.aclose()
    
    async def scrape_player_xg(self, season: str = "2024-2025") -> pd.DataFrame:
        """
        Scrape player xG/xA data for Premier League
        
        Args:
            season: Season in format "2024-2025"
        
        Returns:
            DataFrame with columns: player_name, team, xg, xa, xg_per_90, xa_per_90, minutes
        """
        try:
            # Premier League stats URL
            url = f"{self.base_url}/en/comps/9/stats/Premier-League-Stats"
            
            logger.info(f"Scraping FBRef for season {season}...")
            await self.rate_limiter.wait()
            
            response = await self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the stats table
            stats_table = soup.find('table', {'id': 'stats_standard'})
            
            if not stats_table:
                logger.error("Could not find stats table on FBRef")
                return pd.DataFrame()
            
            # Parse table
            players_data = []
            rows = stats_table.find('tbody').find_all('tr')
            
            for row in rows:
                # Skip header rows
                if row.get('class') and 'thead' in row.get('class'):
                    continue
                
                try:
                    cells = row.find_all(['th', 'td'])
                    
                    player_name = cells[0].text.strip()
                    team = cells[3].text.strip() if len(cells) > 3 else ""
                    
                    # Find xG and xA columns (positions may vary)
                    minutes = self._safe_float(cells[7].text) if len(cells) > 7 else 0
                    
                    # xG is typically around column 20-22
                    xg = 0
                    xa = 0
                    for i, cell in enumerate(cells):
                        if cell.get('data-stat') == 'xg':
                            xg = self._safe_float(cell.text)
                        elif cell.get('data-stat') == 'xg_assist':
                            xa = self._safe_float(cell.text)
                    
                    # Calculate per 90
                    xg_per_90 = (xg / minutes * 90) if minutes > 0 else 0
                    xa_per_90 = (xa / minutes * 90) if minutes > 0 else 0
                    
                    players_data.append({
                        'player_name': player_name,
                        'team': team,
                        'xg': xg,
                        'xa': xa,
                        'xg_per_90': round(xg_per_90, 2),
                        'xa_per_90': round(xa_per_90, 2),
                        'minutes': minutes,
                        'source': 'fbref',
                        'scraped_at': datetime.now()
                    })
                
                except Exception as e:
                    logger.warning(f"Error parsing row: {e}")
                    continue
            
            df = pd.DataFrame(players_data)
            logger.info(f"Scraped {len(df)} players from FBRef")
            
            return df
        
        except Exception as e:
            logger.error(f"Error scraping FBRef: {e}")
            return pd.DataFrame()
    
    def match_player_names(
        self, 
        fbref_df: pd.DataFrame, 
        fpl_players: pd.DataFrame,
        threshold: int = 80
    ) -> pd.DataFrame:
        """
        Match FBRef player names to FPL player IDs using fuzzy matching
        
        Args:
            fbref_df: DataFrame from FBRef scraping
            fpl_players: FPL players DataFrame
            threshold: Minimum fuzzy match score (0-100)
        
        Returns:
            DataFrame with matched player_id and xG data
        """
        matched_data = []
        
        for _, fbref_player in fbref_df.iterrows():
            fbref_name = fbref_player['player_name']
            fbref_team = fbref_player['team']
            
            best_match = None
            best_score = 0
            
            # Try to match by name and team
            for _, fpl_player in fpl_players.iterrows():
                # Create name variations for matching
                fpl_full_name = f"{fpl_player['first_name']} {fpl_player['second_name']}"
                fpl_web_name = fpl_player['web_name']
                fpl_team = fpl_player['team_name']
                
                # Calculate fuzzy match scores
                score_full = fuzz.ratio(fbref_name.lower(), fpl_full_name.lower())
                score_web = fuzz.ratio(fbref_name.lower(), fpl_web_name.lower())
                score = max(score_full, score_web)
                
                # Boost score if teams match
                if fbref_team and fpl_team:
                    team_match = fuzz.partial_ratio(fbref_team.lower(), fpl_team.lower())
                    if team_match > 80:
                        score += 10
                
                if score > best_score:
                    best_score = score
                    best_match = fpl_player
            
            # Only add if match is good enough
            if best_score >= threshold and best_match is not None:
                matched_data.append({
                    'player_id': best_match['id'],
                    'player_name': fbref_name,
                    'fpl_name': f"{best_match['first_name']} {best_match['second_name']}",
                    'match_score': best_score,
                    'xg': fbref_player['xg'],
                    'xa': fbref_player['xa'],
                    'xg_per_90': fbref_player['xg_per_90'],
                    'xa_per_90': fbref_player['xa_per_90'],
                    'source': 'fbref'
                })
            else:
                logger.debug(f"No match found for {fbref_name} (best score: {best_score})")
        
        matched_df = pd.DataFrame(matched_data)
        match_rate = len(matched_df) / len(fbref_df) * 100 if len(fbref_df) > 0 else 0
        logger.info(f"Matched {len(matched_df)}/{len(fbref_df)} players ({match_rate:.1f}%)")
        
        return matched_df
    
    def _safe_float(self, value: str) -> float:
        """Safely convert string to float"""
        try:
            # Remove commas and convert
            cleaned = value.replace(',', '').strip()
            return float(cleaned) if cleaned else 0.0
        except (ValueError, AttributeError):
            return 0.0


# Example usage
async def main():
    """Test the scraper"""
    async with FBRefScraper() as scraper:
        xg_data = await scraper.scrape_player_xg()
        print(f"Scraped {len(xg_data)} players")
        print(xg_data.head())


if __name__ == "__main__":
    asyncio.run(main())
