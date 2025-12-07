import logging
from typing import List, Dict, Any, Optional

import requests

from backend.src.core.config import get_settings


def normalize_team_name(name: str) -> str:
    """Normalize team names for fuzzy matching."""
    if not name:
        return ""
    cleaned = (
        name.lower()
        .replace("fc", "")
        .replace("afc", "")
        .replace(".", "")
        .replace("club", "")
        .strip()
    )
    replacements = {
        "manchester united": "man united",
        "manchester city": "man city",
        "tottenham hotspur": "tottenham",
        "newcastle united": "newcastle",
        "wolverhampton wanderers": "wolves",
    }
    return replacements.get(cleaned, cleaned)


class OddsAPIClient:
    """Client wrapper for TheOddsAPI."""

    SPORT_KEY = "soccer_epl"

    def __init__(self, api_key: Optional[str] = None):
        self.settings = get_settings()
        self.api_key = api_key or self.settings.ODDS_API_KEY
        self.logger = logging.getLogger(__name__)
        self.base_url = self.settings.ODDS_API_BASE_URL.rstrip("/")
        self.region = self.settings.ODDS_API_REGION
        self.markets = self.settings.ODDS_API_MARKETS
        self.bookmakers = self.settings.ODDS_API_BOOKMAKERS

    def fetch_fixture_odds(self) -> List[Dict[str, Any]]:
        """Fetch upcoming fixture odds from TheOddsAPI."""
        if not self.api_key:
            self.logger.warning("ODDS_API_KEY is not configured; skipping odds fetch.")
            return []

        url = f"{self.base_url}/sports/{self.SPORT_KEY}/odds"
        markets_sequence = [self.markets]
        if "team_totals" in self.markets:
            markets_sequence.append("totals")

        for markets in markets_sequence:
            params = {
                "apiKey": self.api_key,
                "regions": self.region,
                "markets": markets,
                "bookmakers": self.bookmakers,
                "oddsFormat": "decimal",
            }

            try:
                response = requests.get(url, params=params, timeout=20)
                if response.status_code == 422 and markets != "totals":
                    self.logger.warning("Markets not available on current plan, falling back to totals.")
                    continue
                response.raise_for_status()
                data = response.json()
                if isinstance(data, dict) and data.get("message"):
                    self.logger.error(f"TheOddsAPI error: {data.get('message')}")
                    return []
                if isinstance(data, list) and data:
                    if markets == "totals":
                        self.logger.info("Using totals market fallback for odds.")
                    return data
            except requests.RequestException as exc:
                self.logger.error(f"Error fetching odds data: {exc}")
                return []

        return []

