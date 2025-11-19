"""
Advanced news and NLP analysis for FPL
Handles press conferences, injury rumors, and real-time news integration
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import re
from dataclasses import dataclass

@dataclass
class NewsAnalysis:
    player_id: int
    player_name: str
    injury_risk: float
    availability_status: str
    news_sentiment: str
    confidence: float
    last_updated: datetime

class NewsAnalyzer:
    """Advanced news and NLP analysis for FPL players"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Injury keywords with weights
        self.injury_keywords = {
            'high_risk': {
                'keywords': ['injured', 'injury', 'out', 'ruled out', 'surgery', 'fracture', 'torn'],
                'weight': 0.9
            },
            'medium_risk': {
                'keywords': ['doubt', 'doubtful', 'fitness test', 'knock', 'strain', 'muscle'],
                'weight': 0.6
            },
            'low_risk': {
                'keywords': ['minor', 'precaution', 'rest', 'fatigue', 'tired'],
                'weight': 0.3
            }
        }
        
        # Positive keywords
        self.positive_keywords = {
            'keywords': ['fit', 'available', 'training', 'ready', 'recovered', 'back'],
            'weight': -0.5  # Negative weight reduces injury risk
        }
        
        # News sources (would be configured in production)
        self.news_sources = [
            'https://www.premierleague.com/news',
            'https://www.bbc.com/sport/football',
            'https://www.skysports.com/football'
        ]
    
    def analyze_player_news(self, players_df: pd.DataFrame) -> List[NewsAnalysis]:
        """Analyze news for all players"""
        try:
            analyses = []
            
            for _, player in players_df.iterrows():
                analysis = self._analyze_single_player(player)
                if analysis:
                    analyses.append(analysis)
            
            self.logger.info(f"Analyzed news for {len(analyses)} players")
            return analyses
            
        except Exception as e:
            self.logger.error(f"Error analyzing player news: {str(e)}")
            return []
    
    def _analyze_single_player(self, player: pd.Series) -> Optional[NewsAnalysis]:
        """Analyze news for a single player"""
        try:
            player_id = player.get('id')
            player_name = player.get('web_name', '')
            news_text = player.get('news', '')
            
            if not news_text or pd.isna(news_text):
                return None
            
            # Analyze injury risk
            injury_risk = self._calculate_injury_risk(news_text)
            
            # Determine availability status
            availability_status = self._determine_availability_status(injury_risk, news_text)
            
            # Analyze sentiment
            sentiment = self._analyze_sentiment(news_text)
            
            # Calculate confidence
            confidence = self._calculate_confidence(news_text, injury_risk)
            
            return NewsAnalysis(
                player_id=player_id,
                player_name=player_name,
                injury_risk=injury_risk,
                availability_status=availability_status,
                news_sentiment=sentiment,
                confidence=confidence,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing player {player.get('web_name', 'Unknown')}: {str(e)}")
            return None
    
    def _calculate_injury_risk(self, news_text: str) -> float:
        """Calculate injury risk from news text"""
        try:
            if not news_text or pd.isna(news_text):
                return 0.0
            
            text_lower = news_text.lower()
            total_risk = 0.0
            keyword_count = 0
            
            # Check injury keywords
            for risk_level, data in self.injury_keywords.items():
                for keyword in data['keywords']:
                    if keyword in text_lower:
                        total_risk += data['weight']
                        keyword_count += 1
            
            # Check positive keywords
            for keyword in self.positive_keywords['keywords']:
                if keyword in text_lower:
                    total_risk += self.positive_keywords['weight']
                    keyword_count += 1
            
            # Normalize risk score
            if keyword_count > 0:
                normalized_risk = min(1.0, max(0.0, total_risk / keyword_count))
            else:
                normalized_risk = 0.0
            
            return normalized_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating injury risk: {str(e)}")
            return 0.0
    
    def _determine_availability_status(self, injury_risk: float, news_text: str) -> str:
        """Determine player availability status"""
        try:
            if injury_risk >= 0.7:
                return "Unavailable"
            elif injury_risk >= 0.4:
                return "Doubtful"
            elif injury_risk >= 0.1:
                return "Minor Doubt"
            else:
                return "Available"
                
        except Exception as e:
            self.logger.error(f"Error determining availability status: {str(e)}")
            return "Unknown"
    
    def _analyze_sentiment(self, news_text: str) -> str:
        """Analyze news sentiment"""
        try:
            if not news_text or pd.isna(news_text):
                return "Neutral"
            
            text_lower = news_text.lower()
            
            # Count positive and negative indicators
            positive_indicators = ['fit', 'available', 'training', 'ready', 'recovered', 'back', 'good']
            negative_indicators = ['injured', 'out', 'doubt', 'problem', 'issue', 'concern']
            
            positive_count = sum(1 for word in positive_indicators if word in text_lower)
            negative_count = sum(1 for word in negative_indicators if word in text_lower)
            
            if positive_count > negative_count:
                return "Positive"
            elif negative_count > positive_count:
                return "Negative"
            else:
                return "Neutral"
                
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {str(e)}")
            return "Neutral"
    
    def _calculate_confidence(self, news_text: str, injury_risk: float) -> float:
        """Calculate confidence in the analysis"""
        try:
            if not news_text or pd.isna(news_text):
                return 0.0
            
            # Base confidence on text length and keyword presence
            text_length = len(news_text)
            keyword_presence = 1.0 if any(keyword in news_text.lower() 
                                         for keyword_group in self.injury_keywords.values() 
                                         for keyword in keyword_group['keywords']) else 0.5
            
            # Confidence increases with more detailed news
            length_factor = min(1.0, text_length / 100)  # Normalize to 0-1
            confidence = (length_factor + keyword_presence) / 2
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0
    
    def get_injury_updates(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Get injury updates for players"""
        try:
            analyses = self.analyze_player_news(players_df)
            
            if not analyses:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for analysis in analyses:
                data.append({
                    'player_id': analysis.player_id,
                    'player_name': analysis.player_name,
                    'injury_risk': analysis.injury_risk,
                    'availability_status': analysis.availability_status,
                    'news_sentiment': analysis.news_sentiment,
                    'confidence': analysis.confidence,
                    'last_updated': analysis.last_updated
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            self.logger.error(f"Error getting injury updates: {str(e)}")
            return pd.DataFrame()
    
    def identify_high_risk_players(self, players_df: pd.DataFrame, threshold: float = 0.6) -> List[Dict]:
        """Identify high-risk players based on news analysis"""
        try:
            injury_updates = self.get_injury_updates(players_df)
            
            if injury_updates.empty:
                return []
            
            high_risk = injury_updates[
                (injury_updates['injury_risk'] >= threshold) &
                (injury_updates['confidence'] >= 0.5)
            ]
            
            return high_risk.to_dict('records')
            
        except Exception as e:
            self.logger.error(f"Error identifying high-risk players: {str(e)}")
            return []
    
    def get_news_summary(self, players_df: pd.DataFrame) -> Dict:
        """Get overall news summary"""
        try:
            injury_updates = self.get_injury_updates(players_df)
            
            if injury_updates.empty:
                return {
                    'total_players': len(players_df),
                    'analyzed_players': 0,
                    'high_risk_count': 0,
                    'doubtful_count': 0,
                    'available_count': 0
                }
            
            summary = {
                'total_players': len(players_df),
                'analyzed_players': len(injury_updates),
                'high_risk_count': len(injury_updates[injury_updates['injury_risk'] >= 0.7]),
                'doubtful_count': len(injury_updates[injury_updates['injury_risk'] >= 0.4]),
                'available_count': len(injury_updates[injury_updates['injury_risk'] < 0.4]),
                'avg_confidence': injury_updates['confidence'].mean(),
                'last_updated': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting news summary: {str(e)}")
            return {}
