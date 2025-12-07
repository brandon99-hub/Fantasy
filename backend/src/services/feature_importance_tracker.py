"""
Feature Importance Tracker

Tracks and analyzes feature importance over time to enable adaptive feature weighting.
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from backend.src.core.postgres_db import PostgresManagerDB


logger = logging.getLogger(__name__)


class FeatureImportanceTracker:
    """Track and analyze feature importance trends"""
    
    def __init__(self):
        self.postgres_db = PostgresManagerDB()
        self.logger = logging.getLogger(__name__)
    
    def get_feature_trends(
        self,
        feature_name: str = None,
        recent_gws: int = 10
    ) -> pd.DataFrame:
        """
        Get feature importance trends over time
        
        Args:
            feature_name: Specific feature to analyze (None for all)
            recent_gws: Number of recent gameweeks to analyze
        
        Returns:
            DataFrame with feature importance trends
        """
        trends = self.postgres_db.get_feature_importance_trends(feature_name, recent_gws)
        
        if not trends:
            return pd.DataFrame()
        
        return pd.DataFrame(trends)
    
    def get_top_features(
        self,
        gameweek: int = None,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get top N most important features
        
        Args:
            gameweek: Specific gameweek (None for latest)
            top_n: Number of features to return
        
        Returns:
            List of top features with importance scores
        """
        try:
            with self.postgres_db.get_connection() as conn:
                with conn.cursor() as cur:
                    if gameweek:
                        where_clause = "WHERE gameweek = %s"
                        params = [gameweek, top_n]
                    else:
                        where_clause = "WHERE gameweek = (SELECT MAX(gameweek) FROM feature_importance_history)"
                        params = [top_n]
                    
                    cur.execute(f"""
                        SELECT 
                            feature_name,
                            AVG(importance_score) as avg_importance,
                            AVG(rank) as avg_rank,
                            COUNT(DISTINCT model_name) as model_count,
                            feature_type
                        FROM feature_importance_history
                        {where_clause}
                        GROUP BY feature_name, feature_type
                        ORDER BY avg_importance DESC
                        LIMIT %s;
                    """, params)
                    
                    return cur.fetchall()
                    
        except Exception as e:
            self.logger.error(f"Error getting top features: {str(e)}")
            return []
    
    def analyze_feature_stability(
        self,
        feature_name: str,
        num_gameweeks: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze stability of a feature's importance over time
        
        Args:
            feature_name: Feature to analyze
            num_gameweeks: Number of gameweeks to analyze
        
        Returns:
            Dictionary with stability metrics
        """
        trends = self.get_feature_trends(feature_name, num_gameweeks)
        
        if trends.empty:
            return {'error': 'No data available'}
        
        importance_values = trends['avg_importance'].values
        
        return {
            'feature_name': feature_name,
            'mean_importance': float(np.mean(importance_values)),
            'std_importance': float(np.std(importance_values)),
            'min_importance': float(np.min(importance_values)),
            'max_importance': float(np.max(importance_values)),
            'coefficient_of_variation': float(np.std(importance_values) / np.mean(importance_values)) if np.mean(importance_values) > 0 else 0,
            'trend': 'increasing' if importance_values[-1] > importance_values[0] else 'decreasing',
            'gameweeks_analyzed': len(importance_values)
        }
    
    def get_feature_type_distribution(
        self,
        gameweek: int = None
    ) -> Dict[str, Any]:
        """
        Get distribution of feature importance by feature type
        
        Args:
            gameweek: Specific gameweek (None for latest)
        
        Returns:
            Dictionary with importance by feature type
        """
        try:
            with self.postgres_db.get_connection() as conn:
                with conn.cursor() as cur:
                    if gameweek:
                        where_clause = "WHERE gameweek = %s"
                        params = [gameweek]
                    else:
                        where_clause = "WHERE gameweek = (SELECT MAX(gameweek) FROM feature_importance_history)"
                        params = []
                    
                    cur.execute(f"""
                        SELECT 
                            feature_type,
                            COUNT(*) as feature_count,
                            AVG(importance_score) as avg_importance,
                            SUM(importance_score) as total_importance
                        FROM feature_importance_history
                        {where_clause}
                        GROUP BY feature_type
                        ORDER BY total_importance DESC;
                    """, params)
                    
                    results = cur.fetchall()
                    
                    return {
                        'by_type': results,
                        'total_features': sum(r['feature_count'] for r in results)
                    }
                    
        except Exception as e:
            self.logger.error(f"Error getting feature type distribution: {str(e)}")
            return {}
    
    def recommend_feature_weights(
        self,
        recent_gws: int = 5
    ) -> Dict[str, float]:
        """
        Recommend feature weights based on recent importance
        
        Args:
            recent_gws: Number of recent gameweeks to consider
        
        Returns:
            Dictionary mapping feature names to recommended weights
        """
        try:
            with self.postgres_db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            feature_name,
                            AVG(importance_score) as avg_importance
                        FROM feature_importance_history
                        WHERE gameweek >= (SELECT MAX(gameweek) FROM feature_importance_history) - %s
                        GROUP BY feature_name
                        ORDER BY avg_importance DESC;
                    """, (recent_gws,))
                    
                    results = cur.fetchall()
                    
                    if not results:
                        return {}
                    
                    # Normalize to sum to 1.0
                    total_importance = sum(r['avg_importance'] for r in results)
                    
                    weights = {}
                    for row in results:
                        weights[row['feature_name']] = float(row['avg_importance']) / total_importance
                    
                    return weights
                    
        except Exception as e:
            self.logger.error(f"Error recommending feature weights: {str(e)}")
            return {}
    
    def detect_feature_drift(
        self,
        threshold: float = 0.3,
        num_gameweeks: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Detect features with significant importance drift
        
        Args:
            threshold: Minimum change in importance to flag (0-1)
            num_gameweeks: Number of gameweeks to analyze
        
        Returns:
            List of features with significant drift
        """
        drifted_features = []
        
        try:
            with self.postgres_db.get_connection() as conn:
                with conn.cursor() as cur:
                    # Get all features
                    cur.execute("""
                        SELECT DISTINCT feature_name
                        FROM feature_importance_history
                        WHERE gameweek >= (SELECT MAX(gameweek) FROM feature_importance_history) - %s;
                    """, (num_gameweeks,))
                    
                    features = [row['feature_name'] for row in cur.fetchall()]
            
            # Analyze each feature for drift
            for feature in features:
                stability = self.analyze_feature_stability(feature, num_gameweeks)
                
                if 'error' not in stability:
                    # Check if coefficient of variation exceeds threshold
                    if stability['coefficient_of_variation'] > threshold:
                        drifted_features.append({
                            'feature_name': feature,
                            'drift_score': stability['coefficient_of_variation'],
                            'trend': stability['trend'],
                            'mean_importance': stability['mean_importance']
                        })
            
            # Sort by drift score
            drifted_features.sort(key=lambda x: x['drift_score'], reverse=True)
            
            return drifted_features
            
        except Exception as e:
            self.logger.error(f"Error detecting feature drift: {str(e)}")
            return []
    
    def generate_feature_report(
        self,
        gameweek: int = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive feature importance report
        
        Args:
            gameweek: Specific gameweek (None for latest)
        
        Returns:
            Comprehensive report dictionary
        """
        return {
            'top_features': self.get_top_features(gameweek, top_n=15),
            'feature_type_distribution': self.get_feature_type_distribution(gameweek),
            'recommended_weights': self.recommend_feature_weights(recent_gws=5),
            'drifted_features': self.detect_feature_drift(threshold=0.3, num_gameweeks=10),
            'gameweek': gameweek or 'latest'
        }
