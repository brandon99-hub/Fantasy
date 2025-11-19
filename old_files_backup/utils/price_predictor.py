"""
Price change prediction for FPL players
Analyzes transfer trends, ownership changes, and performance to predict price movements
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

class PricePredictor:
    """Predicts FPL player price changes based on various factors"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.feature_scaler = None
        self.is_trained = False
        self.feature_names = []
        
        # Price change thresholds (FPL rules)
        self.price_rise_threshold = 0.1  # 0.1M rise threshold
        self.price_fall_threshold = -0.1  # 0.1M fall threshold
        
        # Transfer momentum weights
        self.transfer_momentum_weights = {
            'high_ownership': 0.3,  # >20% ownership
            'medium_ownership': 0.5,  # 5-20% ownership
            'low_ownership': 0.8   # <5% ownership
        }
    
    def _create_price_features(self, players_df: pd.DataFrame, 
                              history_df: pd.DataFrame = None) -> pd.DataFrame:
        """Create features for price prediction"""
        try:
            features = players_df.copy()
            
            # Basic performance metrics
            numeric_features = [
                'form', 'total_points', 'points_per_game', 'goals_scored', 'assists',
                'clean_sheets', 'bonus', 'bps', 'ict_index', 'influence', 
                'creativity', 'threat', 'selected_by_percent'
            ]
            
            for feature in numeric_features:
                if feature in features.columns:
                    features[feature] = pd.to_numeric(features[feature], errors='coerce').fillna(0)
            
            # Transfer momentum features
            if 'transfers_in_event' in features.columns and 'transfers_out_event' in features.columns:
                features['net_transfers'] = features['transfers_in_event'] - features['transfers_out_event']
                features['transfer_ratio'] = features['transfers_in_event'] / (features['transfers_out_event'] + 1)
                
                # Transfer momentum by ownership level
                features['ownership_level'] = pd.cut(
                    features['selected_by_percent'],
                    bins=[0, 5, 20, 100],
                    labels=['low', 'medium', 'high'],
                    include_lowest=True
                )
                
                features['weighted_transfer_momentum'] = features.apply(
                    lambda row: self._calculate_weighted_momentum(row), axis=1
                )
            
            # Price change indicators
            if 'value_form' in features.columns and 'value_season' in features.columns:
                features['price_trend'] = features['value_form'] - features['value_season']
                features['price_acceleration'] = features['value_form'] - features['value_season']
            
            # Performance vs expectation
            if 'ep_this' in features.columns and 'total_points' in features.columns:
                features['performance_vs_expectation'] = features['total_points'] - features['ep_this']
            
            # Recent form trends
            if history_df is not None and not history_df.empty:
                recent_form = self._calculate_recent_form_trends(history_df)
                features = features.merge(recent_form, left_on='id', right_on='element', how='left')
            
            # Position-specific features
            if 'position' in features.columns:
                position_dummies = pd.get_dummies(features['position'], prefix='pos')
                features = pd.concat([features, position_dummies], axis=1)
            
            # Team strength impact
            team_strength_cols = [
                'strength_overall_home', 'strength_overall_away',
                'strength_attack_home', 'strength_attack_away',
                'strength_defence_home', 'strength_defence_away'
            ]
            for col in team_strength_cols:
                if col in features.columns:
                    features[col] = features[col].fillna(3)
            
            # Select final features
            model_features = [
                'form', 'total_points', 'points_per_game', 'goals_scored', 'assists',
                'clean_sheets', 'bonus', 'bps', 'ict_index', 'influence', 'creativity', 'threat',
                'selected_by_percent', 'net_transfers', 'transfer_ratio', 'weighted_transfer_momentum',
                'price_trend', 'performance_vs_expectation'
            ]
            
            # Add position dummies
            pos_cols = [col for col in features.columns if col.startswith('pos_')]
            model_features.extend(pos_cols)
            
            # Add team strength features
            available_strength_cols = [col for col in team_strength_cols if col in features.columns]
            model_features.extend(available_strength_cols)
            
            # Keep only available features
            available_features = [col for col in model_features if col in features.columns]
            result = features[available_features].fillna(0)
            
            self.feature_names = available_features
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating price features: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_weighted_momentum(self, row) -> float:
        """Calculate weighted transfer momentum based on ownership level"""
        try:
            ownership_level = row.get('ownership_level', 'medium')
            net_transfers = row.get('net_transfers', 0)
            selected_by_percent = row.get('selected_by_percent', 0)
            
            weight = self.transfer_momentum_weights.get(ownership_level, 0.5)
            return net_transfers * weight * (1 + selected_by_percent / 100)
            
        except Exception as e:
            self.logger.error(f"Error calculating weighted momentum: {str(e)}")
            return 0.0
    
    def _calculate_recent_form_trends(self, history_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate recent form trends from historical data"""
        try:
            if history_df.empty:
                return pd.DataFrame()
            
            # Get last 5 games for each player
            recent_history = history_df.groupby('element').tail(5)
            
            # Calculate trends
            trends = recent_history.groupby('element').agg({
                'total_points': ['mean', 'std', 'min', 'max'],
                'minutes': 'mean',
                'goals_scored': 'sum',
                'assists': 'sum',
                'clean_sheets': 'sum'
            }).reset_index()
            
            # Flatten column names
            trends.columns = ['element', 'recent_avg_points', 'points_volatility', 
                            'min_points', 'max_points', 'recent_avg_minutes',
                            'recent_goals', 'recent_assists', 'recent_clean_sheets']
            
            # Calculate form trend (slope)
            form_trends = []
            for element in trends['element'].unique():
                player_history = recent_history[recent_history['element'] == element].sort_values('round')
                if len(player_history) >= 3:
                    # Simple linear trend
                    x = np.arange(len(player_history))
                    y = player_history['total_points'].values
                    trend = np.polyfit(x, y, 1)[0] if len(y) > 1 else 0
                else:
                    trend = 0
                form_trends.append({'element': element, 'form_trend': trend})
            
            form_trend_df = pd.DataFrame(form_trends)
            trends = trends.merge(form_trend_df, on='element', how='left')
            
            return trends.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating recent form trends: {str(e)}")
            return pd.DataFrame()
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data for price prediction model"""
        try:
            from database import FPLDatabase
            db = FPLDatabase()
            
            # Use current season data directly
            players_df = db.get_players_with_stats()
            
            if players_df.empty:
                self.logger.warning("No player data available for price prediction")
                return pd.DataFrame(), pd.Series()
            
            # Use current season data for training
            X = self._create_price_features(players_df)
            
            # Create synthetic price change targets based on current performance
            # This is a simplified approach - in production you'd want historical price data
            form_scores = players_df['form'].fillna(0)
            ownership_scores = players_df['selected_by_percent'].fillna(0)
            
            # Simulate price changes based on form and ownership
            # High form + high ownership = likely price rise
            # Low form + low ownership = likely price fall
            price_changes = (
                (form_scores - 3.0) * 0.02 +  # Form effect
                (ownership_scores - 5.0) * 0.001 +  # Ownership effect
                np.random.normal(0, 0.05, len(players_df))  # Random noise
            )
            
            y = pd.Series(price_changes, index=players_df.index)
            
            self.logger.info(f"Prepared price training data from current season: {len(X)} samples, {len(X.columns)} features")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing price training data: {str(e)}")
            return pd.DataFrame(), pd.Series()
    
    def train(self, retrain: bool = False) -> bool:
        """Train the price prediction model"""
        try:
            if self.is_trained and not retrain:
                self.logger.info("Price model already trained. Use retrain=True to retrain.")
                return True
            
            self.logger.info("Starting price prediction model training...")
            
            # Prepare training data
            X, y = self.prepare_training_data()
            
            if X.empty:
                self.logger.error("No training data available for price model")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train ensemble model
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            self.logger.info(f"Price model performance - MAE: {mae:.3f}, RMSE: {rmse:.3f}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error training price model: {str(e)}")
            return False
    
    def predict_price_changes(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Predict price changes for players"""
        try:
            if not self.is_trained:
                self.logger.warning("Price model not trained. Training now...")
                if not self.train():
                    self.logger.error("Failed to train price model")
                    return pd.DataFrame()
            
            # Create features
            X = self._create_price_features(players_df)
            
            if X.empty:
                self.logger.error("No features created for price prediction")
                return pd.DataFrame()
            
            # Predict price changes
            predicted_changes = self.model.predict(X)
            
            # Create results
            results = players_df[['id', 'web_name', 'position', 'team_name', 'now_cost']].copy()
            results['predicted_price_change'] = predicted_changes
            results['price_rise_probability'] = (predicted_changes > self.price_rise_threshold).astype(float)
            results['price_fall_probability'] = (predicted_changes < self.price_fall_threshold).astype(float)
            results['price_stable_probability'] = (
                (predicted_changes >= self.price_fall_threshold) & 
                (predicted_changes <= self.price_rise_threshold)
            ).astype(float)
            
            # Add risk categories
            results['price_risk'] = pd.cut(
                np.abs(predicted_changes),
                bins=[0, 0.05, 0.15, 0.3, 1.0],
                labels=['Low', 'Medium', 'High', 'Very High'],
                include_lowest=True
            )
            
            return results.sort_values('predicted_price_change', ascending=False)
            
        except Exception as e:
            self.logger.error(f"Error predicting price changes: {str(e)}")
            return pd.DataFrame()
    
    def get_transfer_timing_recommendation(self, player_id: int, players_df: pd.DataFrame) -> Dict:
        """Get transfer timing recommendation for a specific player"""
        try:
            player_data = players_df[players_df['id'] == player_id]
            if player_data.empty:
                return {}
            
            player = player_data.iloc[0]
            
            # Get price predictions
            price_predictions = self.predict_price_changes(players_df)
            player_prediction = price_predictions[price_predictions['id'] == player_id]
            
            if player_prediction.empty:
                return {}
            
            pred = player_prediction.iloc[0]
            
            # Determine recommendation
            if pred['price_rise_probability'] > 0.7:
                recommendation = "Transfer in soon - price likely to rise"
                urgency = "High"
            elif pred['price_fall_probability'] > 0.7:
                recommendation = "Wait - price likely to fall"
                urgency = "Low"
            else:
                recommendation = "No immediate price pressure"
                urgency = "Medium"
            
            return {
                'player_id': player_id,
                'player_name': player['web_name'],
                'current_price': player['now_cost'] / 10.0,
                'predicted_change': round(pred['predicted_price_change'], 3),
                'rise_probability': round(pred['price_rise_probability'], 2),
                'fall_probability': round(pred['price_fall_probability'], 2),
                'recommendation': recommendation,
                'urgency': urgency,
                'risk_level': pred['price_risk']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting transfer timing recommendation: {str(e)}")
            return {}
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model"""
        try:
            if not self.is_trained:
                self.logger.error("No trained price model to save")
                return False
            
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Price model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving price model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            
            self.logger.info(f"Price model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading price model: {str(e)}")
            return False
