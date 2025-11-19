import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, OSError):
    lgb = None
    HAS_LIGHTGBM = False
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3

class MinutesPredictor:
    """Predicts player minutes using a two-stage approach:
    1. Classification: Will player start? (>60 minutes)
    2. Regression: Expected minutes if playing
    """
    
    def __init__(self):
        self.start_classifier = None
        self.minutes_regressor = None
        self.feature_scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
        
        self.logger = logging.getLogger(__name__)
        
        # Model parameters
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for minutes prediction"""
        try:
            features = df.copy()
            
            # Basic player info
            if 'position' in features.columns:
                pos_encoder = LabelEncoder()
                features['position_encoded'] = pos_encoder.fit_transform(features['position'].fillna('Unknown'))
            
            # Recent form features (mock - would need historical data)
            features['recent_minutes_avg'] = features.get('minutes', 0)  # Last N games average
            features['recent_starts'] = (features.get('minutes', 0) > 60).astype(int)
            
            # Team strength features
            team_strength_cols = [
                'strength_overall_home', 'strength_overall_away',
                'strength_attack_home', 'strength_attack_away',
                'strength_defence_home', 'strength_defence_away'
            ]
            for col in team_strength_cols:
                if col in features.columns:
                    features[col] = features[col].fillna(3)  # Default strength
            
            # Price as proxy for quality
            if 'now_cost' in features.columns:
                features['price_normalized'] = features['now_cost'] / features['now_cost'].max()
            
            # Form and points features
            numeric_cols = ['form', 'total_points', 'points_per_game', 'ict_index', 
                          'influence', 'creativity', 'threat']
            for col in numeric_cols:
                if col in features.columns:
                    features[col] = features[col].fillna(0)
            
            # Injury/availability flags
            if 'chance_of_playing_this_round' in features.columns:
                features['injury_risk'] = (100 - features['chance_of_playing_this_round'].fillna(100)) / 100
            else:
                features['injury_risk'] = 0
            
            # Playing status
            if 'status' in features.columns:
                features['is_available'] = (features['status'] == 'a').astype(int)
            else:
                features['is_available'] = 1
            
            # Select numeric features for modeling
            feature_cols = [
                'position_encoded', 'recent_minutes_avg', 'recent_starts',
                'strength_overall_home', 'strength_overall_away',
                'price_normalized', 'form', 'total_points', 'points_per_game',
                'ict_index', 'influence', 'creativity', 'threat',
                'injury_risk', 'is_available'
            ]
            
            # Keep only available columns
            available_cols = [col for col in feature_cols if col in features.columns]
            result = features[available_cols].fillna(0)
            
            self.feature_names = available_cols
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating features: {str(e)}")
            return pd.DataFrame()
    
    def _prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare training data from current season data"""
        try:
            from backend.src.core.database import FPLDatabase
            db = FPLDatabase()
            
            # Use current season data directly
            players_df = db.get_players_with_stats()
            
            if players_df.empty:
                self.logger.error("No player data available for training")
                return pd.DataFrame(), pd.Series(), pd.Series()
            
            # Create features from current data
            X = self._create_features(players_df)
            
            # Create synthetic targets based on current performance
            # Use form and points as indicators of playing time
            form_scores = players_df['form'].fillna(0)
            points_scores = players_df['total_points'].fillna(0)
            
            # Simulate minutes based on form and performance
            # High form = more likely to start and play more minutes
            base_minutes = np.clip(form_scores * 15, 0, 90)  # Form * 15, capped at 90
            form_variance = np.random.normal(0, 10, len(players_df))  # Add some randomness
            y_minutes = np.clip(base_minutes + form_variance, 0, 90)
            
            # Add some players who don't play (injured, rotation, etc.)
            no_play_prob = 0.2  # 20% chance of not playing
            no_play_mask = np.random.random(len(players_df)) < no_play_prob
            y_minutes[no_play_mask] = 0
            
            y_starts = (y_minutes > 60).astype(int)
            
            if X.empty or len(y_minutes) == 0:
                self.logger.warning("No valid training data created")
                return pd.DataFrame(), pd.Series(), pd.Series()
            
            self.logger.info(f"Prepared training data from current season: {len(X)} samples, {len(X.columns)} features")
            return X, pd.Series(y_minutes), pd.Series(y_starts)
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            return pd.DataFrame(), pd.Series(), pd.Series()
    
    def train(self, retrain: bool = False) -> bool:
        """Train the minutes prediction models"""
        try:
            if self.is_fitted and not retrain:
                self.logger.info("Model already trained. Use retrain=True to retrain.")
                return True
            
            self.logger.info("Starting minutes model training...")
            
            # Prepare data
            X, y_minutes, y_starts = self._prepare_training_data()
            
            if X.empty:
                self.logger.error("No training data available")
                return False
            
            # Split data
            X_train, X_test, y_minutes_train, y_minutes_test, y_starts_train, y_starts_test = \
                train_test_split(X, y_minutes, y_starts, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Train start classifier
            self.logger.info("Training start classifier...")
            self.start_classifier = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            self.start_classifier.fit(X_train_scaled, y_starts_train)
            
            # Evaluate classifier
            y_starts_pred = self.start_classifier.predict(X_test_scaled)
            start_accuracy = accuracy_score(y_starts_test, y_starts_pred)
            self.logger.info(f"Start classifier accuracy: {start_accuracy:.3f}")
            
            # Train minutes regressor (only on players who played)
            playing_mask = y_minutes_train > 0
            if playing_mask.sum() > 10:  # Need minimum samples
                self.logger.info("Training minutes regressor...")
                
                X_playing = X_train_scaled[playing_mask]
                y_playing = y_minutes_train[playing_mask]
                
                self.minutes_regressor = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
                self.minutes_regressor.fit(X_playing, y_playing)
                
                # Evaluate regressor
                test_playing_mask = y_minutes_test > 0
                if test_playing_mask.sum() > 0:
                    X_test_playing = X_test_scaled[test_playing_mask]
                    y_test_playing = y_minutes_test[test_playing_mask]
                    y_minutes_pred = self.minutes_regressor.predict(X_test_playing)
                    minutes_mae = mean_absolute_error(y_test_playing, y_minutes_pred)
                    self.logger.info(f"Minutes regressor MAE: {minutes_mae:.1f}")
            else:
                self.logger.warning("Insufficient data for minutes regressor training")
            
            self.is_fitted = True
            self.logger.info("Minutes model training completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training minutes model: {str(e)}")
            return False
    
    def predict_minutes(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Predict minutes for players"""
        try:
            if not self.is_fitted:
                self.logger.warning("Model not trained. Training now...")
                if not self.train():
                    self.logger.error("Failed to train model")
                    return pd.DataFrame()
            
            # Create features
            X = self._create_features(players_df)
            
            if X.empty:
                self.logger.error("No features created for prediction")
                return pd.DataFrame()
            
            # Scale features
            X_scaled = self.feature_scaler.transform(X)
            
            # Predict start probability
            start_prob = self.start_classifier.predict_proba(X_scaled)[:, 1]
            
            # Predict minutes for all players
            if self.minutes_regressor is not None:
                predicted_minutes = self.minutes_regressor.predict(X_scaled)
                # Clip to reasonable range
                predicted_minutes = np.clip(predicted_minutes, 0, 90)
            else:
                # Fallback: use start probability * average minutes
                predicted_minutes = start_prob * 75  # Rough estimate
            
            # Expected minutes = start_prob * predicted_minutes
            expected_minutes = start_prob * predicted_minutes
            
            # Create results dataframe
            results = pd.DataFrame({
                'player_id': players_df.get('id', range(len(players_df))),
                'web_name': players_df.get('web_name', ''),
                'position': players_df.get('position', ''),
                'team_name': players_df.get('team_name', ''),
                'start_probability': start_prob,
                'predicted_minutes_if_playing': predicted_minutes,
                'expected_minutes': expected_minutes,
                'minutes_category': pd.cut(expected_minutes, 
                                         bins=[0, 1, 60, 90], 
                                         labels=['No Play', 'Sub', 'Starter'],
                                         include_lowest=True)
            })
            
            return results.sort_values('expected_minutes', ascending=False)
            
        except Exception as e:
            self.logger.error(f"Error predicting minutes: {str(e)}")
            return pd.DataFrame()
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained models"""
        try:
            if not self.is_fitted:
                return pd.DataFrame()
            
            importance_df = pd.DataFrame()
            
            if self.start_classifier is not None and hasattr(self.start_classifier, 'feature_importances_'):
                importance_df['feature'] = self.feature_names
                importance_df['start_classifier_importance'] = self.start_classifier.feature_importances_
            
            if self.minutes_regressor is not None and hasattr(self.minutes_regressor, 'feature_importances_'):
                if importance_df.empty:
                    importance_df['feature'] = self.feature_names
                importance_df['minutes_regressor_importance'] = self.minutes_regressor.feature_importances_
            
            return importance_df.sort_values(
                importance_df.columns[-1], ascending=False
            ) if not importance_df.empty else pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return pd.DataFrame()
    
    def is_trained(self) -> bool:
        """Check if model is trained and ready"""
        return self.is_fitted and self.start_classifier is not None
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model to file"""
        try:
            if not self.is_fitted:
                self.logger.error("No trained model to save")
                return False
            
            model_data = {
                'start_classifier': self.start_classifier,
                'minutes_regressor': self.minutes_regressor,
                'feature_scaler': self.feature_scaler,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.start_classifier = model_data['start_classifier']
            self.minutes_regressor = model_data['minutes_regressor']
            self.feature_scaler = model_data['feature_scaler']
            self.feature_names = model_data['feature_names']
            self.is_fitted = model_data['is_fitted']
            
            self.logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
