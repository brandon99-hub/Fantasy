import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, OSError):
    lgb = None
    HAS_LIGHTGBM = False
import pickle
import logging
from typing import Dict, List, Optional, Tuple
from models.minutes_model import MinutesPredictor

class PointsPredictor:
    """Predicts FPL points using minutes predictions and player features"""
    
    def __init__(self):
        self.regressor = None
        self.feature_scaler = StandardScaler()
        self.minutes_model = MinutesPredictor()
        self.is_fitted = False
        self.feature_names = []
        
        self.logger = logging.getLogger(__name__)
        
        # Position-specific scoring patterns
        self.position_weights = {
            'GKP': {'clean_sheet': 4, 'save': 0.33, 'goals': 6, 'assists': 3},
            'DEF': {'clean_sheet': 4, 'goals': 6, 'assists': 3},
            'MID': {'clean_sheet': 1, 'goals': 5, 'assists': 3},
            'FWD': {'goals': 4, 'assists': 3}
        }
    
    def _create_features(self, df: pd.DataFrame, minutes_predictions: pd.DataFrame = None) -> pd.DataFrame:
        """Create features for points prediction"""
        try:
            features = df.copy()
            
            # Merge minutes predictions if provided
            if minutes_predictions is not None and not minutes_predictions.empty:
                features = features.merge(
                    minutes_predictions[['player_id', 'expected_minutes', 'start_probability']],
                    left_on='id', right_on='player_id', how='left'
                )
            else:
                # Use actual minutes as fallback
                features['expected_minutes'] = features.get('minutes', 0)
                features['start_probability'] = (features.get('minutes', 0) > 60).astype(int)
            
            # Fill missing values
            features['expected_minutes'] = features['expected_minutes'].fillna(0)
            features['start_probability'] = features['start_probability'].fillna(0)
            
            # Basic scoring features
            numeric_features = [
                'form', 'total_points', 'points_per_game', 'goals_scored', 'assists',
                'clean_sheets', 'bonus', 'bps', 'ict_index', 'influence', 
                'creativity', 'threat', 'selected_by_percent'
            ]
            
            for feature in numeric_features:
                if feature in features.columns:
                    features[feature] = pd.to_numeric(features[feature], errors='coerce').fillna(0)
            
            # Position encoding
            if 'position' in features.columns:
                position_dummies = pd.get_dummies(features['position'], prefix='pos')
                features = pd.concat([features, position_dummies], axis=1)
            
            # Price as quality indicator
            if 'now_cost' in features.columns:
                features['price_value'] = features['now_cost'] / 10.0  # Convert to £m
                features['value_ratio'] = features['total_points'] / (features['now_cost'] + 1)
            
            # Recent performance indicators
            features['form_trend'] = features.get('form', 0)
            features['season_avg'] = features.get('points_per_game', 0)
            
            # Team strength features
            team_strength_cols = [
                'strength_overall_home', 'strength_overall_away',
                'strength_attack_home', 'strength_attack_away', 
                'strength_defence_home', 'strength_defence_away'
            ]
            for col in team_strength_cols:
                if col in features.columns:
                    features[col] = features[col].fillna(3)
            
            # Availability and injury risk
            if 'chance_of_playing_this_round' in features.columns:
                features['availability'] = features['chance_of_playing_this_round'].fillna(100) / 100
            else:
                features['availability'] = 1.0
            
            # Select final features for modeling
            model_features = [
                'expected_minutes', 'start_probability', 'form', 'total_points',
                'points_per_game', 'goals_scored', 'assists', 'clean_sheets',
                'bonus', 'bps', 'ict_index', 'influence', 'creativity', 'threat',
                'selected_by_percent', 'price_value', 'value_ratio', 'form_trend',
                'season_avg', 'availability'
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
            self.logger.error(f"Error creating features for points prediction: {str(e)}")
            return pd.DataFrame()
    
    def _prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data with minutes predictions"""
        try:
            from database import FPLDatabase
            db = FPLDatabase()
            
            # Get historical data
            history_df = db.get_player_history_data(limit_gws=50)
            
            if history_df.empty:
                self.logger.warning("No historical data for points model training")
                return pd.DataFrame(), pd.Series()
            
            # Get players data for features
            players_df = db.get_players_with_stats()
            
            # Merge historical performance with player features
            # Note: This is simplified - would need proper time-series alignment
            merged_data = history_df.merge(
                players_df[['id', 'position', 'team_name', 'strength_overall_home', 
                           'strength_overall_away', 'strength_attack_home', 'strength_attack_away',
                           'strength_defence_home', 'strength_defence_away']],
                left_on='element', right_on='id', how='left', suffixes=('', '_current')
            )
            
            # Use historical points as target
            y = merged_data['total_points'].fillna(0)
            
            # Create features (using historical data as proxy for predictions)
            X = self._create_features(merged_data)
            
            if X.empty or len(y) == 0:
                self.logger.warning("No valid training data for points model")
                return pd.DataFrame(), pd.Series()
            
            self.logger.info(f"Prepared points training data: {len(X)} samples, {len(X.columns)} features")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing points training data: {str(e)}")
            return pd.DataFrame(), pd.Series()
    
    def train(self, retrain: bool = False) -> bool:
        """Train the points prediction model"""
        try:
            if self.is_fitted and not retrain:
                self.logger.info("Points model already trained. Use retrain=True to retrain.")
                return True
            
            self.logger.info("Starting points model training...")
            
            # Train minutes model first if not trained
            if not self.minutes_model.is_trained():
                self.logger.info("Training minutes model first...")
                if not self.minutes_model.train():
                    self.logger.error("Failed to train minutes model")
                    return False
            
            # Prepare training data
            X, y = self._prepare_training_data()
            
            if X.empty:
                self.logger.error("No training data available for points model")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Train regressor
            self.regressor = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                random_state=42
            )
            
            self.regressor.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.regressor.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            self.logger.info(f"Points model performance - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
            
            self.is_fitted = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error training points model: {str(e)}")
            return False
    
    def predict_points(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Predict points for players"""
        try:
            if not self.is_fitted:
                self.logger.warning("Points model not trained. Training now...")
                if not self.train():
                    self.logger.error("Failed to train points model")
                    return pd.DataFrame()
            
            # Get minutes predictions first
            minutes_pred = self.minutes_model.predict_minutes(players_df)
            
            if minutes_pred.empty:
                self.logger.error("Failed to get minutes predictions")
                return pd.DataFrame()
            
            # Create features
            X = self._create_features(players_df, minutes_pred)
            
            if X.empty:
                self.logger.error("No features created for points prediction")
                return pd.DataFrame()
            
            # Scale and predict
            X_scaled = self.feature_scaler.transform(X)
            predicted_points = self.regressor.predict(X_scaled)
            
            # Clip to reasonable range (0-25 points)
            predicted_points = np.clip(predicted_points, 0, 25)
            
            # Combine with player info and minutes predictions
            results = players_df[['id', 'web_name', 'position', 'team_name', 'now_cost']].copy()
            results = results.merge(
                minutes_pred[['player_id', 'expected_minutes', 'start_probability']],
                left_on='id', right_on='player_id', how='left'
            )
            
            results['predicted_points'] = predicted_points
            results['points_per_million'] = results['predicted_points'] / (results['now_cost'] / 10.0)
            
            # Add confidence intervals (simple approach)
            results['points_lower'] = np.maximum(0, predicted_points - 2.0)
            results['points_upper'] = predicted_points + 2.0
            
            # Risk categories
            results['risk_category'] = pd.cut(
                results['start_probability'],
                bins=[0, 0.3, 0.7, 1.0],
                labels=['High Risk', 'Medium Risk', 'Low Risk'],
                include_lowest=True
            )
            
            return results.sort_values('predicted_points', ascending=False)
            
        except Exception as e:
            self.logger.error(f"Error predicting points: {str(e)}")
            return pd.DataFrame()
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model"""
        try:
            if not self.is_fitted or self.regressor is None:
                return pd.DataFrame()
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.regressor.feature_importances_
            })
            
            return importance_df.sort_values('importance', ascending=False)
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return pd.DataFrame()
    
    def is_trained(self) -> bool:
        """Check if model is trained"""
        return self.is_fitted and self.regressor is not None
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model"""
        try:
            if not self.is_fitted:
                self.logger.error("No trained points model to save")
                return False
            
            model_data = {
                'regressor': self.regressor,
                'feature_scaler': self.feature_scaler,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted,
                'position_weights': self.position_weights
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Points model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving points model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.regressor = model_data['regressor']
            self.feature_scaler = model_data['feature_scaler']
            self.feature_names = model_data['feature_names']
            self.is_fitted = model_data['is_fitted']
            self.position_weights = model_data.get('position_weights', self.position_weights)
            
            self.logger.info(f"Points model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading points model: {str(e)}")
            return False
