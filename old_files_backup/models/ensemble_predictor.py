"""
Advanced ensemble prediction models for FPL
Combines multiple ML algorithms with proper cross-validation and uncertainty quantification
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
try:
    import xgboost as xgb
    import lightgbm as lgb
    HAS_ADVANCED_MODELS = True
except ImportError:
    HAS_ADVANCED_MODELS = False
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnsemblePredictor:
    """Advanced ensemble prediction system for FPL points"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.is_fitted = False
        self.feature_names = []
        self.model_weights = {}
        self.uncertainty_estimator = None
        
        # Model configurations
        self.model_configs = {
            'gradient_boosting': {
                'model': GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    random_state=42
                ),
                'weight': 0.25
            },
            'random_forest': {
                'model': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42
                ),
                'weight': 0.20
            },
            'ridge': {
                'model': Ridge(alpha=1.0, random_state=42),
                'weight': 0.15
            },
            'neural_network': {
                'model': MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=42
                ),
                'weight': 0.20
            }
        }
        
        # Add advanced models if available
        if HAS_ADVANCED_MODELS:
            self.model_configs.update({
                'xgboost': {
                    'model': xgb.XGBRegressor(
                        n_estimators=200,
                        learning_rate=0.1,
                        max_depth=6,
                        subsample=0.8,
                        random_state=42
                    ),
                    'weight': 0.15
                },
                'lightgbm': {
                    'model': lgb.LGBMRegressor(
                        n_estimators=200,
                        learning_rate=0.1,
                        max_depth=6,
                        subsample=0.8,
                        random_state=42,
                        verbose=-1
                    ),
                    'weight': 0.05
                }
            })
    
    def _create_advanced_features(self, df: pd.DataFrame, 
                                 minutes_predictions: pd.DataFrame = None,
                                 fixture_analysis: pd.DataFrame = None) -> pd.DataFrame:
        """Create advanced features for ensemble prediction"""
        try:
            features = df.copy()
            
            # Merge minutes predictions if provided
            if minutes_predictions is not None and not minutes_predictions.empty:
                features = features.merge(
                    minutes_predictions[['player_id', 'expected_minutes', 'start_probability']],
                    left_on='id', right_on='player_id', how='left'
                )
            else:
                features['expected_minutes'] = features.get('minutes', 0)
                features['start_probability'] = (features.get('minutes', 0) > 60).astype(int)
            
            # Fill missing values
            features['expected_minutes'] = features['expected_minutes'].fillna(0)
            features['start_probability'] = features['start_probability'].fillna(0)
            
            # Basic performance features
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
            
            # Price and value features
            if 'now_cost' in features.columns:
                features['price_value'] = features['now_cost'] / 10.0
                features['value_ratio'] = features['total_points'] / (features['now_cost'] + 1)
                features['price_per_point'] = features['now_cost'] / (features['total_points'] + 1)
            
            # Advanced performance metrics
            features['goals_per_90'] = (features['goals_scored'] * 90) / (features['minutes'] + 1)
            features['assists_per_90'] = (features['assists'] * 90) / (features['minutes'] + 1)
            features['points_per_90'] = (features['total_points'] * 90) / (features['minutes'] + 1)
            
            # Consistency metrics
            features['form_consistency'] = features['form'] / (features['points_per_game'] + 1)
            features['performance_variance'] = features['form'] - features['points_per_game']
            
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
            
            # Ownership features
            if 'selected_by_percent' in features.columns:
                features['ownership_percentile'] = features['selected_by_percent'].rank(pct=True)
                features['is_differential'] = (features['selected_by_percent'] < 5).astype(int)
                features['is_highly_owned'] = (features['selected_by_percent'] > 20).astype(int)
            
            # Transfer momentum
            if 'transfers_in_event' in features.columns and 'transfers_out_event' in features.columns:
                features['net_transfers'] = features['transfers_in_event'] - features['transfers_out_event']
                features['transfer_momentum'] = features['net_transfers'] / (features['selected_by_percent'] + 1)
            
            # Fixture analysis integration
            if fixture_analysis is not None and not fixture_analysis.empty:
                features = features.merge(
                    fixture_analysis[['team_id', 'avg_difficulty', 'has_double_gameweek', 'rotation_risk']],
                    left_on='team', right_on='team_id', how='left'
                )
                features['avg_difficulty'] = features['avg_difficulty'].fillna(3.0)
                features['has_double_gameweek'] = features['has_double_gameweek'].fillna(False)
                features['rotation_risk'] = features['rotation_risk'].fillna(0.1)
            
            # Interaction features
            features['form_x_availability'] = features['form'] * features['availability']
            features['points_x_difficulty'] = features['total_points'] * (1 / features.get('avg_difficulty', 3.0))
            features['minutes_x_start_prob'] = features['expected_minutes'] * features['start_probability']
            
            # Select final features
            model_features = [
                'expected_minutes', 'start_probability', 'form', 'total_points',
                'points_per_game', 'goals_scored', 'assists', 'clean_sheets',
                'bonus', 'bps', 'ict_index', 'influence', 'creativity', 'threat',
                'selected_by_percent', 'price_value', 'value_ratio', 'price_per_point',
                'goals_per_90', 'assists_per_90', 'points_per_90', 'form_consistency',
                'performance_variance', 'availability', 'ownership_percentile',
                'is_differential', 'is_highly_owned', 'net_transfers', 'transfer_momentum',
                'form_x_availability', 'points_x_difficulty', 'minutes_x_start_prob'
            ]
            
            # Add optional fixture analysis features if available
            # Only add them if they were present during training
            if hasattr(self, 'feature_names') and self.feature_names:
                # Only use features that were present during training
                optional_features = ['avg_difficulty', 'has_double_gameweek', 'rotation_risk']
                for feature in optional_features:
                    if feature in features.columns and feature in self.feature_names:
                        model_features.append(feature)
            else:
                # During training, add all available optional features
                optional_features = ['avg_difficulty', 'has_double_gameweek', 'rotation_risk']
                for feature in optional_features:
                    if feature in features.columns:
                        model_features.append(feature)
            
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
            self.logger.error(f"Error creating advanced features: {str(e)}")
            return pd.DataFrame()
    
    def _prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data from current season data"""
        try:
            from database import FPLDatabase
            db = FPLDatabase()
            
            # Use current season data directly
            players_df = db.get_players_with_stats()
            
            if players_df.empty:
                self.logger.error("No player data available for training")
                return pd.DataFrame(), pd.Series()
            
            # Use current season data as training data
            merged_data = players_df.copy()
            
            # Use total_points as target (current season performance)
            y = merged_data['total_points'].fillna(0)
            
            # Create features from current data
            X = self._create_advanced_features(merged_data)
            
            if X.empty or len(y) == 0:
                self.logger.warning("No valid training data for ensemble model")
                return pd.DataFrame(), pd.Series()
            
            self.logger.info(f"Prepared ensemble training data from current season: {len(X)} samples, {len(X.columns)} features")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing ensemble training data: {str(e)}")
            return pd.DataFrame(), pd.Series()
    
    def train(self, retrain: bool = False) -> bool:
        """Train the ensemble model with proper cross-validation"""
        try:
            if self.is_fitted and not retrain:
                self.logger.info("Ensemble model already trained. Use retrain=True to retrain.")
                return True
            
            self.logger.info("Starting ensemble model training...")
            
            # Prepare training data
            X, y = self._prepare_training_data()
            
            if X.empty:
                self.logger.error("No training data available for ensemble model")
                return False
            
            # Use time series split for proper validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Train individual models
            model_scores = {}
            
            for name, config in self.model_configs.items():
                self.logger.info(f"Training {name}...")
                
                model = config['model']
                scaler = RobustScaler()
                
                # Cross-validation scores
                cv_scores = cross_val_score(
                    model, X, y, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1
                )
                
                # Train on full data
                X_scaled = scaler.fit_transform(X)
                model.fit(X_scaled, y)
                
                # Store model and scaler
                self.models[name] = model
                self.scalers[name] = scaler
                
                # Calculate performance
                avg_score = -cv_scores.mean()
                model_scores[name] = avg_score
                
                self.logger.info(f"{name} - CV MAE: {avg_score:.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Calculate model weights based on performance
            total_score = sum(model_scores.values())
            for name, score in model_scores.items():
                self.model_weights[name] = (total_score - score) / (total_score * (len(model_scores) - 1))
            
            # Normalize weights
            total_weight = sum(self.model_weights.values())
            for name in self.model_weights:
                self.model_weights[name] /= total_weight
            
            self.logger.info(f"Model weights: {self.model_weights}")
            
            # Train uncertainty estimator
            self._train_uncertainty_estimator(X, y)
            
            self.is_fitted = True
            self.logger.info("Ensemble model training completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training ensemble model: {str(e)}")
            return False
    
    def _train_uncertainty_estimator(self, X: pd.DataFrame, y: pd.Series):
        """Train Gaussian Process for uncertainty estimation"""
        try:
            # Use a subset of data for GP training (it's computationally expensive)
            n_samples = min(1000, len(X))
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_subset = X.iloc[indices]
            y_subset = y.iloc[indices]
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_subset)
            
            # Train GP
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            self.uncertainty_estimator = GaussianProcessRegressor(
                kernel=kernel,
                random_state=42,
                n_restarts_optimizer=3
            )
            
            self.uncertainty_estimator.fit(X_scaled, y_subset)
            self.gp_scaler = scaler
            
            self.logger.info("Uncertainty estimator trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training uncertainty estimator: {str(e)}")
            self.uncertainty_estimator = None
    
    def predict_points(self, players_df: pd.DataFrame, 
                      minutes_predictions: pd.DataFrame = None,
                      fixture_analysis: pd.DataFrame = None) -> pd.DataFrame:
        """Predict points using ensemble of models"""
        try:
            if not self.is_fitted:
                self.logger.warning("Ensemble model not trained. Training now...")
                if not self.train():
                    self.logger.error("Failed to train ensemble model")
                    return pd.DataFrame()
            
            # Create features
            X = self._create_advanced_features(players_df, minutes_predictions, fixture_analysis)
            
            if X.empty:
                self.logger.error("No features created for ensemble prediction")
                return pd.DataFrame()
            
            # Get predictions from all models
            predictions = {}
            for name, model in self.models.items():
                scaler = self.scalers[name]
                X_scaled = scaler.transform(X)
                pred = model.predict(X_scaled)
                predictions[name] = pred
            
            # Calculate weighted ensemble prediction
            ensemble_pred = np.zeros(len(X))
            for name, pred in predictions.items():
                weight = self.model_weights.get(name, 0)
                ensemble_pred += weight * pred
            
            # Calculate uncertainty if available
            uncertainty = None
            if self.uncertainty_estimator is not None:
                try:
                    X_gp_scaled = self.gp_scaler.transform(X)
                    pred_mean, pred_std = self.uncertainty_estimator.predict(X_gp_scaled, return_std=True)
                    uncertainty = pred_std
                except Exception as e:
                    self.logger.warning(f"Could not calculate uncertainty: {str(e)}")
            
            # Clip predictions to reasonable range
            ensemble_pred = np.clip(ensemble_pred, 0, 25)
            
            # Create results
            results = players_df[['id', 'web_name', 'position', 'team_name', 'now_cost']].copy()
            results['predicted_points'] = ensemble_pred
            results['points_per_million'] = results['predicted_points'] / (results['now_cost'] / 10.0)
            
            # Add confidence intervals
            if uncertainty is not None:
                results['points_lower'] = np.maximum(0, ensemble_pred - 1.96 * uncertainty)
                results['points_upper'] = ensemble_pred + 1.96 * uncertainty
                results['uncertainty'] = uncertainty
            else:
                # Fallback confidence intervals
                results['points_lower'] = np.maximum(0, ensemble_pred - 2.0)
                results['points_upper'] = ensemble_pred + 2.0
                results['uncertainty'] = 2.0
            
            # Risk categories based on uncertainty
            if uncertainty is not None:
                results['risk_category'] = pd.cut(
                    uncertainty,
                    bins=[0, 1.0, 2.0, 3.0, 10.0],
                    labels=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'],
                    include_lowest=True
                )
            else:
                results['risk_category'] = 'Medium Risk'
            
            # Model agreement (variance of individual predictions)
            pred_array = np.array(list(predictions.values()))
            model_variance = np.var(pred_array, axis=0)
            results['model_agreement'] = pd.cut(
                model_variance,
                bins=[0, 0.5, 1.0, 2.0, 10.0],
                labels=['High Agreement', 'Medium Agreement', 'Low Agreement', 'Very Low Agreement'],
                include_lowest=True
            )
            
            return results.sort_values('predicted_points', ascending=False)
            
        except Exception as e:
            self.logger.error(f"Error predicting points with ensemble: {str(e)}")
            return pd.DataFrame()
    
    def get_model_importance(self) -> pd.DataFrame:
        """Get feature importance from all models"""
        try:
            if not self.is_fitted:
                return pd.DataFrame()
            
            importance_data = []
            
            for name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': model.feature_importances_,
                        'model': name
                    })
                    importance_data.append(importance_df)
            
            if not importance_data:
                return pd.DataFrame()
            
            # Combine and average importances
            combined_importance = pd.concat(importance_data)
            avg_importance = combined_importance.groupby('feature')['importance'].mean().reset_index()
            avg_importance = avg_importance.sort_values('importance', ascending=False)
            
            return avg_importance
            
        except Exception as e:
            self.logger.error(f"Error getting model importance: {str(e)}")
            return pd.DataFrame()
    
    def backtest_performance(self, test_periods: int = 5) -> Dict:
        """Backtest model performance on historical data"""
        try:
            from database import FPLDatabase
            db = FPLDatabase()
            
            # Get historical data
            history_df = db.get_player_history_data(limit_gws=50)
            players_df = db.get_players_with_stats()
            
            if history_df.empty or players_df.empty:
                return {}
            
            # Split data by time periods
            unique_gws = sorted(history_df['round'].unique())
            test_gws = unique_gws[-test_periods:]
            train_gws = unique_gws[:-test_periods]
            
            # Train on earlier data
            train_data = history_df[history_df['round'].isin(train_gws)]
            test_data = history_df[history_df['round'].isin(test_gws)]
            
            if train_data.empty or test_data.empty:
                return {}
            
            # This is a simplified backtest - in practice you'd retrain models
            # and test on each period separately
            
            # Merge with player features
            test_merged = test_data.merge(
                players_df[['id', 'position', 'team_name']],
                left_on='element', right_on='id', how='left'
            )
            
            # Get predictions (using current model)
            predictions = self.predict_points(test_merged)
            
            if predictions.empty:
                return {}
            
            # Calculate metrics
            actual_points = test_merged['total_points'].values
            predicted_points = predictions['predicted_points'].values
            
            mae = mean_absolute_error(actual_points, predicted_points)
            rmse = np.sqrt(mean_squared_error(actual_points, predicted_points))
            r2 = r2_score(actual_points, predicted_points)
            
            return {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'test_periods': test_periods,
                'n_predictions': len(predictions)
            }
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {str(e)}")
            return {}
    
    def save_model(self, filepath: str) -> bool:
        """Save trained ensemble model"""
        try:
            if not self.is_fitted:
                self.logger.error("No trained ensemble model to save")
                return False
            
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'model_weights': self.model_weights,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted,
                'uncertainty_estimator': self.uncertainty_estimator,
                'gp_scaler': getattr(self, 'gp_scaler', None)
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Ensemble model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving ensemble model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained ensemble model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.model_weights = model_data['model_weights']
            self.feature_names = model_data['feature_names']
            self.is_fitted = model_data['is_fitted']
            self.uncertainty_estimator = model_data.get('uncertainty_estimator')
            self.gp_scaler = model_data.get('gp_scaler')
            
            self.logger.info(f"Ensemble model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading ensemble model: {str(e)}")
            return False
