"""
Advanced ensemble prediction models for FPL
Combines multiple ML algorithms with proper cross-validation and uncertainty quantification
"""

import logging
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler

from backend.src.core.config import get_settings
from backend.src.core.database import FPLDatabase
from backend.src.core.training_data import TrainingDataBuilder
from backend.src.models.minutes_model import MinutesPredictor

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
        self.builder = TrainingDataBuilder(lookback_gws=5)
        self.minutes_model = MinutesPredictor()
        self.settings = get_settings()
        self.db = FPLDatabase()
        self.model_version: Optional[str] = None
        self.calibration_factor: float = 1.0
        
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
    
    def _get_feature_columns(self) -> List[str]:
        if not self.feature_names:
            self.feature_names = self.builder.get_feature_columns()
        return self.feature_names

    def _build_feature_frame(self, players_df: pd.DataFrame) -> pd.DataFrame:
        feature_frame = self.builder.build_prediction_features(players_df["id"].tolist())
        if feature_frame.empty:
            return pd.DataFrame()
        feature_cols = self._get_feature_columns()
        return feature_frame[["player_id", *feature_cols]]
    
    def _prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare training data from historical player history."""
        try:
            dataset = self.builder.build_points_training_set()
            if dataset.empty:
                self.logger.error("No historical data available for ensemble training")
                return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=int)

            feature_cols = self._get_feature_columns()
            X = dataset[feature_cols].fillna(0)
            y = dataset["points_target"]
            rounds = dataset["round"]
            self.logger.info(
                "Prepared ensemble training data: %d samples, %d features",
                len(X),
                len(feature_cols),
            )
            return X, y, rounds

        except Exception as e:
            self.logger.error(f"Error preparing ensemble training data: {str(e)}")
            return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=int)
    
    def train(self, retrain: bool = False) -> bool:
        """Train the ensemble model with chronological splits."""
        try:
            if self.is_fitted and not retrain:
                self.logger.info("Ensemble model already trained. Use retrain=True to retrain.")
                return True

            self.logger.info("Starting ensemble model training...")
            data_tuple = self._prepare_training_data()
            if data_tuple[0].empty:
                return False

            X, y, rounds = data_tuple
            feature_cols = self._get_feature_columns()
            dataset = pd.DataFrame(X, columns=feature_cols)
            dataset["points_target"] = y.values
            dataset["round"] = rounds.values

            splits = self.builder.split_by_gameweek(dataset)
            if not splits.train_rounds or not splits.test_rounds:
                self.logger.error("Insufficient rounds for ensemble training splits")
                return False

            train_mask = dataset["round"].isin(splits.train_rounds)
            val_mask = dataset["round"].isin(splits.validation_rounds)
            test_mask = dataset["round"].isin(splits.test_rounds)

            if not val_mask.any():
                val_mask = train_mask

            X_train = dataset.loc[train_mask, feature_cols]
            y_train = dataset.loc[train_mask, "points_target"]
            X_val = dataset.loc[val_mask, feature_cols]
            y_val = dataset.loc[val_mask, "points_target"]
            X_test = dataset.loc[test_mask, feature_cols]
            y_test = dataset.loc[test_mask, "points_target"]
            if X_test.empty:
                X_test = X_val
                y_test = y_val

            model_scores = {}
            for name, config in self.model_configs.items():
                self.logger.info("Training %s...", name)
                model = config["model"]
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                model.fit(X_train_scaled, y_train)
                self.models[name] = model
                self.scalers[name] = scaler

                val_pred = model.predict(scaler.transform(X_val))
                mae = mean_absolute_error(y_val, val_pred)
                model_scores[name] = mae
                self.logger.info("%s validation MAE: %.3f", name, mae)

            inv_scores = {name: 1 / (score + 1e-6) for name, score in model_scores.items()}
            total_weight = sum(inv_scores.values())
            self.model_weights = {name: score / total_weight for name, score in inv_scores.items()}
            self.logger.info("Model weights: %s", self.model_weights)

            test_predictions = {}
            for name, model in self.models.items():
                scaler = self.scalers[name]
                test_predictions[name] = model.predict(scaler.transform(X_test))

            ensemble_pred = np.zeros(len(X_test))
            for name, pred in test_predictions.items():
                ensemble_pred += self.model_weights.get(name, 0) * pred

            mae = mean_absolute_error(y_test, ensemble_pred)
            rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
            r2 = r2_score(y_test, ensemble_pred)
            
            # Calibrate predictions to match actual distribution (Platt-style scaling)
            actual_mean = float(y_test.mean())
            pred_mean = float(np.mean(ensemble_pred)) if len(ensemble_pred) else 1.0
            calibration_factor = actual_mean / pred_mean if pred_mean > 0 else 1.0
            calibration_factor = float(np.clip(calibration_factor, 0.5, 3.0))
            self.calibration_factor = calibration_factor
            self.logger.info(
                "Ensemble evaluation — MAE: %.2f, RMSE: %.2f, R2: %.3f", mae, rmse, r2
            )

            self._train_uncertainty_estimator(pd.DataFrame(X_train, columns=feature_cols), y_train)

            self.is_fitted = True
            artifact_path = str(self.settings.ML_MODELS_DIR / "ensemble_model.pkl")
            self.model_version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            self.save_model(artifact_path)
            self.db.save_model_metrics(
                model_name="ensemble_model",
                version=self.model_version,
                train_start_gw=splits.train_rounds[0],
                train_end_gw=splits.train_rounds[-1],
                validation_gw=splits.validation_rounds[-1] if splits.validation_rounds else None,
                test_gw=splits.test_rounds[-1] if splits.test_rounds else None,
                metrics={"mae": mae, "rmse": rmse, "r2": r2},
                artifact_path=artifact_path,
            )

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
    
    def predict_points(
        self,
        players_df: pd.DataFrame,
        minutes_predictions: pd.DataFrame = None,
        fixture_analysis: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Predict points using the trained ensemble."""
        try:
            if not self.is_fitted and not self.train():
                return pd.DataFrame()

            feature_frame = self._build_feature_frame(players_df)
            if feature_frame.empty:
                self.logger.error("Unable to build features for ensemble prediction")
                return pd.DataFrame()

            feature_cols = self._get_feature_columns()
            X = feature_frame[feature_cols].fillna(0)

            predictions = {}
            for name, model in self.models.items():
                scaler = self.scalers[name]
                predictions[name] = model.predict(scaler.transform(X))

            ensemble_pred = np.zeros(len(X))
            for name, pred in predictions.items():
                ensemble_pred += self.model_weights.get(name, 0) * pred

            uncertainty = None
            if self.uncertainty_estimator is not None:
                try:
                    X_gp_scaled = self.gp_scaler.transform(X)
                    _, pred_std = self.uncertainty_estimator.predict(X_gp_scaled, return_std=True)
                    uncertainty = pred_std
                except Exception:
                    self.logger.warning("Gaussian process uncertainty estimation failed")

            minutes_pred = minutes_predictions
            if minutes_pred is None:
                minutes_pred = self.minutes_model.predict_minutes(players_df)

            results = players_df[['id', 'web_name', 'position', 'team_name', 'now_cost']].copy()
            if minutes_pred is not None and not minutes_pred.empty:
                results = results.merge(
                    minutes_pred[['player_id', 'expected_minutes', 'start_probability']],
                    left_on='id',
                    right_on='player_id',
                    how='left'
                )

            ensemble_pred = np.clip(ensemble_pred * self.calibration_factor, 0, 25)
            results['predicted_points'] = ensemble_pred
            results['points_per_million'] = results['predicted_points'] / (results['now_cost'] / 10.0)

            if uncertainty is not None:
                results['points_lower'] = np.maximum(0, ensemble_pred - 1.96 * uncertainty)
                results['points_upper'] = ensemble_pred + 1.96 * uncertainty
                results['uncertainty'] = uncertainty
                results['risk_category'] = pd.cut(
                    uncertainty,
                    bins=[0, 1.0, 2.0, 3.0, 10.0],
                    labels=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'],
                    include_lowest=True
                )
            else:
                results['points_lower'] = np.maximum(0, ensemble_pred - 2.0)
                results['points_upper'] = ensemble_pred + 2.0
                results['uncertainty'] = 2.0
                results['risk_category'] = 'Medium Risk'

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
            from backend.src.core.database import FPLDatabase
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
