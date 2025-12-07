import logging
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from backend.src.core.config import get_settings
from backend.src.core.database import FPLDatabase
from backend.src.core.training_data import TrainingDataBuilder
from backend.src.models.minutes_model import MinutesPredictor


class PointsPredictor:
    """Predicts FPL points using historical trends and minutes predictions."""

    def __init__(self):
        self.regressor: Optional[GradientBoostingRegressor] = None
        self.feature_scaler = StandardScaler()
        self.minutes_model = MinutesPredictor()
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.builder = TrainingDataBuilder(lookback_gws=5)
        self.logger = logging.getLogger(__name__)
        self.settings = get_settings()
        self.db = FPLDatabase()
        self.model_version: Optional[str] = None
        self.position_weights = {
            "GKP": {"clean_sheet": 4, "save": 0.33, "goals": 6, "assists": 3},
            "DEF": {"clean_sheet": 4, "goals": 6, "assists": 3},
            "MID": {"clean_sheet": 1, "goals": 5, "assists": 3},
            "FWD": {"goals": 4, "assists": 3},
        }

    def _feature_columns(self) -> List[str]:
        if not self.feature_names:
            self.feature_names = self.builder.get_feature_columns()
        return self.feature_names

    def _prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        dataset = self.builder.build_points_training_set()
        if dataset.empty:
            return (
                pd.DataFrame(),
                pd.Series(dtype=float),
                pd.Series(dtype=int),
            )

        feature_cols = self._feature_columns()
        X = dataset[feature_cols].fillna(0)
        y = dataset["points_target"]
        rounds = dataset["round"]
        return X, y, rounds

    def _log_metrics(
        self,
        mae: float,
        rmse: float,
        r2: float,
        splits,
        artifact_path: Optional[str],
    ) -> None:
        train_start = splits.train_rounds[0] if splits.train_rounds else None
        train_end = splits.train_rounds[-1] if splits.train_rounds else None
        val_gw = splits.validation_rounds[-1] if splits.validation_rounds else None
        test_gw = splits.test_rounds[-1] if splits.test_rounds else None

        self.db.save_model_metrics(
            model_name="points_model",
            version=self.model_version or datetime.utcnow().strftime("%Y%m%d%H%M%S"),
            train_start_gw=train_start or 0,
            train_end_gw=train_end or 0,
            validation_gw=val_gw,
            test_gw=test_gw,
            metrics={"mae": mae, "rmse": rmse, "r2": r2},
            artifact_path=artifact_path,
        )

    def train(self, retrain: bool = False) -> bool:
        try:
            if self.is_fitted and not retrain:
                self.logger.info("Points model already trained. Use retrain=True to retrain.")
                return True

            self.logger.info("Training points model with historical data...")
            if not self.minutes_model.is_trained():
                if not self.minutes_model.train():
                    self.logger.error("Minutes model training failed; cannot train points model")
                    return False

            data_tuple = self._prepare_training_data()
            if data_tuple[0].empty:
                self.logger.error("No historical data available for points model")
                return False

            X, y, rounds = data_tuple
            feature_cols = self._feature_columns()

            dataset = pd.DataFrame(X, columns=feature_cols).copy()
            dataset["points_target"] = y.values
            dataset["round"] = rounds.values

            splits = self.builder.split_by_gameweek(dataset)
            if not splits.train_rounds or not splits.test_rounds:
                self.logger.error("Insufficient rounds for chronological splits in points model")
                return False

            train_mask = dataset["round"].isin(splits.train_rounds)
            test_mask = dataset["round"].isin(splits.test_rounds)

            X_train = dataset.loc[train_mask, feature_cols]
            y_train = dataset.loc[train_mask, "points_target"]

            if X_train.empty:
                self.logger.error("Training set is empty for points model")
                return False

            self.feature_scaler.fit(X_train)
            X_train_scaled = self.feature_scaler.transform(X_train)
            self.regressor = GradientBoostingRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=42,
            )
            self.regressor.fit(X_train_scaled, y_train)

            X_test = dataset.loc[test_mask, feature_cols]
            y_test = dataset.loc[test_mask, "points_target"]
            if X_test.empty:
                X_test = X_train
                y_test = y_train

            X_test_scaled = self.feature_scaler.transform(X_test)
            predictions = self.regressor.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            self.logger.info(
                "Points model evaluation — MAE: %.2f, RMSE: %.2f, R2: %.3f", mae, rmse, r2
            )

            self.is_fitted = True
            artifact_path = str(self.settings.ML_MODELS_DIR / "points_model.pkl")
            self.model_version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            self.save_model(artifact_path)
            self._log_metrics(mae, rmse, r2, splits, artifact_path)
            return True

        except Exception as e:
            self.logger.error(f"Error training points model: {str(e)}")
            return False

    def _build_feature_frame(self, players_df: pd.DataFrame) -> pd.DataFrame:
        feature_frame = self.builder.build_prediction_features(players_df["id"].tolist())
        if feature_frame.empty:
            return pd.DataFrame()

        feature_cols = self._feature_columns()
        result = feature_frame[["player_id", *feature_cols]].copy()
        return result

    def predict_points(self, players_df: pd.DataFrame) -> pd.DataFrame:
        try:
            if not self.is_fitted:
                if not self.train():
                    return pd.DataFrame()

            minutes_pred = self.minutes_model.predict_minutes(players_df)
            if minutes_pred.empty:
                self.logger.error("Minutes predictions unavailable; aborting points prediction")
                return pd.DataFrame()

            feature_frame = self._build_feature_frame(players_df)
            if feature_frame.empty:
                self.logger.error("Unable to build feature frame for points prediction")
                return pd.DataFrame()

            feature_cols = self._feature_columns()
            X = feature_frame[feature_cols].fillna(0)
            X_scaled = self.feature_scaler.transform(X)
            predicted_points = self.regressor.predict(X_scaled)
            predicted_points = np.clip(predicted_points, 0, 25)

            results = players_df[['id', 'web_name', 'position', 'team_name', 'now_cost']].copy()
            results = results.merge(
                minutes_pred[['player_id', 'expected_minutes', 'start_probability']],
                left_on='id',
                right_on='player_id',
                how='left'
            ).merge(
                feature_frame,
                left_on='id',
                right_on='player_id',
                how='left'
            )

            results['predicted_points'] = predicted_points
            results['points_per_million'] = results['predicted_points'] / (results['now_cost'] / 10.0).replace(0, np.nan)
            results['points_lower'] = np.maximum(0, predicted_points - 2.5)
            results['points_upper'] = predicted_points + 2.5
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
