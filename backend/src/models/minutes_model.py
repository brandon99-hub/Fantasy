import logging
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from backend.src.core.config import get_settings
from backend.src.core.database import FPLDatabase
from backend.src.core.training_data import TrainingDataBuilder


class MinutesPredictor:
    """Predicts player minutes using historical gameweek data."""

    def __init__(self):
        self.start_classifier: Optional[GradientBoostingClassifier] = None
        self.minutes_regressor: Optional[GradientBoostingRegressor] = None
        self.feature_scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.builder = TrainingDataBuilder(lookback_gws=5)
        self.logger = logging.getLogger(__name__)
        self.settings = get_settings()
        self.db = FPLDatabase()
        self.model_version: Optional[str] = None
        self.min_playing_samples = 50

    def _get_feature_columns(self) -> List[str]:
        if not self.feature_names:
            self.feature_names = self.builder.get_feature_columns()
        return self.feature_names

    def _prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        dataset = self.builder.build_minutes_training_set()
        if dataset.empty:
            return (
                pd.DataFrame(),
                pd.Series(dtype=float),
                pd.Series(dtype=int),
                pd.Series(dtype=int),
            )

        feature_cols = self._get_feature_columns()
        X = dataset[feature_cols].fillna(0)
        y_minutes = dataset["minutes_target"]
        y_starts = (y_minutes >= 60).astype(int)
        return X, y_minutes, y_starts, dataset["round"]

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
            model_name="minutes_model",
            version=self.model_version or datetime.utcnow().strftime("%Y%m%d%H%M%S"),
            train_start_gw=train_start or 0,
            train_end_gw=train_end or 0,
            validation_gw=val_gw,
            test_gw=test_gw,
            metrics={"mae": mae, "rmse": rmse, "r2": r2},
            artifact_path=artifact_path,
        )

    def train(self, retrain: bool = False) -> bool:
        """Train the minutes prediction models using chronological splits."""
        try:
            if self.is_fitted and not retrain:
                self.logger.info("Minutes model already trained. Use retrain=True to retrain.")
                return True

            self.logger.info("Starting minutes model training with historical data...")
            data_tuple = self._prepare_training_data()
            if not data_tuple[0].size:
                self.logger.error("No historical player history available for training")
                return False

            X, y_minutes, y_starts, rounds = data_tuple
            feature_cols = self._get_feature_columns()
            dataset = pd.DataFrame(X, columns=feature_cols)
            dataset["minutes_target"] = y_minutes.values
            dataset["starts_target"] = y_starts.values
            dataset["round"] = rounds.values

            splits = self.builder.split_by_gameweek(dataset)
            if not splits.train_rounds or not splits.test_rounds:
                self.logger.error("Insufficient historical gameweeks for chronological split")
                return False

            train_mask = dataset["round"].isin(splits.train_rounds)
            val_mask = dataset["round"].isin(splits.validation_rounds) if splits.validation_rounds else None
            test_mask = dataset["round"].isin(splits.test_rounds)

            X_train = dataset.loc[train_mask, feature_cols]
            y_minutes_train = dataset.loc[train_mask, "minutes_target"]
            y_starts_train = dataset.loc[train_mask, "starts_target"]

            if X_train.empty:
                self.logger.error("Training split is empty")
                return False

            self.feature_scaler.fit(X_train)
            X_train_scaled = self.feature_scaler.transform(X_train)

            self.logger.info("Training start probability classifier...")
            self.start_classifier = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
            )
            self.start_classifier.fit(X_train_scaled, y_starts_train)

            playing_mask = y_minutes_train > 0
            if playing_mask.sum() < self.min_playing_samples:
                self.logger.warning("Not enough positive samples for minutes regressor")
                return False

            self.minutes_regressor = GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
            )
            self.minutes_regressor.fit(
                X_train_scaled[playing_mask], y_minutes_train[playing_mask]
            )

            eval_mask = test_mask if not dataset.loc[test_mask, feature_cols].empty else train_mask
            X_eval = dataset.loc[eval_mask, feature_cols]
            y_eval = dataset.loc[eval_mask, "minutes_target"]
            X_eval_scaled = self.feature_scaler.transform(X_eval)
            minutes_pred = self.minutes_regressor.predict(X_eval_scaled)

            mae = mean_absolute_error(y_eval, minutes_pred)
            rmse = np.sqrt(mean_squared_error(y_eval, minutes_pred))
            r2 = r2_score(y_eval, minutes_pred)
            self.logger.info(
                "Minutes model evaluation — MAE: %.2f, RMSE: %.2f, R2: %.3f", mae, rmse, r2
            )

            self.is_fitted = True
            artifact_path = str(self.settings.ML_MODELS_DIR / "minutes_model.pkl")
            self.model_version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            self.save_model(artifact_path)
            self._log_metrics(mae, rmse, r2, splits, artifact_path)
            return True

        except Exception as e:
            self.logger.error(f"Error training minutes model: {str(e)}")
            return False

    def predict_minutes(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Predict minutes for players using latest historical features."""
        try:
            if not self.is_fitted:
                if not self.train():
                    return pd.DataFrame()

            player_ids = players_df["id"].tolist()
            feature_frame = self.builder.build_prediction_features(player_ids)
            if feature_frame.empty:
                self.logger.error("Insufficient historical data to build prediction features")
                return pd.DataFrame()

            feature_cols = self._get_feature_columns()
            missing_cols = [col for col in feature_cols if col not in feature_frame.columns]
            if missing_cols:
                self.logger.error(f"Missing feature columns for prediction: {missing_cols}")
                return pd.DataFrame()

            X = feature_frame[feature_cols].fillna(0)
            X_scaled = self.feature_scaler.transform(X)
            start_prob = self.start_classifier.predict_proba(X_scaled)[:, 1]

            predicted_minutes = self.minutes_regressor.predict(X_scaled)
            predicted_minutes = np.clip(predicted_minutes, 0, 90)
            expected_minutes = predicted_minutes * start_prob

            metadata = feature_frame[["player_id"]].copy()
            metadata["start_probability"] = start_prob
            metadata["predicted_minutes_if_playing"] = predicted_minutes
            metadata["expected_minutes"] = expected_minutes
            metadata["minutes_category"] = pd.cut(
                expected_minutes,
                bins=[0, 1, 60, 90],
                labels=["No Play", "Sub", "Starter"],
                include_lowest=True,
            )

            output = players_df.merge(
                metadata,
                left_on="id",
                right_on="player_id",
                how="inner",
            )

            return output[
                [
                    "id",
                    "web_name",
                    "position",
                    "team_name",
                    "start_probability",
                    "predicted_minutes_if_playing",
                    "expected_minutes",
                    "minutes_category",
                ]
            ].rename(columns={"id": "player_id"})

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
