"""Utilities for building historical training datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.src.core.database import FPLDatabase


@dataclass
class DatasetSplits:
    train_rounds: List[int]
    validation_rounds: List[int]
    test_rounds: List[int]


class TrainingDataBuilder:
    """Builds chronological datasets from stored FPL history."""

    def __init__(self, lookback_gws: int = 5):
        self.lookback = max(1, lookback_gws)
        self.db = FPLDatabase()
        self._history_cache: Optional[pd.DataFrame] = None
        self._players_cache: Optional[pd.DataFrame] = None
        self._feature_history: Optional[pd.DataFrame] = None
        self._ewm_feature_names: List[str] = []

    def _get_history(self) -> pd.DataFrame:
        if self._history_cache is None:
            self._history_cache = self.db.get_player_history_data(limit_gws=None)
        return self._history_cache.copy()

    def _get_players_snapshot(self) -> pd.DataFrame:
        if self._players_cache is None:
            self._players_cache = self.db.get_players_with_stats()
        return self._players_cache.copy()

    def _prepare_feature_history(self) -> pd.DataFrame:
        if self._feature_history is not None:
            return self._feature_history.copy()

        history = self._get_history()
        if history.empty:
            self._feature_history = history
            return history

        history = history.sort_values(["element", "round"]).reset_index(drop=True)
        history["played_flag"] = (history["minutes"] > 0).astype(int)
        history["starter_flag"] = (history["minutes"] >= 60).astype(int)

        rolling_cols = [
            "minutes",
            "total_points",
            "goals_scored",
            "assists",
            "clean_sheets",
            "bps",
            "ict_index",
        ]

        for col in rolling_cols:
            history[f"{col}_rolling_mean"] = history.groupby("element")[col].transform(
                lambda s: s.shift(1)
                .rolling(self.lookback, min_periods=1)
                .mean()
            )

        history["starts_rolling_mean"] = history.groupby("element")[
            "starter_flag"
        ].transform(
            lambda s: s.shift(1)
            .rolling(self.lookback, min_periods=1)
            .mean()
        )
        history["appearance_rate"] = history.groupby("element")[
            "played_flag"
        ].transform(
            lambda s: s.shift(1)
            .rolling(self.lookback, min_periods=1)
            .mean()
        )

        history["recent_minutes"] = history.groupby("element")["minutes"].shift(1)
        history["recent_points"] = history.groupby("element")["total_points"].shift(1)

        # Exponentially weighted features to bias towards current season form
        alpha = 0.7  # heavier weight on latest matches
        self._ewm_feature_names = []
        for col in rolling_cols:
            ewm_col = f"{col}_ewm_recent"
            history[ewm_col] = history.groupby("element")[col].transform(
                lambda s: s.shift(1).ewm(alpha=alpha, adjust=False).mean()
            )
            self._ewm_feature_names.append(ewm_col)

        self._feature_history = history
        return history.copy()

    def _base_feature_columns(self) -> List[str]:
        base_cols = [
            "minutes_rolling_mean",
            "total_points_rolling_mean",
            "goals_scored_rolling_mean",
            "assists_rolling_mean",
            "clean_sheets_rolling_mean",
            "bps_rolling_mean",
            "ict_index_rolling_mean",
            "starts_rolling_mean",
            "appearance_rate",
            "recent_minutes",
            "recent_points",
        ]
        if hasattr(self, "_ewm_feature_names"):
            base_cols.extend(self._ewm_feature_names)
        return base_cols

    def get_feature_columns(self) -> List[str]:
        """Expose feature column ordering for models."""
        return self._base_feature_columns()

    def build_minutes_training_set(self) -> pd.DataFrame:
        history = self._prepare_feature_history()
        if history.empty:
            return pd.DataFrame()

        feature_cols = self._base_feature_columns()
        dataset = history.dropna(subset=feature_cols + ["minutes"]).copy()
        if dataset.empty:
            return pd.DataFrame()

        dataset = dataset.rename(columns={"element": "player_id"})
        dataset["minutes_target"] = dataset["minutes"]
        return dataset[["player_id", "round", "minutes_target", *feature_cols]]

    def build_points_training_set(self) -> pd.DataFrame:
        history = self._prepare_feature_history()
        if history.empty:
            return pd.DataFrame()

        feature_cols = self._base_feature_columns()
        dataset = history.dropna(subset=feature_cols + ["total_points"]).copy()
        if dataset.empty:
            return pd.DataFrame()

        dataset = dataset.rename(columns={"element": "player_id"})
        dataset["points_target"] = dataset["total_points"]
        return dataset[["player_id", "round", "points_target", *feature_cols]]

    def build_prediction_features(
        self, player_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        feature_history = self._prepare_feature_history()
        if feature_history.empty:
            return pd.DataFrame()

        feature_cols = self._base_feature_columns()
        latest = (
            feature_history.sort_values("round")
            .groupby("element")
            .tail(1)
            .rename(columns={"element": "player_id"})
        )

        if latest.empty:
            return pd.DataFrame()

        latest[feature_cols] = latest[feature_cols].fillna(0)

        if player_ids:
            latest = latest[latest["player_id"].isin(player_ids)]

        players_snapshot = self._get_players_snapshot()
        metadata_cols = ["position", "team", "team_name", "now_cost", "status"]
        if not players_snapshot.empty:
            latest = latest.merge(
                players_snapshot,
                left_on="player_id",
                right_on="id",
                how="left",
                suffixes=("", "_player"),
            )

        available_metadata = [
            col for col in metadata_cols if col in latest.columns
        ]

        return latest[["player_id", "round", *feature_cols, *available_metadata]].copy()

    def split_by_gameweek(
        self, dataset: pd.DataFrame, round_column: str = "round", val_gws: int = 2, test_gws: int = 1
    ) -> DatasetSplits:
        if dataset.empty:
            return DatasetSplits([], [], [])

        unique_rounds = sorted(dataset[round_column].unique())
        if len(unique_rounds) <= val_gws + test_gws:
            return DatasetSplits(unique_rounds, [], [])

        test_rounds = unique_rounds[-test_gws:] if test_gws else []
        val_rounds = (
            unique_rounds[-(test_gws + val_gws) : -test_gws] if val_gws else []
        )
        train_rounds = [
            gw
            for gw in unique_rounds
            if gw not in set(val_rounds) | set(test_rounds)
        ]

        return DatasetSplits(train_rounds, val_rounds, test_rounds)

