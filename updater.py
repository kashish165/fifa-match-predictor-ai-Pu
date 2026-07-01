"""
updater.py
==========
FIFA Match Outcome Prediction System — Dynamic Model Updater

Simulates a production ML system that stays current: after every new result
is received, it appends the match to the master dataset, recomputes features,
retrains the model, and logs rolling accuracy.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import numpy as np

from src.config import (
    MODELS_DIR,
    MODEL_FILENAME,
    PROCESSED_DIR,
    RANDOM_STATE,
    TARGET_CLASSES,
    UPDATER_LOG_WINDOW,
)
from src.data_pipeline import DataPipeline
from src.feature_engineering import FeatureEngineer, FORM_WINDOWS
from src.model import ModelTrainer, encode_labels, decode_labels
from src.utils import format_probability_output

logger = logging.getLogger(__name__)

UPDATE_LOG_PATH = MODELS_DIR / "update_log.jsonl"


class ModelUpdater:
    """
    Manages incremental updates to the master dataset and model after each match.

    Parameters
    ----------
    master_df : pd.DataFrame
        Current master dataframe (loaded from processed/master.csv).
    shootouts_df : pd.DataFrame
        Raw shootouts table (needed for shootout win-rate features).
    model_bundle : dict
        Current model bundle (from ``ModelTrainer.load_model()``).

    Attributes
    ----------
    master_df : pd.DataFrame
    shootouts_df : pd.DataFrame
    model_bundle : dict
    update_history : list[dict]
        Log of all updates applied in this session.
    """

    def __init__(
        self,
        master_df: pd.DataFrame,
        shootouts_df: pd.DataFrame,
        model_bundle: dict,
    ) -> None:
        self.master_df = master_df.copy()
        self.shootouts_df = shootouts_df.copy()
        self.model_bundle = model_bundle
        self.update_history: list[dict] = []

    # ------------------------------------------------------------------
    # Core update method
    # ------------------------------------------------------------------

    def update_after_match(
        self,
        date: str,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        tournament: str = "Friendly",
        city: str = "",
        country: str = "",
        neutral: bool = False,
        shootout_winner: Optional[str] = None,
    ) -> dict:
        """
        Process a new match result, update the dataset, retrain the model.

        Steps
        -----
        1. Validate inputs.
        2. Append the new result row to ``self.master_df``.
        3. Append shootout record if applicable.
        4. Recompute all rolling and H2H features for the two teams involved.
        5. Retrain model on the updated dataset.
        6. Serialise the updated model to disk.
        7. Log the update with timestamp and rolling accuracy.

        Parameters
        ----------
        date : str
            Match date, ``YYYY-MM-DD``.
        home_team : str
        away_team : str
        home_score : int
        away_score : int
        tournament : str, optional
        city : str, optional
        country : str, optional
        neutral : bool, optional
        shootout_winner : str, optional
            Team that won the shootout (if applicable).

        Returns
        -------
        dict
            Update summary with keys: status, match, new_model_accuracy,
            rolling_accuracy, timestamp.

        Raises
        ------
        ValueError
            If scores are negative or teams are the same.
        """
        # --- Input validation ---
        if home_score < 0 or away_score < 0:
            raise ValueError("Scores cannot be negative.")
        if home_team == away_team:
            raise ValueError("home_team and away_team must be different.")
        if home_score != away_score and shootout_winner is not None:
            raise ValueError("Shootout winner only valid when scores are equal.")

        match_date = pd.Timestamp(date)
        logger.info(
            "Updating with new match: %s %d–%d %s (%s)",
            home_team, home_score, away_score, away_team, date,
        )

        # --- Step 1: Derive result ---
        if home_score > away_score:
            result = "home_win"
        elif home_score == away_score:
            result = "draw"
        else:
            result = "away_win"

        # Tournament tier
        from src.data_pipeline import DataPipeline as _DP
        tmp = _DP.__new__(_DP)
        tier_fn = _DP.split_by_tournament_weight
        tier = 3  # default
        t_lower = tournament.lower()
        for kw in ["fifa world cup", "uefa euro", "copa america",
                   "africa cup of nations", "afc asian cup"]:
            if kw in t_lower and "qualif" not in t_lower:
                tier = 1
                break
        for kw in ["qualif", "qualification", "nations league", "confederations cup"]:
            if kw in t_lower:
                tier = 2
                break

        weight_map = {1: 3, 2: 2, 3: 1}
        new_row = pd.DataFrame([{
            "date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "tournament": tournament,
            "city": city,
            "country": country,
            "neutral": neutral,
            "shootout_winner": shootout_winner,
            "first_shooter": None,
            "home_goal_count": home_score,
            "home_penalty_count": 0,
            "away_goal_count": away_score,
            "away_penalty_count": 0,
            "home_own_goals": 0,
            "away_own_goals": 0,
            "total_goals": home_score + away_score,
            "is_incomplete": False,
            "home_goals": home_score,
            "away_goals": away_score,
            "result": result,
            "tournament_tier": tier,
            "match_weight": weight_map[tier],
        }])

        # --- Step 2: Append to master ---
        self.master_df = pd.concat([self.master_df, new_row], ignore_index=True)
        self.master_df = self.master_df.sort_values("date").reset_index(drop=True)

        # --- Step 3: Append shootout if applicable ---
        if shootout_winner is not None:
            new_sht = pd.DataFrame([{
                "date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "winner": shootout_winner,
                "first_shooter": None,
            }])
            self.shootouts_df = pd.concat([self.shootouts_df, new_sht], ignore_index=True)

        # --- Step 4 & 5: Retrain ---
        new_bundle = self._retrain()

        # --- Step 6: Save ---
        out_path = MODELS_DIR / MODEL_FILENAME
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(new_bundle, out_path)
        self.model_bundle = new_bundle
        logger.info("Updated model saved to %s", out_path)

        # --- Step 7: Log ---
        rolling_acc = self._compute_rolling_accuracy(new_bundle, window=UPDATER_LOG_WINDOW)
        update_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "match": f"{home_team} {home_score}-{away_score} {away_team}",
            "date": date,
            "result": result,
            "new_cv_f1": new_bundle["cv_results"].get(new_bundle["model_name"], {}).get("f1_weighted", None),
            "rolling_accuracy_last_100": rolling_acc,
        }
        self.update_history.append(update_record)
        self._write_log(update_record)

        logger.info(
            "Update complete — rolling accuracy (last %d): %.4f",
            UPDATER_LOG_WINDOW, rolling_acc,
        )
        return update_record

    # ------------------------------------------------------------------
    # Retrain helper
    # ------------------------------------------------------------------

    def _retrain(self) -> dict:
        """
        Retrain all models on the current master_df and return new bundle.

        Returns
        -------
        dict
            New model bundle.
        """
        df = self.master_df
        complete = df[~df["is_incomplete"] & df["result"].notna()].reset_index(drop=True)

        logger.info("Retraining on %d complete matches …", len(complete))

        fe = FeatureEngineer(
            form_windows=FORM_WINDOWS,
            master_df=df,
            shootouts_df=self.shootouts_df,
        )
        fe.fit(complete)
        X = fe.transform(complete)
        y = complete["result"]
        weights = complete["match_weight"]

        trainer = ModelTrainer(random_state=RANDOM_STATE)
        trainer.train_all(X, y, sample_weight=weights, tune_hyperparams=False)

        return {
            "model": trainer.best_model,
            "model_name": trainer.best_model_name,
            "cv_results": trainer.cv_results,
            "label_encoder": trainer.label_encoder,
            "feature_engineer": fe,
        }

    # ------------------------------------------------------------------
    # Rolling accuracy
    # ------------------------------------------------------------------

    def _compute_rolling_accuracy(self, bundle: dict, window: int = 100) -> float:
        """
        Compute prediction accuracy on the last ``window`` complete matches.

        Parameters
        ----------
        bundle : dict
        window : int

        Returns
        -------
        float
        """
        df = self.master_df
        recent = (
            df[~df["is_incomplete"] & df["result"].notna()]
            .sort_values("date")
            .tail(window)
            .reset_index(drop=True)
        )

        if len(recent) == 0:
            return 0.0

        fe = bundle.get("feature_engineer")
        if fe is None:
            fe = FeatureEngineer(
                form_windows=FORM_WINDOWS,
                master_df=self.master_df,
                shootouts_df=self.shootouts_df,
            )
            fe.fit(recent)

        try:
            X_recent = fe.transform(recent)
            model = bundle["model"]
            y_pred = decode_labels(model.predict(X_recent.values))
            y_true = recent["result"].tolist()
            correct = sum(p == t for p, t in zip(y_pred, y_true))
            return correct / len(y_true)
        except Exception as exc:
            logger.warning("Rolling accuracy computation failed: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _write_log(self, record: dict) -> None:
        """Append update record to JSONL log file."""
        UPDATE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(UPDATE_LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")

    def save_master(self, path: Optional[Path] = None) -> Path:
        """
        Persist updated master dataframe to disk.

        Parameters
        ----------
        path : Path, optional
            Output path (default: ``data/processed/master.csv``).

        Returns
        -------
        Path
        """
        out = path or (PROCESSED_DIR / "master.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        self.master_df.to_csv(out, index=False)
        logger.info("Updated master dataframe saved to %s", out)
        return out
