"""
model.py
========
FIFA Match Outcome Prediction System — Model Training & Evaluation

Trains four classifiers (Logistic Regression, Random Forest, Gradient Boosting,
Extra Trees as XGBoost surrogate) using time-series cross-validation with
tournament-tier sample weights.

The best model is serialised to ``models/best_model.pkl`` and exposes a
``predict_match()`` convenience function used by the Streamlit app.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import (
    GB_LEARNING_RATE,
    GB_MAX_DEPTH,
    GB_N_ESTIMATORS,
    GB_PARAM_DIST,
    LR_C_VALUES,
    LR_MAX_ITER,
    MODELS_DIR,
    MODEL_FILENAME,
    N_CV_SPLITS,
    RANDOM_STATE,
    RANDOMIZED_SEARCH_ITER,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
    RF_PARAM_DIST,
    SCORING_METRIC,
    TARGET_CLASSES,
)
from src.feature_engineering import FeatureEngineer
from src.utils import decode_result, encode_result, format_probability_output

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label encoding helpers
# ---------------------------------------------------------------------------

_LABEL_ENCODER = LabelEncoder()
_LABEL_ENCODER.classes_ = np.array(TARGET_CLASSES)   # home_win=0, draw=1, away_win=2


def encode_labels(y: pd.Series) -> np.ndarray:
    """Encode result strings to integers 0/1/2."""
    return _LABEL_ENCODER.transform(y)


def decode_labels(y: np.ndarray) -> list[str]:
    """Decode integer labels back to result strings."""
    return _LABEL_ENCODER.inverse_transform(y).tolist()


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _build_models() -> dict[str, Pipeline]:
    """
    Return a dictionary of sklearn Pipelines, one per model.

    Each pipeline includes a StandardScaler followed by the classifier.
    The scaler is included here for completeness; if the upstream
    FeatureEngineer already scales, the scaler here is a no-op (identity on
    already-standardised data is harmless).

    Returns
    -------
    dict[str, Pipeline]
    """
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                # multi_class removed in sklearn 1.7+; multinomial is default for lbfgs
                class_weight="balanced",
                max_iter=LR_MAX_ITER,
                C=1.0,
                random_state=RANDOM_STATE,
                solver="lbfgs",
            )),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS,
                max_depth=RF_MAX_DEPTH,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GradientBoostingClassifier(
                n_estimators=GB_N_ESTIMATORS,
                learning_rate=GB_LEARNING_RATE,
                max_depth=GB_MAX_DEPTH,
                random_state=RANDOM_STATE,
            )),
        ]),
        "ExtraTrees": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", ExtraTreesClassifier(
                n_estimators=RF_N_ESTIMATORS,
                max_depth=RF_MAX_DEPTH,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }
    return models


# ---------------------------------------------------------------------------
# ModelTrainer class
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    Orchestrates training, cross-validation, hyperparameter tuning, and
    evaluation of all four models.

    Parameters
    ----------
    n_cv_splits : int, optional
        Number of TimeSeriesSplit folds (default from config).
    random_state : int, optional
        Global random seed (default 42).

    Attributes
    ----------
    models : dict[str, Pipeline]
        Trained pipelines keyed by model name.
    cv_results : dict[str, dict]
        Cross-validation metrics for each model.
    best_model_name : str
        Name of the best-performing model by weighted F1.
    best_model : Pipeline
        The best trained pipeline.
    label_encoder : LabelEncoder
    """

    def __init__(
        self,
        n_cv_splits: int = N_CV_SPLITS,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.n_cv_splits = n_cv_splits
        self.random_state = random_state
        self.models: dict[str, Pipeline] = {}
        self.cv_results: dict[str, dict] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Pipeline] = None
        self.label_encoder = _LABEL_ENCODER

    # ------------------------------------------------------------------
    # Train all models
    # ------------------------------------------------------------------

    def train_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
        tune_hyperparams: bool = True,
    ) -> dict[str, dict]:
        """
        Train and evaluate all four models using time-series CV.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (output of FeatureEngineer.transform).
        y : pd.Series
            Target labels (``'home_win'``, ``'draw'``, ``'away_win'``).
        sample_weight : pd.Series, optional
            Per-sample weights (tournament tier weights).
        tune_hyperparams : bool, optional
            If True, run RandomizedSearchCV on RF and GB (default True).

        Returns
        -------
        dict[str, dict]
            CV metrics for each model — accuracy, f1_weighted, f1_macro, log_loss.
        """
        y_enc = encode_labels(y)
        sw = sample_weight.values if sample_weight is not None else None
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)

        all_pipelines = _build_models()

        for name, pipe in all_pipelines.items():
            logger.info("Training %s …", name)
            t0 = time.time()

            if tune_hyperparams and name in ("RandomForest", "GradientBoosting"):
                pipe = self._tune_model(name, pipe, X, y_enc, sw, tscv)

            # Final fit on full training data
            fit_params = {}
            if sw is not None:
                fit_params["classifier__sample_weight"] = sw

            pipe.fit(X, y_enc, **fit_params)

            # CV evaluation
            cv_metrics = self._cross_validate(name, pipe, X, y_enc, sw, tscv)
            self.cv_results[name] = cv_metrics
            self.models[name] = pipe

            elapsed = time.time() - t0
            logger.info(
                "%s trained in %.1fs — CV F1(weighted)=%.4f, Acc=%.4f",
                name, elapsed,
                cv_metrics["f1_weighted"],
                cv_metrics["accuracy"],
            )

        # Select best model
        self.best_model_name = max(
            self.cv_results,
            key=lambda n: self.cv_results[n]["f1_weighted"],
        )
        self.best_model = self.models[self.best_model_name]
        logger.info("Best model: %s (F1=%.4f)", self.best_model_name,
                    self.cv_results[self.best_model_name]["f1_weighted"])

        return self.cv_results

    def _tune_model(
        self,
        name: str,
        pipe: Pipeline,
        X: pd.DataFrame,
        y_enc: np.ndarray,
        sw: Optional[np.ndarray],
        tscv: TimeSeriesSplit,
    ) -> Pipeline:
        """
        Run RandomizedSearchCV for RF or GB and return the best pipeline.

        Parameters
        ----------
        name : str
        pipe : Pipeline
        X : pd.DataFrame
        y_enc : np.ndarray
        sw : np.ndarray or None
        tscv : TimeSeriesSplit

        Returns
        -------
        Pipeline
            Best estimator from the search.
        """
        param_dist = RF_PARAM_DIST if name == "RandomForest" else GB_PARAM_DIST
        fit_params = {}
        if sw is not None:
            fit_params["classifier__sample_weight"] = sw

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=RANDOMIZED_SEARCH_ITER,
            cv=tscv,
            scoring=SCORING_METRIC,
            random_state=self.random_state,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X, y_enc, **fit_params)
        logger.info(
            "%s best params: %s (score=%.4f)",
            name, search.best_params_, search.best_score_,
        )
        return search.best_estimator_

    def _cross_validate(
        self,
        name: str,
        pipe: Pipeline,
        X: pd.DataFrame,
        y_enc: np.ndarray,
        sw: Optional[np.ndarray],
        tscv: TimeSeriesSplit,
    ) -> dict:
        """
        Evaluate a fitted pipeline with time-series cross-validation.

        Parameters
        ----------
        name : str
        pipe : Pipeline
        X : pd.DataFrame
        y_enc : np.ndarray
        sw : np.ndarray or None
        tscv : TimeSeriesSplit

        Returns
        -------
        dict
            Keys: accuracy, f1_weighted, f1_macro, log_loss (all CV-averaged).
        """
        acc_scores, f1w_scores, f1m_scores, ll_scores = [], [], [], []

        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_arr)):
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_val = y_enc[train_idx], y_enc[val_idx]
            sw_tr = sw[train_idx] if sw is not None else None

            fit_params = {}
            if sw_tr is not None:
                fit_params["classifier__sample_weight"] = sw_tr

            pipe.fit(X_tr, y_tr, **fit_params)
            y_pred = pipe.predict(X_val)
            y_proba = pipe.predict_proba(X_val)

            acc_scores.append(accuracy_score(y_val, y_pred))
            f1w_scores.append(f1_score(y_val, y_pred, average="weighted", zero_division=0))
            f1m_scores.append(f1_score(y_val, y_pred, average="macro", zero_division=0))
            ll_scores.append(log_loss(y_val, y_proba))

        return {
            "accuracy": float(np.mean(acc_scores)),
            "accuracy_std": float(np.std(acc_scores)),
            "f1_weighted": float(np.mean(f1w_scores)),
            "f1_weighted_std": float(np.std(f1w_scores)),
            "f1_macro": float(np.mean(f1m_scores)),
            "log_loss": float(np.mean(ll_scores)),
        }

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def evaluate_on_test(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:
        """
        Evaluate the best model on a held-out test set.

        Parameters
        ----------
        X_test : pd.DataFrame
        y_test : pd.Series

        Returns
        -------
        dict
            Full classification report plus confusion matrix.
        """
        if self.best_model is None:
            raise RuntimeError("Call train_all() before evaluate_on_test().")

        y_enc = encode_labels(y_test)
        y_pred = self.best_model.predict(X_test.values)
        y_proba = self.best_model.predict_proba(X_test.values)

        report = classification_report(
            y_enc, y_pred,
            target_names=TARGET_CLASSES,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_enc, y_pred)

        return {
            "accuracy": accuracy_score(y_enc, y_pred),
            "f1_weighted": f1_score(y_enc, y_pred, average="weighted", zero_division=0),
            "f1_macro": f1_score(y_enc, y_pred, average="macro", zero_division=0),
            "log_loss": log_loss(y_enc, y_proba),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "y_pred": decode_labels(y_pred),
            "y_proba": y_proba.tolist(),
        }

    def get_feature_importances(self, feature_names: list[str]) -> pd.DataFrame:
        """
        Extract feature importances from the best tree-based model.

        Parameters
        ----------
        feature_names : list[str]
            Names of input features (from ``FeatureEngineer.get_feature_names_out()``).

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``importance``, sorted descending.
        """
        if self.best_model is None:
            raise RuntimeError("Call train_all() first.")

        clf = self.best_model.named_steps["classifier"]
        if not hasattr(clf, "feature_importances_"):
            logger.warning(
                "%s does not have feature_importances_; skipping.",
                self.best_model_name,
            )
            return pd.DataFrame(columns=["feature", "importance"])

        importances = clf.feature_importances_
        return (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_best_model(self, path: Optional[Path] = None) -> Path:
        """
        Serialise the best model (pipeline) to disk with joblib.

        Parameters
        ----------
        path : Path, optional
            Output path (default: ``models/best_model.pkl``).

        Returns
        -------
        Path
            Actual path written to.

        Raises
        ------
        RuntimeError
            If no model has been trained yet.
        """
        if self.best_model is None:
            raise RuntimeError("No trained model to save. Call train_all() first.")

        out_path = path or (MODELS_DIR / MODEL_FILENAME)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.best_model,
                "model_name": self.best_model_name,
                "cv_results": self.cv_results,
                "label_encoder": self.label_encoder,
            },
            out_path,
        )
        logger.info("Best model ('%s') saved to %s", self.best_model_name, out_path)
        return out_path

    @staticmethod
    def load_model(path: Optional[Path] = None) -> dict:
        """
        Load a serialised model bundle from disk.

        Parameters
        ----------
        path : Path, optional
            Path to ``.pkl`` file (default: ``models/best_model.pkl``).

        Returns
        -------
        dict
            Keys: ``model``, ``model_name``, ``cv_results``, ``label_encoder``.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        """
        load_path = path or (MODELS_DIR / MODEL_FILENAME)
        if not load_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {load_path}\n"
                "Run model training first."
            )
        bundle = joblib.load(load_path)
        logger.info("Model '%s' loaded from %s", bundle["model_name"], load_path)
        return bundle


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def predict_match(
    home_team: str,
    away_team: str,
    feature_row: pd.DataFrame,
    model_bundle: dict,
) -> dict:
    """
    Run inference for a single match using a loaded model bundle.

    Parameters
    ----------
    home_team : str
    away_team : str
    feature_row : pd.DataFrame
        Single-row DataFrame of engineered features (from FeatureEngineer).
    model_bundle : dict
        Output of ``ModelTrainer.load_model()``.

    Returns
    -------
    dict
        Formatted prediction — see ``format_probability_output``.
    """
    model = model_bundle["model"]
    proba = model.predict_proba(feature_row.values)[0]
    return format_probability_output(proba, home_team, away_team)
