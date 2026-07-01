"""
feature_engineering.py
======================
FIFA Match Outcome Prediction System — Feature Engineering

Builds 25+ features across four groups:
  A. Historical performance (all-time career stats per team)
  B. Recent form (rolling last-N-match windows)
  C. Head-to-head (fixture-specific historical record)
  D. Contextual (venue, tournament, rest days, experience)

All features obey a strict temporal constraint: only information available
BEFORE the match date is used, preventing any data leakage.

The ``FeatureEngineer`` class is sklearn ``TransformerMixin``-compatible so
it can be embedded in a ``Pipeline`` and serialised with ``joblib``.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    FORM_WINDOWS,
    MIN_H2H_MATCHES,
    RANDOM_STATE,
    WIN_POINTS,
    DRAW_POINTS,
    LOSS_POINTS,
)
from src.utils import (
    get_team_matches,
    safe_divide,
    log_experience,
    normalise_points,
    compute_points,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level stat computers (pure functions, no side-effects)
# ---------------------------------------------------------------------------

def _career_stats(team_matches: pd.DataFrame) -> dict:
    """
    Compute all-time career statistics for a team from their match history.

    Parameters
    ----------
    team_matches : pd.DataFrame
        Output of ``get_team_matches()`` — one row per match, with
        ``team_result`` (win/draw/loss), ``goals_scored``, ``goals_conceded``.

    Returns
    -------
    dict
        Keys: win_pct, draw_pct, avg_goals_scored, avg_goals_conceded,
        goal_diff_avg, clean_sheet_pct, total_matches.
    """
    tm = team_matches.dropna(subset=["team_result", "goals_scored", "goals_conceded"])
    n = len(tm)
    if n == 0:
        return {
            "win_pct": 0.0, "draw_pct": 0.0,
            "avg_goals_scored": 0.0, "avg_goals_conceded": 0.0,
            "goal_diff_avg": 0.0, "clean_sheet_pct": 0.0,
            "total_matches": 0,
        }

    wins = (tm["team_result"] == "win").sum()
    draws = (tm["team_result"] == "draw").sum()
    gs = tm["goals_scored"].fillna(0).astype(float)
    gc = tm["goals_conceded"].fillna(0).astype(float)

    return {
        "win_pct": safe_divide(wins, n),
        "draw_pct": safe_divide(draws, n),
        "avg_goals_scored": gs.mean(),
        "avg_goals_conceded": gc.mean(),
        "goal_diff_avg": (gs - gc).mean(),
        "clean_sheet_pct": safe_divide((gc == 0).sum(), n),
        "total_matches": n,
    }


def _shootout_stats(team: str, shootouts_df: pd.DataFrame, before_date: pd.Timestamp) -> dict:
    """
    Compute shootout win rate for a team up to (not including) ``before_date``.

    Parameters
    ----------
    team : str
    shootouts_df : pd.DataFrame
        Raw shootouts table with columns date, home_team, away_team, winner.
    before_date : pd.Timestamp

    Returns
    -------
    dict
        Keys: shootout_win_pct, shootout_appearances.
    """
    past = shootouts_df[
        (shootouts_df["date"] < before_date)
        & (
            (shootouts_df["home_team"] == team)
            | (shootouts_df["away_team"] == team)
        )
    ]
    n = len(past)
    wins = (past["winner"] == team).sum()
    return {
        "shootout_win_pct": safe_divide(wins, n),
        "shootout_appearances": n,
    }


def _form_features(team_matches: pd.DataFrame, window: int) -> dict:
    """
    Compute rolling form features over the last ``window`` matches.

    Parameters
    ----------
    team_matches : pd.DataFrame
        Temporal-ordered matches for the team (most recent last).
    window : int
        Number of matches to look back.

    Returns
    -------
    dict
        Keys: form_{window}, goals_scored_last{window}, goals_conceded_last{window},
        momentum.
    """
    recent = team_matches.dropna(subset=["team_result"]).tail(window)
    n = len(recent)

    if n == 0:
        return {
            f"form_{window}": 0.0,
            f"goals_scored_last{window}": 0.0,
            f"goals_conceded_last{window}": 0.0,
            "momentum": 0,
        }

    points = recent["team_result"].map({"win": WIN_POINTS, "draw": DRAW_POINTS, "loss": LOSS_POINTS})
    form_normalised = normalise_points(points.sum(), n)

    gs = recent["goals_scored"].fillna(0).astype(float).mean()
    gc = recent["goals_conceded"].fillna(0).astype(float).mean()

    # Momentum: current win streak length
    results_list = recent["team_result"].tolist()
    streak = 0
    for r in reversed(results_list):
        if r == "win":
            streak += 1
        else:
            break

    return {
        f"form_{window}": form_normalised,
        f"goals_scored_last{window}": gs,
        f"goals_conceded_last{window}": gc,
        "momentum": streak,
    }


def _h2h_features(
    home_team: str,
    away_team: str,
    master_df: pd.DataFrame,
    before_date: pd.Timestamp,
) -> dict:
    """
    Compute head-to-head features between two specific teams.

    Parameters
    ----------
    home_team, away_team : str
    master_df : pd.DataFrame
    before_date : pd.Timestamp

    Returns
    -------
    dict
        Keys: h2h_home_win_pct, h2h_draw_pct, h2h_away_win_pct,
        h2h_matches_played, h2h_avg_goals.
    """
    h2h = master_df[
        (master_df["date"] < before_date)
        & (
            ((master_df["home_team"] == home_team) & (master_df["away_team"] == away_team))
            | ((master_df["home_team"] == away_team) & (master_df["away_team"] == home_team))
        )
    ].dropna(subset=["result"])

    n = len(h2h)
    if n < MIN_H2H_MATCHES:
        return {
            "h2h_home_win_pct": 0.33,
            "h2h_draw_pct": 0.33,
            "h2h_away_win_pct": 0.33,
            "h2h_matches_played": n,
            "h2h_avg_goals": 2.5,
        }

    # Normalise from perspective of the current fixture (home_team = home side)
    hw = 0; draws = 0; aw = 0
    for _, row in h2h.iterrows():
        res = row["result"]
        if row["home_team"] == home_team:
            if res == "home_win": hw += 1
            elif res == "draw": draws += 1
            else: aw += 1
        else:
            # Fixture was reversed
            if res == "home_win": aw += 1
            elif res == "draw": draws += 1
            else: hw += 1

    avg_goals = h2h["total_goals"].fillna(0).mean() if "total_goals" in h2h.columns else 2.5

    return {
        "h2h_home_win_pct": safe_divide(hw, n),
        "h2h_draw_pct": safe_divide(draws, n),
        "h2h_away_win_pct": safe_divide(aw, n),
        "h2h_matches_played": n,
        "h2h_avg_goals": avg_goals,
    }


def _is_knockout(tournament: str) -> int:
    """Heuristic detection of knockout-stage matches from tournament name."""
    if pd.isna(tournament):
        return 0
    t = tournament.lower()
    knockout_keywords = ["final", "semi-final", "quarter-final", "round of 16",
                         "knockout", "elimination", "last 16", "last 8"]
    return int(any(kw in t for kw in knockout_keywords))


# ---------------------------------------------------------------------------
# FeatureEngineer — sklearn-compatible transformer
# ---------------------------------------------------------------------------

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that builds all 25+ predictive features.

    Must be ``fit`` on a training slice before being applied to test data.
    Internally caches per-team career statistics indexed by team × date to
    avoid redundant recomputation during cross-validation.

    Parameters
    ----------
    form_windows : list[int], optional
        Rolling windows for recent-form features (default [5, 10]).
    master_df : pd.DataFrame, optional
        Full historical dataframe. Required before calling ``transform``.
    shootouts_df : pd.DataFrame, optional
        Raw shootouts dataframe for shootout-win-rate features.

    Attributes
    ----------
    feature_names_ : list[str]
        List of output feature column names, set after ``fit``.

    Examples
    --------
    >>> fe = FeatureEngineer(form_windows=[5, 10])
    >>> fe.fit(train_df)
    >>> X_train = fe.transform(train_df)
    """

    def __init__(
        self,
        form_windows: list[int] = None,
        master_df: Optional[pd.DataFrame] = None,
        shootouts_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.form_windows = form_windows or FORM_WINDOWS
        self.master_df = master_df
        self.shootouts_df = shootouts_df
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        """
        Fit the transformer (stores training data for look-back queries).

        Parameters
        ----------
        X : pd.DataFrame
            Training slice of the master dataframe.
        y : ignored

        Returns
        -------
        self
        """
        if self.master_df is None:
            self.master_df = X.copy()
        logger.info("FeatureEngineer.fit() called on %d rows", len(X))
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Build all features for every row in ``X``.

        Uses strict temporal look-back: for each row, only matches whose
        ``date < row['date']`` are used in any computation.

        Parameters
        ----------
        X : pd.DataFrame
            Slice of the master dataframe (must have date, home_team,
            away_team, result, home_goals, away_goals, neutral,
            tournament, tournament_tier, total_goals columns).

        Returns
        -------
        pd.DataFrame
            One row per input row, columns = all engineered features.
            Shape: ``(len(X), n_features)``.
        """
        logger.info("FeatureEngineer.transform() — building features for %d matches", len(X))
        records = []

        for idx, row in X.iterrows():
            feat = self._build_row_features(row)
            records.append(feat)

        result_df = pd.DataFrame(records, index=X.index)
        self.feature_names_ = list(result_df.columns)
        logger.info("Features built — shape %s", result_df.shape)
        return result_df

    def get_feature_names_out(self) -> list[str]:
        """Return list of output feature names (set after ``transform``)."""
        return self.feature_names_

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_row_features(self, row: pd.Series) -> dict:
        """Build the full feature vector for a single match row."""
        date = row["date"]
        home = row["home_team"]
        away = row["away_team"]

        ref_df = self.master_df
        sht_df = self.shootouts_df

        # -- Group A: Historical career stats --
        home_hist = get_team_matches(ref_df, home, before_date=date)
        away_hist = get_team_matches(ref_df, away, before_date=date)

        home_career = _career_stats(home_hist)
        away_career = _career_stats(away_hist)

        home_sht = _shootout_stats(home, sht_df, date) if sht_df is not None else {"shootout_win_pct": 0.0, "shootout_appearances": 0}
        away_sht = _shootout_stats(away, sht_df, date) if sht_df is not None else {"shootout_win_pct": 0.0, "shootout_appearances": 0}

        feats: dict = {}

        for prefix, career, sht in [("home", home_career, home_sht), ("away", away_career, away_sht)]:
            feats[f"{prefix}_win_pct"] = career["win_pct"]
            feats[f"{prefix}_draw_pct"] = career["draw_pct"]
            feats[f"{prefix}_avg_goals_scored"] = career["avg_goals_scored"]
            feats[f"{prefix}_avg_goals_conceded"] = career["avg_goals_conceded"]
            feats[f"{prefix}_goal_diff_avg"] = career["goal_diff_avg"]
            feats[f"{prefix}_clean_sheet_pct"] = career["clean_sheet_pct"]
            feats[f"{prefix}_shootout_win_pct"] = sht["shootout_win_pct"]
            feats[f"{prefix}_team_experience"] = log_experience(career["total_matches"])

        # -- Group B: Recent form --
        for window in self.form_windows:
            home_form = _form_features(home_hist, window)
            away_form = _form_features(away_hist, window)

            feats[f"home_form_{window}"] = home_form[f"form_{window}"]
            feats[f"away_form_{window}"] = away_form[f"form_{window}"]
            feats[f"home_goals_scored_last{window}"] = home_form[f"goals_scored_last{window}"]
            feats[f"away_goals_scored_last{window}"] = away_form[f"goals_scored_last{window}"]
            feats[f"home_goals_conceded_last{window}"] = home_form[f"goals_conceded_last{window}"]
            feats[f"away_goals_conceded_last{window}"] = away_form[f"goals_conceded_last{window}"]

        feats["home_momentum"] = _form_features(home_hist, 5)["momentum"]
        feats["away_momentum"] = _form_features(away_hist, 5)["momentum"]

        # Rest days
        home_dates = home_hist["date"].dropna().sort_values()
        away_dates = away_hist["date"].dropna().sort_values()
        feats["days_since_last_match_home"] = int((date - home_dates.iloc[-1]).days) if len(home_dates) > 0 else 365
        feats["days_since_last_match_away"] = int((date - away_dates.iloc[-1]).days) if len(away_dates) > 0 else 365

        # -- Group C: Head-to-head --
        h2h = _h2h_features(home, away, ref_df, before_date=date)
        feats.update(h2h)

        # -- Group D: Contextual --
        feats["is_neutral"] = int(bool(row.get("neutral", False)))
        feats["tournament_tier"] = int(row.get("tournament_tier", 3))
        feats["is_knockout"] = _is_knockout(row.get("tournament", ""))

        # Differential features (often the most informative)
        feats["win_pct_diff"] = feats["home_win_pct"] - feats["away_win_pct"]
        feats["goal_diff_avg_diff"] = feats["home_goal_diff_avg"] - feats["away_goal_diff_avg"]
        feats["form_5_diff"] = feats["home_form_5"] - feats["away_form_5"]
        feats["form_10_diff"] = feats["home_form_10"] - feats["away_form_10"]
        feats["experience_diff"] = feats["home_team_experience"] - feats["away_team_experience"]

        return feats


# ---------------------------------------------------------------------------
# Full sklearn Pipeline factory
# ---------------------------------------------------------------------------

def build_feature_pipeline(
    master_df: pd.DataFrame,
    shootouts_df: Optional[pd.DataFrame] = None,
    form_windows: list[int] = None,
    scale: bool = True,
) -> Pipeline:
    """
    Build a serialisable sklearn ``Pipeline`` that transforms raw match rows
    into scaled numeric feature matrices.

    Parameters
    ----------
    master_df : pd.DataFrame
        Full historical dataframe used as the look-back reference.
    shootouts_df : pd.DataFrame, optional
        Raw shootouts table.
    form_windows : list[int], optional
        Rolling windows (default from config).
    scale : bool, optional
        Whether to append a ``StandardScaler`` step (default True).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Steps: ``feature_engineer`` → (optional) ``scaler``.
    """
    steps = [
        (
            "feature_engineer",
            FeatureEngineer(
                form_windows=form_windows or FORM_WINDOWS,
                master_df=master_df,
                shootouts_df=shootouts_df,
            ),
        )
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))

    pipe = Pipeline(steps)
    logger.info(
        "Feature pipeline built — steps: %s, scale=%s",
        [s[0] for s in steps],
        scale,
    )
    return pipe


# ---------------------------------------------------------------------------
# Convenience: build feature matrix from master dataframe
# ---------------------------------------------------------------------------

def build_feature_matrix(
    master_df: pd.DataFrame,
    shootouts_df: Optional[pd.DataFrame] = None,
    complete_only: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build feature matrix ``X``, target vector ``y``, and sample weights
    from the master dataframe in one call.

    Parameters
    ----------
    master_df : pd.DataFrame
    shootouts_df : pd.DataFrame, optional
    complete_only : bool, optional
        If True (default), drop rows with ``is_incomplete=True`` or null result.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix — one row per match.
    y : pd.Series
        Target labels (``'home_win'``, ``'draw'``, ``'away_win'``).
    weights : pd.Series
        Sample weights from ``match_weight`` column.
    """
    df = master_df.copy()
    if complete_only:
        df = df[~df["is_incomplete"] & df["result"].notna()].reset_index(drop=True)

    logger.info("Building feature matrix from %d complete matches …", len(df))

    fe = FeatureEngineer(
        form_windows=FORM_WINDOWS,
        master_df=master_df,   # pass FULL df for look-back
        shootouts_df=shootouts_df,
    )
    fe.fit(df)
    X = fe.transform(df)
    y = df["result"].reset_index(drop=True)
    weights = df["match_weight"].reset_index(drop=True)

    logger.info(
        "Feature matrix ready — X: %s, y distribution: %s",
        X.shape,
        y.value_counts().to_dict(),
    )
    return X, y, weights
