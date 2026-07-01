"""
utils.py
========
Shared utility functions for the FIFA Match Outcome Prediction System.

All helpers here are pure functions with no side-effects, making them
trivially testable and reusable across pipeline, feature, and model modules.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def to_decade(date: pd.Timestamp) -> str:
    """
    Return the decade string for a given date.

    Parameters
    ----------
    date : pd.Timestamp

    Returns
    -------
    str
        E.g. ``'1990s'``, ``'2000s'``.
    """
    decade = (date.year // 10) * 10
    return f"{decade}s"


def days_between(d1: Optional[pd.Timestamp], d2: Optional[pd.Timestamp]) -> Optional[int]:
    """
    Return integer number of days between two timestamps.

    Parameters
    ----------
    d1, d2 : pd.Timestamp or None

    Returns
    -------
    int or None
        Returns ``None`` if either argument is ``None`` / ``NaT``.
    """
    if d1 is None or d2 is None:
        return None
    if pd.isna(d1) or pd.isna(d2):
        return None
    return int(abs((d2 - d1).days))


# ---------------------------------------------------------------------------
# Team helpers
# ---------------------------------------------------------------------------

def get_all_teams(df: pd.DataFrame) -> list[str]:
    """
    Return sorted list of all unique team names in a master dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``home_team`` and ``away_team`` columns.

    Returns
    -------
    list[str]
    """
    teams = set(df["home_team"].dropna().unique()) | set(df["away_team"].dropna().unique())
    return sorted(teams)


def get_team_matches(
    df: pd.DataFrame,
    team: str,
    before_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Return all matches involving a given team, optionally filtered before a date.

    Each row is augmented with ``is_home`` (bool), ``goals_scored``, and
    ``goals_conceded`` columns from the perspective of ``team``.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataframe.
    team : str
        Team name to filter on.
    before_date : pd.Timestamp, optional
        If supplied, only matches strictly before this date are returned.

    Returns
    -------
    pd.DataFrame
    """
    home_mask = df["home_team"] == team
    away_mask = df["away_team"] == team

    home_df = df[home_mask].copy()
    home_df["is_home"] = True
    home_df["goals_scored"] = home_df["home_goals"]
    home_df["goals_conceded"] = home_df["away_goals"]
    home_df["team_result"] = home_df["result"].map(
        {"home_win": "win", "draw": "draw", "away_win": "loss"}
    )

    away_df = df[away_mask].copy()
    away_df["is_home"] = False
    away_df["goals_scored"] = away_df["away_goals"]
    away_df["goals_conceded"] = away_df["home_goals"]
    away_df["team_result"] = away_df["result"].map(
        {"home_win": "loss", "draw": "draw", "away_win": "win"}
    )

    combined = pd.concat([home_df, away_df], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    if before_date is not None:
        combined = combined[combined["date"] < before_date].reset_index(drop=True)

    return combined


def compute_points(result: str, is_home: bool) -> int:
    """
    Convert a match result string to points for a team.

    Parameters
    ----------
    result : str
        One of ``'home_win'``, ``'draw'``, ``'away_win'``.
    is_home : bool

    Returns
    -------
    int
        3 for a win, 1 for a draw, 0 for a loss.
    """
    if result == "draw":
        return 1
    if (result == "home_win" and is_home) or (result == "away_win" and not is_home):
        return 3
    return 0


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Divide numerator by denominator, returning ``default`` on zero division.

    Parameters
    ----------
    numerator : float
    denominator : float
    default : float, optional
        Value returned when denominator is 0 (default 0.0).

    Returns
    -------
    float
    """
    if denominator == 0:
        return default
    return numerator / denominator


def log_experience(n_matches: int) -> float:
    """
    Return natural-log-transformed match count as an experience proxy.

    Parameters
    ----------
    n_matches : int
        Total matches played by a team.

    Returns
    -------
    float
        ``ln(n_matches + 1)`` to avoid log(0).
    """
    return math.log(max(n_matches, 0) + 1)


def normalise_points(points: float, n_matches: int, win_points: int = 3) -> float:
    """
    Normalise accumulated points to the [0, 1] range.

    Parameters
    ----------
    points : float
        Raw points total.
    n_matches : int
        Number of matches played (denominator = n_matches * win_points).
    win_points : int, optional
        Points for a win (default 3).

    Returns
    -------
    float
        0.0 if ``n_matches == 0``.
    """
    max_possible = n_matches * win_points
    return safe_divide(points, max_possible)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def encode_result(result: str) -> int:
    """
    Encode result string to integer label.

    Parameters
    ----------
    result : str
        ``'home_win'``, ``'draw'``, or ``'away_win'``.

    Returns
    -------
    int
        0 = home_win, 1 = draw, 2 = away_win.

    Raises
    ------
    ValueError
        If ``result`` is not one of the three valid strings.
    """
    mapping = {"home_win": 0, "draw": 1, "away_win": 2}
    if result not in mapping:
        raise ValueError(f"Unknown result '{result}'. Expected one of {list(mapping)}")
    return mapping[result]


def decode_result(label: int) -> str:
    """
    Decode integer label back to result string.

    Parameters
    ----------
    label : int
        0, 1, or 2.

    Returns
    -------
    str
    """
    mapping = {0: "home_win", 1: "draw", 2: "away_win"}
    return mapping.get(label, "unknown")


def format_probability_output(probs: np.ndarray, home_team: str, away_team: str) -> dict:
    """
    Format model probability output into a human-readable dictionary.

    Parameters
    ----------
    probs : np.ndarray
        Shape ``(3,)`` array of [home_win_prob, draw_prob, away_win_prob].
    home_team : str
    away_team : str

    Returns
    -------
    dict
        Keys: ``home_win``, ``draw``, ``away_win``, ``predicted_outcome``,
        ``confidence``, ``summary``.
    """
    home_prob, draw_prob, away_prob = float(probs[0]), float(probs[1]), float(probs[2])
    max_prob = max(home_prob, draw_prob, away_prob)

    if max_prob == home_prob:
        predicted = "home_win"
        winner = home_team
    elif max_prob == away_prob:
        predicted = "away_win"
        winner = away_team
    else:
        predicted = "draw"
        winner = "Draw"

    if max_prob >= 0.60:
        confidence = "High"
    elif max_prob >= 0.40:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "home_win": round(home_prob, 4),
        "draw": round(draw_prob, 4),
        "away_win": round(away_prob, 4),
        "predicted_outcome": predicted,
        "predicted_winner": winner,
        "confidence": confidence,
        "summary": (
            f"{home_team} vs {away_team} → "
            f"{home_team} Win: {home_prob:.1%} | "
            f"Draw: {draw_prob:.1%} | "
            f"{away_team} Win: {away_prob:.1%} "
            f"[{confidence} confidence]"
        ),
    }
