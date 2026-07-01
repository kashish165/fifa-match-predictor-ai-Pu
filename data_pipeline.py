"""
data_pipeline.py
================
FIFA International Match Outcome Prediction System — Data Pipeline

Handles loading, cleaning, entity resolution, and merging of all four raw datasets
into a single analytical master dataframe ready for feature engineering.

Author: Senior Data Scientist
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (all magic numbers live here; override via config.yaml later)
# ---------------------------------------------------------------------------
TOURNAMENT_TIERS: dict[str, int] = {
    # Tier 1 — Elite competitive matches (weight = 3)
    "FIFA World Cup": 1,
    "UEFA Euro": 1,
    "Copa América": 1,
    "Africa Cup of Nations": 1,
    "AFC Asian Cup": 1,
    "CONCACAF Gold Cup": 1,
    "OFC Nations Cup": 1,
    # Tier 2 — Qualification and secondary competitions (weight = 2)
    "FIFA World Cup qualification": 2,
    "UEFA Euro qualification": 2,
    "Copa América qualification": 2,
    "Africa Cup of Nations qualification": 2,
    "AFC Asian Cup qualification": 2,
    "CONCACAF Gold Cup qualification": 2,
    "Confederations Cup": 2,
    "UEFA Nations League": 2,
    "CONCACAF Nations League": 2,
}
TIER_WEIGHTS: dict[int, int] = {1: 3, 2: 2, 3: 1}


# ---------------------------------------------------------------------------
# DataPipeline class
# ---------------------------------------------------------------------------
class DataPipeline:
    """
    End-to-end data pipeline for the FIFA Match Outcome Prediction System.

    Loads the four raw CSV files, resolves historical team name changes,
    handles missing values, engineers the target label, and merges everything
    into one analytical master dataframe.

    Parameters
    ----------
    data_dir : str or Path
        Root directory that contains a ``raw/`` subdirectory with the four CSVs.

    Attributes
    ----------
    data_dir : Path
    results : pd.DataFrame
    goalscorers : pd.DataFrame
    shootouts : pd.DataFrame
    former_names : pd.DataFrame
    master : pd.DataFrame
        Populated after calling :meth:`build_master`.

    Examples
    --------
    >>> pipeline = DataPipeline("data")
    >>> pipeline.load_raw_data()
    >>> master = pipeline.build_master()
    >>> master.shape
    (49477, ...)
    """

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # DataFrames — populated by load_raw_data()
        self.results: Optional[pd.DataFrame] = None
        self.goalscorers: Optional[pd.DataFrame] = None
        self.shootouts: Optional[pd.DataFrame] = None
        self.former_names: Optional[pd.DataFrame] = None
        self.master: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # 1. load_raw_data
    # ------------------------------------------------------------------
    def load_raw_data(self) -> None:
        """
        Load all four CSV files from the raw data directory.

        Parses date columns as ``pd.Timestamp`` and casts score columns to
        nullable ``Int64`` so that missing values remain ``pd.NA`` rather than
        ``NaN`` (which would force a float dtype).

        Returns
        -------
        None
            DataFrames are stored on ``self``.

        Raises
        ------
        FileNotFoundError
            If any of the four expected CSV files is missing.
        ValueError
            If a required column is absent after loading.
        """
        files = {
            "results": self.raw_dir / "results.csv",
            "goalscorers": self.raw_dir / "goalscorers.csv",
            "shootouts": self.raw_dir / "shootouts.csv",
            "former_names": self.raw_dir / "former_names.csv",
        }

        for name, path in files.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Expected raw data file not found: {path}\n"
                    f"Place all four CSVs inside {self.raw_dir}/"
                )

        logger.info("Loading raw CSV files from %s", self.raw_dir)

        # --- results.csv ---
        try:
            self.results = pd.read_csv(
                files["results"],
                parse_dates=["date"],
                dtype={"neutral": bool},
            )
            # Cast scores to nullable integer (preserves pd.NA for missing values)
            self.results["home_score"] = self.results["home_score"].astype("Int64")
            self.results["away_score"] = self.results["away_score"].astype("Int64")
            logger.info(
                "results.csv loaded — %d rows, %d columns",
                len(self.results),
                self.results.shape[1],
            )
        except Exception as exc:
            raise ValueError(f"Failed to load results.csv: {exc}") from exc

        # --- goalscorers.csv ---
        try:
            self.goalscorers = pd.read_csv(
                files["goalscorers"],
                parse_dates=["date"],
                dtype={"own_goal": bool, "penalty": bool},
            )
            logger.info(
                "goalscorers.csv loaded — %d rows, %d columns",
                len(self.goalscorers),
                self.goalscorers.shape[1],
            )
        except Exception as exc:
            raise ValueError(f"Failed to load goalscorers.csv: {exc}") from exc

        # --- shootouts.csv ---
        try:
            self.shootouts = pd.read_csv(
                files["shootouts"],
                parse_dates=["date"],
            )
            logger.info(
                "shootouts.csv loaded — %d rows, %d columns",
                len(self.shootouts),
                self.shootouts.shape[1],
            )
        except Exception as exc:
            raise ValueError(f"Failed to load shootouts.csv: {exc}") from exc

        # --- former_names.csv ---
        try:
            self.former_names = pd.read_csv(
                files["former_names"],
                parse_dates=["start_date", "end_date"],
            )
            logger.info(
                "former_names.csv loaded — %d rows, %d columns",
                len(self.former_names),
                self.former_names.shape[1],
            )
        except Exception as exc:
            raise ValueError(f"Failed to load former_names.csv: {exc}") from exc

    # ------------------------------------------------------------------
    # 2. resolve_team_names
    # ------------------------------------------------------------------
    def resolve_team_names(
        self,
        df: pd.DataFrame,
        name_col: str,
        date_col: str = "date",
    ) -> pd.DataFrame:
        """
        Standardise historical team names to their current official names.

        Uses a **date-aware** lookup: a former name is applied only when the
        match date falls within ``[start_date, end_date]`` of that former-name
        record.  This prevents incorrectly renaming "West Germany" records that
        already use "Germany" post-1990.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the team name column and a date column.
        name_col : str
            Name of the column holding team names (e.g. ``"home_team"``).
        date_col : str, optional
            Name of the date column (default ``"date"``).

        Returns
        -------
        pd.DataFrame
            A copy of ``df`` with the ``name_col`` values standardised.

        Raises
        ------
        RuntimeError
            If :meth:`load_raw_data` has not been called first.
        """
        if self.former_names is None:
            raise RuntimeError("Call load_raw_data() before resolve_team_names().")

        df = df.copy()
        resolved_count = 0

        for _, row in self.former_names.iterrows():
            former = row["former"]
            current = row["current"]
            start = row["start_date"]
            end = row["end_date"]

            # Boolean mask: rows where the team name matches AND the date is in range
            mask = (
                (df[name_col] == former)
                & (df[date_col] >= start)
                & (df[date_col] <= end)
            )
            n = mask.sum()
            if n > 0:
                df.loc[mask, name_col] = current
                resolved_count += n
                logger.debug(
                    "Renamed %d '%s' → '%s' in column '%s'",
                    n,
                    former,
                    current,
                    name_col,
                )

        if resolved_count == 0:
            logger.info(
                "resolve_team_names('%s'): 0 renames needed — dataset already uses "
                "current team names (former_names.csv serves as reference metadata).",
                name_col,
            )
        else:
            logger.info(
                "resolve_team_names('%s'): %d name occurrences standardised",
                name_col,
                resolved_count,
            )
        return df

    # ------------------------------------------------------------------
    # 3. handle_missing_values
    # ------------------------------------------------------------------
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values across the master dataframe with domain-aware strategy.

        Strategy by column
        ------------------
        - ``home_score`` / ``away_score``: Do **not** impute or drop.
          Flag with ``is_incomplete = True``; these may be in-progress fixtures.
        - ``scorer`` (goalscorers): Leave as ``NaN`` — own-goal records
          intentionally omit the player name.
        - ``minute`` (goalscorers): Impute with the **median minute for that
          tournament type** to preserve temporal patterns within each competition.

        Parameters
        ----------
        df : pd.DataFrame
            The merged (or results) dataframe to clean.

        Returns
        -------
        pd.DataFrame
            Dataframe with an added ``is_incomplete`` boolean column and
            imputed ``minute`` values in goalscorers (if present).
        """
        df = df.copy()

        # --- Flag incomplete matches (scores not yet recorded) ---
        score_null = df["home_score"].isna() | df["away_score"].isna()
        df["is_incomplete"] = score_null
        n_incomplete = score_null.sum()
        logger.info(
            "handle_missing_values: %d incomplete match(es) flagged (scores not imputed)",
            n_incomplete,
        )

        # --- Impute goal minutes in the goalscorers table (if column present) ---
        if "minute" in df.columns:
            n_null_minutes = df["minute"].isna().sum()
            if n_null_minutes > 0:
                if "tournament" in df.columns:
                    # Median minute by tournament type
                    tournament_medians = df.groupby("tournament")["minute"].transform(
                        "median"
                    )
                    df["minute"] = df["minute"].fillna(tournament_medians)
                # Fallback: global median for any remaining nulls
                global_median = df["minute"].median()
                df["minute"] = df["minute"].fillna(global_median)
                logger.info(
                    "handle_missing_values: %d missing goal minutes imputed "
                    "(tournament-median with global fallback)",
                    n_null_minutes,
                )

        return df

    # ------------------------------------------------------------------
    # 4. create_match_result_label
    # ------------------------------------------------------------------
    def create_match_result_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive the three-class target variable and clean goal columns.

        Adds the following columns to ``df``:
        - ``result``: ``'home_win'`` | ``'draw'`` | ``'away_win'``
        - ``home_goals``: ``Int64`` goals scored by home team
        - ``away_goals``: ``Int64`` goals scored by away team

        Rows with missing scores (``is_incomplete == True``) receive ``NaN``
        in all three new columns so they can be excluded from training.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with ``home_score``, ``away_score``, and ``is_incomplete``.

        Returns
        -------
        pd.DataFrame
            Input dataframe with three new columns appended.
        """
        df = df.copy()

        # Clean integer copies (for arithmetic)
        df["home_goals"] = df["home_score"].astype("Int64")
        df["away_goals"] = df["away_score"].astype("Int64")

        # Derive three-class result label
        # Cast to float first so comparisons return standard numpy bool arrays
        # (Int64 nullable integer comparisons return pandas BooleanArray which
        #  np.select does not accept in NumPy 2.x)
        home_f = df["home_goals"].astype(float)
        away_f = df["away_goals"].astype(float)

        conditions = [
            (home_f > away_f).to_numpy(dtype=bool, na_value=False),
            (home_f == away_f).to_numpy(dtype=bool, na_value=False),
            (home_f < away_f).to_numpy(dtype=bool, na_value=False),
        ]
        choices = ["home_win", "draw", "away_win"]
        df["result"] = np.select(conditions, choices, default=None)
        df["result"] = df["result"].replace({None: pd.NA})

        # Overwrite incomplete rows with NA
        df.loc[df["is_incomplete"], "result"] = pd.NA

        result_counts = df["result"].value_counts(dropna=True)
        logger.info(
            "create_match_result_label: result distribution — %s",
            result_counts.to_dict(),
        )
        return df

    # ------------------------------------------------------------------
    # 5. _aggregate_goal_features
    # ------------------------------------------------------------------
    def _aggregate_goal_features(self) -> pd.DataFrame:
        """
        Aggregate goal-level data from ``goalscorers.csv`` to match level.

        Returns
        -------
        pd.DataFrame
            One row per (date, home_team, away_team) with columns:
            - ``home_goal_count``, ``away_goal_count``
            - ``home_penalty_count``, ``away_penalty_count``
            - ``home_own_goals``, ``away_own_goals``
            - ``total_goals``
        """
        if self.goalscorers is None:
            raise RuntimeError("Call load_raw_data() before _aggregate_goal_features().")

        gs = self.goalscorers.copy()

        # A goal by team X appearing in a match (home_team=A, away_team=B)
        # is a "home goal" if team X == home_team, else "away goal".
        # Own goals count for the *conceding* team, so we flip the team.

        # Regular goals (not own goals)
        reg = gs[~gs["own_goal"]].copy()
        own = gs[gs["own_goal"]].copy()

        def _agg_for_side(sub: pd.DataFrame, side: str) -> pd.DataFrame:
            """side = 'home' or 'away'."""
            team_col = f"{side}_team"
            mask = sub["team"] == sub[team_col]
            side_goals = sub[mask].groupby(
                ["date", "home_team", "away_team"]
            ).agg(
                **{
                    f"{side}_goal_count": ("scorer", "count"),
                    f"{side}_penalty_count": ("penalty", "sum"),
                }
            ).reset_index()
            return side_goals

        home_reg = _agg_for_side(reg, "home")
        away_reg = _agg_for_side(reg, "away")

        # Own goals: scored *against* the named team
        home_og = (
            own[own["team"] == own["away_team"]]  # own goal by away player → home team gets it
            .groupby(["date", "home_team", "away_team"])
            .size()
            .reset_index(name="home_own_goals")
        )
        away_og = (
            own[own["team"] == own["home_team"]]
            .groupby(["date", "home_team", "away_team"])
            .size()
            .reset_index(name="away_own_goals")
        )

        # Merge all aggregations
        join_keys = ["date", "home_team", "away_team"]
        agg = (
            home_reg
            .merge(away_reg, on=join_keys, how="outer")
            .merge(home_og, on=join_keys, how="outer")
            .merge(away_og, on=join_keys, how="outer")
            .fillna(0)
        )

        # Ensure integer types
        count_cols = [
            "home_goal_count", "away_goal_count",
            "home_penalty_count", "away_penalty_count",
            "home_own_goals", "away_own_goals",
        ]
        for col in count_cols:
            if col not in agg.columns:
                agg[col] = 0
            agg[col] = agg[col].astype(int)

        # Total goals (including own goals)
        agg["total_goals"] = (
            agg["home_goal_count"] + agg["away_goal_count"]
            + agg["home_own_goals"] + agg["away_own_goals"]
        )

        logger.info(
            "_aggregate_goal_features: %d match-level goal aggregations produced",
            len(agg),
        )
        return agg

    # ------------------------------------------------------------------
    # 6. merge_datasets
    # ------------------------------------------------------------------
    def merge_datasets(self) -> pd.DataFrame:
        """
        Produce the master analytical dataframe.

        Joins:
        1. ``results.csv`` (base)  +  shootout winner flag  (left join)
        2. + aggregated goal-level features               (left join)

        Also applies team name resolution to all team name columns and
        assigns tournament tiers.

        Returns
        -------
        pd.DataFrame
            Master dataframe with all columns from results, a
            ``shootout_winner`` column, goal aggregates, ``is_incomplete``,
            ``result``, and ``match_weight``.

        Raises
        ------
        RuntimeError
            If :meth:`load_raw_data` has not been called first.
        """
        if any(
            df is None
            for df in [self.results, self.goalscorers, self.shootouts, self.former_names]
        ):
            raise RuntimeError("Call load_raw_data() before merge_datasets().")

        logger.info("Starting dataset merge pipeline …")
        join_keys = ["date", "home_team", "away_team"]

        # ---- Step 1: Resolve names in results ----
        res = self.results.copy()
        res = self.resolve_team_names(res, "home_team")
        res = self.resolve_team_names(res, "away_team")

        # ---- Step 2: Resolve names in shootouts ----
        sht = self.shootouts.copy()
        sht = self.resolve_team_names(sht, "home_team")
        sht = self.resolve_team_names(sht, "away_team")
        sht = self.resolve_team_names(sht, "winner")
        sht = sht.rename(columns={"winner": "shootout_winner"})
        sht = sht[join_keys + ["shootout_winner", "first_shooter"]]

        # ---- Step 3: Resolve names in goalscorers ----
        self.goalscorers = self.resolve_team_names(self.goalscorers, "home_team")
        self.goalscorers = self.resolve_team_names(self.goalscorers, "away_team")
        self.goalscorers = self.resolve_team_names(self.goalscorers, "team")

        # ---- Step 4: Merge results + shootouts ----
        master = res.merge(sht, on=join_keys, how="left")
        logger.info(
            "After results + shootouts merge: %d rows, %d cols",
            len(master), master.shape[1],
        )

        # ---- Step 5: Merge goal aggregates ----
        goal_agg = self._aggregate_goal_features()
        master = master.merge(goal_agg, on=join_keys, how="left")
        logger.info(
            "After goal aggregation merge: %d rows, %d cols",
            len(master), master.shape[1],
        )

        # ---- Step 6: Handle missing values ----
        master = self.handle_missing_values(master)

        # ---- Step 7: Create result label ----
        master = self.create_match_result_label(master)

        # ---- Step 8: Assign tournament tiers and weights ----
        master = self.split_by_tournament_weight(master)

        logger.info(
            "Master dataframe ready — %d rows × %d columns",
            len(master), master.shape[1],
        )
        return master

    # ------------------------------------------------------------------
    # 7. split_by_tournament_weight
    # ------------------------------------------------------------------
    def split_by_tournament_weight(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign tournament tiers and sample weights to each match.

        Tier assignment (``tournament_tier`` column):
        - **Tier 1** (weight 3): FIFA World Cup, Continental Championships
        - **Tier 2** (weight 2): World Cup / Continental Qualification, Nations Leagues
        - **Tier 3** (weight 1): Friendly matches and all others

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a ``tournament`` column.

        Returns
        -------
        pd.DataFrame
            Input dataframe with ``tournament_tier`` (int) and
            ``match_weight`` (int) columns added.
        """
        df = df.copy()

        def _assign_tier(tournament: str) -> int:
            """Return tier 1, 2, or 3 for a given tournament name string."""
            if pd.isna(tournament):
                return 3
            t_lower = tournament.lower()
            # Tier 1 patterns
            tier1_patterns = [
                "fifa world cup", "uefa european championship", "uefa euro",
                "copa america", "africa cup of nations", "african cup of nations",
                "afc asian cup", "concacaf gold cup", "ofc nations cup",
            ]
            for pattern in tier1_patterns:
                if pattern in t_lower and "qualif" not in t_lower:
                    return 1
            # Tier 2 patterns
            tier2_patterns = [
                "qualif", "qualification", "nations league",
                "confederations cup", "concacaf championship",
            ]
            for pattern in tier2_patterns:
                if pattern in t_lower:
                    return 2
            return 3

        df["tournament_tier"] = df["tournament"].apply(_assign_tier)
        df["match_weight"] = df["tournament_tier"].map(TIER_WEIGHTS)

        tier_counts = df["tournament_tier"].value_counts().sort_index()
        logger.info(
            "split_by_tournament_weight: tier distribution — %s",
            tier_counts.to_dict(),
        )
        return df

    # ------------------------------------------------------------------
    # 8. build_master  (convenience orchestrator)
    # ------------------------------------------------------------------
    def build_master(self, save: bool = True) -> pd.DataFrame:
        """
        Orchestrate the full pipeline: load → resolve → merge → save.

        Parameters
        ----------
        save : bool, optional
            If ``True`` (default), writes the master dataframe to
            ``data/processed/master.parquet`` for fast downstream loading.

        Returns
        -------
        pd.DataFrame
            The fully processed master dataframe.
        """
        self.load_raw_data()
        self.master = self.merge_datasets()

        if save:
            out_path = self.processed_dir / "master.parquet"
            self.master.to_parquet(out_path, index=False)
            logger.info("Master dataframe saved to %s", out_path)

        return self.master

    # ------------------------------------------------------------------
    # 9. get_summary
    # ------------------------------------------------------------------
    def get_summary(self) -> dict:
        """
        Return a concise summary dictionary of the master dataframe.

        Returns
        -------
        dict
            Keys: ``n_matches``, ``date_range``, ``n_teams``,
            ``n_tournaments``, ``result_distribution``, ``n_shootouts``,
            ``n_incomplete``.
        """
        if self.master is None:
            raise RuntimeError("Call build_master() first.")

        m = self.master
        return {
            "n_matches": len(m),
            "date_range": (m["date"].min(), m["date"].max()),
            "n_teams": len(
                set(m["home_team"].unique()) | set(m["away_team"].unique())
            ),
            "n_tournaments": m["tournament"].nunique(),
            "result_distribution": m["result"].value_counts(dropna=True).to_dict(),
            "n_shootouts": m["shootout_winner"].notna().sum(),
            "n_incomplete": int(m["is_incomplete"].sum()),
            "columns": list(m.columns),
        }
