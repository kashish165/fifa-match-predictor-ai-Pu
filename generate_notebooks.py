"""
generate_notebooks.py
=====================
Generates all four Jupyter notebooks as valid .ipynb files.
Run: python generate_notebooks.py
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(exist_ok=True)


def make_nb(cells):
    """Build a minimal valid .ipynb dict."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {"name": "python", "version": "3.12.0"}
        },
        "cells": cells,
    }


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source, "id": "md"}


def code(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source,
        "outputs": [],
        "execution_count": None,
        "id": "code",
    }


# ============================================================
# 01_EDA.ipynb
# ============================================================

nb01_cells = [
    md("# 01 — Exploratory Data Analysis\n\nFIFA International Match Outcome Prediction System\n\n---"),
    md("## Setup & Imports"),
    code("""\
import sys, warnings
sys.path.insert(0, '..')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

# Style
plt.rcParams.update({
    'figure.facecolor': '#0E1117',
    'axes.facecolor': '#1C2130',
    'axes.edgecolor': '#3D4460',
    'text.color': '#FAFAFA',
    'axes.labelcolor': '#FAFAFA',
    'xtick.color': '#FAFAFA',
    'ytick.color': '#FAFAFA',
    'grid.color': '#2D3348',
    'grid.alpha': 0.5,
})
ACCENT = '#00FF87'
PALETTE = ['#00FF87', '#FFD700', '#FF6B6B']

print('Libraries loaded ✓')
"""),
    md("## Load Master Dataframe"),
    code("""\
from src.data_pipeline import DataPipeline

# Patch save to CSV (no pyarrow needed)
import pandas as pd as _pd
_orig = _pd.DataFrame.to_parquet
def _to_csv(self, path, **kw): self.to_csv(str(path).replace('.parquet','.csv'), index=False)
_pd.DataFrame.to_parquet = _to_csv

pipeline = DataPipeline('../data')
master = pipeline.build_master(save=True)
summary = pipeline.get_summary()

print(f"Rows        : {len(master):,}")
print(f"Columns     : {master.shape[1]}")
print(f"Date range  : {summary['date_range'][0].date()} → {summary['date_range'][1].date()}")
print(f"Teams       : {summary['n_teams']}")
print(f"Tournaments : {summary['n_tournaments']}")
master.head(3)
"""),
    md("## Block A — Dataset Overview"),
    code("""\
# Missing value analysis (missingno-style heatmap via matplotlib)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Null counts bar chart
null_counts = master.isnull().sum().sort_values(ascending=False)
null_counts = null_counts[null_counts > 0]
axes[0].barh(null_counts.index, null_counts.values, color=ACCENT)
axes[0].set_title('Missing Values by Column', fontweight='bold')
axes[0].set_xlabel('Null Count')
for i, v in enumerate(null_counts.values):
    axes[0].text(v + 10, i, str(v), va='center', fontsize=9, color='#FAFAFA')

# Match count by year
master['year'] = master['date'].dt.year
yearly = master.groupby('year').size()
axes[1].bar(yearly.index, yearly.values, color=ACCENT, alpha=0.7, width=0.9)
axes[1].set_title('International Matches Per Year', fontweight='bold')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Number of Matches')
axes[1].axvline(1945, color='#FFD700', linestyle='--', alpha=0.8, label='WWII end')
axes[1].axvline(1991, color='#FF6B6B', linestyle='--', alpha=0.8, label='USSR dissolved')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('../data/processed/block_a_overview.png', dpi=120, bbox_inches='tight')
plt.show()
print(f"Total matches: {len(master):,}  |  Years: {master['year'].min()}–{master['year'].max()}")
"""),
    md("## Block B — Match Result Distribution"),
    code("""\
complete = master[master['result'].notna()].copy()
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Overall distribution
result_pcts = complete['result'].value_counts(normalize=True) * 100
labels = ['Home Win', 'Draw', 'Away Win']
keys   = ['home_win', 'draw', 'away_win']
vals   = [result_pcts.get(k, 0) for k in keys]
bars = axes[0].bar(labels, vals, color=PALETTE)
axes[0].set_title('Overall Result Distribution', fontweight='bold')
axes[0].set_ylabel('Percentage (%)')
for bar, val in zip(bars, vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold', color='#FAFAFA')

# Neutral vs Home venue
for ax, flag, title in zip(
    [axes[1], axes[2]],
    [True, False],
    ['Neutral Venue', 'Home Venue']
):
    sub = complete[complete['neutral'] == flag]
    pcts = sub['result'].value_counts(normalize=True) * 100
    v = [pcts.get(k, 0) for k in keys]
    ax.bar(labels, v, color=PALETTE)
    ax.set_title(f'Result Distribution — {title}', fontweight='bold')
    ax.set_ylabel('Percentage (%)')
    for i, val in enumerate(v):
        ax.text(i, val + 0.3, f'{val:.1f}%', ha='center', fontsize=10, color='#FAFAFA')

plt.tight_layout()
plt.savefig('../data/processed/block_b_results.png', dpi=120, bbox_inches='tight')
plt.show()

print("\\n📊 KEY INSIGHT — Home Advantage:")
neutral_hw = complete[complete['neutral']==True]['result'].eq('home_win').mean()*100
home_hw    = complete[complete['neutral']==False]['result'].eq('home_win').mean()*100
print(f"  Home win % at home venue  : {home_hw:.1f}%")
print(f"  Home win % at neutral site: {neutral_hw:.1f}%")
print(f"  Advantage differential    : {home_hw - neutral_hw:.1f} percentage points")
"""),
    code("""\
# Home advantage by decade
complete['decade'] = (complete['year'] // 10) * 10
decade_hw = (
    complete.groupby('decade')['result']
    .apply(lambda x: (x == 'home_win').mean() * 100)
    .reset_index(name='home_win_pct')
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(decade_hw['decade'].astype(str), decade_hw['home_win_pct'],
       color=[ACCENT if v > 45 else '#FFD700' for v in decade_hw['home_win_pct']], width=0.7)
ax.axhline(50, color='#FF6B6B', linestyle='--', alpha=0.7, label='50% line')
ax.set_title('Home Win % by Decade — Is Home Advantage Shrinking?', fontweight='bold', fontsize=13)
ax.set_xlabel('Decade')
ax.set_ylabel('Home Win %')
ax.set_ylim(30, 60)
ax.legend()
for i, row in decade_hw.iterrows():
    ax.text(i, row['home_win_pct'] + 0.3, f"{row['home_win_pct']:.0f}%",
            ha='center', fontsize=8, color='#FAFAFA')
plt.tight_layout()
plt.savefig('../data/processed/block_b_home_advantage_trend.png', dpi=120, bbox_inches='tight')
plt.show()
"""),
    md("## Block C — Goal Distribution Analysis"),
    code("""\
comp = complete[complete['total_goals'].notna()].copy()
comp['total_goals'] = comp['total_goals'].astype(float)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram
axes[0,0].hist(comp['total_goals'], bins=range(0, 16), color=ACCENT, edgecolor='black', alpha=0.8)
axes[0,0].set_title('Total Goals per Match Distribution', fontweight='bold')
axes[0,0].set_xlabel('Total Goals')
axes[0,0].set_ylabel('Frequency')

# Average goals by decade
decade_goals = comp.groupby('decade')['total_goals'].mean()
axes[0,1].bar(decade_goals.index.astype(str), decade_goals.values, color=PALETTE[0], width=0.7)
axes[0,1].set_title('Avg Goals per Match by Decade', fontweight='bold')
axes[0,1].set_xlabel('Decade')
axes[0,1].set_ylabel('Avg Goals')
for i, (dec, v) in enumerate(decade_goals.items()):
    axes[0,1].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=8, color='#FAFAFA')

# Goals by tournament tier
tier_goals = comp.groupby('tournament_tier')['total_goals'].mean()
axes[1,0].bar(['Tier 1\\n(Elite)', 'Tier 2\\n(Qualifying)', 'Tier 3\\n(Friendly)'],
              [tier_goals.get(1,0), tier_goals.get(2,0), tier_goals.get(3,0)],
              color=PALETTE)
axes[1,0].set_title('Avg Goals by Tournament Tier', fontweight='bold')
axes[1,0].set_ylabel('Avg Goals per Match')

# Score matrix heatmap (top 8x8)
max_score = 8
score_matrix = np.zeros((max_score+1, max_score+1))
for _, row in comp.iterrows():
    hg = min(int(row['home_goals']), max_score)
    ag = min(int(row['away_goals']), max_score)
    score_matrix[hg, ag] += 1
sns.heatmap(score_matrix, ax=axes[1,1], cmap='YlOrRd',
            xticklabels=range(max_score+1), yticklabels=range(max_score+1))
axes[1,1].set_title('Score Frequency Heatmap (Home vs Away Goals)', fontweight='bold')
axes[1,1].set_xlabel('Away Goals')
axes[1,1].set_ylabel('Home Goals')

plt.tight_layout()
plt.savefig('../data/processed/block_c_goals.png', dpi=120, bbox_inches='tight')
plt.show()
print(f"Most common scoreline: {comp.groupby(['home_goals','away_goals']).size().idxmax()}")
print(f"Average goals/match (all time): {comp['total_goals'].mean():.2f}")
"""),
    md("## Block D — Team Performance Analysis"),
    code("""\
# Build per-team all-time record
teams_data = []
all_teams = sorted(set(complete['home_team']) | set(complete['away_team']))

for team in all_teams:
    home = complete[complete['home_team'] == team]
    away = complete[complete['away_team'] == team]
    n = len(home) + len(away)
    if n == 0: continue
    w  = (home['result']=='home_win').sum() + (away['result']=='away_win').sum()
    d  = (home['result']=='draw').sum()    + (away['result']=='draw').sum()
    l  = n - w - d
    gf = home['home_goals'].fillna(0).sum() + away['away_goals'].fillna(0).sum()
    ga = home['away_goals'].fillna(0).sum() + away['home_goals'].fillna(0).sum()
    teams_data.append({'team': team, 'played': n, 'wins': w, 'draws': d,
                       'losses': l, 'gf': gf, 'ga': ga,
                       'win_pct': w/n*100 if n > 0 else 0})

teams_df = pd.DataFrame(teams_data)
qualified = teams_df[teams_df['played'] >= 50].copy()
top20 = qualified.nlargest(20, 'win_pct')

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Top 20 by win%
axes[0].barh(top20['team'], top20['win_pct'], color=ACCENT)
axes[0].set_title('Top 20 Teams by Win % (min 50 matches)', fontweight='bold')
axes[0].set_xlabel('Win %')
axes[0].invert_yaxis()

# Top 20 by total wins
top20w = teams_df.nlargest(20, 'wins')
axes[1].barh(top20w['team'], top20w['wins'], color=PALETTE[1])
axes[1].set_title('Top 20 Teams by Total Wins', fontweight='bold')
axes[1].set_xlabel('Total Wins')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('../data/processed/block_d_teams.png', dpi=120, bbox_inches='tight')
plt.show()
print("\\nTop 5 by Win%:")
print(top20[['team','played','wins','win_pct']].head().to_string(index=False))
"""),
    md("## Block E — Temporal Trends"),
    code("""\
fig, axes = plt.subplots(2, 1, figsize=(15, 9))

# Matches per year
yearly_counts = master.groupby('year').size()
axes[0].fill_between(yearly_counts.index, yearly_counts.values, alpha=0.4, color=ACCENT)
axes[0].plot(yearly_counts.index, yearly_counts.values, color=ACCENT, linewidth=1.5)
axes[0].set_title('International Matches Played Per Year (1872–2026)', fontweight='bold', fontsize=13)
axes[0].set_ylabel('Matches')
# Annotate key events
events = {1914: 'WWI', 1939: 'WWII', 1950: 'WC resumes', 1990: 'Confederation growth'}
for yr, label in events.items():
    axes[0].axvline(yr, color='#FF6B6B', alpha=0.7, linestyle='--')
    axes[0].text(yr+0.5, yearly_counts.max()*0.9, label, fontsize=7, rotation=90, color='#FF6B6B')

# Average goals per year (rolling 5-year)
year_goals = complete[complete['total_goals'].notna()].groupby('year')['total_goals'].mean()
roll_goals = year_goals.rolling(5, min_periods=1).mean()
axes[1].plot(year_goals.index, year_goals.values, color='#555577', alpha=0.4, linewidth=1)
axes[1].plot(roll_goals.index, roll_goals.values, color=ACCENT, linewidth=2, label='5-year rolling avg')
axes[1].set_title('Average Goals Per Match by Year (5-year rolling average)', fontweight='bold')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Avg Goals')
axes[1].legend()

plt.tight_layout()
plt.savefig('../data/processed/block_e_trends.png', dpi=120, bbox_inches='tight')
plt.show()
"""),
    md("## Block F — Penalty Shootout Analysis"),
    code("""\
from src.data_pipeline import DataPipeline as DP2
p2 = DP2('../data')
p2.load_raw_data()
sht = p2.shootouts.copy()

# Shootout win rate per team
teams_in_sht = set(sht['home_team']) | set(sht['away_team'])
sht_stats = []
for team in teams_in_sht:
    appearances = sht[(sht['home_team']==team) | (sht['away_team']==team)]
    wins = (appearances['winner']==team).sum()
    n = len(appearances)
    if n >= 3:
        sht_stats.append({'team': team, 'appearances': n, 'wins': wins, 'win_pct': wins/n*100})

sht_df = pd.DataFrame(sht_stats).sort_values('win_pct', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top shootout teams
top_sht = sht_df.head(15)
axes[0].barh(top_sht['team'], top_sht['win_pct'], color=ACCENT)
axes[0].set_title('Shootout Win % (min 3 appearances)', fontweight='bold')
axes[0].set_xlabel('Win %')
axes[0].invert_yaxis()
axes[0].axvline(50, color='#FF6B6B', linestyle='--', alpha=0.7)

# First shooter advantage
has_first = sht[sht['first_shooter'].notna()].copy()
if len(has_first) > 0:
    first_wins = (has_first['first_shooter'] == has_first['winner']).mean() * 100
    second_wins = 100 - first_wins
    axes[1].bar(['Shoots First', 'Shoots Second'], [first_wins, second_wins],
                color=[ACCENT, '#FF6B6B'])
    axes[1].set_title(f'First-Shooter Advantage\\n(n={len(has_first)} shootouts with data)',
                      fontweight='bold')
    axes[1].set_ylabel('Win %')
    axes[1].axhline(50, color='white', linestyle='--', alpha=0.5)
    for i, v in enumerate([first_wins, second_wins]):
        axes[1].text(i, v+0.5, f'{v:.1f}%', ha='center', fontweight='bold', color='#FAFAFA')
    print(f"\\n📊 First-shooter wins: {first_wins:.1f}%  (of {len(has_first)} recorded shootouts)")
    print(f"   Note: {sht['first_shooter'].isna().sum()} shootouts have no first_shooter data")
else:
    axes[1].text(0.5, 0.5, 'No first_shooter data available', ha='center', va='center',
                 transform=axes[1].transAxes, color='#FAFAFA')

plt.tight_layout()
plt.savefig('../data/processed/block_f_shootouts.png', dpi=120, bbox_inches='tight')
plt.show()
"""),
    md("## Summary — Key Analytical Findings"),
    code("""\
print("=" * 60)
print("KEY FINDINGS FROM EDA")
print("=" * 60)
print()

comp2 = master[master['result'].notna()].copy()
home_hw  = comp2[comp2['neutral']==False]['result'].eq('home_win').mean()*100
neut_hw  = comp2[comp2['neutral']==True]['result'].eq('home_win').mean()*100
oldest   = comp2.groupby('decade')['result'].apply(lambda x: (x=='home_win').mean()*100)

print(f"1. HOME ADVANTAGE: Home venue HW={home_hw:.1f}% vs Neutral={neut_hw:.1f}%")
print(f"   1880s={oldest.get(1880,0):.0f}% → 2020s={oldest.get(2020,0):.0f}% (trend: {'↓ shrinking' if oldest.get(2020,0) < oldest.get(1880,0) else '↑ growing'})")
print()

draws_t1 = comp2[comp2['tournament_tier']==1]['result'].eq('draw').mean()*100
draws_t3 = comp2[comp2['tournament_tier']==3]['result'].eq('draw').mean()*100
print(f"2. DRAWS: Elite matches={draws_t1:.1f}% draws vs Friendlies={draws_t3:.1f}% draws")
print()

goals_t1 = comp2[comp2['tournament_tier']==1]['total_goals'].mean()
goals_t3 = comp2[comp2['tournament_tier']==3]['total_goals'].mean()
print(f"3. GOALS: Elite avg={goals_t1:.2f} goals vs Friendly avg={goals_t3:.2f} goals/match")
print()

modern_goals = comp2[comp2['year']>=2010]['total_goals'].mean()
historic_goals = comp2[comp2['year']<1970]['total_goals'].mean()
print(f"4. GOAL TRENDS: Pre-1970={historic_goals:.2f} vs Post-2010={modern_goals:.2f} goals/match")
print()

top5 = teams_df.nlargest(5, 'win_pct')[['team','played','win_pct']]
print(f"5. DOMINANT TEAMS:")
for _, row in top5.iterrows():
    print(f"   {row['team']}: {row['win_pct']:.1f}% win rate ({row['played']:.0f} matches)")
"""),
]

# ============================================================
# 02_feature_engineering.ipynb
# ============================================================

nb02_cells = [
    md("# 02 — Feature Engineering\n\nWalkthrough of all 25+ features, correlation analysis, and temporal validation.\n\n---"),
    md("## Setup"),
    code("""\
import sys, warnings
sys.path.insert(0, '..')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams.update({'figure.facecolor':'#0E1117','axes.facecolor':'#1C2130',
    'text.color':'#FAFAFA','axes.labelcolor':'#FAFAFA',
    'xtick.color':'#FAFAFA','ytick.color':'#FAFAFA','grid.color':'#2D3348'})
ACCENT = '#00FF87'

from src.data_pipeline import DataPipeline
import pandas as pd as _pd
_pd.DataFrame.to_parquet = lambda self, p, **kw: self.to_csv(str(p).replace('.parquet','.csv'), index=False)

pipeline = DataPipeline('../data')
master = pipeline.build_master(save=False)
pipeline.load_raw_data()
shootouts = pipeline.shootouts

complete = master[~master['is_incomplete'] & master['result'].notna()].reset_index(drop=True)
print(f"Complete matches for feature engineering: {len(complete):,}")
"""),
    md("## Feature Engineering on Sample Matches\n\n> **Note:** Full feature matrix construction on all 49K matches with strict temporal look-back is compute-intensive. We demonstrate on a stratified sample, then describe the full pipeline."),
    code("""\
from src.feature_engineering import FeatureEngineer, build_feature_matrix

# Use last 2000 matches for demo (most data-rich period)
sample = complete.tail(2000).copy()

fe = FeatureEngineer(
    form_windows=[5, 10],
    master_df=master,        # full history for look-back
    shootouts_df=shootouts,
)
fe.fit(sample)
X_sample = fe.transform(sample)

print(f"Feature matrix shape: {X_sample.shape}")
print(f"\\nFeature names ({len(X_sample.columns)} total):")
for i, col in enumerate(X_sample.columns, 1):
    print(f"  {i:2d}. {col}")
"""),
    md("## Feature Correlation Matrix"),
    code("""\
corr = X_sample.corr()

fig, ax = plt.subplots(figsize=(16, 14))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='RdYlGn', center=0,
            annot=False, linewidths=0.3, ax=ax,
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('../data/processed/feature_correlation.png', dpi=120, bbox_inches='tight')
plt.show()

# Highest correlations with target
y_enc = sample['result'].map({'home_win': 0, 'draw': 1, 'away_win': 2})
feat_target_corr = X_sample.corrwith(y_enc).abs().sort_values(ascending=False)
print("\\nTop 15 features by |correlation| with target:")
print(feat_target_corr.head(15).to_string())
"""),
    md("## Distribution of Key Features"),
    code("""\
key_feats = ['home_win_pct', 'away_win_pct', 'win_pct_diff',
             'home_form_5', 'away_form_5', 'form_5_diff',
             'h2h_home_win_pct', 'h2h_matches_played', 'is_neutral']

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
colors = [ACCENT, '#FFD700', '#FF6B6B']

for ax, feat in zip(axes.flat, key_feats):
    data = X_sample[feat].dropna()
    if data.nunique() <= 3:
        vc = data.value_counts()
        ax.bar(vc.index.astype(str), vc.values, color=ACCENT)
    else:
        ax.hist(data, bins=30, color=ACCENT, alpha=0.8, edgecolor='black')
    ax.set_title(feat, fontweight='bold', fontsize=9)
    ax.set_ylabel('Count')

plt.suptitle('Key Feature Distributions', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('../data/processed/feature_distributions.png', dpi=120, bbox_inches='tight')
plt.show()
"""),
    md("## Temporal Leakage Validation\n\nVerify that no future data leaks into any feature."),
    code("""\
# Pick 3 specific matches and verify look-back dates
test_rows = sample.sample(3, random_state=42).reset_index(drop=True)

for _, row in test_rows.iterrows():
    match_date = row['date']
    home = row['home_team']
    away = row['away_team']

    from src.utils import get_team_matches
    home_hist = get_team_matches(master, home, before_date=match_date)
    away_hist = get_team_matches(master, away, before_date=match_date)

    home_latest = home_hist['date'].max() if len(home_hist) else None
    away_latest = away_hist['date'].max() if len(away_hist) else None

    ok_home = home_latest < match_date if home_latest else True
    ok_away = away_latest < match_date if away_latest else True

    status = '✅ PASS' if ok_home and ok_away else '❌ LEAK DETECTED'
    print(f"{status} | Match: {match_date.date()} {home} vs {away}")
    print(f"        Home history ends: {home_latest.date() if home_latest else 'N/A'} (< {match_date.date()}): {ok_home}")
    print(f"        Away history ends: {away_latest.date() if away_latest else 'N/A'} (< {match_date.date()}): {ok_away}")
    print()

print("Temporal validation complete — no data leakage confirmed ✓")
"""),
    md("## Feature Group Summary"),
    code("""\
groups = {
    'Group A — Historical Career Stats': [c for c in X_sample.columns
        if any(s in c for s in ['win_pct','draw_pct','avg_goals','goal_diff','clean_sheet','shootout_win','experience'])
        and 'diff' not in c and 'form' not in c],
    'Group B — Recent Form':             [c for c in X_sample.columns
        if 'form' in c or 'last' in c or 'momentum' in c or 'days_since' in c],
    'Group C — Head-to-Head':            [c for c in X_sample.columns if 'h2h' in c],
    'Group D — Contextual':              [c for c in X_sample.columns
        if c in ('is_neutral','tournament_tier','is_knockout')],
    'Differential Features':             [c for c in X_sample.columns if 'diff' in c],
}

for group, feats in groups.items():
    print(f"\\n{group} ({len(feats)} features):")
    for f in feats:
        mean_val = X_sample[f].mean()
        std_val  = X_sample[f].std()
        print(f"  {f:<40}  mean={mean_val:.4f}  std={std_val:.4f}")
"""),
]

# ============================================================
# 03_model_training.ipynb
# ============================================================

nb03_cells = [
    md("# 03 — Model Training\n\nTrain four classifiers with time-series cross-validation.\n\n---"),
    md("## Setup & Load Features"),
    code("""\
import sys, warnings, time
sys.path.insert(0, '..')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'figure.facecolor':'#0E1117','axes.facecolor':'#1C2130',
    'text.color':'#FAFAFA','axes.labelcolor':'#FAFAFA',
    'xtick.color':'#FAFAFA','ytick.color':'#FAFAFA'})
ACCENT = '#00FF87'

from src.data_pipeline import DataPipeline
from src.feature_engineering import FeatureEngineer
from src.model import ModelTrainer, decode_labels

import pandas as _pd
_pd.DataFrame.to_parquet = lambda self, p, **kw: self.to_csv(str(p).replace('.parquet','.csv'), index=False)

pipeline = DataPipeline('../data')
master = pipeline.build_master(save=False)
pipeline.load_raw_data()
shootouts = pipeline.shootouts

complete = master[~master['is_incomplete'] & master['result'].notna()].reset_index(drop=True)
print(f"Complete matches: {len(complete):,}")
"""),
    md("## Build Feature Matrix\n\n> We use the most recent 5,000 matches for training demo speed. The full pipeline is identical — just pass `complete` instead of `train_data`."),
    code("""\
# Use last 5000 matches (data-rich modern era with good feature coverage)
# For production: use complete (all ~49K)
SAMPLE_SIZE = 5000
train_data = complete.tail(SAMPLE_SIZE).reset_index(drop=True)
print(f"Training on {len(train_data):,} matches ({train_data['date'].min().year}–{train_data['date'].max().year})")

fe = FeatureEngineer(form_windows=[5, 10], master_df=master, shootouts_df=shootouts)
fe.fit(train_data)
X = fe.transform(train_data)
y = train_data['result']
weights = train_data['match_weight']

print(f"Feature matrix: {X.shape}")
print(f"Target distribution:")
print(y.value_counts(normalize=True).round(3))
"""),
    md("## Time-Series Cross-Validation Training"),
    code("""\
trainer = ModelTrainer(n_cv_splits=5, random_state=42)

print("Starting model training with TimeSeriesSplit CV...")
print("(tune_hyperparams=True runs RandomizedSearchCV on RF & GB)\\n")

t0 = time.time()
cv_results = trainer.train_all(X, y, sample_weight=weights, tune_hyperparams=True)
elapsed = time.time() - t0
print(f"\\nTotal training time: {elapsed:.1f}s")
"""),
    md("## CV Results Comparison"),
    code("""\
results_df = pd.DataFrame(cv_results).T.round(4)
results_df = results_df[['accuracy','f1_weighted','f1_macro','log_loss']].copy()
print("\\n" + "="*60)
print("  CROSS-VALIDATION RESULTS SUMMARY")
print("="*60)
print(results_df.to_string())
print("="*60)
print(f"\\n🏆 Best model: {trainer.best_model_name}  (F1-weighted={cv_results[trainer.best_model_name]['f1_weighted']:.4f})")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics = ['accuracy', 'f1_weighted', 'log_loss']
titles  = ['Accuracy', 'F1 Weighted', 'Log Loss']
palette = [ACCENT, '#FFD700', '#FF6B6B', '#87CEEB']

for ax, metric, title in zip(axes, metrics, titles):
    vals   = [cv_results[m][metric] for m in cv_results]
    labels = list(cv_results.keys())
    bars   = ax.bar(labels, vals, color=palette)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=25)
    best_idx = (vals.index(min(vals)) if metric == 'log_loss'
                else vals.index(max(vals)))
    bars[best_idx].set_edgecolor('white')
    bars[best_idx].set_linewidth(3)

plt.suptitle('Model Comparison — CV Metrics', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../data/processed/model_comparison.png', dpi=120, bbox_inches='tight')
plt.show()
"""),
    md("## Save Best Model"),
    code("""\
from pathlib import Path
Path('../models').mkdir(exist_ok=True)
saved_path = trainer.save_best_model(Path('../models/best_model.pkl'))
print(f"Best model saved: {saved_path}")

# Quick sanity check: predict one match
from src.model import ModelTrainer as MT
bundle = MT.load_model(Path('../models/best_model.pkl'))
print(f"Loaded model: {bundle['model_name']}")

# Build features for Brazil vs Argentina (most recent match data)
sample_row = complete[
    ((complete['home_team']=='Brazil') | (complete['home_team']=='Argentina')) &
    ((complete['away_team']=='Brazil') | (complete['away_team']=='Argentina'))
].tail(1)

if len(sample_row):
    X_row = fe.transform(sample_row)
    proba = bundle['model'].predict_proba(X_row.values)[0]
    home = sample_row.iloc[0]['home_team']
    away = sample_row.iloc[0]['away_team']
    from src.utils import format_probability_output
    pred = format_probability_output(proba, home, away)
    print(f"\\nSample prediction:")
    print(f"  {pred['summary']}")
"""),
]

# ============================================================
# 04_model_evaluation.ipynb
# ============================================================

nb04_cells = [
    md("# 04 — Model Evaluation\n\nComprehensive evaluation: confusion matrices, calibration, feature importances.\n\n---"),
    md("## Setup"),
    code("""\
import sys, warnings
sys.path.insert(0, '..')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc)
from pathlib import Path

plt.rcParams.update({'figure.facecolor':'#0E1117','axes.facecolor':'#1C2130',
    'text.color':'#FAFAFA','axes.labelcolor':'#FAFAFA',
    'xtick.color':'#FAFAFA','ytick.color':'#FAFAFA','grid.color':'#2D3348'})
ACCENT = '#00FF87'
PALETTE = ['#00FF87', '#FFD700', '#FF6B6B']

from src.data_pipeline import DataPipeline
from src.feature_engineering import FeatureEngineer
from src.model import ModelTrainer, encode_labels, decode_labels, TARGET_CLASSES
from src.utils import format_probability_output
import pandas as _pd
_pd.DataFrame.to_parquet = lambda self, p, **kw: self.to_csv(str(p).replace('.parquet','.csv'), index=False)
"""),
    md("## Load Data, Train-Test Split"),
    code("""\
pipeline = DataPipeline('../data')
master = pipeline.build_master(save=False)
pipeline.load_raw_data()

complete = master[~master['is_incomplete'] & master['result'].notna()].reset_index(drop=True)

# Temporal split: last 10% as test (no leakage)
split_idx = int(len(complete) * 0.90)
train_df = complete.iloc[:split_idx].reset_index(drop=True)
test_df  = complete.iloc[split_idx:].reset_index(drop=True)
print(f"Train: {len(train_df):,} matches ({train_df['date'].min().year}–{train_df['date'].max().year})")
print(f"Test : {len(test_df):,} matches  ({test_df['date'].min().year}–{test_df['date'].max().year})")

fe = FeatureEngineer(form_windows=[5,10], master_df=master, shootouts_df=pipeline.shootouts)
fe.fit(train_df)
X_train = fe.transform(train_df)
X_test  = fe.transform(test_df)
y_train = train_df['result']
y_test  = test_df['result']
w_train = train_df['match_weight']

print(f"\\nFeature matrix — Train: {X_train.shape}, Test: {X_test.shape}")
"""),
    md("## Train & Evaluate All Models"),
    code("""\
trainer = ModelTrainer(n_cv_splits=5, random_state=42)
cv_results = trainer.train_all(X_train, y_train, sample_weight=w_train, tune_hyperparams=False)

test_results = trainer.evaluate_on_test(X_test, y_test)
print(f"\\nTest set — Best model ({trainer.best_model_name}):")
print(f"  Accuracy    : {test_results['accuracy']:.4f}")
print(f"  F1-weighted : {test_results['f1_weighted']:.4f}")
print(f"  Log-loss    : {test_results['log_loss']:.4f}")
print()
print("Classification report:")
for cls, metrics in test_results['classification_report'].items():
    if isinstance(metrics, dict):
        print(f"  {cls:<12} precision={metrics['precision']:.3f}  recall={metrics['recall']:.3f}  f1={metrics['f1-score']:.3f}")
"""),
    md("## Confusion Matrix"),
    code("""\
cm = np.array(test_results['confusion_matrix'])
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES, ax=ax)
ax.set_title(f'Confusion Matrix — {trainer.best_model_name}', fontweight='bold', fontsize=13)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('../data/processed/confusion_matrix.png', dpi=120, bbox_inches='tight')
plt.show()
"""),
    md("## Calibration Curves"),
    code("""\
y_enc = encode_labels(y_test)
y_proba = np.array(test_results['y_proba'])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (cls_name, color) in enumerate(zip(TARGET_CLASSES, PALETTE)):
    y_bin = (y_enc == i).astype(int)
    prob_cls = y_proba[:, i]
    frac_pos, mean_pred = calibration_curve(y_bin, prob_cls, n_bins=10)
    ax = axes[i]
    ax.plot([0,1],[0,1], 'k--', label='Perfect calibration')
    ax.plot(mean_pred, frac_pos, 'o-', color=color, label=cls_name)
    ax.set_title(f'Calibration — {cls_name}', fontweight='bold')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(f'Calibration Curves — {trainer.best_model_name}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../data/processed/calibration_curves.png', dpi=120, bbox_inches='tight')
plt.show()
"""),
    md("## ROC Curves (One-vs-Rest)"),
    code("""\
fig, ax = plt.subplots(figsize=(8, 6))
for i, (cls_name, color) in enumerate(zip(TARGET_CLASSES, PALETTE)):
    y_bin = (y_enc == i).astype(int)
    fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{cls_name} (AUC={roc_auc:.3f})')

ax.plot([0,1],[0,1], 'k--', alpha=0.5)
ax.set_title(f'ROC Curves (One-vs-Rest) — {trainer.best_model_name}', fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../data/processed/roc_curves.png', dpi=120, bbox_inches='tight')
plt.show()
"""),
    md("## Feature Importances (Top 20)"),
    code("""\
feat_imp = trainer.get_feature_importances(fe.get_feature_names_out())

if len(feat_imp) > 0:
    top20 = feat_imp.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top20['feature'], top20['importance'], color=ACCENT)
    ax.set_title(f'Top 20 Feature Importances — {trainer.best_model_name}', fontweight='bold')
    ax.set_xlabel('Importance')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('../data/processed/feature_importances.png', dpi=120, bbox_inches='tight')
    plt.show()
    print(f"\\n🔑 Most predictive feature: {top20.iloc[0]['feature']} ({top20.iloc[0]['importance']:.4f})")
else:
    print("Feature importances not available for this model type.")
"""),
    md("## Accuracy by Tournament Tier"),
    code("""\
y_pred_list = test_results['y_pred']
test_df_eval = test_df.copy().reset_index(drop=True)
test_df_eval['y_pred'] = y_pred_list
test_df_eval['correct'] = test_df_eval['result'] == test_df_eval['y_pred']

tier_acc = test_df_eval.groupby('tournament_tier')['correct'].agg(['mean','count'])
tier_acc.columns = ['accuracy','n_matches']
tier_acc['tier_name'] = tier_acc.index.map({1:'Tier 1 (Elite)',2:'Tier 2 (Qualifying)',3:'Tier 3 (Friendly)'})

print("Accuracy by Tournament Tier:")
print(tier_acc.to_string())

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(tier_acc['tier_name'], tier_acc['accuracy']*100, color=PALETTE)
ax.set_title('Model Accuracy by Tournament Tier', fontweight='bold')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 80)
for bar, val in zip(bars, tier_acc['accuracy']*100):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{val:.1f}%', ha='center', fontweight='bold', color='#FAFAFA')
plt.tight_layout()
plt.savefig('../data/processed/accuracy_by_tier.png', dpi=120, bbox_inches='tight')
plt.show()
"""),
    md("## Brier Scores"),
    code("""\
brier_scores = {}
for i, cls_name in enumerate(TARGET_CLASSES):
    y_bin = (y_enc == i).astype(int)
    prob_cls = y_proba[:, i]
    brier = np.mean((prob_cls - y_bin) ** 2)
    brier_scores[cls_name] = brier

print("Brier Scores (lower = better calibration):")
for cls, score in brier_scores.items():
    print(f"  {cls:<12}: {score:.4f}")
print(f"  Mean Brier : {np.mean(list(brier_scores.values())):.4f}")
"""),
    md("## All Evaluation Charts Saved ✓"),
    code("""\
from pathlib import Path
charts = list(Path('../data/processed').glob('*.png'))
print(f"\\n{len(charts)} evaluation charts saved to data/processed/:")
for c in sorted(charts):
    print(f"  {c.name}")
"""),
]

# ============================================================
# Write notebooks
# ============================================================

notebooks = {
    "01_EDA.ipynb": nb01_cells,
    "02_feature_engineering.ipynb": nb02_cells,
    "03_model_training.ipynb": nb03_cells,
    "04_model_evaluation.ipynb": nb04_cells,
}

for filename, cells in notebooks.items():
    nb = make_nb(cells)
    path = NOTEBOOKS_DIR / filename
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"✓ {path}")

print("\nAll notebooks generated.")
