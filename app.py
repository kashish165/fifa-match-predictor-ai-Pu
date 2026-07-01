"""
streamlit_app.py  —  FIFA Match Outcome Prediction System
==========================================================
Professional dark-pitch dashboard with 4 tabs:
  Tab 1 — Match Predictor        (flags · probability bars · H2H · form)
  Tab 2 — Team Analysis          (records · win-trend · goals · opponents)
  Tab 3 — Tournament Bracket     (Monte-Carlo knockout / group stage)
  Tab 4 — Historical Explorer    (nations · goals · home-advantage · scorers · shootouts)
"""

from __future__ import annotations
import random, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

from src.data_pipeline import DataPipeline
from src.feature_engineering import FeatureEngineer
from src.model import ModelTrainer, TARGET_CLASSES
from src.utils import format_probability_output, get_team_matches

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Country flag emoji map (ISO → emoji) ─────────────────────────────────────
FLAG_MAP: dict[str, str] = {
    "Brazil": "🇧🇷", "Argentina": "🇦🇷", "France": "🇫🇷", "Germany": "🇩🇪",
    "Spain": "🇪🇸", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Italy": "🇮🇹", "Netherlands": "🇳🇱",
    "Portugal": "🇵🇹", "Belgium": "🇧🇪", "Croatia": "🇭🇷", "Uruguay": "🇺🇾",
    "Mexico": "🇲🇽", "USA": "🇺🇸", "Japan": "🇯🇵", "South Korea": "🇰🇷",
    "Senegal": "🇸🇳", "Morocco": "🇲🇦", "Australia": "🇦🇺", "Poland": "🇵🇱",
    "Denmark": "🇩🇰", "Switzerland": "🇨🇭", "Sweden": "🇸🇪", "Colombia": "🇨🇴",
    "Chile": "🇨🇱", "Ecuador": "🇪🇨", "Peru": "🇵🇪", "Paraguay": "🇵🇾",
    "Bolivia": "🇧🇴", "Venezuela": "🇻🇪", "Tunisia": "🇹🇳", "Cameroon": "🇨🇲",
    "Nigeria": "🇳🇬", "Ghana": "🇬🇭", "Egypt": "🇪🇬", "Algeria": "🇩🇿",
    "South Africa": "🇿🇦", "Iran": "🇮🇷", "Saudi Arabia": "🇸🇦", "Qatar": "🇶🇦",
    "China": "🇨🇳", "India": "🇮🇳", "Turkey": "🇹🇷", "Greece": "🇬🇷",
    "Czech Republic": "🇨🇿", "Hungary": "🇭🇺", "Romania": "🇷🇴", "Serbia": "🇷🇸",
    "Ukraine": "🇺🇦", "Russia": "🇷🇺", "Austria": "🇦🇹", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿", "Ireland": "🇮🇪", "Norway": "🇳🇴", "Finland": "🇫🇮",
    "Iceland": "🇮🇸", "Canada": "🇨🇦", "New Zealand": "🇳🇿", "Costa Rica": "🇨🇷",
    "Panama": "🇵🇦", "Honduras": "🇭🇳", "Jamaica": "🇯🇲", "Trinidad and Tobago": "🇹🇹",
    "Ivory Coast": "🇨🇮", "Mali": "🇲🇱", "Burkina Faso": "🇧🇫", "Angola": "🇦🇴",
    "Zambia": "🇿🇲", "Zimbabwe": "🇿🇼", "Uganda": "🇺🇬", "Kenya": "🇰🇪",
    "Israel": "🇮🇱", "Jordan": "🇯🇴", "Iraq": "🇮🇶", "Syria": "🇸🇾",
    "Indonesia": "🇮🇩", "Thailand": "🇹🇭", "Vietnam": "🇻🇳", "Philippines": "🇵🇭",
    "Bolivia": "🇧🇴", "West Germany": "🇩🇪", "Soviet Union": "🇷🇺",
    "Yugoslavia": "🇷🇸", "Czechoslovakia": "🇨🇿",
}

def flag(team: str) -> str:
    return FLAG_MAP.get(team, "🏳️")

# ── HD Flag images via flagcdn.com (free, no API key needed) ─────────────────
COUNTRY_CODES: dict[str, str] = {
    "Brazil": "br", "Argentina": "ar", "France": "fr", "Germany": "de",
    "Spain": "es", "England": "gb-eng", "Italy": "it", "Netherlands": "nl",
    "Portugal": "pt", "Belgium": "be", "Croatia": "hr", "Uruguay": "uy",
    "Mexico": "mx", "USA": "us", "Japan": "jp", "South Korea": "kr",
    "Senegal": "sn", "Morocco": "ma", "Australia": "au", "Poland": "pl",
    "Denmark": "dk", "Switzerland": "ch", "Sweden": "se", "Colombia": "co",
    "Chile": "cl", "Ecuador": "ec", "Peru": "pe", "Paraguay": "py",
    "Tunisia": "tn", "Cameroon": "cm", "Nigeria": "ng", "Ghana": "gh",
    "Egypt": "eg", "Algeria": "dz", "South Africa": "za", "Iran": "ir",
    "Saudi Arabia": "sa", "Qatar": "qa", "Turkey": "tr", "Greece": "gr",
    "Czech Republic": "cz", "Hungary": "hu", "Romania": "ro", "Serbia": "rs",
    "Ukraine": "ua", "Russia": "ru", "Austria": "at", "Scotland": "gb-sct",
    "Wales": "gb-wls", "Ireland": "ie", "Norway": "no", "Finland": "fi",
    "Iceland": "is", "Canada": "ca", "New Zealand": "nz", "Costa Rica": "cr",
    "Panama": "pa", "Honduras": "hn", "Jamaica": "jm", "Israel": "il",
    "Jordan": "jo", "Iraq": "iq", "Indonesia": "id", "Thailand": "th",
    "Vietnam": "vn", "Bolivia": "bo", "Venezuela": "ve", "West Germany": "de",
    "Soviet Union": "ru", "Yugoslavia": "rs", "Czechoslovakia": "cz",
    "China": "cn", "India": "in", "Ivory Coast": "ci", "Mali": "ml",
    "Angola": "ao", "Zambia": "zm", "Zimbabwe": "zw", "Uganda": "ug",
    "Kenya": "ke", "Syria": "sy", "Philippines": "ph", "Trinidad and Tobago": "tt",
    "Burkina Faso": "bf",
}

def flag_img(team: str, width: int = 90) -> str:
    """Returns HD flag <img> tag from flagcdn.com — free, no API key needed."""
    code = COUNTRY_CODES.get(team, "un")
    url  = f"https://flagcdn.com/w160/{code}.png"
    return (
        f'<img src="{url}" width="{width}" '
        f'style="border-radius:8px;box-shadow:0 4px 16px #0006;'
        f'object-fit:cover;display:block;margin:0 auto;" '
        f'onerror="this.style.display=\'none\'" />'
    )

# ── matplotlib dark theme ─────────────────────────────────────────────────────
PLT_STYLE = {
    "figure.facecolor": "#0A0E1A", "axes.facecolor": "#111827",
    "axes.edgecolor": "#1F2937", "text.color": "#F9FAFB",
    "axes.labelcolor": "#9CA3AF", "xtick.color": "#6B7280",
    "ytick.color": "#6B7280", "grid.color": "#1F2937", "grid.alpha": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
}
plt.rcParams.update(PLT_STYLE)
ACCENT  = "#00FF87"
GOLD    = "#FFD700"
RED     = "#FF4B4B"
BLUE    = "#38BDF8"
PURPLE  = "#A78BFA"
PALETTE = [ACCENT, GOLD, RED, BLUE, PURPLE, "#FB923C"]

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Oswald:wght@400;500;600;700&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"] {
    background: #0A0E1A !important;
    color: #F9FAFB !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: #060A14 !important;
    border-right: 1px solid #1F2937 !important;
}
[data-testid="stHeader"] { background: transparent !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #111827 !important;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1F2937;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #6B7280 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 8px 16px !important;
    border: none !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00FF8722, #00FF8711) !important;
    color: #00FF87 !important;
    border: 1px solid #00FF8744 !important;
}
[data-testid="stTabsContent"] { padding-top: 1.5rem !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00FF87, #00CC6A) !important;
    color: #000 !important;
    font-weight: 800 !important;
    font-size: 0.9rem !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 20px #00FF8730 !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 0 30px #00FF8760 !important;
}
[data-testid="stBaseButton-secondary"] > button {
    background: transparent !important;
    color: #00FF87 !important;
    border: 1px solid #00FF8744 !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #111827 !important;
    border: 1px solid #1F2937 !important;
    border-radius: 10px !important;
    color: #F9FAFB !important;
}
.stSelectbox label { color: #9CA3AF !important; font-size: 0.8rem !important; font-weight: 500 !important; letter-spacing: 0.5px !important; }

/* Toggle */
.stToggle > label { color: #9CA3AF !important; font-size: 0.8rem !important; }

/* Slider */
.stSlider > label { color: #9CA3AF !important; font-size: 0.8rem !important; }

/* Divider */
hr { border-color: #1F2937 !important; margin: 1.5rem 0 !important; }

/* Spinner */
.stSpinner { color: #00FF87 !important; }

/* --- Custom component classes --- */
.pitch-header {
    background: linear-gradient(135deg, #0F172A 0%, #111827 50%, #0F172A 100%);
    border: 1px solid #1F2937;
    border-radius: 16px;
    padding: 2rem 2rem 1.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.pitch-header::before {
    content: '';
    position: absolute; top: 0; left: 50%; transform: translateX(-50%);
    width: 1px; height: 100%; background: #1F2937;
}
.pitch-title {
    font-family: 'Oswald', sans-serif;
    font-size: 1.8rem; font-weight: 700;
    color: #F9FAFB; letter-spacing: 1px;
    text-align: center; margin-bottom: 0.3rem;
}
.pitch-subtitle {
    color: #4B5563; font-size: 0.8rem;
    text-align: center; letter-spacing: 2px; text-transform: uppercase;
}
.vs-badge {
    font-family: 'Oswald', sans-serif;
    font-size: 2.5rem; font-weight: 900;
    color: #00FF87; text-align: center;
    text-shadow: 0 0 20px #00FF8760;
    line-height: 1;
}
.team-flag-box {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 14px;
    padding: 1.5rem 1rem;
    text-align: center;
    transition: border-color 0.2s;
}
.team-flag-box:hover { border-color: #00FF8744; }
.flag-emoji { font-size: 3.5rem; display: block; margin-bottom: 0.5rem; }
.team-name-display {
    font-family: 'Oswald', sans-serif;
    font-size: 1.3rem; font-weight: 600;
    color: #F9FAFB; letter-spacing: 0.5px;
}
.team-sub { font-size: 0.72rem; color: #4B5563; text-transform: uppercase; letter-spacing: 1px; }

/* Metric cards */
.metric-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    height: 100%;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #374151; }
.metric-label {
    color: #4B5563; font-size: 0.7rem;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin-bottom: 0.4rem; display: block;
}
.metric-val {
    font-family: 'Oswald', sans-serif;
    font-size: 2rem; font-weight: 700; line-height: 1;
    color: #00FF87;
}
.metric-val.white { color: #F9FAFB; }
.metric-val.red   { color: #FF4B4B; }
.metric-val.gold  { color: #FFD700; }
.metric-sub {
    font-size: 0.7rem; color: #4B5563;
    margin-top: 0.3rem; display: block;
}

/* Probability result block */
.result-block {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    position: relative;
}
.result-block.winner { border-color: #00FF87; box-shadow: 0 0 24px #00FF8720; }
.result-block.draw   { border-color: #FFD700; box-shadow: 0 0 24px #FFD70015; }
.result-block.away   { border-color: #A78BFA; box-shadow: 0 0 24px #A78BFA15; }
.result-pct {
    font-family: 'Oswald', sans-serif;
    font-size: 3.2rem; font-weight: 700; line-height: 1;
}
.result-label {
    font-size: 0.72rem; letter-spacing: 2px;
    text-transform: uppercase; color: #6B7280;
    margin-bottom: 0.8rem; display: block;
}
.prob-track {
    background: #1F2937; border-radius: 6px;
    height: 8px; overflow: hidden; margin-top: 0.8rem;
}
.prob-fill {
    height: 100%; border-radius: 6px;
    transition: width 0.6s ease;
}

/* Confidence badge */
.conf-badge {
    display: inline-block;
    padding: 4px 14px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 700;
    letter-spacing: 1px; text-transform: uppercase;
}
.conf-high   { background: #00FF8720; color: #00FF87; border: 1px solid #00FF8744; }
.conf-medium { background: #FFD70020; color: #FFD700; border: 1px solid #FFD70044; }
.conf-low    { background: #FF4B4B20; color: #FF4B4B; border: 1px solid #FF4B4B44; }

/* Form badges */
.form-pill {
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px; border-radius: 50%;
    font-size: 0.7rem; font-weight: 800;
    margin: 2px;
}
.fw { background: #00FF87; color: #000; }
.fd { background: #FFD700; color: #000; }
.fl { background: #FF4B4B; color: #fff; }

/* Section header */
.sec-head {
    font-family: 'Oswald', sans-serif;
    font-size: 0.75rem; font-weight: 600;
    color: #4B5563; letter-spacing: 3px;
    text-transform: uppercase;
    border-bottom: 1px solid #1F2937;
    padding-bottom: 8px; margin: 1.5rem 0 1rem;
}

/* H2H table */
.h2h-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.h2h-table thead tr { color: #4B5563; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 1px; }
.h2h-table tbody tr { border-bottom: 1px solid #1F2937; }
.h2h-table td, .h2h-table th { padding: 7px 10px; }
.score-cell { text-align: center; font-weight: 800; font-size: 1rem; }

/* Sidebar metric */
.sb-metric {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
}
.sb-metric-label { color: #4B5563; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.sb-metric-val { font-family: 'Oswald', sans-serif; font-size: 1.5rem; font-weight: 700; color: #00FF87; }
.sb-metric-sub { color: #374151; font-size: 0.68rem; margin-top: 2px; }

/* Winner announcement */
.winner-row {
    background: linear-gradient(135deg, #00FF8710, #00FF8705);
    border: 1px solid #00FF8730;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
    margin-top: 1rem;
}
.winner-label { color: #4B5563; font-size: 0.72rem; letter-spacing: 2px; text-transform: uppercase; }
.winner-name  { font-family: 'Oswald', sans-serif; font-size: 1.6rem; font-weight: 700; color: #00FF87; }

/* Pill tag */
.pill {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.7rem; font-weight: 600;
    background: #1F2937; color: #9CA3AF;
    margin: 2px; border: 1px solid #374151;
}
</style>
""", unsafe_allow_html=True)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data"

# ── data loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading match database…")
def load_data():
    pipeline = DataPipeline(str(DATA_DIR))
    orig = pd.DataFrame.to_parquet
    def _to_csv(self, path, **kw):
        self.to_csv(str(path).replace(".parquet", ".csv"), index=False)
    pd.DataFrame.to_parquet = _to_csv

    master = pipeline.build_master(save=False)
    pipeline.load_raw_data()
    shootouts = pipeline.shootouts

    complete = master[~master["is_incomplete"] & master["result"].notna()].copy()
    teams = sorted(set(master["home_team"].dropna()) | set(master["away_team"].dropna()))

    records = {}
    for team in teams:
        home = complete[complete["home_team"] == team]
        away = complete[complete["away_team"] == team]
        n  = len(home) + len(away)
        w  = (home["result"] == "home_win").sum() + (away["result"] == "away_win").sum()
        d  = (home["result"] == "draw").sum()     + (away["result"] == "draw").sum()
        l  = n - w - d
        gf = home["home_goals"].fillna(0).sum() + away["away_goals"].fillna(0).sum()
        ga = home["away_goals"].fillna(0).sum() + away["home_goals"].fillna(0).sum()
        records[team] = {"played": n, "wins": int(w), "draws": int(d), "losses": int(l),
                         "gf": int(gf), "ga": int(ga), "gd": int(gf - ga)}

    model_bundle = None
    model_path = MODELS_DIR / "best_model.pkl"
    if model_path.exists():
        import joblib
        model_bundle = joblib.load(model_path)

    recent = complete.tail(2000).reset_index(drop=True)
    fe = FeatureEngineer(form_windows=[5, 10], master_df=master, shootouts_df=shootouts)
    fe.fit(recent)

    return {"master": master, "complete": complete, "shootouts": shootouts,
            "teams": teams, "records": records,
            "feature_engineer": fe, "model_bundle": model_bundle}


# ── helpers ───────────────────────────────────────────────────────────────────
def _predict(home_team, away_team, is_neutral, tournament_tier, data):
    fe = data["feature_engineer"]
    row = pd.DataFrame([{
        "date": pd.Timestamp.now(), "home_team": home_team, "away_team": away_team,
        "neutral": is_neutral,
        "tournament": ["FIFA World Cup","FIFA World Cup qualification","Friendly"][tournament_tier-1],
        "tournament_tier": tournament_tier,
        "total_goals": 2.5, "home_goals": None, "away_goals": None,
        "result": None, "match_weight": 1, "is_incomplete": False,
    }])
    X_row = fe.transform(row)
    bundle = data.get("model_bundle")
    if bundle is None:
        rh = data["records"].get(home_team, {"played":1,"wins":0})
        ra = data["records"].get(away_team, {"played":1,"wins":0})
        wh = rh["wins"] / max(rh["played"],1)
        wa = ra["wins"] / max(ra["played"],1)
        if not is_neutral: wh *= 1.15
        total = wh + wa + 0.25
        ph = wh/total; pa = wa/total; pd_ = max(0,1-ph-pa)
        t2 = ph+pd_+pa
        return format_probability_output(np.array([ph/t2, pd_/t2, pa/t2]), home_team, away_team)
    return format_probability_output(bundle["model"].predict_proba(X_row.values)[0], home_team, away_team)


def _form_html(team, complete_df, n=5):
    tm = get_team_matches(complete_df, team)
    last = tm.dropna(subset=["team_result"]).tail(n)
    parts = []
    for _, row in last.iterrows():
        r   = row["team_result"]
        opp = row["away_team"] if row["is_home"] else row["home_team"]
        gs  = int(row["goals_scored"]) if pd.notna(row["goals_scored"]) else "?"
        gc  = int(row["goals_conceded"]) if pd.notna(row["goals_conceded"]) else "?"
        cls = {"win":"fw","draw":"fd","loss":"fl"}[r]
        parts.append(f'<span class="form-pill {cls}" title="{flag(opp)} {opp} {gs}-{gc}">{r[0].upper()}</span>')
    return "".join(parts) or "<span style='color:#4B5563'>—</span>"


def _h2h_html(home, away, complete_df, n=5):
    h2h = complete_df[
        ((complete_df["home_team"]==home) & (complete_df["away_team"]==away)) |
        ((complete_df["home_team"]==away) & (complete_df["away_team"]==home))
    ].sort_values("date").tail(n)
    if len(h2h) == 0:
        return "<p style='color:#4B5563;font-size:0.82rem'>No previous meetings found.</p>"
    rows = []
    for _, r in h2h.iterrows():
        hg   = int(r["home_goals"]) if pd.notna(r["home_goals"]) else "-"
        ag   = int(r["away_goals"]) if pd.notna(r["away_goals"]) else "-"
        date = r["date"].strftime("%d %b %Y")
        res  = r.get("result","")
        if res == "home_win":
            sc = f"<span style='color:#00FF87'>{hg}</span> – {ag}"
        elif res == "away_win":
            sc = f"{hg} – <span style='color:#00FF87'>{ag}</span>"
        else:
            sc = f"<span style='color:#FFD700'>{hg} – {ag}</span>"
        hfl = flag(r["home_team"]); afl = flag(r["away_team"])
        rows.append(
            f"<tr><td style='color:#4B5563;white-space:nowrap'>{date}</td>"
            f"<td>{hfl} <b>{r['home_team']}</b></td>"
            f"<td class='score-cell'>{sc}</td>"
            f"<td>{afl} <b>{r['away_team']}</b></td>"
            f"<td style='color:#374151;font-size:0.72rem'>{r.get('tournament','')[:28]}</td></tr>"
        )
    return (
        "<table class='h2h-table'><thead><tr>"
        "<th>Date</th><th>Home</th><th style='text-align:center'>Score</th>"
        "<th>Away</th><th>Tournament</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _confidence_html(conf: str) -> str:
    cls = {"High":"conf-high","Medium":"conf-medium","Low":"conf-low"}.get(conf,"conf-medium")
    return f'<span class="conf-badge {cls}">{conf} confidence</span>'


def _mini_donut(values, colors, labels, title, figsize=(3.2, 3.2)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    wedges, texts = ax.pie(values, colors=colors, startangle=90,
                           wedgeprops=dict(width=0.45, edgecolor="#111827", linewidth=2))
    max_i = values.index(max(values))
    ax.text(0, -0.05, f"{max(values):.0f}%", ha="center", va="center",
            color=colors[max_i], fontsize=16, fontweight="bold",
            fontfamily="sans-serif")
    ax.text(0, -0.38, title, ha="center", va="center",
            color="#6B7280", fontsize=7, fontfamily="sans-serif")
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.25),
              ncol=1, frameon=False, fontsize=7,
              labelcolor="#9CA3AF")
    plt.tight_layout(pad=0.2)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MATCH PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
def tab_predictor(data):
    teams = data["teams"]

    # ── header ──
    st.markdown("""
    <div class='pitch-header'>
      <div class='pitch-title'>⚽ MATCH PREDICTION SYSTEM</div>
      <div class='pitch-subtitle'>Advanced Machine Learning · International Football</div>
    </div>
    """, unsafe_allow_html=True)

    # ── team selectors ──
    col_home, col_vs, col_away = st.columns([5, 1, 5])
    with col_home:
        home_team = st.selectbox("🏠 Home Team", teams,
                                  index=teams.index("Brazil") if "Brazil" in teams else 0,
                                  key="pred_home")
    with col_vs:
        st.markdown("<div style='height:2.2rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='vs-badge'>VS</div>", unsafe_allow_html=True)
    with col_away:
        away_opts = [t for t in teams if t != home_team]
        away_team = st.selectbox("✈️ Away Team", away_opts,
                                  index=away_opts.index("Argentina") if "Argentina" in away_opts else 0,
                                  key="pred_away")

    # ── flag display ──
    c1, c2, c3 = st.columns([4, 2, 4])
    
    # Calculate attack and defense percentages for home team
    rec_h = data["records"].get(home_team, {})
    n_h = max(rec_h.get("played", 1), 1)
    attack_h = (rec_h.get("gf", 0) / n_h) / 2.0 * 100  # Normalize to a % (assuming ~2 goals/game average)
    defense_h = (1 - (rec_h.get("ga", 0) / n_h) / 2.0) * 100
    attack_h = min(max(attack_h, 0), 100)
    defense_h = min(max(defense_h, 0), 100)

    # Calculate attack and defense percentages for away team
    rec_a = data["records"].get(away_team, {})
    n_a = max(rec_a.get("played", 1), 1)
    attack_a = (rec_a.get("gf", 0) / n_a) / 2.0 * 100
    defense_a = (1 - (rec_a.get("ga", 0) / n_a) / 2.0) * 100
    attack_a = min(max(attack_a, 0), 100)
    defense_a = min(max(defense_a, 0), 100)
    
    with c1:
        st.markdown(f"""
        <div class='team-flag-box'>
          {flag_img(home_team, width=100)}
          <div class='team-name-display' style='margin-top:12px'>{home_team}</div>
          <div class='team-sub'>Home Side</div>
          <div style='display: flex; justify-content: space-around; margin-top: 8px;'>
            <div><span style='color:#00FF87; font-weight:700;'>⚔ {attack_h:.0f}%</span></div>
            <div><span style='color:#38BDF8; font-weight:700;'>🛡 {defense_h:.0f}%</span></div>
          </div>
        </div>""", unsafe_allow_html=True)
    
    with c2:
        n_h = max(rec_h.get("played",1), 1)
        n_a = max(rec_a.get("played",1), 1)
        wp_h = rec_h.get("wins",0) / n_h * 100
        wp_a = rec_a.get("wins",0) / n_a * 100
        st.markdown(f"""
        <div style='text-align:center;padding:1.5rem 0'>
          <div style='color:#4B5563;font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px'>Win Rate</div>
          <div style='font-family:Oswald,sans-serif;font-size:1.1rem;color:#00FF87;font-weight:700'>{wp_h:.0f}%</div>
          <div style='color:#1F2937;font-size:1.3rem;font-weight:900;margin:4px 0'>vs</div>
          <div style='font-family:Oswald,sans-serif;font-size:1.1rem;color:#A78BFA;font-weight:700'>{wp_a:.0f}%</div>
        </div>""", unsafe_allow_html=True)
    
    with c3:
        st.markdown(f"""
        <div class='team-flag-box'>
          {flag_img(away_team, width=100)}
          <div class='team-name-display' style='margin-top:12px'>{away_team}</div>
          <div class='team-sub'>Away Side</div>
          <div style='display: flex; justify-content: space-around; margin-top: 8px;'>
            <div><span style='color:#00FF87; font-weight:700;'>⚔ {attack_a:.0f}%</span></div>
            <div><span style='color:#38BDF8; font-weight:700;'>🛡 {defense_a:.0f}%</span></div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── match context ──
    ct1, ct2, ct3 = st.columns([2, 2, 2])
    with ct1:
        is_neutral = st.toggle("⚖️ Neutral Venue", value=False)
    with ct2:
        t_type = st.selectbox("🏆 Tournament Type",
                               ["Major Tournament (Tier 1)", "Qualification (Tier 2)", "Friendly (Tier 3)"],
                               label_visibility="collapsed")
        tier_map = {"Major Tournament (Tier 1)":1, "Qualification (Tier 2)":2, "Friendly (Tier 3)":3}
        tournament_tier = tier_map[t_type]
    with ct3:
        predict_btn = st.button("🔮  PREDICT MATCH", use_container_width=True)

    # ── prediction output ──
    if predict_btn:
        with st.spinner("Computing prediction…"):
            pred = _predict(home_team, away_team, is_neutral, tournament_tier, data)

        hp = pred["home_win"] * 100
        dp = pred["draw"]     * 100
        ap = pred["away_win"] * 100
        conf = pred["confidence"]

        st.markdown("<div class='sec-head'>Predicted Outcome</div>", unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns([3,3,3,2])

        with r1:
            cls = "winner" if hp >= dp and hp >= ap else ""
            st.markdown(f"""
            <div class='result-block {cls}'>
              <span class='result-label'>{flag(home_team)} {home_team} Win</span>
              <div class='result-pct' style='color:#00FF87'>{hp:.1f}%</div>
              <div class='prob-track'><div class='prob-fill' style='width:{hp}%;background:#00FF87'></div></div>
            </div>""", unsafe_allow_html=True)

        with r2:
            cls = "draw" if dp >= hp and dp >= ap else ""
            st.markdown(f"""
            <div class='result-block {cls}'>
              <span class='result-label'>🤝 Draw</span>
              <div class='result-pct' style='color:#FFD700'>{dp:.1f}%</div>
              <div class='prob-track'><div class='prob-fill' style='width:{dp}%;background:#FFD700'></div></div>
            </div>""", unsafe_allow_html=True)

        with r3:
            cls = "away" if ap >= hp and ap >= dp else ""
            st.markdown(f"""
            <div class='result-block {cls}'>
              <span class='result-label'>{flag(away_team)} {away_team} Win</span>
              <div class='result-pct' style='color:#A78BFA'>{ap:.1f}%</div>
              <div class='prob-track'><div class='prob-fill' style='width:{ap}%;background:#A78BFA'></div></div>
            </div>""", unsafe_allow_html=True)

        with r4:
            fig = _mini_donut(
                [hp, dp, ap], ["#00FF87","#FFD700","#A78BFA"],
                [f"{home_team}", "Draw", f"{away_team}"], "Confidence"
            )
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Winner + confidence row
        winner_label = pred["predicted_winner"] if pred["predicted_outcome"] != "draw" else "Draw"
        conf_cls = {"High":"conf-high","Medium":"conf-medium","Low":"conf-low"}.get(conf,"conf-medium")
        st.markdown(f"""
        <div class='winner-row'>
          <div class='winner-label'>Predicted Result</div>
          <div class='winner-name'>{flag(home_team) if pred["predicted_outcome"]=="home_win" else (flag(away_team) if pred["predicted_outcome"]=="away_win" else "🤝")} &nbsp;{winner_label}</div>
          <div style='margin-top:6px'><span class='conf-badge {conf_cls}'>{conf} Confidence</span></div>
        </div>""", unsafe_allow_html=True)

        # ── Key metrics row ──
        st.markdown("<div class='sec-head'>Key Match Metrics</div>", unsafe_allow_html=True)
        rec_h = data["records"].get(home_team, {})
        rec_a = data["records"].get(away_team, {})
        n_h   = max(rec_h.get("played",1), 1)
        n_a   = max(rec_a.get("played",1), 1)

        m_cols = st.columns(6)
        metrics = [
            ("Home Win %",      f'{rec_h.get("wins",0)/n_h*100:.1f}%',   "#00FF87"),
            ("Away Win %",      f'{rec_a.get("wins",0)/n_a*100:.1f}%',   "#A78BFA"),
            ("Avg Goals (H)",   f'{rec_h.get("gf",0)/n_h:.2f}',          "#38BDF8"),
            ("Avg Goals (A)",   f'{rec_a.get("gf",0)/n_a:.2f}',          "#FB923C"),
            ("Goal Diff (H)",   f'{rec_h.get("gd",0):+d}',               "#00FF87"),
            ("Clean Sheet*",    f'{(1-(rec_h.get("ga",0)/n_h)/3)*100:.0f}%', "#FFD700"),
        ]
        for col, (lbl, val, clr) in zip(m_cols, metrics):
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                  <span class='metric-label'>{lbl}</span>
                  <div class='metric-val' style='color:{clr};font-size:1.6rem'>{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── H2H + Form ──
        left, right = st.columns(2)
        with left:
            st.markdown(f"<div class='sec-head'>Head-to-Head · Last 5 Meetings</div>", unsafe_allow_html=True)
            st.markdown(_h2h_html(home_team, away_team, data["complete"]), unsafe_allow_html=True)

        with right:
            st.markdown("<div class='sec-head'>Recent Form · Last 5 Matches</div>", unsafe_allow_html=True)
            complete = data["complete"]
            for tm in [home_team, away_team]:
                badges = _form_html(tm, complete)
                t_matches = get_team_matches(complete, tm)
                last5 = t_matches.dropna(subset=["team_result"]).tail(5)
                pts   = last5["team_result"].map({"win":3,"draw":1,"loss":0}).sum()
                st.markdown(f"""
                <div style='background:#111827;border:1px solid #1F2937;border-radius:10px;
                            padding:12px 16px;margin-bottom:10px;'>
                  <div style='display:flex;align-items:center;justify-content:space-between'>
                    <span style='font-weight:700;font-size:0.95rem'>{flag(tm)} {tm}</span>
                    <span style='color:#4B5563;font-size:0.75rem'>{pts}/15 pts</span>
                  </div>
                  <div style='margin-top:8px'>{badges}</div>
                </div>""", unsafe_allow_html=True)

        # ── Footer credit ──
        st.markdown("""
        <div style='text-align: center; margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid #1F2937;'>
            <span style='color: #374151; font-size: 0.7rem; letter-spacing: 1px;'>
                ⚽ FIFA Prediction System · Made by Waqar Khan · COMSATS University Islamabad
            </span>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TEAM ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def tab_team_analysis(data):
    st.markdown("<div class='pitch-title' style='margin-bottom:1.5rem'>📊 TEAM ANALYSIS DASHBOARD</div>",
                unsafe_allow_html=True)

    team = st.selectbox("Select Team", data["teams"],
                        index=data["teams"].index("Brazil") if "Brazil" in data["teams"] else 0,
                        key="analysis_team")
    rec      = data["records"].get(team, {})
    complete = data["complete"]

    n = max(rec.get("played",1), 1)

    # ── All-time record cards ──
    st.markdown(f"<div class='sec-head'>{flag(team)} {team} · All-Time Record</div>",
                unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    stats = [
        ("Played",  f'{rec.get("played",0):,}',  "#F9FAFB", "matches"),
        ("Won",     f'{rec.get("wins",0):,}',     "#00FF87", "victories"),
        ("Drawn",   f'{rec.get("draws",0):,}',    "#FFD700", "draws"),
        ("Lost",    f'{rec.get("losses",0):,}',   "#FF4B4B", "defeats"),
        ("GF",      f'{rec.get("gf",0):,}',       "#38BDF8", "goals for"),
        ("GA",      f'{rec.get("ga",0):,}',       "#FB923C", "goals against"),
        ("GD",      f'{rec.get("gd",0):+,}',      "#00FF87" if rec.get("gd",0)>=0 else "#FF4B4B", "goal diff"),
    ]
    for col, (lbl, val, clr, sub) in zip([c1,c2,c3,c4,c5,c6,c7], stats):
        with col:
            st.markdown(f"""<div class='metric-card'>
              <span class='metric-label'>{lbl}</span>
              <div class='metric-val' style='color:{clr}'>{val}</div>
              <span class='metric-sub'>{sub}</span>
            </div>""", unsafe_allow_html=True)

    # ── Charts ──
    tm_all = get_team_matches(complete, team)
    if len(tm_all) == 0:
        st.warning(f"No match data found for {team}.")
        return
    tm_all = tm_all.dropna(subset=["team_result","goals_scored","goals_conceded"])
    tm_all["year"] = tm_all["date"].dt.year

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("<div class='sec-head'>Win % Trend · 3-Year Rolling</div>", unsafe_allow_html=True)
        yearly = tm_all[tm_all["year"]>=1950].groupby("year").apply(
            lambda x: (x["team_result"]=="win").mean()*100
        ).reset_index(name="win_pct")
        roll = yearly.set_index("year")["win_pct"].rolling(3, min_periods=1).mean()
        fig, ax = plt.subplots(figsize=(7,3.8))
        ax.fill_between(roll.index, roll.values, alpha=0.12, color=ACCENT)
        ax.plot(roll.index, roll.values, color=ACCENT, linewidth=2.5)
        ax.axhline(50, color=RED, linestyle="--", alpha=0.4, linewidth=1)
        ax.set_ylabel("Win %")
        ax.set_xlabel("Year")
        ax.grid(True, alpha=0.2)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig); plt.close()

    with col_c2:
        st.markdown("<div class='sec-head'>Avg Goals Scored vs Conceded</div>", unsafe_allow_html=True)
        yearly_goals = tm_all[tm_all["year"]>=1960].groupby("year").agg(
            scored=("goals_scored","mean"), conceded=("goals_conceded","mean")
        )
        fig, ax = plt.subplots(figsize=(7,3.8))
        ax.bar(yearly_goals.index,  yearly_goals["scored"],   color=ACCENT,  label="Scored",   alpha=0.85, width=0.7)
        ax.bar(yearly_goals.index, -yearly_goals["conceded"], color=RED,     label="Conceded", alpha=0.8,  width=0.7)
        ax.axhline(0, color="#374151", linewidth=0.8)
        ax.set_ylabel("Goals/match"); ax.set_xlabel("Year"); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig); plt.close()

    # ── Top opponents ──
    st.markdown("<div class='sec-head'>Top 10 Opponents by Matches Played</div>", unsafe_allow_html=True)
    opp_col = tm_all.apply(lambda r: r["away_team"] if r["is_home"] else r["home_team"], axis=1)
    top_opp = opp_col.value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10,3.2))
    bars = ax.barh(top_opp.index, top_opp.values, color=[ACCENT]*len(top_opp), height=0.6)
    for bar, val in zip(bars, top_opp.values):
        ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2, str(val),
                va="center", fontsize=8, color="#6B7280")
    ax.set_xlabel("Matches"); ax.invert_yaxis()
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout(pad=0.5)
    st.pyplot(fig); plt.close()

    # ── Performance by tier ──
    st.markdown("<div class='sec-head'>Performance by Tournament Tier</div>", unsafe_allow_html=True)
    tier_cols = st.columns(3)
    for i, (tier, name, clr) in enumerate([(1,"Elite / World Cup",ACCENT),(2,"Qualifying",GOLD),(3,"Friendlies",BLUE)]):
        sub_t = tm_all[tm_all["tournament_tier"]==tier]
        wp  = (sub_t["team_result"]=="win").mean()*100 if len(sub_t)>0 else 0
        nm  = len(sub_t)
        with tier_cols[i]:
            st.markdown(f"""<div class='metric-card'>
              <span class='metric-label'>{name}</span>
              <div class='metric-val' style='color:{clr}'>{wp:.0f}%</div>
              <span class='metric-sub'>{nm} matches</span>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TOURNAMENT BRACKET
# ═══════════════════════════════════════════════════════════════════════════════
def tab_bracket(data):
    st.markdown("<div class='pitch-title' style='margin-bottom:1.5rem'>🏆 TOURNAMENT BRACKET PREDICTOR</div>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#4B5563;font-size:0.85rem'>Simulate knockout or group stage using Monte Carlo.</p>",
                unsafe_allow_html=True)

    teams_all = data["teams"]
    mode = st.radio("Mode", ["Quick Knockout (8 Teams)", "Group Stage Simulation"], horizontal=True)

    if mode == "Quick Knockout (8 Teams)":
        st.markdown("<div class='sec-head'>Select 8 Teams</div>", unsafe_allow_html=True)
        defaults = ["Brazil","Argentina","France","Germany","Spain","England","Italy","Netherlands"]
        defaults = [t for t in defaults if t in teams_all]
        while len(defaults) < 8:
            for t in teams_all:
                if t not in defaults: defaults.append(t)
                if len(defaults)==8: break

        selected = []
        cols = st.columns(4)
        for i in range(8):
            with cols[i%4]:
                picked = st.selectbox(f"Team {i+1}", teams_all,
                                      index=teams_all.index(defaults[i]) if defaults[i] in teams_all else i,
                                      key=f"bt_{i}")
                selected.append(picked)

        n_sims = st.slider("Monte Carlo Simulations", 1000, 10000, 5000, step=1000)

        if st.button("🎲  RUN SIMULATION", use_container_width=True):
            with st.spinner(f"Running {n_sims:,} simulations…"):
                results = {t: {"qf":0,"sf":0,"final":0,"winner":0} for t in selected}
                def gwp(t1,t2):
                    r1 = data["records"].get(t1,{"played":1,"wins":0})
                    r2 = data["records"].get(t2,{"played":1,"wins":0})
                    w1 = r1["wins"]/max(r1["played"],1)
                    w2 = r2["wins"]/max(r2["played"],1)
                    total = w1+w2
                    return 0.5 if total==0 else w1/total

                for _ in range(n_sims):
                    bracket = list(selected)
                    for stage in ["qf","sf","final","winner"]:
                        winners=[]
                        random.shuffle(bracket)
                        for j in range(0,len(bracket),2):
                            if j+1>=len(bracket):
                                winners.append(bracket[j])
                                results[bracket[j]][stage]+=1
                                continue
                            p = gwp(bracket[j],bracket[j+1])
                            w = bracket[j] if random.random()<p else bracket[j+1]
                            winners.append(w)
                            results[bracket[j]][stage]+=1
                            results[bracket[j+1]][stage]+=1
                        bracket=winners
                    if bracket: results[bracket[0]]["winner"]+=1

            sim_df = pd.DataFrame([{
                "Team": f"{flag(t)} {t}",
                "QF %": f'{results[t]["qf"]/n_sims*100:.1f}%',
                "SF %": f'{results[t]["sf"]/n_sims*100:.1f}%',
                "Final %": f'{results[t]["final"]/n_sims*100:.1f}%',
                "Win %": f'{results[t]["winner"]/n_sims*100:.1f}%',
                "_raw": results[t]["winner"]/n_sims*100,
            } for t in selected]).sort_values("_raw", ascending=False).drop(columns=["_raw"])

            st.markdown("<div class='sec-head'>Simulation Results</div>", unsafe_allow_html=True)
            st.dataframe(sim_df.reset_index(drop=True), use_container_width=True)

            win_probs  = {t: results[t]["winner"]/n_sims*100 for t in selected}
            sorted_t   = sorted(win_probs, key=win_probs.get, reverse=True)
            bar_colors = [ACCENT,GOLD,RED] + [BLUE]*(len(sorted_t)-3)

            fig, ax = plt.subplots(figsize=(10,4))
            bars = ax.bar([f"{flag(t)}\n{t}" for t in sorted_t],
                          [win_probs[t] for t in sorted_t], color=bar_colors)
            ax.set_title(f"Champion Probability · {n_sims:,} Simulations",
                         fontweight="bold", fontsize=12)
            ax.set_ylabel("Probability (%)")
            for bar, t in zip(bars, sorted_t):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                        f"{win_probs[t]:.1f}%", ha="center", fontsize=9, color="#9CA3AF")
            ax.grid(True, alpha=0.2, axis="y")
            plt.tight_layout(pad=0.5)
            st.pyplot(fig); plt.close()

    else:
        st.markdown("<div class='sec-head'>Configure Group</div>", unsafe_allow_html=True)
        n_teams = st.slider("Group Size", 3, 6, 4)
        group_teams = []
        cols = st.columns(n_teams)
        for i in range(n_teams):
            with cols[i]:
                t = st.selectbox(f"Team {i+1}", teams_all, key=f"gt_{i}", index=i%len(teams_all))
                group_teams.append(t)

        n_sims2 = st.slider("Simulations", 1000, 5000, 2000, step=500)

        if st.button("🎲  SIMULATE GROUP", use_container_width=True):
            with st.spinner("Simulating…"):
                pts_total = {t:0 for t in group_teams}
                gd_total  = {t:0 for t in group_teams}
                for _ in range(n_sims2):
                    pts = {t:0 for t in group_teams}
                    gd  = {t:0 for t in group_teams}
                    for i, t1 in enumerate(group_teams):
                        for j, t2 in enumerate(group_teams):
                            if i>=j: continue
                            r1 = data["records"].get(t1,{"played":1,"gf":0})
                            r2 = data["records"].get(t2,{"played":1,"gf":0})
                            s1 = int(np.random.poisson(max(r1["gf"]/max(r1["played"],1),0.5)))
                            s2 = int(np.random.poisson(max(r2["gf"]/max(r2["played"],1),0.5)))
                            if s1>s2:   pts[t1]+=3
                            elif s1==s2: pts[t1]+=1; pts[t2]+=1
                            else:        pts[t2]+=3
                            gd[t1]+=s1-s2; gd[t2]+=s2-s1
                    for t in group_teams:
                        pts_total[t]+=pts[t]; gd_total[t]+=gd[t]

            res_df = pd.DataFrame([{
                "Team": f"{flag(t)} {t}",
                "Avg Points": f'{pts_total[t]/n_sims2:.2f}',
                "Avg GD":     f'{gd_total[t]/n_sims2:.2f}',
            } for t in sorted(group_teams, key=lambda x:-pts_total[x])])
            st.markdown("<div class='sec-head'>Group Table · Averaged Over Simulations</div>",
                        unsafe_allow_html=True)
            st.dataframe(res_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HISTORICAL EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
def tab_history(data):
    st.markdown("<div class='pitch-title' style='margin-bottom:1.5rem'>🗂️ HISTORICAL STATISTICS EXPLORER</div>",
                unsafe_allow_html=True)

    complete = data["complete"].copy()
    complete["year"]   = complete["date"].dt.year
    complete["decade"] = (complete["year"] // 10) * 10

    subtab = st.selectbox("Explorer Section", [
        "Most Successful Nations",
        "Goal Trends Over Time",
        "Home Advantage Evolution",
        "Top Individual Scorers",
        "Penalty Shootout Specialists",
    ])

    if subtab == "Most Successful Nations":
        c1, c2 = st.columns(2)
        with c1:
            decade_opts = sorted(complete["decade"].unique())
            dec = st.selectbox("Decade Filter", ["All"] + [str(d) for d in decade_opts])
        with c2:
            tier_f = st.selectbox("Tier", ["All","Tier 1 (Elite)","Tier 2 (Qualifying)","Tier 3 (Friendly)"])
        df_f = complete.copy()
        if dec != "All": df_f = df_f[df_f["decade"]==int(dec)]
        if "1" in tier_f: df_f = df_f[df_f["tournament_tier"]==1]
        elif "2" in tier_f: df_f = df_f[df_f["tournament_tier"]==2]
        elif "3" in tier_f: df_f = df_f[df_f["tournament_tier"]==3]

        ts = []
        for team in sorted(set(df_f["home_team"].dropna())|set(df_f["away_team"].dropna())):
            h = df_f[df_f["home_team"]==team]; a = df_f[df_f["away_team"]==team]
            n = len(h)+len(a)
            if n<10: continue
            w = (h["result"]=="home_win").sum()+(a["result"]=="away_win").sum()
            ts.append({"team":team,"played":n,"wins":int(w),"win_pct":w/n*100})
        ts_df = pd.DataFrame(ts).nlargest(20,"win_pct")

        fig, ax = plt.subplots(figsize=(10,7))
        colors = [ACCENT if i==0 else GOLD if i==1 else RED if i==2 else "#374151"
                  for i in range(len(ts_df))]
        ax.barh(ts_df["team"], ts_df["win_pct"], color=colors, height=0.7)
        ax.set_title("Top 20 Nations by Win %  (min 10 matches)", fontweight="bold", fontsize=12)
        ax.set_xlabel("Win %"); ax.invert_yaxis()
        ax.grid(True, alpha=0.2, axis="x")
        plt.tight_layout(pad=0.5)
        st.pyplot(fig); plt.close()

    elif subtab == "Goal Trends Over Time":
        c1, c2 = st.columns(2)
        with c1: start_yr = st.slider("From Year", 1872, 2020, 1950)
        with c2: roll_w   = st.slider("Rolling Window (years)", 1, 10, 5)
        gby = complete[(complete["year"]>=start_yr)&(complete["total_goals"].notna())
                       ].groupby("year")["total_goals"].mean()
        rolling = gby.rolling(roll_w, min_periods=1).mean()
        fig, ax = plt.subplots(figsize=(12,4.5))
        ax.fill_between(gby.index, gby.values, alpha=0.1, color=ACCENT)
        ax.plot(gby.index, gby.values, color="#374151", alpha=0.5, linewidth=1)
        ax.plot(rolling.index, rolling.values, color=ACCENT, linewidth=2.5,
                label=f"{roll_w}-yr rolling avg")
        for yr, lbl in [(1914,"WWI"),(1939,"WWII"),(1966,"England WC"),(1990,"Low-scoring era"),(2010,"SA WC")]:
            if yr>=start_yr:
                ax.axvline(yr, color=RED, alpha=0.4, linestyle="--")
                ax.text(yr+0.3, rolling.max()*0.9, lbl, fontsize=7, rotation=90, color=RED, va="top")
        ax.set_title("Average Goals per Match Over Time", fontweight="bold", fontsize=12)
        ax.set_xlabel("Year"); ax.set_ylabel("Avg Goals"); ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig); plt.close()

    elif subtab == "Home Advantage Evolution":
        dhw = (
            complete[complete["neutral"]==False]
            .groupby("decade")["result"]
            .apply(lambda x: (x=="home_win").mean()*100)
            .reset_index(name="home_win_pct")
        )
        fig, ax = plt.subplots(figsize=(12,4.5))
        bar_clrs = [ACCENT if v>45 else GOLD for v in dhw["home_win_pct"]]
        ax.bar(dhw["decade"].astype(str), dhw["home_win_pct"], color=bar_clrs, width=0.7, alpha=0.9)
        ax.axhline(50, color=RED, linestyle="--", alpha=0.6, linewidth=1, label="50%")
        x = np.arange(len(dhw))
        z = np.polyfit(x, dhw["home_win_pct"], 1)
        p = np.poly1d(z)
        ax.plot(dhw["decade"].astype(str), p(x), color=GOLD, linestyle="-",
                linewidth=2, label=f"Trend ({z[0]:+.2f}%/decade)")
        ax.set_title("Home Win % by Decade — Home Advantage Evolution", fontweight="bold", fontsize=12)
        ax.set_xlabel("Decade"); ax.set_ylabel("Home Win %"); ax.set_ylim(30,65)
        ax.legend(fontsize=9)
        for i, row in dhw.iterrows():
            ax.text(i, row["home_win_pct"]+0.4, f'{row["home_win_pct"]:.0f}%',
                    ha="center", fontsize=8, color="#9CA3AF")
        ax.grid(True, alpha=0.2, axis="y")
        plt.tight_layout(pad=0.5)
        st.pyplot(fig); plt.close()
        trend_word = "shrinking" if z[0]<0 else "stable/growing"
        st.markdown(f"<p style='color:#4B5563;font-size:0.82rem'>Home advantage is <b style='color:#00FF87'>{trend_word}</b> ({z[0]:+.2f}% per decade).</p>",
                    unsafe_allow_html=True)

    elif subtab == "Top Individual Scorers":
        pipeline_gs = DataPipeline(str(DATA_DIR))
        pipeline_gs.load_raw_data()
        gs = pipeline_gs.goalscorers.copy()
        gs = gs[~gs["own_goal"] & gs["scorer"].notna()]
        team_f = st.selectbox("Filter by Team", ["All Teams"]+sorted(gs["team"].dropna().unique().tolist()))
        if team_f!="All Teams": gs = gs[gs["team"]==team_f]
        top_s = (gs.groupby("scorer").agg(
            goals=("scorer","count"), penalties=("penalty","sum"),
            team=("team", lambda x: x.mode()[0] if len(x)>0 else "")
        ).reset_index().nlargest(20,"goals"))

        fig, ax = plt.subplots(figsize=(10,7))
        ax.barh(top_s["scorer"], top_s["goals"],     color=ACCENT, alpha=0.9, label="Open Play", height=0.7)
        ax.barh(top_s["scorer"], top_s["penalties"], color=GOLD,   alpha=0.8, label="Penalties",  height=0.7)
        ax.set_title("Top 20 International Goal Scorers", fontweight="bold", fontsize=12)
        ax.set_xlabel("Goals"); ax.invert_yaxis(); ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis="x")
        plt.tight_layout(pad=0.5)
        st.pyplot(fig); plt.close()
        st.dataframe(top_s.reset_index(drop=True), use_container_width=True)

    elif subtab == "Penalty Shootout Specialists":
        sht = data["shootouts"].copy()
        teams_sht = set(sht["home_team"].dropna())|set(sht["away_team"].dropna())
        stats = []
        for team in teams_sht:
            app = sht[(sht["home_team"]==team)|(sht["away_team"]==team)]
            wins = (sht["winner"]==team).sum(); n = len(app)
            if n>=2: stats.append({"team":team,"appearances":n,"wins":wins,"win_pct":wins/n*100})
        sht_df = pd.DataFrame(stats).sort_values("win_pct", ascending=False)
        min_app = st.slider("Min Shootout Appearances", 1, 10, 3)
        sht_df  = sht_df[sht_df["appearances"]>=min_app].head(20)
        fig, ax = plt.subplots(figsize=(10,5.5))
        ax.barh(sht_df["team"], sht_df["win_pct"], color=ACCENT, height=0.65)
        ax.axvline(50, color=RED, linestyle="--", alpha=0.6, linewidth=1, label="50%")
        ax.set_title(f"Shootout Win %  (min {min_app} appearances)", fontweight="bold", fontsize=12)
        ax.set_xlabel("Win %"); ax.invert_yaxis(); ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis="x")
        plt.tight_layout(pad=0.5)
        st.pyplot(fig); plt.close()
        st.dataframe(sht_df.reset_index(drop=True), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
def sidebar(data):
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align:center;padding:1.2rem 0 0.5rem'>
          <div style='font-size:2.2rem'>⚽</div>
          <div style='font-family:Oswald,sans-serif;font-size:1.3rem;
                      font-weight:700;color:#00FF87;letter-spacing:1px'>FIFA PREDICTOR</div>
          <div style='color:#374151;font-size:0.65rem;letter-spacing:3px;
                      text-transform:uppercase;margin-top:3px'>AI Powered Predictions</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#1F2937'>", unsafe_allow_html=True)

        complete = data["complete"]
        model_st = "✅ ML Model Loaded" if data.get("model_bundle") else "⚠️ Heuristic Mode"
        model_clr = "#00FF87" if data.get("model_bundle") else "#FFD700"

        for lbl, val, sub in [
            ("Matches in DB", f'{len(data["master"]):,}', "1872 – 2026"),
            ("National Teams", str(len(data["teams"])), "all confederations"),
            ("Model Status", model_st, "Run notebook 03 to train"),
        ]:
            st.markdown(f"""
            <div class='sb-metric'>
              <div class='sb-metric-label'>{lbl}</div>
              <div class='sb-metric-val' style='color:{model_clr if lbl=="Model Status" else "#00FF87"};
                   font-size:{"0.85rem" if lbl=="Model Status" else "1.5rem"}'>{val}</div>
              <div class='sb-metric-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#1F2937'>", unsafe_allow_html=True)
        st.markdown("""
        <div style='color:#374151;font-size:0.72rem;line-height:1.8'>
          <div style='color:#4B5563;font-size:0.68rem;letter-spacing:1px;
               text-transform:uppercase;margin-bottom:6px'>Built With</div>
          <span class='pill'>scikit-learn</span>
          <span class='pill'>pandas</span>
          <span class='pill'>matplotlib</span>
          <span class='pill'>Streamlit</span>
          <br><br>
          <div style='color:#374151;font-size:0.68rem;margin-top:4px'>
            Data: Kaggle International Football Results 1872–2026
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#1F2937'>", unsafe_allow_html=True)
        # Quick stat
        hw_pct = (complete["result"]=="home_win").mean()*100
        d_pct  = (complete["result"]=="draw").mean()*100
        aw_pct = (complete["result"]=="away_win").mean()*100
        st.markdown(f"""
        <div style='font-size:0.68rem;color:#4B5563;letter-spacing:1px;
             text-transform:uppercase;margin-bottom:8px'>Overall Result Split</div>
        <div style='display:flex;gap:4px;margin-bottom:4px'>
          <div style='flex:{hw_pct};background:#00FF87;height:6px;border-radius:3px'></div>
          <div style='flex:{d_pct};background:#FFD700;height:6px;border-radius:3px'></div>
          <div style='flex:{aw_pct};background:#A78BFA;height:6px;border-radius:3px'></div>
        </div>
        <div style='display:flex;justify-content:space-between;font-size:0.68rem'>
          <span style='color:#00FF87'>H {hw_pct:.0f}%</span>
          <span style='color:#FFD700'>D {d_pct:.0f}%</span>
          <span style='color:#A78BFA'>A {aw_pct:.0f}%</span>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    data = load_data()
    sidebar(data)

    tab1, tab2, tab3, tab4 = st.tabs([
        "⚽  Match Predictor",
        "📊  Team Analysis",
        "🏆  Tournament Bracket",
        "🗂️  Historical Explorer",
    ])
    with tab1: tab_predictor(data)
    with tab2: tab_team_analysis(data)
    with tab3: tab_bracket(data)
    with tab4: tab_history(data)


if __name__ == "__main__":
    main()
