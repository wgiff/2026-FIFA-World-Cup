import base64
import html
import itertools
import os
import re
import textwrap
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import simulation

HOSTS = ["United States", "Mexico", "Canada"]
N_SIMS = 10000
FLAG_DIR = "FIFA"
APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="2026 FIFA World Cup Simulator",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def inject_global_css():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Sora:wght@600;700;800&display=swap');
:root {
    --wc-blue: #0b4cff;
    --wc-blue-dark: #06215f;
    --wc-blue-mid: #0f3fb5;
    --wc-red: #ef233c;
    --wc-gold: #f8c537;
    --wc-bg: #f5f8ff;
    --wc-card: #ffffff;
    --wc-text: #0f172a;
    --wc-muted: #64748b;
    --wc-border: #dbe4f0;
}
/* Hide Streamlit chrome */
[data-testid="stHeader"] {
    display: none !important;
}

[data-testid="stToolbar"] {
    display: none !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

[data-testid="stStatusWidget"] {
    display: none !important;
}

#MainMenu {
    visibility: hidden !important;
}

footer {
    visibility: hidden !important;
}

header {
    visibility: hidden !important;
}
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.stApp {
    background:
        radial-gradient(circle at top left, rgba(11, 76, 255, 0.14), transparent 34rem),
        linear-gradient(180deg, #f7faff 0%, #eef4ff 48%, #f8fbff 100%);
    color: var(--wc-text);
}
.block-container {
    padding-top: 0.75rem !important;
    padding-bottom: 2rem !important;
    max-width: 1500px;
}
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Sora', 'Inter', sans-serif;
    letter-spacing: -0.045em;
    color: var(--wc-text);
}
h2, .stMarkdown h2 {
    font-size: 1.75rem;
    font-weight: 800;
    margin-top: 0.65rem;
    margin-bottom: 0.65rem;
}
h3, .stMarkdown h3 {
    font-size: 1.25rem;
    font-weight: 800;
    margin-top: 0.8rem;
    margin-bottom: 0.45rem;
}
.wc-hero {
    position: relative;
    overflow: hidden;
    border-radius: 22px;
    padding: 1.65rem 1.5rem 1.25rem 1.5rem;
    margin-bottom: 0.95rem;
    background:
        linear-gradient(125deg, rgba(5, 25, 72, 0.98) 0%, rgba(9, 61, 177, 0.97) 58%, rgba(11, 76, 255, 0.95) 100%);
    box-shadow: 0 20px 45px rgba(15, 49, 120, 0.22);
    color: white;
}
.wc-hero::after {
    content: "";
    position: absolute;
    right: -35px;
    top: -55px;
    width: 225px;
    height: 225px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(248, 197, 55, 0.42), rgba(248, 197, 55, 0) 68%);
}
.wc-hero-title {
    font-family: 'Sora', 'Inter', sans-serif;
    font-size: clamp(2rem, 4vw, 3.35rem);
    line-height: 1.12;
    font-weight: 900;
    letter-spacing: -0.055em;
    margin: 0;
}
.wc-hero-subtitle {
    margin-top: 0.45rem;
    color: rgba(255,255,255,0.82);
    font-size: 1.02rem;
    font-weight: 500;
}
.wc-hero-pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.8rem;
}
.wc-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    border: 1px solid rgba(255,255,255,0.22);
    background: rgba(255,255,255,0.12);
    color: white;
    border-radius: 999px;
    padding: 0.34rem 0.65rem;
    font-size: 0.82rem;
    font-weight: 700;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    background: rgba(255,255,255,0.74);
    border: 1px solid rgba(197, 211, 232, 0.8);
    border-radius: 16px;
    padding: 0.35rem;
    box-shadow: 0 10px 25px rgba(30, 64, 175, 0.08);
}
.stTabs [data-baseweb="tab"] {
    height: 2.65rem;
    border-radius: 12px;
    padding: 0 0.85rem;
    color: #334155;
    font-weight: 800;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--wc-blue), #073a9f) !important;
    color: white !important;
    box-shadow: 0 8px 18px rgba(11, 76, 255, 0.25);
}
.stButton > button {
    border: 0 !important;
    border-radius: 13px !important;
    padding: 0.55rem 1rem !important;
    font-weight: 850 !important;
    background: linear-gradient(135deg, var(--wc-blue), #062b86) !important;
    color: white !important;
    box-shadow: 0 10px 18px rgba(11, 76, 255, 0.25);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 13px 25px rgba(11, 76, 255, 0.30);
}
[data-testid="stSelectbox"] label, [data-testid="stNumberInput"] label {
    font-weight: 800;
    color: #1e293b;
}
[data-baseweb="select"] > div {
    border-radius: 13px !important;
    border-color: #c9d7ee !important;
    background: rgba(255,255,255,0.95) !important;
    min-height: 2.65rem;
}
[data-testid="stAlert"] {
    border-radius: 14px;
    border: 1px solid rgba(96, 165, 250, 0.25);
}
.wc-section-card {
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(210, 222, 240, 0.9);
    border-radius: 18px;
    padding: 0.85rem 1rem;
    box-shadow: 0 12px 28px rgba(30, 64, 175, 0.08);
}
.wc-note {
    color: var(--wc-muted);
    font-size: 0.9rem;
    font-weight: 500;
}
[data-testid="stHorizontalBlock"] {
    gap: 0.85rem;
}
hr {
    border-color: rgba(148, 163, 184, 0.28);
}

.score-matrix-card {
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid rgba(210, 222, 240, 0.9);
    border-radius: 18px;
    padding: 0.9rem 0.9rem 1rem 0.9rem;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}
.score-matrix-top-label {
    text-align: center;
    font-weight: 850;
    font-size: 1.05rem;
    margin-bottom: 0.5rem;
}
.score-matrix-side-label {
    min-height: 360px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding-right: 0.45rem;
}
@media (max-width: 700px) {
    .score-matrix-card {
        padding: 0.7rem 0.55rem;
    }
    .score-matrix-top-label {
        text-align: left;
        font-size: 0.92rem;
        margin-left: 0.1rem;
    }
    .score-matrix-side-label {
        min-height: auto;
        justify-content: flex-start;
        padding: 0 0 0.4rem 0;
    }
    .score-matrix-side-label .matrix-side-label {
        max-width: 100%;
        text-align: left;
        font-size: 0.86rem;
    }
}

</style>
        """,
        unsafe_allow_html=True
    )


inject_global_css()

st.markdown(
    """
<div class="wc-hero">
    <div class="wc-hero-title">🏆 2026 FIFA World Cup Simulator</div>
    <div class="wc-hero-subtitle">Interactive tournament forecasts, group-stage odds, and matchup score distributions.</div>
    <div class="wc-hero-pill-row">
        <span class="wc-pill">⚽ 48 Teams</span>
        <span class="wc-pill">📊 Odds from 10,000 sims</span>
        <span class="wc-pill">🔥 Score Heatmaps</span>
        <span class="wc-pill">🏟️ Full Bracket</span>
    </div>
</div>
    """,
    unsafe_allow_html=True
)


def normalize_flag_name(value):
    value = str(value).strip()
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"\s+", " ", value)
    return value.casefold()


def candidate_flag_dirs():
    return [
        APP_DIR / FLAG_DIR,
        Path.cwd() / FLAG_DIR,
    ]


def image_mime_type(raw_bytes):
    if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw_bytes.startswith(b"RIFF") and raw_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def find_flag_path(team):
    team_clean = str(team).strip()
    wanted = normalize_flag_name(team_clean)
    exact_names = [
        f"{team_clean}.png",
        f"{team_clean}.PNG",
        f"{team_clean}.jpg",
        f"{team_clean}.jpeg",
        f"{team_clean}.webp",
    ]
    for folder in candidate_flag_dirs():
        for name in exact_names:
            path = folder / name
            if path.exists():
                return path
        if folder.exists():
            for path in folder.iterdir():
                if not path.is_file():
                    continue
                if path.suffix.casefold() not in [".png", ".jpg", ".jpeg", ".webp"]:
                    continue
                if normalize_flag_name(path.stem) == wanted:
                    return path
    return None


def flag_data_uri(team):
    """Return a base64 data URI for a local flag file.

    Not cached, so corrected/replaced flag files are picked up without clearing
    Streamlit cache. The MIME type is detected from file bytes, which also fixes
    cases where a JPG/WEBP image was accidentally saved with a .png extension.
    """
    path = find_flag_path(team)
    if path is None:
        return ""
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{image_mime_type(raw)};base64,{encoded}"


def flag_img_html(team, width=22):
    team = str(team).strip()
    uri = flag_data_uri(team)
    if not uri:
        return ""
    safe_team = html.escape(team)
    return (
        f'<img class="wc-flag" src="{uri}" alt="{safe_team}" style="width:{width}px;">'
    )


def team_html(team, width=24):
    team = str(team).strip()
    return (
        f'<span class="wc-team">'
        f'{flag_img_html(team, width)}'
        f'<span class="wc-team-name">{html.escape(team)}</span>'
        f'</span>'
    )


def render_html_table(df, height=None):
    table_html = df.to_html(
        escape=False,
        index=False,
        border=0,
        classes="wc-table"
    )
    height_css = f"max-height:{height}px;overflow:auto;" if height else ""
    html_block = f"""
<style>
.wc-table-wrap {{
    {height_css}
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid rgba(210, 222, 240, 0.96);
    border-radius: 16px;
    overflow: auto;
    box-shadow: 0 10px 24px rgba(30, 64, 175, 0.075);
}}
table.wc-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.91rem;
    table-layout: auto;
}}
table.wc-table th {{
    background: linear-gradient(180deg, #f9fbff 0%, #edf4ff 100%);
    color: #183153;
    font-weight: 850;
    text-align: left;
    border-bottom: 1px solid #d8e3f5;
    padding: 0.48rem 0.58rem;
    position: sticky;
    top: 0;
    z-index: 1;
}}
table.wc-table td {{
    border-bottom: 1px solid #edf2fb;
    padding: 0.42rem 0.58rem;
    vertical-align: middle;
    color: #172033;
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: normal;
}}
table.wc-table tr:nth-child(even) td {{ background: rgba(247, 250, 255, 0.62); }}
table.wc-table tr:hover td {{ background: #eaf3ff; }}
table.wc-table td:not(:first-child) {{
    font-variant-numeric: tabular-nums;
}}
.wc-team {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    white-space: normal;
    font-weight: 750;
    max-width: 100%;
    min-width: 0;
    line-height: 1.22;
}}
.wc-team-name {{
    display: inline-block;
    min-width: 0;
    max-width: 100%;
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: normal;
}}
.matrix-side-label {{
    font-weight: 800;
    font-size: 0.98rem;
    line-height: 1.35;
    max-width: 180px;
    white-space: normal;
    overflow-wrap: anywhere;
}}
.matrix-side-label .wc-team {{
    justify-content: center;
    flex-wrap: wrap;
    gap: 0.25rem;
}}

.wc-flag {{
    display: inline-block;
    height: auto;
    max-height: 15px;
    object-fit: contain;
    border: 0;
    border-radius: 1px;
    flex: 0 0 auto;
    box-shadow: none;
    vertical-align: -2px;
}}
@media (max-width: 700px) {{
    .wc-table-wrap {{
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }}
    table.wc-table {{
        width: max-content;
        min-width: 100%;
        font-size: 0.78rem;
    }}
    table.wc-table th,
    table.wc-table td {{
        padding: 0.36rem 0.42rem;
        max-width: 115px;
    }}
    table.wc-table th:first-child,
    table.wc-table td:first-child {{
        max-width: 145px;
    }}
    .wc-team {{
        gap: 5px;
        align-items: flex-start;
        font-size: 0.78rem;
    }}
    .wc-team .wc-flag {{
        width: 17px !important;
        max-height: 12px;
        margin-top: 2px;
    }}
}}
</style>
<div class="wc-table-wrap">{table_html}</div>
"""
    st.markdown(textwrap.dedent(html_block).strip(), unsafe_allow_html=True)


def inject_bracket_css():
    st.markdown(
        '''
<style>
.bracket-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(220px, 1fr));
    gap: 16px;
    align-items: start;
    overflow-x: auto;
    padding: 0.35rem 0 0.8rem 0;
}
.bracket-round-title {
    font-weight: 800;
    color: #08245f;
    margin-bottom: 9px;
    font-size: 0.98rem;
    letter-spacing: -0.02em;
}
.bracket-round {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.bracket-game {
    position: relative;
    border: 1px solid #d5e3f7;
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.96);
    box-shadow: 0 10px 22px rgba(30, 64, 175, 0.08);
    overflow: hidden;
}
.bracket-game::after {
    content: "";
    position: absolute;
    right: -14px;
    top: 50%;
    width: 14px;
    border-top: 1px solid #cbd5e1;
}
.bracket-round:last-child .bracket-game::after {
    display: none;
}
.bracket-team-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 8px 10px;
    border-bottom: 1px solid #eef2f7;
    color: #52627a;
}
.bracket-team-row:last-of-type {
    border-bottom: 0;
}
.bracket-team-row.winner {
    color: #071c4d;
    font-weight: 900;
    background: linear-gradient(90deg, rgba(11,76,255,0.12), rgba(255,255,255,0.95));
    border-left: 4px solid #0b4cff;
    padding-left: 6px;
}
.bracket-team-name {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
}
.bracket-score {
    font-variant-numeric: tabular-nums;
    font-weight: 800;
    color: #071c4d;
}
.bracket-note {
    padding: 6px 10px 8px 10px;
    font-size: 0.78rem;
    color: #64748b;
    background: #f7fbff;
}
.bracket-note.pk {
    color: #b45309;
    font-weight: 850;
}
@media (max-width: 1200px) {
    .bracket-grid {
        grid-template-columns: repeat(5, 240px);
    }
}
</style>
        ''',
        unsafe_allow_html=True
    )


def bracket_game_html(game):
    team1 = html.escape(game["team1"])
    team2 = html.escape(game["team2"])
    winner = game["winner"]
    t1_class = "bracket-team-row winner" if winner == game["team1"] else "bracket-team-row"
    t2_class = "bracket-team-row winner" if winner == game["team2"] else "bracket-team-row"
    note = "Advanced on PKs" if game.get("pk") else ""
    note_class = "bracket-note pk" if game.get("pk") else "bracket-note"
    return f'''
<div class="bracket-game">
    <div class="{t1_class}">
        <span class="bracket-team-name">{flag_img_html(game["team1"])}<span>{team1}</span></span>
        <span class="bracket-score">{game["g1"]}</span>
    </div>
    <div class="{t2_class}">
        <span class="bracket-team-name">{flag_img_html(game["team2"])}<span>{team2}</span></span>
        <span class="bracket-score">{game["g2"]}</span>
    </div>
    <div class="{note_class}">{note}</div>
</div>
'''.strip()


def render_bracket(bracket, round_order):
    inject_bracket_css()
    rounds_html = []
    for round_name in round_order:
        games_html = "".join(bracket_game_html(game) for game in bracket[round_name])
        rounds_html.append(f'''
<div class="bracket-round">
    <div class="bracket-round-title">{html.escape(round_name)}</div>
    {games_html}
</div>
'''.strip())
    st.markdown(
        f'<div class="bracket-grid">{"".join(rounds_html)}</div>',
        unsafe_allow_html=True
    )


def tournament_teams():
    return sorted({
        team
        for group_teams in simulation.world_cup_groups.values()
        for team in group_teams
    })


def format_percent_columns(df, columns):
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x:.1f}%")
    return out


@st.cache_data(show_spinner=False)
def load_tournament_odds(csv_path="Tournament Sims.csv"):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"team": "Team"})
    stage_cols = ["R32", "R16", "QF", "SF", "Final", "Champion"]
    for col in stage_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("Champion", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_group_odds(csv_path="Group Sims.csv", tournament_csv_path="Tournament Sims.csv"):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"team": "Team", "group": "Group"})
    finish_cols = ["1st", "2nd", "3rd", "4th"]
    for col in finish_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    tournament_df = pd.read_csv(tournament_csv_path)
    tournament_df = tournament_df.rename(columns={"team": "Team"})
    tournament_df["R32"] = pd.to_numeric(tournament_df["R32"], errors="coerce")
    df = df.merge(
        tournament_df[["Team", "R32"]],
        on="Team",
        how="left"
    )
    df = df.rename(columns={"R32": "Advance"})
    return df.sort_values(["Group", "1st", "2nd"], ascending=[True, False, False]).reset_index(drop=True)



@st.cache_data(show_spinner=False)
def load_group_stage_matchups(csv_path="Group Stage Matchups.csv"):
    df = pd.read_csv(csv_path)
    required_cols = [
        "Round",
        "Date",
        "Group",
        "Team A",
        "Team B",
        "Quality Rank",
        "Offense Rank",
        "Closeness Rank",
        "Overall Rank"
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Group Stage Matchups.csv is missing columns: {missing}")
    df = df.copy()
    df["Date Parsed"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Round", "Quality Rank", "Offense Rank", "Closeness Rank", "Overall Rank"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["Date Parsed", "Round", "Group", "Team A", "Team B"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def cached_matchup_summary(home_team, away_team, n_sims):
    summary, matrix = simulation.matchup_matrix(
        home_team=home_team,
        away_team=away_team,
        sims=n_sims
    )
    summary["Probability"] = summary["Probability"].round(1)
    matrix = matrix.iloc[:9, :9].copy()
    matrix.index = [str(i) for i in matrix.index]
    matrix.columns = [str(i) for i in matrix.columns]
    return summary, matrix


@st.cache_data(show_spinner=False)
def cached_match_score_distribution(home_team, away_team, n_sims):
    counts = {}
    neutral = not (home_team in HOSTS or away_team in HOSTS)
    for _ in range(n_sims):
        h_goals, a_goals = simulation.simulate_game_once(
            home_team,
            away_team,
            neutral=neutral
        )
        score = f"{home_team} {h_goals}-{a_goals} {away_team}"
        counts[score] = counts.get(score, 0) + 1
    rows = [
        {
            "Score": score,
            "Frequency": count,
            "Probability": round(count / n_sims * 100, 2)
        }
        for score, count in counts.items()
    ]
    return pd.DataFrame(rows).sort_values(
        ["Frequency", "Score"],
        ascending=[False, True]
    ).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def cached_group_match_probabilities(group_name, n_sims):
    schedule = load_group_stage_matchups()
    group_schedule = (
        schedule[schedule["Group"].astype(str) == str(group_name)]
        .sort_values(["Date Parsed", "Round", "Team A", "Team B"])
        .reset_index(drop=True)
    )
    if group_schedule.empty:
        teams = simulation.world_cup_groups[group_name]
        fallback_rows = []
        for round_idx, (team_a, team_b) in enumerate(itertools.combinations(teams, 2), start=1):
            fallback_rows.append({
                "Round": round_idx,
                "Date": "",
                "Group": group_name,
                "Team A": team_a,
                "Team B": team_b
            })
        group_schedule = pd.DataFrame(fallback_rows)
    rows = []
    for _, game in group_schedule.iterrows():
        team_a = str(game["Team A"]).strip()
        team_b = str(game["Team B"]).strip()
        host_team = next((team for team in HOSTS if team in [team_a, team_b]), None)
        if host_team is not None:
            home_team = host_team
            away_team = team_b if team_a == host_team else team_a
            neutral = False
        else:
            home_team = team_a
            away_team = team_b
            neutral = True
        team_a_wins = 0
        draws = 0
        team_b_wins = 0
        for _ in range(n_sims):
            h_goals, a_goals = simulation.simulate_game_once(
                home_team,
                away_team,
                neutral=neutral
            )
            if home_team == team_a:
                team_a_goals = h_goals
                team_b_goals = a_goals
            else:
                team_a_goals = a_goals
                team_b_goals = h_goals
            if team_a_goals > team_b_goals:
                team_a_wins += 1
            elif team_a_goals < team_b_goals:
                team_b_wins += 1
            else:
                draws += 1
        rows.append({
            "Date": game.get("Date", ""),
            "Team A": team_a,
            "Team A Win %": round(team_a_wins / n_sims * 100, 1),
            "Draw %": round(draws / n_sims * 100, 1),
            "Team B Win %": round(team_b_wins / n_sims * 100, 1),
            "Team B": team_b
        })
    return pd.DataFrame(rows)


def simulate_knockout_match_with_pk_indicator(team1, team2):
    g1, g2 = simulation.simulate_knockout_match(team1, team2)
    pk = False
    if g1 > g2:
        winner = team1
    elif g2 > g1:
        winner = team2
    else:
        pk = True
        winner = np.random.choice([team1, team2])
    return g1, g2, winner, pk


def simulate_one_tournament_bracket():
    wc = simulation.simulate_world_cup_once(
        simulation.world_cup_groups,
        simulation.conditions
    )
    teams = wc["r32"].copy()
    round_names = {
        32: "Round of 32",
        16: "Round of 16",
        8: "Quarterfinals",
        4: "Semifinals",
        2: "Final"
    }
    bracket = {}
    while len(teams) > 1:
        round_name = round_names[len(teams)]
        bracket[round_name] = []
        next_round = []
        for i in range(0, len(teams), 2):
            team1 = teams[i]
            team2 = teams[i + 1]
            g1, g2, winner, pk = simulate_knockout_match_with_pk_indicator(
                team1,
                team2
            )
            bracket[round_name].append({
                "team1": team1,
                "team2": team2,
                "g1": g1,
                "g2": g2,
                "winner": winner,
                "pk": pk
            })
            next_round.append(winner)
        teams = next_round
    return wc, bracket, teams[0]


single_tab, tournament_tab, group_tab, matchup_tab, best_matchups_tab, team_ratings_tab = st.tabs([
    "One Tournament",
    "Tournament Odds",
    "Group Stage Odds",
    "Matchup Score Projections",
    "Best Matchups",
    "Team Ratings"
])

with single_tab:
    st.header("Run One Tournament Simulation")
    if "one_tournament_result" not in st.session_state:
        st.session_state.one_tournament_result = None
    if st.button("Generate Tournament", type="primary"):
        with st.spinner("Simulating one World Cup..."):
            st.session_state.one_tournament_result = simulate_one_tournament_bracket()
    if st.session_state.one_tournament_result is None:
        st.info("Click the button to generate one full tournament simulation.")
    else:
        wc, bracket, champion = st.session_state.one_tournament_result
        st.subheader("Group Stage")
        group_cols = st.columns(3)
        for idx, group in enumerate(sorted(wc["group_results"].keys())):
            standings = wc["group_results"][group][[
                "rank",
                "team",
                "points",
                "goal_diff"
            ]].copy()
            standings = standings.rename(columns={
                "rank": "Rank",
                "team": "Team",
                "points": "Points",
                "goal_diff": "GD"
            })
            standings["Team"] = standings["Team"].apply(team_html)
            with group_cols[idx % 3]:
                st.markdown(f"**Group {group}**")
                render_html_table(standings)
        st.subheader("Advancing Third-Place Teams")
        thirds = wc["qualified_thirds"][[
            "group",
            "team",
            "points",
            "goal_diff",
            "goals_for"
        ]].copy()
        thirds = thirds.rename(columns={
            "group": "Group",
            "team": "Team",
            "points": "Points",
            "goal_diff": "GD",
            "goals_for": "GF"
        })
        thirds["Team"] = thirds["Team"].apply(team_html)
        render_html_table(thirds)
        st.subheader("Knockout Bracket")
        round_order = [
            "Round of 32",
            "Round of 16",
            "Quarterfinals",
            "Semifinals",
            "Final"
        ]
        render_bracket(bracket, round_order)
        st.success(f"🏆 Champion: {champion}")
        st.markdown(f"<h3>{team_html(champion, 32)}</h3>", unsafe_allow_html=True)

with tournament_tab:
    st.header("Results of 10,000 Tournament Simulations")
    try:
        tournament_results = load_tournament_odds()
        display = tournament_results.copy()
        display.insert(0, "Flag", display["Team"].map(flag_data_uri))
        stage_cols = ["R32", "R16", "QF", "SF", "Final", "Champion"]
        display = display[["Flag", "Team"] + [col for col in stage_cols if col in display.columns]]
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            height=720,
            row_height=34,
            column_config={
                "Flag": st.column_config.ImageColumn("", width="small"),
                "Team": st.column_config.TextColumn("Team", width="medium"),
                "R32": st.column_config.NumberColumn("R32", format="%.1f%%"),
                "R16": st.column_config.NumberColumn("R16", format="%.1f%%"),
                "QF": st.column_config.NumberColumn("QF", format="%.1f%%"),
                "SF": st.column_config.NumberColumn("SF", format="%.1f%%"),
                "Final": st.column_config.NumberColumn("Final", format="%.1f%%"),
                "Champion": st.column_config.NumberColumn("Champion", format="%.1f%%"),
            }
        )
    except FileNotFoundError:
        st.error("Tournament Sims.csv was not found. Put it in the same folder as app.py.")

with group_tab:
    st.header("Results of 10,000 Group Stage Simulations")
    group_name = st.selectbox(
        "Select group",
        sorted(simulation.world_cup_groups.keys()),
        format_func=lambda x: f"Group {x}"
    )
    try:
        group_odds = load_group_odds()
        finish_df = group_odds[group_odds["Group"] == group_name].copy()
        match_probs = cached_group_match_probabilities(group_name, N_SIMS)
        finish_display = finish_df.copy()
        finish_display["Team"] = finish_display["Team"].apply(team_html)
        st.subheader(f"Group {group_name} Finish Probabilities")
        render_html_table(
            format_percent_columns(finish_display, ["1st", "2nd", "3rd", "4th", "Advance"])
        )
        match_display = match_probs.copy()
        match_display["Team A"] = match_display["Team A"].apply(team_html)
        match_display["Team B"] = match_display["Team B"].apply(team_html)
        st.subheader(f"Group {group_name} Match Win/Draw/Loss Probabilities")
        render_html_table(
            format_percent_columns(match_display, ["Team A Win %", "Draw %", "Team B Win %"])
        )
    except FileNotFoundError as exc:
        st.error(f"Required CSV was not found: {exc.filename}. Put it in the same folder as app.py.")

with matchup_tab:
    st.header("10,000-Simulation Matchup Score Projections")
    all_teams = tournament_teams()
    col1, col2 = st.columns(2)
    with col1:
        home_default = all_teams.index("Mexico") if "Mexico" in all_teams else 0
        home_team = st.selectbox("Team 1", all_teams, index=home_default)
    with col2:
        away_default = all_teams.index("South Africa") if "South Africa" in all_teams else min(1, len(all_teams) - 1)
        away_team = st.selectbox("Team 2", all_teams, index=away_default)
    if home_team == away_team:
        st.warning("Choose two different teams.")
    else:
        with st.spinner("Loading cached matchup simulations..."):
            summary, score_matrix = cached_matchup_summary(home_team, away_team, N_SIMS)
            score_distribution = cached_match_score_distribution(home_team, away_team, N_SIMS)
        st.subheader("Overall Win/Draw/Loss Probabilities")
        summary_display = summary.copy()
        summary_display["Probability"] = summary_display["Probability"].map(lambda x: f"{x:.1f}%")
        render_html_table(summary_display)
        st.subheader("Score Probability Matrix")
        st.markdown(
            f"""
            <div class="score-matrix-card">
                <div class="score-matrix-top-label">{team_html(away_team)} Goals</div>
            """,
            unsafe_allow_html=True
        )
        label_col, matrix_col = st.columns([1.8, 10])
        with label_col:
            st.markdown(
                textwrap.dedent(f"""
                <div class="score-matrix-side-label">
                    <div class="matrix-side-label">
                        {team_html(home_team)}<br>Goals
                    </div>
                </div>
                """).strip(),
                unsafe_allow_html=True
            )
        with matrix_col:
            styled_matrix = (
                score_matrix
                .style
                .background_gradient(cmap="YlOrRd", axis=None)
                .format("{:.2f}")
            )
            st.dataframe(
                styled_matrix,
                use_container_width=True,
                height=360
            )
        st.markdown("</div>", unsafe_allow_html=True)


with best_matchups_tab:
    st.header("Best Group Stage Matchups")
    try:
        matchups = load_group_stage_matchups()
        metric_options = {
            "Overall": "Overall Rank",
            "Quality": "Quality Rank",
            "Offense": "Offense Rank",
            "Closeness": "Closeness Rank"
        }
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_metric = st.radio(
                "Rank matchups by",
                list(metric_options.keys()),
                horizontal=True,
                index=0
            )
        with col2:
            top_n = st.selectbox(
                "Show top",
                options=[10, 25, 50, 72],
                index=1
            )
        rank_col = metric_options[selected_metric]
        display = (
            matchups
            .sort_values([rank_col, "Date Parsed", "Group"])
            .head(top_n)
            .copy()
        )
        display = display.rename(columns={rank_col: "Rank"})
        display["Team A"] = display["Team A"].apply(team_html)
        display["Team B"] = display["Team B"].apply(team_html)
        columns = [
            "Rank",
            "Date",
            "Group",
            "Team A",
            "Team B",
            "Quality Rank",
            "Offense Rank",
            "Closeness Rank",
            "Overall Rank"
        ]
        # Avoid showing the selected rank twice under both "Rank" and its original name.
        selected_original = metric_options[selected_metric]
        if selected_original in display.columns and selected_original != "Rank":
            columns = [col for col in columns if col != selected_original]
        columns = [col for col in columns if col in display.columns]
        st.subheader(f"Top {top_n} Matchups by {selected_metric} Rank")
        render_html_table(display[columns], height=720)
    except FileNotFoundError as exc:
        st.error(f"Required CSV was not found: {exc.filename}. Put Group Stage Matchups.csv in the same folder as app.py.")
    except ValueError as exc:
        st.error(str(exc))

with team_ratings_tab:
    st.header("Team Ratings")
    st.caption("""
    These ratings are used to power the site's simulations. They are based off of results from matches played between FIFA Top 100 teams from 2023-Present, as well as from the 2010, 2014, 2018, and 2022 World Cups.
    
    A value iteration method was used to calculate teams' offensive and defensive efficiencies based on goals scored and allowed in a game, and the strength of the opponent.
    
    The offensive efficiency represents the number of goals a team would be expected to score against an "average" team in the dataset. The inverse is true for defensive efficiency.
    
    Games are then simulated by using the below ratings to fit a Diagonally-Inflated Bivariate Poisson (DIBP) distribution model, which forecasts the likelihood of every possible score for a game.
    
    The tournament simulations also depend on the format of the 2026 FIFA World Cup, ensuring that the Knockout Round matchups are generated according to the bracket published by FIFA.
    """)
    @st.cache_data
    def load_team_ratings():
        ratings_df = pd.read_csv("fifa_efficiencies_app.csv")
        ratings_df["team"] = ratings_df["team"].astype(str).str.strip()
        ratings_df = ratings_df.rename(
            columns={
                "team": "Team",
                "off_eff": "Offensive Rating",
                "def_eff": "Defensive Rating",
                "overall_rating": "Overall Rating"
            }
        )
        return ratings_df
    ratings_df = load_team_ratings()
    
    ratings_df = ratings_df[
        [
            "Group",
            "Team",
            "Overall Rating",
            "Offensive Rating",
            "Defensive Rating"
        ]
    ]
    ratings_df = ratings_df.sort_values(
        "Overall Rating",
        ascending=False
    )
    st.dataframe(
        ratings_df,
        use_container_width=True,
        hide_index=True
    )
