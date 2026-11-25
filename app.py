"""82-0 NBA Challenge (2025-26 Season Edition) - Streamlit app.

Run:  streamlit run app.py
"""
import random
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from lineup_features import FEATURES, POSITIONS, STATS, lineup_features

SEASON = "2025-26"
TOTAL_SKIPS = 2
POS_NAMES = {"PG": "Point Guard", "SG": "Shooting Guard", "SF": "Small Forward",
             "PF": "Power Forward", "C": "Center"}

st.set_page_config(page_title="82-0 NBA Challenge", page_icon="🏀", layout="wide")


APP_DIR = Path(__file__).parent  # works no matter which folder the app is started from


@st.cache_data
def load_players():
    return pd.read_csv(APP_DIR / "data" / "nba_2025_26_players.csv")


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / "model" / "win_predictor.joblib")


players = load_players()
model = load_model()
ALL_TEAMS = sorted(players.team.unique())


# ---------- Game state ----------
def new_game():
    st.session_state.lineup = {pos: None for pos in POSITIONS}
    st.session_state.used_teams = []
    st.session_state.current_team = None
    st.session_state.skips = TOTAL_SKIPS


def spin():
    options = [t for t in ALL_TEAMS if t not in st.session_state.used_teams]
    team = random.choice(options)
    st.session_state.current_team = team
    st.session_state.used_teams.append(team)  # never spin the same team twice


def skip_team():
    st.session_state.skips -= 1
    spin()


def pick(player_name, pos):
    row = players[players.player_name == player_name].iloc[0]
    st.session_state.lineup[pos] = row.to_dict()
    st.session_state.current_team = None


if "lineup" not in st.session_state:
    new_game()

lineup = st.session_state.lineup
open_slots = [pos for pos in POSITIONS if lineup[pos] is None]
round_no = len(POSITIONS) - len(open_slots) + 1


# ---------- Sidebar ----------
with st.sidebar:
    st.header("How to play")
    st.markdown(
        f"1. **Spin** to land on a random {SEASON} NBA team.\n"
        "2. **Pick one player** from that team for an open position.\n"
        "3. Don't like the team? Use one of your **2 skips**.\n"
        "4. Fill all 5 spots, then the AI predicts your 82-game record.\n\n"
        "Player stats stay hidden until your lineup is complete. Go with your gut!"
    )
    st.button("🔄 Restart game", on_click=new_game, width="stretch")
    st.divider()
    st.subheader("⚠️ Responsible AI")
    st.caption(
        "The predicted record comes from a machine-learning model trained on "
        "simulated lineups. **AI predictions can be wrong.** The model only sees "
        f"5 box-score stats (PPG, RPG, APG, SPG, BPG) from the {SEASON} season. It "
        "knows nothing about chemistry, coaching, injuries or off-ball defense. "
        "It's for fun and learning, not a real forecast."
    )


# ---------- Header + roster board ----------
st.title("🏀 82-0 NBA Challenge")
st.caption(f"{SEASON} Season Edition · Build a starting five. Can your team go undefeated?")

cols = st.columns(5)
for col, pos in zip(cols, POSITIONS):
    with col.container(border=True):
        st.markdown(f"**{pos}** · {POS_NAMES[pos]}")
        p = lineup[pos]
        if p:
            st.markdown(f"### {p['player_name']}")
            st.caption(p["team"])
        else:
            st.markdown("### ―")
            st.caption("Open slot")


# ---------- Draft phase ----------
if open_slots:
    st.subheader(f"Round {round_no} of 5")
    st.write(f"Team skips left: **{'🟢 ' * st.session_state.skips or 'none'}**")

    team = st.session_state.current_team
    if team is None:
        st.button("🎰 Spin Team", on_click=spin, type="primary")
    else:
        st.markdown(f"## You landed on: {team}")
        st.write("Pick one player for an open slot (stats are hidden!):")
        roster = players[players.team == team].sort_values(
            "position", key=lambda s: s.map(POSITIONS.index))
        grid = st.columns(3)
        for i, row in enumerate(roster.itertuples()):
            is_open = row.position in open_slots
            grid[i % 3].button(
                f"{row.player_name} · {row.position}",
                key=f"pick_{row.player_name}",
                on_click=pick, args=(row.player_name, row.position),
                disabled=not is_open,
                help=None if is_open else f"Your {row.position} slot is already filled",
                width="stretch",
            )
        st.button(f"⏭️ Skip this team ({st.session_state.skips} left)",
                  on_click=skip_team, disabled=st.session_state.skips == 0)

# ---------- Results ----------
else:
    team_df = pd.DataFrame([lineup[pos] for pos in POSITIONS])
    X = pd.DataFrame([lineup_features(team_df)])[FEATURES]
    wins = int(round(min(max(model.predict(X)[0], 0), 82)))
    losses = 82 - wins

    if wins == 82:
        grade, msg = "S", "PERFECT SEASON! The elusive 82-0!"
    elif wins >= 72:
        grade, msg = "A+", "Historic juggernaut. Title favorite."
    elif wins >= 62:
        grade, msg = "A", "Elite contender."
    elif wins >= 50:
        grade, msg = "B", "Solid playoff team."
    elif wins >= 38:
        grade, msg = "C", "Fighting for a play-in spot."
    else:
        grade, msg = "D", "Lottery bound. Better luck next spin!"

    st.divider()
    with st.container(border=True):
        a, b = st.columns([2, 1])
        a.markdown(f"# Predicted record: {wins}-{losses}")
        a.markdown(f"### {msg}")
        b.metric("Grade", grade)
        b.metric("Win %", f"{wins / 82:.1%}")
    if wins == 82:
        st.balloons()

    st.subheader("Your starting five, revealed")
    table = team_df[["position", "player_name", "team"] + STATS].copy()
    table.columns = ["Pos", "Player", "Team", "PPG", "RPG", "APG", "SPG", "BPG"]
    st.dataframe(table, hide_index=True, width="stretch")

    st.subheader("Team stat totals (stacked by player)")
    chart = team_df.set_index("player_name")[STATS].T
    chart.index = [s.upper() for s in STATS]
    st.bar_chart(chart)

    st.button("🔁 Play Again", on_click=new_game, type="primary")

st.divider()
st.caption("⚠️ AI predictions can be wrong. This is a school AI Literacy project, "
           f"not an official NBA forecast. Stats: NBA.com, {SEASON} regular season.")
