"""Sanity-check the saved model with a weak, an average and a superteam lineup.

Run after train_model.py:  python test_model.py
"""
from pathlib import Path

import joblib
import pandas as pd

from lineup_features import FEATURES, lineup_features

APP_DIR = Path(__file__).parent
players = pd.read_csv(APP_DIR / "data" / "nba_2025_26_players.csv").set_index("player_name")
model = joblib.load(APP_DIR / "model" / "win_predictor.joblib")

LINEUPS = {
    "Role players": ["Collin Gillespie", "Cody Williams", "Jaylen Wells",
                     "Sam Hauser", "GG Jackson"],
    "Average starters": ["Donte DiVincenzo", "Christian Braun", "Andrew Wiggins",
                         "Keegan Murray", "Jarrett Allen"],
    "Superteam": ["Shai Gilgeous-Alexander", "Anthony Edwards", "Kawhi Leonard",
                  "Giannis Antetokounmpo", "Nikola Jokić"],
    "Best possible?": ["Luka Dončić", "Donovan Mitchell", "Kawhi Leonard",
                       "Giannis Antetokounmpo", "Victor Wembanyama"],
}

for name, roster in LINEUPS.items():
    lineup = players.loc[roster].reset_index()
    X = pd.DataFrame([lineup_features(lineup)])[FEATURES]
    wins = int(round(min(max(model.predict(X)[0], 0), 82)))
    print(f"{name:18s} -> {wins}-{82 - wins}")
