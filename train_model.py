"""Train the 82-0 win predictor.

Steps:
1. Load the 2025-26 player CSV (official NBA.com per-game stats).
2. Simulate thousands of random 5-man lineups (one PG, SG, SF, PF, C).
3. Give each lineup a "true" win total (0-82) using a basketball formula:
   all 5 box-score stats + positional balance + some random luck.
4. Train a scikit-learn GradientBoostingRegressor to predict wins from
   the lineup's combined stats, then save it to model/win_predictor.joblib.

Run:  python train_model.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from lineup_features import FEATURES, POSITIONS, STATS, lineup_features

APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "data" / "nba_2025_26_players.csv"
MODEL_PATH = APP_DIR / "model" / "win_predictor.joblib"
N_LINEUPS = 20_000
rng = np.random.default_rng(42)


def player_impact(df: pd.DataFrame) -> pd.Series:
    """One number per player. Defensive stats are rarer, so they count more."""
    return df.ppg + 1.2 * df.rpg + 1.5 * df.apg + 3.0 * df.spg + 3.0 * df.bpg


def true_wins(feats: pd.DataFrame, impact_sum: pd.Series) -> pd.Series:
    """The 'answer key' the model learns from (with random luck added)."""
    score = impact_sum.copy()
    # Positional balance: reward a real playmaker, rebounding, and rim protection.
    score += 2.0 * np.minimum(feats.guard_apg - 8, 6)
    score += 1.5 * np.minimum(feats.big_rpg - 14, 8)
    score += 3.0 * np.minimum(feats.big_bpg - 1.6, 2)
    # A lineup with a non-scorer is easier to defend.
    score += 1.0 * np.minimum(feats.min_ppg - 10, 0)
    luck = rng.normal(0, 0.3, len(score))  # same lineup won't win the exact same games every season
    # Squash the score into 0-82 wins with an S-curve. Stacked superteams
    # (score above 260) get an extra push so 82-0 is possible but rare.
    logit = (score - 119) / 53 + 0.045 * np.maximum(score - 260, 0) + luck
    return 82 / (1 + np.exp(-logit))


def simulate_lineups(players: pd.DataFrame) -> pd.DataFrame:
    """Draw all lineups at once (one column of picks per position)."""
    players = players.assign(impact=player_impact(players)).reset_index(drop=True)
    # Mix of fully random lineups and star-heavy lineups, so the model
    # sees both bad teams and superteams during training.
    star_bias = rng.choice([0, 2, 5], N_LINEUPS)
    picks = {}
    for pos in POSITIONS:
        pool = players[players.position == pos]
        idx = np.empty(N_LINEUPS, dtype=int)
        for b in (0, 2, 5):
            w = pool.impact.to_numpy() ** b
            mask = star_bias == b
            idx[mask] = rng.choice(pool.index.to_numpy(), mask.sum(), p=w / w.sum())
        picks[pos] = players.loc[idx].reset_index(drop=True)

    data = pd.DataFrame({f"total_{s}": sum(picks[p][s] for p in POSITIONS) for s in STATS})
    data["guard_apg"] = picks["PG"].apg + picks["SG"].apg
    data["big_rpg"] = picks["PF"].rpg + picks["C"].rpg
    data["big_bpg"] = picks["PF"].bpg + picks["C"].bpg
    data["min_ppg"] = np.minimum.reduce([picks[p].ppg for p in POSITIONS])
    data["wins"] = true_wins(data, sum(picks[p].impact for p in POSITIONS))

    # Self-check: these fast features must match what the app computes.
    first = pd.DataFrame([picks[p].iloc[0] for p in POSITIONS])
    assert np.allclose(pd.Series(lineup_features(first))[FEATURES], data.loc[0, FEATURES])
    return data


def main():
    players = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(players)} players from {players.team.nunique()} teams")

    data = simulate_lineups(players)
    print(f"Simulated {len(data)} lineups. Wins: min {data.wins.min():.0f}, "
          f"median {data.wins.median():.0f}, max {data.wins.max():.0f}")

    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data.wins, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(
        n_estimators=400, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(f"Test MAE: {mean_absolute_error(y_test, pred):.1f} wins | "
          f"R^2: {r2_score(y_test, pred):.3f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
