"""Turns a 5-man lineup into the numbers the AI model reads.

Shared by train_model.py and app.py so training and the app always use
exactly the same features.
"""
import pandas as pd

POSITIONS = ["PG", "SG", "SF", "PF", "C"]
STATS = ["ppg", "rpg", "apg", "spg", "bpg"]

FEATURES = [
    "total_ppg", "total_rpg", "total_apg", "total_spg", "total_bpg",
    "guard_apg",   # playmaking from the PG + SG
    "big_rpg",     # rebounding from the PF + C
    "big_bpg",     # rim protection from the PF + C
    "min_ppg",     # weakest scorer (is there a "hole" in the lineup?)
]


def lineup_features(lineup: pd.DataFrame) -> dict:
    """lineup: 5 rows (one per position) with the STATS columns."""
    by_pos = lineup.set_index("position")
    guards = by_pos.loc[["PG", "SG"]]
    bigs = by_pos.loc[["PF", "C"]]
    feats = {f"total_{s}": lineup[s].sum() for s in STATS}
    feats["guard_apg"] = guards["apg"].sum()
    feats["big_rpg"] = bigs["rpg"].sum()
    feats["big_bpg"] = bigs["bpg"].sum()
    feats["min_ppg"] = lineup["ppg"].min()
    return feats
