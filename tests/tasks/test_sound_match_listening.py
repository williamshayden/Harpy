"""Listening examples follow the declared numerical selection without cherry picking."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "sound_matching_listening", Path(__file__).parents[2] / "research/sound_matching/listening.py"
)
listening = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(listening)


def test_lower_median_and_worst_are_stable_under_order_and_ties():
    rows = [
        {"episode_id": "e_c", "distance": 2.0},
        {"episode_id": "e_b", "distance": 1.0},
        {"episode_id": "e_a", "distance": 1.0},
        {"episode_id": "e_d", "distance": 4.0},
    ]
    median, worst = listening.select(rows)
    assert median["episode_id"] == "e_b"
    assert worst["episode_id"] == "e_d"
    assert listening.select(list(reversed(rows))) == (median, worst)
    assert [row["episode_id"] for row in rows] == ["e_c", "e_b", "e_a", "e_d"]
