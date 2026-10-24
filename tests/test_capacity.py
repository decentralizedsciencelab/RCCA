"""Capacity from measurements (rcca.capacity.service_times_measured): each policy's realized service time
and escalation rate come from the replay's decisions, not from nominal action costs."""
import json
import math

from rcca import capacity as C

DECISIONS = [
    {"policy": "rcca", "ceiling": "q60", "action": "esc", "wall_s": 2.0, "overhead_s": 0.5},
    {"policy": "rcca", "ceiling": "q60", "action": "acc", "wall_s": 6.0, "overhead_s": 0.5},
    {"policy": "rcca", "ceiling": "q90", "action": "acc", "wall_s": 600.0, "overhead_s": 0.0},     # another ceiling: ignored
    {"policy": "standard_ci", "ceiling": "q60", "action": "acc", "wall_s": 3.0, "overhead_s": 0.0},
    {"policy": "standard_ci", "ceiling": "q60", "action": "rej", "wall_s": 3.0, "overhead_s": 0.0},
]


def test_service_times_come_from_the_decisions_of_one_ceiling():
    st = C.service_times_measured(DECISIONS, ceiling="q60")
    assert st["rcca"]["n"] == 2 and st["rcca"]["wall_s"] == 4.5 and st["rcca"]["esc_rate"] == 0.5
    assert st["rcca"]["cs2"] == (((2.5 - 4.5) ** 2 + (6.5 - 4.5) ** 2) / 2) / 4.5 ** 2
    assert st["standard_ci"]["cs2"] == 0.0 and st["standard_ci"]["esc_rate"] == 0.0


def test_measured_sweep_waits_grow_with_load_and_price_the_reviewer_queue():
    st = C.service_times_measured(DECISIONS, ceiling="q60")
    rows = C.sweep_measured(st, [10.0, 100.0, 1000.0])
    rcca = [r for r in rows if r["policy"] == "rcca"]
    assert rcca[0]["wait_min"] < rcca[1]["wait_min"] and math.isinf(rcca[2]["wait_min"])   # 1000/h exceeds 800/h capacity
    assert rcca[0]["reviewer_hours_per_h"] == 10.0 * 0.5 * 12 / 60.0
    assert all(r["utilization"] == r["arrivals_per_h"] * r["mean_service_s"] / 3600.0 for r in rows)


def test_measured_main_writes_a_file_that_says_it_is_measured(tmp_path):
    (tmp_path / "decisions.jsonl").write_text("".join(json.dumps(d) + "\n" for d in DECISIONS))
    res = C.main_measured(tmp_path / "decisions.jsonl", tmp_path / "cap.json", ceiling="q60", lam_grid=[10.0, 50.0])
    on_disk = json.loads((tmp_path / "cap.json").read_text())
    assert "measured" in on_disk["note"] and set(on_disk["service_times"]) == {"rcca", "standard_ci"}
    assert len(on_disk["sweep"]) == 4 and res["ceiling"] == "q60"
