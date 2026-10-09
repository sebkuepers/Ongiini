"""Divergence guard against the logged histories of real runs (scripts/train_guard.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_guard import DivergenceGuard, retry_argv  # noqa: E402


def run(history):
    g = DivergenceGuard()
    for step, loss, norm in history:
        if g.check(step, loss, norm):
            return step, g.reason
    return None, ""


def warning(history):
    g = DivergenceGuard()
    for step, loss, norm in history:
        g.check(step, loss, norm)
    return g.warning


# T4 (lr 2e-4, 2026-10-09), logged every 10 steps; values from its trainer_state
T4 = ([(s, 1.0, 2.5) for s in range(10, 450, 10)] + [(450, 1.36, 16.92)]
      + [(s, 0.9, 3.0) for s in range(460, 1100, 10)] + [(1020, 0.92, 24.35), (1100, 1.30, 308.84)]
      + [(s, 0.85, 2.6) for s in range(1110, 1140, 10)] + [(1140, 0.85, 112.85), (1150, 1.09, 14.55)]
      + [(1240, 7.16, 40.0)])


def test_t4_warns_on_spikes_and_trips_on_loss():
    assert "grad-norm spikes" in warning(sorted(T4))
    step, reason = run(sorted(T4))
    assert step == 1240 and reason.startswith("loss")


def test_c1_b10kr_like_spikes_only_warn():
    # C1-B10kR finished fine although its grad norm peaked at 883
    hist = [(s, 1.0, 3.8) for s in range(10, 2080, 10)]
    hist[23] = (240, 1.0, 883.0)
    hist[34] = (350, 1.0, 163.6)
    assert run(hist) == (None, "") and warning(hist)


def test_loss_rule_still_trips():
    step, reason = run([(s, 1.0, 2.0) for s in range(10, 100, 10)] + [(100, 11.7, 3.0)])
    assert step == 100 and reason.startswith("loss")


def test_t4b_like_run_does_not_trip():
    # T4b (lr 1e-4): norms 1-5 with single peaks up to 21
    hist = [(s, 1.0, 2.0 + (s % 70) / 20) for s in range(10, 4000, 10)]
    hist[10] = (110, 1.0, 12.3)
    hist[111] = (1120, 1.0, 21.0)
    assert run(hist) == (None, "") and not warning(hist)


def test_single_spike_does_not_trip():
    hist = [(s, 1.0, 2.0) for s in range(10, 2000, 10)]
    hist[100] = (1010, 1.0, 300.0)
    assert run(hist) == (None, "")


def test_spikes_far_apart_do_not_trip():
    hist = [(s, 1.0, 2.0) for s in range(10, 2000, 10)]
    hist[30] = (310, 1.0, 300.0)
    hist[150] = (1510, 1.0, 300.0)
    assert run(hist) == (None, "")


def test_no_trip_before_enough_history():
    assert run([(60, 1.0, 400.0), (70, 1.0, 400.0)]) == (None, "")


def test_retry_argv_halves_lr_and_counts_down():
    argv = ["--model", "m", "--lr", "2e-4", "--out", "o", "--divergence-retries", "1"]
    assert retry_argv(argv, 2e-4, 1) == ["--model", "m", "--out", "o", "--lr", "0.0001", "--divergence-retries", "0"]
    assert retry_argv(["--lr=1e-4"], 1e-4, 2)[-4:] == ["--lr", "5e-05", "--divergence-retries", "1"]
