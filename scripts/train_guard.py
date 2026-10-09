"""Divergence guard and automatic retry for LoRA training (scripts/train_lora.py).

Two signals, checked at every logging step:
- loss: C2r (2026-10-06) jumped from 1.2 to 11.7 at step ~90 and stayed at ~7. Normal
  SFT runs log 0.6-1.5, so a loss above `limit` after step `after` means the run broke.
- gradient-norm spikes, an early WARNING only: T4 (2026-10-09, lr 2e-4) showed them from
  step ~480 and its loss exploded at 1240 — but C1-B10kR (CPT start adapter) had spikes up
  to 883 from step 240 and finished fine (replay over all runs, 2026-10-09), so spikes
  alone do not stop a run. A spike is a grad norm above `spike_min` AND `spike_factor` x
  the median of the recent normal ones; `spikes` of them within `window` steps raise the
  warning (printed once as UNSTABLE, kept in run.json).
On a trip (loss), train_lora.py restarts from scratch at half the learning rate
(`retry_argv`), keeping the diverged run next to it, as often as --divergence-retries allows.
No heavy imports here so the logic is testable without a GPU (scripts/test_train_guard.py).
"""
from __future__ import annotations

from collections import deque
from statistics import median


class DivergenceGuard:
    def __init__(self, limit: float = 4.0, after: int = 50, spike_factor: float = 20.0,
                 spike_min: float = 50.0, spikes: int = 2, window: int = 300, history: int = 50):
        self.limit, self.after = limit, after
        self.spike_factor, self.spike_min, self.spikes_needed, self.window = spike_factor, spike_min, spikes, window
        self.norms: deque[float] = deque(maxlen=history)
        self.spike_steps: list[int] = []
        self.tripped, self.reason = False, ""
        self.warning = ""  # set once when the spike rule fires

    def check(self, step: int, loss=None, grad_norm=None) -> bool:
        if self.tripped:
            return True
        if loss is not None and step >= self.after and float(loss) > self.limit:
            self.tripped, self.reason = True, f"loss {float(loss):.3f} at step {step}"
            return True
        if grad_norm is not None:
            g = float(grad_norm)
            ref = median(self.norms) if len(self.norms) >= 5 else None
            if step >= self.after and ref is not None and g > max(self.spike_min, self.spike_factor * ref):
                self.spike_steps = [s for s in self.spike_steps if step - s <= self.window] + [step]
                if len(self.spike_steps) >= self.spikes_needed and not self.warning:
                    self.warning = (f"grad-norm spikes at steps {self.spike_steps} (last {g:.1f}, "
                                    f"recent median {ref:.2f})")
            else:
                self.norms.append(g)
        return False


def retry_argv(argv: list[str], lr: float, retries: int) -> list[str]:
    """The same command line with --lr halved and --divergence-retries decremented."""
    out, skip = [], False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        if a in ("--lr", "--divergence-retries"):
            skip = True
            continue
        if a.startswith(("--lr=", "--divergence-retries=")):
            continue
        out.append(a)
    return out + ["--lr", f"{lr / 2:g}", "--divergence-retries", str(retries - 1)]
