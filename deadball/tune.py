"""Re-tune the detector on Metrica's sample games only, over stress scenarios that owe nothing to
the challenge match (organisers' rules clarification, 2026-11-06: every tuned value must come from
external data).

    python tune.py            # all three stages, then the final scores (about 15 minutes)

Scenarios, each on all three sample games:
    ball visible in stoppages 0% / 50% / 100%, the visible ones with and without balls lying past
    the line (5), times live-play ball dropout 20% / 40% (DROPOUT_RATES)  ->  10 scenarios

Rule, fixed before running: the winner is the highest mean stoppage F1 over the 10 scenarios.
Anything within TIE of the best counts as a tie, and a tie keeps the current value (fewest
changes). Stages run in the order they were first tuned: base rules, stoppage evidence, latch.
"""

import itertools
import multiprocessing
import sys
import time
import warnings

import numpy as np

from deadball import evaluate, features, labels, rules
from run import restart_file, sample_games

warnings.filterwarnings("ignore")

TIE = 0.002
SCENARIOS = [(v, beyond, rate) for v, beyond in [(0.0, False), (0.5, False), (1.0, False),
                                                  (0.5, True), (1.0, True)]
             for rate in evaluate.DROPOUT_RATES]
STAGES = [
    ("base rules (was scratch/sweep.py)",
     dict(enter=[1.5, 2.0], exit=[0.5, 1.0], min_dead_s=[2, 3, 4], min_live_s=[1, 2, 3],
          long_gap_s=[3, 4, 5], slow_speed=[1.6, 2.0], w_slow=[0.5, 1.0])),
    ("stoppage evidence (was scratch/sweep_robust.py)",
     dict(enter=[1.5, 2.0], w_ball_still=[1.0, 1.5], ball_still_s=[1.0, 1.5, 2.5],
          w_out_seen=[1.0, 2.0], w_slow=[0.5, 1.0], w_walking=[0.5, 1.0])),
    ("out-of-play latch (was scratch/sweep_latch.py)",
     dict(set_speed=[2.0, 2.5, 3.5], set_s=[0.6, 1.0, 1.5], strike_speed=[4.0, 5.0, 6.5],
          w_out_latch=[0.5, 1.0], latch_max_s=[30.0, 60.0, 90.0])),
]

GAMES = []  # (scenario, features, dead, fps), built once and shared with forked workers


def build():
    for name, frames, dead in sample_games():
        restarts = labels.restarts_from_events(restart_file(name))
        for visible, beyond, rate in SCENARIOS:
            x = frames
            if visible:
                x = evaluate.show_dead_ball(frames, dead, visible, jitter_m=0.3,
                                            restarts=restarts if beyond else None)
            x = evaluate.add_ball_dropout(x, dead, rate=rate)
            GAMES.append(((visible, beyond, rate), features.compute(x), dead, frames.fps))


def evaluate_params(p):
    by = {}
    for scenario, f, dead, fps in GAMES:
        by.setdefault(scenario, []).append(evaluate.scores(dead, rules.detect(f, fps, p))["stoppage_f1"])
    means = [np.mean(by[s]) for s in SCENARIOS]
    return float(np.mean(means)), float(min(means))


def _job(p):
    return evaluate_params(p)


def run_stage(pool, title, grid, params):
    keys = list(grid)
    combos = [c for c in itertools.product(*grid.values())
              if dict(params, **dict(zip(keys, c)))["exit"] < dict(params, **dict(zip(keys, c)))["enter"]]
    started = time.time()
    results = pool.map(_job, [dict(params, **dict(zip(keys, c))) for c in combos])
    current = tuple(params[k] for k in keys)
    best = max(r[0] for r in results)
    ties = [(c, r) for c, r in zip(combos, results) if r[0] >= best - TIE]
    pick, score = min(ties, key=lambda t: sum(a != b for a, b in zip(t[0], current)))
    print("\n== {}  ({} combinations, {:.0f} s)".format(title, len(combos), time.time() - started))
    print("   current {}  mean {:.3f}  worst {:.3f}".format(
        dict(zip(keys, current)), *results[combos.index(current)]))
    print("   best mean {:.3f}; picked {}  mean {:.3f}  worst {:.3f}".format(
        best, dict(zip(keys, pick)), *score))
    return dict(params, **dict(zip(keys, pick)))


def main():
    started = time.time()
    build()
    print("{} scenario runs built in {:.0f} s".format(len(GAMES), time.time() - started))
    params = dict(rules.PARAMS)
    with multiprocessing.get_context("fork").Pool() as pool:
        for title, grid in STAGES:
            params = run_stage(pool, title, grid, params)
    changed = {k: (rules.PARAMS[k], v) for k, v in params.items() if rules.PARAMS[k] != v}
    print("\nchanged (old -> new):", changed or "nothing")
    print("final mean {:.3f}  worst {:.3f}".format(*evaluate_params(params)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
