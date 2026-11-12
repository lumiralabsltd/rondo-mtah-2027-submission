"""Re-tune Sukh's dead-ball settings for the score we are ranked on, on external data only.

    .venv/bin/python -m rondo.tune_dead_ball          # about 20 minutes on 10 cores

deadball/tune.py tunes the detector for its own stoppage F1. But what is ranked is spellF1 of the
whole file, and a dead-ball mistake costs spellF1 in its own way: a false stoppage inside a
possession splits it into two, a missed one merges two. So this scores every candidate setting by
running the whole pipeline (dead ball, then the v3 possession model as it stands) on Metrica's
open sample games and scoring the result with the challenge's spellF1, against possession from
their event data (rondo/train_possession.py). Nothing here touches the challenge match.

Runs: 3 games x deadball/tune.py's 10 stress scenarios (ball visible in stoppages 0 / 50 / 100%,
with and without balls past the line, x live-play dropout 20% / 40%) x tracking clean / noisy
(train_possession's "both") = 60 runs.

Rule, fixed before running: one setting at a time, in the order of SEARCH, each over its list of
values with the others held; the value with the highest mean spellF1 over the 60 runs wins; within
TIE of the best counts as a tie, and a tie keeps the current value. Two passes over the list. The
Smart Tagging settings are left alone, as the sample games have no tags.

Prints aggregates only, and the games are external, so the output is safe to share.
"""

import multiprocessing
import os
import sys
import time
import warnings

import numpy as np

from rondo import possession
from rondo import train_possession as tp
from rondo.match import ROOT

sys.path.insert(0, os.path.join(ROOT, "deadball"))
from deadball import evaluate, features, labels, rules  # noqa: E402
import run as deadball_run  # noqa: E402
import tune as deadball_tune  # noqa: E402

warnings.filterwarnings("ignore")

TIE = 0.002
PASSES = 2
NOISE = ("clean", "both")
SEARCH = [
    ("enter", [1.5, 2.0, 2.5, 3.0]),
    ("exit", [0.5, 1.0, 1.5]),
    ("min_dead_s", [2.0, 3.0, 4.0, 5.0]),
    ("min_live_s", [0.5, 1.0, 2.0, 3.0]),
    ("long_gap_s", [3.0, 4.0, 5.0, 7.0, 10.0]),
    ("w_long_gap", [0.5, 1.0, 1.5]),
    ("slow_speed", [1.2, 1.6, 2.0]),
    ("w_slow", [0.25, 0.5, 1.0]),
    ("w_walking", [0.5, 1.0, 1.5]),
    ("w_ball_still", [0.5, 1.0, 1.5]),
    ("ball_still_s", [1.0, 1.5, 2.5]),
    ("w_out", [1.0, 2.0, 3.0]),
    ("w_out_seen", [1.0, 2.0, 3.0]),
    ("w_out_latch", [0.5, 1.0, 1.5]),
    ("latch_max_s", [15.0, 30.0, 60.0]),
    ("snap_start_back_s", [1.0, 2.0, 3.0, 4.0]),
    ("snap_start_fwd_s", [1.0, 2.0, 3.0]),
    ("snap_end_back_s", [2.0, 3.0, 4.0]),
    ("snap_end_fwd_s", [0.5, 1.0, 2.0]),
]

RUNS = []    # (game, deadball features, fps, match, truth), built once, shared with forked workers
MODEL = None


def build():
    global MODEL
    MODEL = possession.load_model()
    for g, (name, frames, dead) in enumerate(deadball_run.sample_games()):
        truth = tp.sample_truth(name, dead)
        restarts = labels.restarts_from_events(deadball_run.restart_file(name))
        for s, (visible, beyond, rate) in enumerate(deadball_tune.SCENARIOS):
            x = frames
            if visible:
                x = evaluate.show_dead_ball(frames, dead, visible, jitter_m=0.3,
                                            restarts=restarts if beyond else None)
            x = evaluate.add_ball_dropout(x, dead, rate=rate)
            f = features.compute(x)
            clean = tp.match_from_frames(x)
            for k, noise in enumerate(NOISE):
                match = tp.add_noise(clean, *tp.NOISE[noise], seed=5000 + 1000 * g + 10 * s + k)
                RUNS.append((name, f, x.fps, match, truth))


def _job(args):
    params, i = args
    name, f, fps, match, truth = RUNS[i]
    alive = ~rules.detect(f, fps, params)
    return name, tp.spell_f1(truth, alive, possession.predict_team(match, alive, MODEL))


def evaluate_params(pool, params):
    results = pool.map(_job, [(params, i) for i in range(len(RUNS))])
    by_game = {}
    for name, score in results:
        by_game.setdefault(name, []).append(score)
    return float(np.mean([s for _, s in results])), {g: float(np.mean(v)) for g, v in by_game.items()}


def main():
    started = time.time()
    build()
    print("{} runs built in {:.0f} s".format(len(RUNS), time.time() - started), flush=True)
    params = dict(rules.PARAMS)
    with multiprocessing.get_context("fork").Pool() as pool:
        start_mean, start_games = evaluate_params(pool, params)
        print("current settings: mean spellF1 {:.4f}  {}".format(
            start_mean, "  ".join("{} {:.3f}".format(g, v) for g, v in sorted(start_games.items()))), flush=True)
        current = start_mean
        for p in range(PASSES):
            changed = False
            for key, values in SEARCH:
                scores = {}
                for value in values:
                    trial = dict(params, **{key: value})
                    if trial["exit"] >= trial["enter"]:
                        continue
                    scores[value] = current if value == params[key] else evaluate_params(pool, trial)[0]
                best = max(scores.values())
                pick = params[key] if scores.get(params[key], -1) >= best - TIE else \
                    max(scores, key=lambda v: scores[v])
                note = ""
                if pick != params[key]:
                    note = "  -> {} (was {})".format(pick, params[key])
                    params[key], current, changed = pick, scores[pick], True
                print("pass {} {:<18} {}{}".format(p + 1, key, "  ".join(
                    "{}: {:.4f}".format(v, s) for v, s in scores.items()), note), flush=True)
            if not changed:
                break
        final_mean, final_games = evaluate_params(pool, params)
    print("\nchanged (old -> new):", {k: (rules.PARAMS[k], v) for k, v in params.items()
                                     if rules.PARAMS[k] != v} or "nothing")
    print("mean spellF1 {:.4f} -> {:.4f}".format(start_mean, final_mean))
    for g in sorted(final_games):
        print("  {:<10} {:.3f} -> {:.3f}".format(g, start_games[g], final_games[g]))
    print("{:.0f} min".format((time.time() - started) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
