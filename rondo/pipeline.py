"""ROLE 3 — pipeline: combine dead ball + possession into Rondo.csv, validate it, score it.

    .venv/bin/python -m rondo.pipeline                    # build, validate, score
    .venv/bin/python -m rondo.pipeline --check-baseline   # also compare to the official baseline

Writes evaluation/results/Rondo.csv (gitignored — a submission must never be committed), runs the
official validator on it, then the official scorer against the public 5 minutes.

Remember the public 5 minutes are 17 possessions: use them to check the pipeline is correct, not
to pick between two versions (EVALUATION.md, "The public window is not a leaderboard").

Everything printed here is aggregate — counts and scores — so it is safe to paste to anyone.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

from rondo.dead_ball import predict_alive
from rondo.match import ROOT, load_match
from rondo.possession import predict_team

TEAM_NAME = "Rondo"
OUT_PATH = os.path.join(ROOT, "evaluation", "results", TEAM_NAME + ".csv")
BASELINE_PATH = os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")


def combine(match, alive, team):
    """One label per frame: D where dead, the possession answer where alive, X off the graded halves."""
    _check_shape("predict_alive", alive, match.n_frames)
    _check_shape("predict_team", team, match.n_frames)
    labels = np.where(alive, team, "D").astype("<U1")
    labels[~match.graded] = "X"
    return labels


def _check_shape(name, array, n_frames):
    if len(array) != n_frames:
        raise ValueError("{} returned {} values, expected {}".format(name, len(array), n_frames))


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


def summarise(match, labels):
    """Label counts and possession count on graded frames. Aggregates only."""
    graded = labels[match.graded]
    counts = {label: int(np.sum(graded == label)) for label in "HADX"}
    spells = 0
    for half in (1, 2):
        current = None
        for label in labels[match.half == half]:
            if label == "X":
                continue  # the scorer carries the previous answer through an X
            if label != current and label in "HA":
                spells += 1
            current = label
    print("graded frames  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("possessions    {}  (on graded frames, counted the way the scorer counts them)"
          .format(spells))


def compare_to_baseline(match, labels):
    """How many graded frames agree with the committed baseline_smooth50.csv (v0 matched 100%)."""
    with open(BASELINE_PATH) as handle:
        next(handle)
        baseline = np.array([line.rstrip().split(",")[1] for line in handle])
    same = int(np.sum(baseline[match.graded] == labels[match.graded]))
    total = int(np.sum(match.graded))
    print("matches baseline_smooth50 on {} of {} graded frames ({:.3f}%)".format(
        same, total, 100.0 * same / total))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", help="where the match data is (default: ./data)")
    parser.add_argument("--out", default=OUT_PATH, help="where to write the CSV")
    parser.add_argument("--check-baseline", action="store_true",
                        help="how many graded frames agree with the official baseline")
    parser.add_argument("--no-score", action="store_true", help="skip validator and scorer")
    args = parser.parse_args()

    match = load_match(data_dir=args.data_dir)
    alive = predict_alive(match)
    labels = combine(match, alive, predict_team(match, alive))
    path = write_submission(args.out, labels)
    print("wrote {}".format(os.path.relpath(path, ROOT)))
    summarise(match, labels)
    if args.check_baseline:
        compare_to_baseline(match, labels)

    if not args.no_score:
        print()
        subprocess.check_call([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), path])
        print()
        subprocess.check_call([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py")])


if __name__ == "__main__":
    main()
