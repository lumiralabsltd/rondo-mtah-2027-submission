"""Independent check on DFL open data: the whole pipeline, scored against the tracking provider's
own ball status and possession flag. Nothing is tuned or trained here.

    .venv/bin/python -m rondo.check_dfl

Needs the DFL files in deadball/data/dfl/ (see deadball/get_dfl.py; the first load parses each XML
for a few minutes and caches it). These labels come from tracking, like the challenge's ground
truth, and from a different provider than the sample games everything was tuned on, so this is the
closest thing we have to a held-out test. Prints aggregates only (external data).

Runs per match: ball in stoppages as tracked or blanked x live-play ball dropout 20% / 40% x
train_possession's 4 tracking conditions (clean, ball jitter, player jitter and loss, both) = 16.
"""

import glob
import os
import sys
import warnings

import numpy as np

from rondo import possession
from rondo import train_possession as tp
from rondo.match import HOME, ROOT

sys.path.insert(0, os.path.join(ROOT, "deadball"))
from deadball import evaluate, features, loaders, rules  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402

warnings.filterwarnings("ignore")

DFL = os.path.join(ROOT, "deadball", "data", "dfl")


def truth_codes(dead, owner):
    """D where dead, else the provider's possession team (carried forward where it has none)."""
    known = np.where(owner >= 0, np.arange(len(owner)), -1)
    carry = np.maximum.accumulate(known)
    team = np.where(carry >= 0, owner[np.maximum(carry, 0)], HOME)
    codes = np.where(team == HOME, pm.H, pm.A).astype(np.int8)
    codes[dead] = pm.D
    return codes


def blank_dead(frames, dead):
    ball = frames.ball.copy()
    ball[dead] = np.nan
    return loaders.Frames(ball=ball, players=frames.players, team=frames.team, pitch=frames.pitch,
                          fps=frames.fps)


def main(predict_team=None):
    predict_team = predict_team or possession.predict_team
    metas = sorted(glob.glob(os.path.join(DFL, "*_meta.xml")))
    if not metas:
        print("no DFL files in {} (see deadball/get_dfl.py)".format(DFL))
        return 1
    rows = []
    for g, meta in enumerate(metas):
        name = os.path.basename(meta).split("_")[0]
        frames, dead, owner = loaders.load_dfl_full(meta, meta.replace("_meta", "_tracking"))
        truth = truth_codes(dead, owner)
        for s, (in_stoppages, rate) in enumerate([(v, r) for v in ("tracked", "blanked")
                                                  for r in evaluate.DROPOUT_RATES]):
            x = blank_dead(frames, dead) if in_stoppages == "blanked" else frames
            x = evaluate.add_ball_dropout(x, dead, rate=rate)
            alive = ~rules.detect(features.compute(x), x.fps)
            stop = evaluate.scores(dead, ~alive)["stoppage_f1"]
            clean = tp.match_from_frames(x)
            for k, noise in enumerate(sorted(tp.NOISE)):
                match = tp.add_noise(clean, *tp.NOISE[noise], seed=7000 + 1000 * g + 10 * s + k)
                full = tp.spell_f1(truth, alive, predict_team(match, alive))
                true_alive = ~dead
                perfect_dead = tp.spell_f1(truth, true_alive, predict_team(match, true_alive))
                rows.append(dict(match=name, stoppages=in_stoppages, noise=noise, stop=stop,
                                 full=full, perfect_dead=perfect_dead))
        print("  {}: {} frames, dead {:.0%}, possession flag on {:.1%} of live frames".format(
            name, frames.n, dead.mean(), (owner[~dead] >= 0).mean()), flush=True)

    def mean(key, **where):
        picked = [r[key] for r in rows if all(r[k] == v for k, v in where.items())]
        return np.mean(picked)

    print("\nspellF1 of the whole pipeline (and with a perfect dead ball), stoppage F1 of the detector")
    for key, values in (("match", sorted({r["match"] for r in rows})), ("stoppages", ("tracked", "blanked")),
                        ("noise", sorted(tp.NOISE))):
        for value in values:
            print("  {:<9} {:<8} spellF1 {:.3f}   with perfect dead ball {:.3f}   stoppage F1 {:.3f}".format(
                key, value, mean("full", **{key: value}), mean("perfect_dead", **{key: value}),
                mean("stop", **{key: value})))
    print("  mean               spellF1 {:.3f}   with perfect dead ball {:.3f}   stoppage F1 {:.3f}   "
          "worst run {:.3f}".format(mean("full"), mean("perfect_dead"), mean("stop"),
                                    min(r["full"] for r in rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
