"""Start/end offsets (predicted - true, seconds) of stoppages that overlap one-to-one."""
import sys, warnings
import numpy as np
sys.path.insert(0, '.')
warnings.filterwarnings("ignore")
from run import sample_games
from deadball import evaluate, features, rules
from deadball.export import runs

def offsets(true, pred, fps):
    out = []
    for a, b in runs(true):
        hits = [(c, d) for c, d in runs(pred) if c < b and d > a]
        if len(hits) == 1:
            c, d = hits[0]
            out.append(((c - a) / fps, (d - b) / fps))
    return np.array(out)

for name, fr, dead in sample_games():
    dr = evaluate.add_ball_dropout(fr, dead)
    f = features.compute(dr)
    pred = rules.detect(f, fr.fps)
    o = offsets(dead, pred, fr.fps)
    print(name, "matched", len(o), "| start median {:+.1f}s p10 {:+.1f} p90 {:+.1f} | end median {:+.1f}s p10 {:+.1f} p90 {:+.1f}".format(
        *np.median(o[:, :1], 0), *np.percentile(o[:, 0], [10, 90]), np.median(o[:, 1]), *np.percentile(o[:, 1], [10, 90])))
    # what does the clean ball do at true boundaries?
    miss = np.isnan(fr.ball[:, 0])
    seen_before = [ (a - np.flatnonzero(~miss[:a])[-1]) / fr.fps for a, b in runs(dead) if a > 0 and (~miss[:a]).any()]
    seen_after = [ (np.flatnonzero(~miss[b:])[0]) / fr.fps for a, b in runs(dead) if b < fr.n and (~miss[b:]).any()]
    print("   clean ball: last seen before true start median {:.2f}s, first seen after true end median {:.2f}s".format(np.median(seen_before), np.median(seen_after)))
