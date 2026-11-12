"""Where stoppage F1 is lost: per-IoU F1, whole misses, inventions, and IoU of matched pairs."""
import sys, warnings
import numpy as np
sys.path.insert(0, '.')
warnings.filterwarnings("ignore")
from run import sample_games
from deadball import evaluate, features, rules, loaders
from deadball.export import runs
loaders.mtah_repo()
from evaluation import possession_metrics as pm

for name, fr, dead in sample_games():
    f = features.compute(evaluate.add_ball_dropout(fr, dead))
    pred = rules.detect(f, fr.fps)
    codes = lambda d: np.where(d, pm.H, pm.D).astype(np.int8)
    s = pm.spell_scores(codes(dead), codes(pred), [(0, len(dead))])
    t, p = runs(dead), runs(pred)
    missed = [r for r in t if not pred[r[0]:r[1]].any()]
    invented = [r for r in p if not dead[r[0]:r[1]].any()]
    ious, short = [], 0
    for a, b in t:
        best = 0
        for c, d in p:
            if c < b and d > a:
                best = max(best, (min(b, d) - max(a, c)) / (max(b, d) - min(a, c)))
        if best: ious.append(best)
    ious = np.array(ious)
    split = sum(1 for a, b in t if sum(1 for c, d in p if c < b and d > a) > 1)
    merged = sum(1 for c, d in p if sum(1 for a, b in t if a < d and b > c) > 1)
    print(name, "F1@IoU", {k: round(v["f1"], 2) for k, v in s["per_iou"].items()})
    print("   missed whole {} (median {:.1f}s) | invented {} (median {:.1f}s) | split {} merged {} | matched IoU<0.7: {}/{}".format(
        len(missed), np.median([b - a for a, b in missed] or [0]) / 25, len(invented),
        np.median([b - a for a, b in invented] or [0]) / 25, split, merged, (ious < 0.7).sum(), len(ious)))
