"""Grid search of the edge-snapping windows on the sample games (other games: allowed)."""
import itertools, sys, warnings
import numpy as np
sys.path.insert(0, '.')
warnings.filterwarnings("ignore")
from run import sample_games
from deadball import evaluate, features, rules

games = []
for seed in (0, 7):
    for name, fr, dead in sample_games():
        games.append((features.compute(evaluate.add_ball_dropout(fr, dead, seed=seed)), dead, fr.fps))
grid = dict(snap_start_back_s=[1, 2, 3, 5], snap_start_fwd_s=[0, 1, 2, 4],
            snap_end_back_s=[1, 3, 5], snap_end_fwd_s=[0, 1, 2, 4], min_seen_s=[0.2, 0.4, 1.0])
res = []
for combo in itertools.product(*grid.values()):
    p = dict(rules.PARAMS, **dict(zip(grid, combo)))
    r = [evaluate.scores(d, rules.detect(f, fps, p)) for f, d, fps in games]
    res.append((np.mean([x["stoppage_f1"] for x in r]), np.mean([x["dead_f1"] for x in r]), combo))
res.sort(key=lambda x: -x[0])
print(list(grid))
cur = tuple(rules.PARAMS[k] for k in grid)
print("current", [round(x, 3) for x in next((a, b) for a, b, c in res if c == cur)], cur)
for a, b, c in res[:10]: print(round(a, 3), round(b, 3), c)
