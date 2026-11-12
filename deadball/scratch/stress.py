"""Stress grid: stoppage F1 by share of stoppage frames with the ball visible and its jitter."""
import sys, warnings
import numpy as np
sys.path.insert(0, '.')
warnings.filterwarnings("ignore")
from run import sample_games
from deadball import evaluate, features, rules

import json, os
games = list(sample_games())
by_visible = {}
print("visible  jitter_m  stopF1 per game          mean")
for visible in (0.0, 0.5, 1.0):
    for jitter in ((0.1, 0.3, 0.6) if visible else (0.1,)):
        r = []
        for name, fr, dead in games:
            x = evaluate.show_dead_ball(fr, dead, visible, jitter_m=jitter) if visible else fr
            x = evaluate.add_ball_dropout(x, dead)
            r.append(evaluate.scores(dead, rules.detect(features.compute(x), fr.fps))["stoppage_f1"])
        print("{:>6}  {:>8}  {}  {:.3f}".format(visible, jitter, " ".join("{:.3f}".format(v) for v in r), np.mean(r)))
        if jitter in (0.1, 0.3) and (visible == 0 or jitter == 0.3):
            by_visible[str(visible)] = round(float(np.mean(r)), 3)
os.makedirs("viewer/data", exist_ok=True)
json.dump({"by_visible": by_visible, "jitter_m": 0.3}, open("viewer/data/stress.json", "w"))
print("wrote viewer/data/stress.json", by_visible)
