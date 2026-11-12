"""Simulated Smart Tagging: no tags vs end-anchoring only vs anchoring + inserting stoppages."""
import sys, warnings
import numpy as np
sys.path.insert(0, '.')
warnings.filterwarnings("ignore")
from run import sample_games, restart_file
from deadball import evaluate, features, labels, rules

SCENARIOS = [(0.0, False), (0.5, False), (1.0, False), (0.5, True), (1.0, True)]
rows = {k: {} for k in ("none", "anchor", "anchor+insert")}
for name, fr, dead in sample_games():
    events = labels.restarts_from_events(restart_file(name))
    tags = evaluate.simulate_tags(events, fr.fps)
    for visible, beyond in SCENARIOS:
        x = evaluate.show_dead_ball(fr, dead, visible, jitter_m=0.3,
                                    restarts=events if beyond else None) if visible else fr
        f = features.compute(evaluate.add_ball_dropout(x, dead))
        for key, t, ins in (("none", None, True), ("anchor", tags, False), ("anchor+insert", tags, True)):
            p = dict(rules.PARAMS, tag_insert=ins)
            rows[key].setdefault((visible, beyond), []).append(
                evaluate.scores(dead, rules.detect(f, fr.fps, p, tags=t))["stoppage_f1"])
for key, by in rows.items():
    means = [np.mean(by[sc]) for sc in SCENARIOS]
    print("{:<14} mean {:.3f} worst {:.3f} | {}".format(key, np.mean(means), min(means),
                                                         " ".join("{:.3f}".format(m) for m in means)))
