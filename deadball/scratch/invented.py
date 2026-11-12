import sys, warnings, json
import numpy as np
sys.path.insert(0, '.')
warnings.filterwarnings("ignore")
from run import sample_games
from deadball import evaluate, features, rules
from deadball.export import runs
for name, fr, dead in sample_games():
    f = features.compute(evaluate.add_ball_dropout(fr, dead))
    pred = rules.detect(f, fr.fps)
    clean_missing = np.isnan(fr.ball[:, 0])
    inv = [r for r in runs(pred) if not dead[r[0]:r[1]].any()]
    blank = np.array([clean_missing[a:b].mean() for a, b in inv])
    print(name, "invented", len(inv), "| clean ball blanked >90% of the run:", int((blank > 0.9).sum()),
          "| examples (start mm:ss, len s, blanked):", [("{}:{:02d}".format(a // 1500, a // 25 % 60), round((b - a) / 25, 1), round(x, 2)) for (a, b), x in list(zip(inv, blank))[:6]])
