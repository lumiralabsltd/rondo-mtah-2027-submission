"""Train role 2's model (rondo/possession_model.json) on external data, and say how good it is.

    .venv/bin/python -m rondo.train_possession            # leave-one-game-out report, then train
    .venv/bin/python -m rondo.train_possession --report   # report only, keep the current model

The organisers' rules clarification (2026-11-06) says anything trained or tuned must use external
data, never our match's ground truth or video. So this uses Metrica's open sample games only,
where the event data says who has the ball, and scores with the challenge's own spellF1 code.
Needs deadball/data/metrica/ (`.venv/bin/python deadball/get_metrica.py`). Prints aggregates only,
and the games are external, so the output is safe to share.

Who has the ball on a sample game: the team of the latest on-ball event (pass, carry, shot, set
piece, recovery, ball lost), on every frame deadball's event labels call live. Challenges are
skipped because both teams are in them.

Our match's tracking is automatic, so each sample game is made worse first, in every combination:
  - dead ball: live-play ball dropout 20% / 40%, ball blanked or fully visible in stoppages,
    then Sukh's detector decides what is alive, exactly as on our match
  - tracking: clean / ball jitter (0.5 m) / player jitter (0.2 m) with 15% of player positions
    lost in bursts / both
That is 3 games x 4 x 4 = 48 runs.

What is learned, all from those runs: the scene table and the logistic weights (counted and
fitted, nothing to choose), and SWITCH_COST. Rule for the switch cost, fixed before running:
leave one game out (fit on two, score the third, for each game in turn); the cost with the highest
mean spellF1 over the held-out runs wins; within TIE of the best counts as a tie, and a tie goes to
the higher cost (fewer changes). The final model is then fitted on all three games.
"""

import argparse
import csv
import json
import os
import sys
import warnings

import numpy as np

from rondo import possession
from rondo.match import AWAY, HOME, ROOT, Match

sys.path.insert(0, os.path.join(ROOT, "deadball"))
from deadball import evaluate, features, rules  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402
import run as deadball_run  # noqa: E402

warnings.filterwarnings("ignore")

TIE = 0.002
COSTS = (1, 2, 3, 5, 7, 10, 15, 20)
STRIDE = 5          # train on every 5th frame (neighbouring frames say the same thing)
L2 = 1.0
ON_BALL = {"PASS", "CARRY", "SHOT", "SET PIECE", "RECOVERY", "BALL LOST"}
DEAD_BALL = [(rate, visible) for rate in evaluate.DROPOUT_RATES for visible in (0.0, 1.0)]
NOISE = dict(clean=(0.0, 0.0, 0.0), ball=(0.5, 0.0, 0.0), players=(0.0, 0.2, 0.15),
             both=(0.5, 0.2, 0.15))   # ball jitter m, player jitter m, share of player positions lost


# -- the sample games ---------------------------------------------------------------------------

def on_ball_events(path):
    """[(start frame index, HOME/AWAY)] for the on-ball events of a Metrica event file."""
    out = []
    if path.endswith(".json"):
        with open(path) as handle:
            for e in json.load(handle)["data"]:
                if e["type"]["name"] in ON_BALL:
                    out.append((e["start"]["frame"] - 1, HOME if e["team"]["name"] == "Team A" else AWAY))
    else:
        with open(path) as handle:
            for row in csv.DictReader(handle):
                if row["Type"] in ON_BALL:
                    out.append((int(row["Start Frame"]) - 1, HOME if row["Team"] == "Home" else AWAY))
    return sorted(out)


def truth_codes(events, dead):
    """H/A/D per frame: D where dead, otherwise the team of the latest on-ball event."""
    codes = np.full(len(dead), pm.D, dtype=np.int8)
    starts = np.array([frame for frame, _ in events])
    teams = np.array([team for _, team in events])
    latest = np.searchsorted(starts, np.arange(len(dead)), side="right") - 1
    known = (latest >= 0) & ~dead
    codes[known] = np.where(teams[latest[known]] == HOME, pm.H, pm.A)
    return codes


def match_from_frames(frames):
    """deadball Frames (centred metres) as our Match (0-1 fractions, origin top-left)."""
    length, width = frames.pitch

    def fractions(xy):
        out = np.full(xy.shape[:-1] + (3,), np.nan, dtype=np.float32)
        out[..., 0] = xy[..., 0] / length + 0.5
        out[..., 1] = 0.5 - xy[..., 1] / width
        return out

    return Match(ball=fractions(frames.ball), players=fractions(frames.players),
                 team_of_slot=frames.team.astype(np.int8), half=np.ones(frames.n, dtype=np.int8),
                 tags=[], pitch=frames.pitch, fps=frames.fps)


def add_noise(match, ball_sd, player_sd, player_lost, seed):
    """Jitter the ball and players, and lose a share of player positions in bursts (~2 s)."""
    if not (ball_sd or player_sd or player_lost):
        return match
    rng = np.random.RandomState(seed)
    length, width = match.pitch
    scale = np.array([length, width], dtype=np.float32)
    ball = match.ball.copy()
    ball[:, :2] += (rng.normal(0, ball_sd, (match.n_frames, 2)) / scale).astype(np.float32)
    players = match.players.copy()
    players[..., :2] += (rng.normal(0, player_sd, players.shape[:2] + (2,)) / scale).astype(np.float32)
    if player_lost:
        burst = 2 * match.fps
        for slot in range(players.shape[1]):
            frame = 0
            while frame < match.n_frames:
                frame += int(rng.exponential(burst * (1 - player_lost) / player_lost))
                gap = int(rng.exponential(burst))
                players[frame:frame + gap, slot, :] = np.nan
                frame += gap
    return Match(ball=ball, players=players, team_of_slot=match.team_of_slot, half=match.half,
                 tags=[], pitch=match.pitch, fps=match.fps)


def build_runs():
    """One dict per game x dead-ball scenario x tracking noise, holding only what training needs."""
    runs = []
    for g, (name, frames, dead) in enumerate(deadball_run.sample_games()):
        truth = truth_codes(on_ball_events(deadball_run.restart_file(name)), dead)
        for s, (rate, visible) in enumerate(DEAD_BALL):
            x = evaluate.show_dead_ball(frames, dead, visible, jitter_m=0.3) if visible else frames
            x = evaluate.add_ball_dropout(x, dead, rate=rate)
            alive = ~rules.detect(features.compute(x), x.fps)
            clean = match_from_frames(x)
            for k, (noise, levels) in enumerate(sorted(NOISE.items())):
                match = add_noise(clean, *levels, seed=1000 * g + 10 * s + k)
                runs.append(dict(game=name, noise=noise, alive=alive, truth=truth,
                                 geo=possession.geometry(match, alive)))
        print("  {}: {} runs".format(name, len(DEAD_BALL) * len(NOISE)), flush=True)
    return runs


# -- fitting --------------------------------------------------------------------------------------

def fit_scenes(runs):
    """Log-odds that the nearest player's team has the ball, for each scene (+1 smoothing)."""
    same = np.zeros(possession.N_SCENES)
    total = np.zeros(possession.N_SCENES)
    for run in runs:
        truth, geo = run["truth"], run["geo"]
        labelled = (truth == pm.H) | (truth == pm.A)
        team = np.where(truth == pm.H, HOME, AWAY)
        np.add.at(total, geo["scene"][labelled], 1)
        np.add.at(same, geo["scene"][labelled], (team == geo["team"])[labelled])
    llr = np.log((same + 1) / (total - same + 1))
    llr[0] = 0.0
    return llr


def fit_weights(runs, scene_llr):
    """Logistic regression, no intercept, L2, by Newton's method (deterministic)."""
    xs, ys = [], []
    for run in runs:
        truth, alive = run["truth"], run["alive"]
        rows = np.flatnonzero(alive & ((truth == pm.H) | (truth == pm.A)))[::STRIDE]
        xs.append(possession.features(run["geo"], alive, scene_llr)[rows])
        ys.append((truth[rows] == pm.H).astype(np.float64))
    x, y = np.concatenate(xs), np.concatenate(ys)
    w = np.zeros(x.shape[1])
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-(x @ w)))
        gradient = x.T @ (p - y) + L2 * w
        hessian = (x * (p * (1 - p))[:, None]).T @ x + L2 * np.eye(len(w))
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


def spell_f1(truth, alive, team):
    codes = pm.label_codes(np.where(alive, team, "D"))
    runs = pm.valid_runs(truth)
    return pm.spell_scores(truth, pm.fill_abstentions(codes, runs), runs)["mean_f1"]


def score_runs(runs, scene_llr, weights, costs):
    """{cost: [spellF1 per run]}."""
    out = {c: [] for c in costs}
    model = dict(scene_llr=scene_llr, weights=weights)
    for run in runs:
        alive, geo = run["alive"], run["geo"]
        score = possession.frame_scores(geo, alive, model)
        for c in costs:
            team = possession.decode(score, alive, geo["scene"] > 0, c)
            out[c].append(spell_f1(run["truth"], alive, team))
    return out


def leave_one_game_out(runs):
    """{cost: [(run, spellF1)]} with each game scored by a model fitted on the other two."""
    out = {c: [] for c in COSTS}
    for game in sorted({r["game"] for r in runs}):
        train = [r for r in runs if r["game"] != game]
        test = [r for r in runs if r["game"] == game]
        llr = fit_scenes(train)
        scores = score_runs(test, llr, fit_weights(train, llr), COSTS)
        for c in COSTS:
            out[c] += list(zip(test, scores[c]))
    return out


def pick_cost(held_out):
    means = {c: float(np.mean([s for _, s in v])) for c, v in held_out.items()}
    best = max(means.values())
    return max(c for c, m in means.items() if m >= best - TIE), means


def report(held_out, cost):
    rows = held_out[cost]
    print("\nheld-out spellF1 at switch cost {} (each game scored by a model that never saw it)".format(cost))
    for key in ("game", "noise"):
        for value in sorted({r[key] for r, _ in rows}):
            print("  {:<8} {:<10} {:.3f}".format(key, value, np.mean([s for r, s in rows if r[key] == value])))
    print("  mean               {:.3f}   worst run {:.3f}".format(np.mean([s for _, s in rows]),
                                                                  min(s for _, s in rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true", help="report only, keep the current model")
    args = parser.parse_args()

    print("building runs (3 games x {} dead-ball x {} tracking)".format(len(DEAD_BALL), len(NOISE)))
    runs = build_runs()
    held_out = leave_one_game_out(runs)
    cost, means = pick_cost(held_out)
    print("\nmean held-out spellF1 by switch cost: " +
          "  ".join("{}: {:.3f}".format(c, means[c]) for c in COSTS))
    print("picked switch cost {}".format(cost))
    report(held_out, cost)
    if args.report:
        return 0

    llr = fit_scenes(runs)
    weights = fit_weights(runs, llr)
    model = dict(
        version=3,
        trained_on="Metrica sample games 1-3 (github.com/metrica-sports/sample-data), "
                   "{} runs: dead-ball dropout x tracking noise; nothing from the challenge match".format(len(runs)),
        switch_cost=cost,
        held_out_spell_f1=round(means[cost], 4),
        scene_llr=[float(v) for v in llr],
        weights=[float(v) for v in weights],
    )
    with open(possession.MODEL_PATH, "w") as handle:
        json.dump(model, handle, indent=0)
        handle.write("\n")
    print("\nwrote {}".format(os.path.relpath(possession.MODEL_PATH, ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
