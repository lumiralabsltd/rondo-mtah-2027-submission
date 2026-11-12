"""Scores for a dead/live prediction, plus the ball dropout used to make clean data look like ours.

Two numbers matter:
- frame F1 of the dead class, and the mean of dead/live F1 (the per-frame view);
- stoppage F1: stoppages matched one-to-one by temporal IoU, averaged over IoU 0.3-0.7. It is
  the challenge's own spellF1 code (evaluation/possession_metrics.py) pointed at dead runs instead
  of possessions, because every stoppage we miss or invent splits or merges possessions.
"""

import numpy as np

from .loaders import mtah_repo

# Params
# None of these come from the challenge match: the rules (organisers' clarification, 2026-11-06)
# say every tuned value must come from external data, and these shape the scenarios we tune on.
TARGET_LIVE_MISSING = 0.30   # share of live frames with no ball: a middle value for automatic
DROPOUT_RATES = (0.2, 0.4)   # tracking; tune.py averages over both of these instead
MEAN_GAP_S = 1.5             # mean length of an added dropout burst
THROW_BEYOND_M = (0.5, 3.0)  # stress test: how far past the touchline a ball out for a throw lies
GOAL_BEYOND_M = (1.0, 8.0)   # ...and past the goal line for a goal kick or corner (runs on behind the goal)
ROLL_M, ROLL_PERIOD_S = 2.0, 6.0   # back inside, the ball is rolled about this much (~2 m/s)
ROLL_STILL_S = 1.0           # ...and lies still only this long before it is struck
TAG_SHARE = 0.55             # simulated Smart Tagging: share of restarts that get a set-piece tag
TAG_LEAD_S = (0.0, 8.0)      # ...starting this long before the restart (the starter: tags mark a
                             # window of a few seconds around the restart, not its instant)
TAG_LEN_S = 35.0             # ...and lasting this long (Smart Tagging set pieces are long phases)


def add_ball_dropout(frames, dead, rate=TARGET_LIVE_MISSING, mean_gap_s=MEAN_GAP_S, seed=0):
    """Blank the ball in bursts on live frames until about `rate` of live frames have no ball."""
    rng = np.random.default_rng(seed)
    ball = frames.ball.copy()
    live = ~dead
    already = np.isnan(ball[live, 0]).mean()
    extra = max(0.0, (rate - already) / max(1e-9, 1 - already))
    mean_len = mean_gap_s * frames.fps
    p_start = extra / (mean_len * (1 - extra) + 1e-9)

    i = 0
    while i < len(ball):
        if live[i] and rng.random() < p_start:
            stop = min(len(ball), i + max(1, int(rng.exponential(mean_len))))
            ball[i:stop] = np.nan
            i = stop
        else:
            i += 1
    return type(frames)(ball=ball, players=frames.players, team=frames.team,
                        pitch=frames.pitch, fps=frames.fps)


def _beyond_line(exit_xy, restart_xy, kind, pitch, rng):
    """Where a ball that left the pitch is tracked lying: past the touchline for a throw-in, past
    the goal line for a goal kick or corner, on the side the restart is taken from."""
    length, width = pitch
    if kind == "Throw-in":
        return np.array([restart_xy[0], np.sign(restart_xy[1]) * (width / 2 + rng.uniform(*THROW_BEYOND_M))])
    if kind in ("Goal kick", "Corner"):
        return np.array([np.sign(restart_xy[0]) * (length / 2 + rng.uniform(*GOAL_BEYOND_M)), exit_xy[1]])
    return exit_xy


def show_dead_ball(frames, dead, visible=1.0, jitter_m=0.3, seed=0, restarts=None):
    """Stress test: put the ball back on screen during stoppages, as automatic tracking might.

    Metrica's sample data blanks the ball for a whole stoppage, which makes "ball missing" nearly
    the answer. Here each stoppage instead shows the ball where it went dead for the first 60%
    (lying out of play, or held for the throw), carried to the restart spot over the next 20%, and
    still on the spot for the last 20%. `visible` is the share of stoppage frames that keep it;
    the rest are blanked in bursts.

    With `restarts` ([(frame, type)] from labels.restarts_from_events), a stoppage ended by a
    throw-in, goal kick or corner has the ball lying a few metres past the line it crossed and
    carried back onto the pitch, as automatic tracking that keeps following the ball would show.
    """
    rng = np.random.default_rng(seed)
    ball = frames.ball.copy()
    seen = np.flatnonzero(~np.isnan(ball[:, 0]))
    kinds = dict(restarts or [])
    edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        before = seen[seen < start]
        after = seen[seen >= stop]
        if not len(before) or not len(after):
            continue
        exit_xy, restart_xy = ball[before[-1]], ball[after[0]]
        kind = next((kinds[f] for f in range(stop - frames.fps, stop + frames.fps) if f in kinds), None)
        n = stop - start
        if kind:
            # Out past the line, fetched back early, then rolled about near the spot (the keeper
            # setting up a goal kick, a thrower waiting) and only still for the last moment.
            exit_xy = _beyond_line(exit_xy, restart_xy, kind, frames.pitch, rng)
            hold, move = int(0.25 * n), max(1, int(0.15 * n))
            t = np.arange(n) / frames.fps
            roll = ROLL_M * np.sin(2 * np.pi * t / ROLL_PERIOD_S)
            roll[n - int(ROLL_STILL_S * frames.fps):] = 0
            roll[:hold + move] = 0
        else:
            hold, move = int(0.6 * n), max(1, int(0.2 * n))
            roll = np.zeros(n)
        w = np.clip((np.arange(n) - hold) / move, 0, 1)[:, None]
        path = exit_xy + (restart_xy - exit_xy) * w
        path[:, 1] += roll
        ball[start:stop] = path + rng.normal(0, jitter_m, path.shape)

    if visible < 1.0:
        hide = np.zeros(len(ball), dtype=bool)
        mean_len = MEAN_GAP_S * frames.fps
        p_start = (1 - visible) / (mean_len * visible + 1e-9)
        i = 0
        while i < len(ball):
            if dead[i] and rng.random() < p_start:
                stop = min(len(ball), i + max(1, int(rng.exponential(mean_len))))
                hide[i:stop] = dead[i:stop]
                i = stop
            else:
                i += 1
        ball[hide] = np.nan
    return type(frames)(ball=ball, players=frames.players, team=frames.team,
                        pitch=frames.pitch, fps=frames.fps)


def _f1(truth, pred):
    tp = np.sum(truth & pred)
    precision = tp / max(1, pred.sum())
    recall = tp / max(1, truth.sum())
    return 2 * precision * recall / max(1e-9, precision + recall)


def scores(dead_true, dead_pred, graded=None):
    mtah_repo()
    from evaluation import possession_metrics as pm

    graded = np.ones(len(dead_true), dtype=bool) if graded is None else graded
    t, p = dead_true[graded], dead_pred[graded]
    dead_f1, live_f1 = _f1(t, p), _f1(~t, ~p)

    # Dead runs as "H" spells, live as "D", ungraded as "X": spell_scores then matches stoppages.
    def codes(dead):
        c = np.where(dead, pm.H, pm.D).astype(np.int8)
        c[~graded] = pm.X
        return c

    gt = codes(dead_true)
    spell = pm.spell_scores(gt, codes(dead_pred), pm.valid_runs(gt))
    return {"dead_f1": dead_f1, "macro_f1": (dead_f1 + live_f1) / 2,
            "stoppage_f1": spell["mean_f1"], "n_true": spell["n_gt"], "n_pred": spell["n_pred"]}


def simulate_tags(restarts, fps=25, share=TAG_SHARE, seed=0):
    """Smart Tagging set pieces for a sample game, from its restart events: [(start, end)] frames.

    The sample games have no Smart Tagging, so the tag anchors are scored on these instead."""
    rng = np.random.default_rng(seed)
    out = []
    for frame, _ in restarts:
        if rng.random() < share:
            start = int(frame - rng.uniform(*TAG_LEAD_S) * fps)
            out.append((start, start + int(TAG_LEN_S * fps)))
    return out
