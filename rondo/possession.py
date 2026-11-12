"""ROLE 2 — who has it: while the ball is in play, which team is in possession?

Contract (do not change without telling role 3):

    predict_team(match, alive) -> np.ndarray of str, shape (match.n_frames,), each "H", "A" or "X"
        H = Barcelona, A = Athletic, X = not sure
        `alive` is role 1's answer, so each stretch of play can be handled on its own.

X is a real tool on the ranking metric: the scorer replaces an X with the previous answer. So
inside a possession (ball lost behind a crowd) X is free and safer than a guess; at a hand-over,
guess, or the old team gets carried across. See EVALUATION.md "How an abstention is scored".

Version 4 — a model learned on external data only (Metrica's open sample games):

  1. Clean the ball track: each seen ball position is averaged with its neighbours (5 frames,
     which takes out tracking jitter), and gaps of up to 1 s inside live play are filled with a
     straight line.
  2. Scene: on every frame, how far the nearest player is from the ball, how much closer they
     are than the nearest opponent, how fast the ball moves, and whether the ball is seen or
     filled in. A table counted on the sample games turns each scene into evidence (log-odds)
     for the nearest player's team: a slow ball at a player's feet with no opponent near is
     strong evidence, a fast ball flying past someone is almost none.
  3. Context: that evidence summed over windows before and after the frame (0.04 s to 10 s), the
     team of the last and next strong piece of evidence, and the team of the last and next touch
     (the ball changes speed or direction right next to a player), fading with time.
  4. Touch sequence: the team of the last two and the next two touches, and touches per team in
     the 0.5 / 1 / 3 s either side. A lone touch by one team between two of the other's is a
     deflection; two in a row is control (v4).
  5. A logistic regression with no intercept (so neither team is favoured) turns those into one
     score per frame: the log-odds that Barça has the ball.
  6. For each stretch of live play, a two-state model (H / A) picks the labelling that agrees most
     with the scores, paying SWITCH_COST for every change of possession (Viterbi).
  7. A stretch of play with no ball information at all is X.

Everything learned (the table, the weights, the switch cost) is in possession_model.json, written
by rondo/train_possession.py from the sample games under simulated dropout and tracking noise.
v4 learns the official definition of a change (a touch without control is not one), and weights
every possession equally-ish in training (1 / sqrt(length)), because spellF1 counts a 1 s
possession as much as a 30 s one. The organisers' rules clarification (2026-11-06) forbids
training or tuning on our match's ground truth or video, and nothing here has seen either. The
constants below are fixed design choices, not tuned values.

Version 3 (commit a25bec1) was steps 1-3 and 5-7, trained on every touch with equal frame weights.
Version 2 (commit f91eff9) was step 6 alone, on "nearest player within 3 m" evidence.

Ideas not done yet:
  - team shape (the side in possession spreads out) for the stretches with no ball
"""

import json
import os

import numpy as np

from rondo.match import AWAY, HOME

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "possession_model.json")

# Fixed design choices, not tuned
SMOOTH_HALF = 2          # ball smoothing: centred mean over 2 * 2 + 1 = 5 frames (0.2 s)
MAX_GAP = 25             # fill ball gaps up to 25 frames (1 s) inside live play
SPEED_K = 3              # ball speed from positions 3 frames either side
D_BINS = (0.5, 1, 1.5, 2, 3, 4, 6, 10)   # metres, nearest player to the ball
M_BINS = (0.5, 1, 2, 4)                  # metres, nearest opponent minus nearest player
V_BINS = (2, 5, 10, 18)                  # m/s, ball speed (one more bin for "unknown")
N_SCENES = 1 + (len(D_BINS) + 1) * (len(M_BINS) + 1) * (len(V_BINS) + 2) * 2
WINDOWS = (1, 2, 3, 6, 12, 25, 50, 100, 250)   # frames, evidence summed before / after
STRONG = (0.75, 1.5, 2.5)                      # log-odds, what counts as strong evidence
TAUS = (5, 25, 125)                            # frames, how fast the last / next team fades
TOUCHES = ((3, 5.0, 1.5), (3, 10.0, 2.0),      # (frames either side, velocity change m/s,
           (6, 5.0, 1.5), (6, 10.0, 2.0))      #  nearest player within m) = a touch
SEQUENCE = ((3, 5.0, 1.5), (6, 10.0, 2.0))     # the touches the touch-sequence features use
SEQUENCE_FADE = 50                             # frames
SEQUENCE_WINDOWS = (12, 25, 75)                # frames, touches per team before / after


def load_model(path=MODEL_PATH):
    with open(path) as handle:
        model = json.load(handle)
    return dict(scene_llr=np.array(model["scene_llr"], dtype=np.float64),
                weights=np.array(model["weights"], dtype=np.float64),
                switch_cost=float(model["switch_cost"]))


def predict_team(match, alive, model=None):
    model = model or load_model()
    geo = geometry(match, alive)
    return decode(frame_scores(geo, alive, model), alive & match.graded, geo["scene"] > 0,
                  model["switch_cost"])


def frame_scores(geo, alive, model):
    """Per-frame log-odds that Barça has the ball. einsum rather than @: same numbers on every
    machine (no BLAS), and no spurious macOS matmul warnings."""
    return np.einsum("ij,j->i", features(geo, alive, model["scene_llr"]), model["weights"])


def decode(score, playing, informed, switch_cost):
    """H/A per stretch of play from the per-frame scores, X where a stretch has no information."""
    labels = np.full(len(score), "X", dtype="<U1")
    for start, stop in _runs(playing):
        if not np.any(informed[start:stop]):
            continue
        states = viterbi_two_state(score[start:stop], switch_cost)
        labels[start:stop] = np.where(states == HOME, "H", "A")
    return labels


# -- the scene around the ball ------------------------------------------------------------------

def geometry(match, alive):
    """Per-frame arrays the features are built from. Never printed."""
    seen_xy = match.metres(smooth_ball(match.ball[:, :2].astype(np.float64)))
    ball = fill_gaps(seen_xy, alive, MAX_GAP)
    filled = np.isnan(seen_xy[:, 0]) & ~np.isnan(ball[:, 0])

    players = match.metres(match.players[:, :, :2].astype(np.float64))
    distance = np.sqrt(np.sum((players - ball[:, None, :]) ** 2, axis=2))
    distance = np.where(np.isnan(distance), np.inf, distance)
    home = match.team_of_slot == HOME
    no_one = np.full(len(ball), np.inf)
    d_home = distance[:, home].min(axis=1) if home.any() else no_one
    d_away = distance[:, ~home].min(axis=1) if (~home).any() else no_one
    nearest = np.minimum(d_home, d_away)
    team = np.where(d_home <= d_away, HOME, AWAY).astype(np.int8)

    k = SPEED_K
    speed = np.full(len(ball), np.nan)
    speed[k:-k] = np.sqrt(np.sum((ball[2 * k:] - ball[:-2 * k]) ** 2, axis=1)) * match.fps / (2 * k)
    with np.errstate(invalid="ignore"):
        d_bin = np.digitize(nearest, D_BINS)
        m_bin = np.digitize(np.abs(d_home - d_away), M_BINS)
    v_bin = np.where(np.isnan(speed), len(V_BINS) + 1, np.digitize(np.nan_to_num(speed), V_BINS))
    scene = 1 + ((d_bin * (len(M_BINS) + 1) + m_bin) * (len(V_BINS) + 2) + v_bin) * 2 + filled
    scene = np.where(np.isfinite(nearest), scene, 0).astype(np.int32)

    change = {}
    for k in sorted({touch[0] for touch in TOUCHES}):
        before = np.full_like(seen_xy, np.nan)
        after = np.full_like(seen_xy, np.nan)
        before[k:] = (seen_xy[k:] - seen_xy[:-k]) * match.fps / k
        after[:-k] = (seen_xy[k:] - seen_xy[:-k]) * match.fps / k
        change[k] = np.nan_to_num(np.sqrt(np.sum((after - before) ** 2, axis=1)))
    return dict(scene=scene, team=team, nearest=nearest, change=change)


def smooth_ball(xy, half=SMOOTH_HALF):
    """Each seen position replaced by the mean of the seen positions within `half` frames."""
    seen = ~np.isnan(xy[:, 0])
    total = np.concatenate((np.zeros((1, 2)), np.cumsum(np.where(seen[:, None], xy, 0.0), axis=0)))
    count = np.concatenate(([0], np.cumsum(seen)))
    index = np.arange(len(xy))
    lo, hi = np.maximum(index - half, 0), np.minimum(index + half + 1, len(xy))
    out = (total[hi] - total[lo]) / np.maximum(count[hi] - count[lo], 1)[:, None]
    out[~seen] = np.nan
    return out


def fill_gaps(xy, alive, max_gap):
    """Straight line over ball gaps of up to max_gap frames inside each stretch of live play."""
    out = xy.copy()
    seen = ~np.isnan(xy[:, 0])
    for start, stop in _runs(alive):
        known = start + np.flatnonzero(seen[start:stop])
        if len(known) < 2:
            continue
        gaps = np.diff(known)
        for i in np.flatnonzero((gaps > 1) & (gaps <= max_gap + 1)):
            a, b = known[i], known[i + 1]
            t = (np.arange(a + 1, b) - a) / float(b - a)
            out[a + 1:b] = xy[a] + (xy[b] - xy[a]) * t[:, None]
    return out


# -- features -------------------------------------------------------------------------------------

def features(geo, alive, scene_llr):
    """(n, n_features) per-frame features, every one signed so + means Barça (home)."""
    sign = np.where(geo["team"] == HOME, 1.0, -1.0)
    evidence = np.where(alive, scene_llr[geo["scene"]] * sign, 0.0)
    columns = [evidence]
    for k in WINDOWS:
        before, after = window_sums(evidence, k)
        columns += [before / k, after / k]
    for level in STRONG:
        strong = np.where(np.abs(evidence) > level, np.sign(evidence), 0.0)
        for tau in TAUS:
            columns += list(last_and_next(strong, tau))
    for k, change, within in TOUCHES:
        touch = touches(geo, alive, k, change, within)
        for tau in TAUS:
            columns += list(last_and_next(touch, tau))
    for k, change, within in SEQUENCE:
        columns += touch_sequence(touches(geo, alive, k, change, within))
    return np.stack(columns, axis=1)


def touches(geo, alive, k, change, within):
    """+1 / -1 on frames where a Barça / Athletic player touches the ball, 0 elsewhere."""
    sign = np.where(geo["team"] == HOME, 1.0, -1.0)
    return np.where((geo["change"][k] > change) & (geo["nearest"] < within) & alive, sign, 0.0)


def touch_sequence(touch):
    """Team of the last two and next two touches (fading), and touches per team either side."""
    n = len(touch)
    index = np.arange(n)
    hits = np.flatnonzero(touch != 0)
    columns = []
    at_or_after = np.searchsorted(hits, index, side="left")
    for back in (1, 2):
        h = at_or_after - back
        ok = h >= 0
        frame = hits[np.maximum(h, 0)] if len(hits) else index
        columns.append(np.where(ok, touch[frame] * np.exp(-(index - frame) / SEQUENCE_FADE), 0.0))
    after = np.searchsorted(hits, index, side="right")
    for ahead in (0, 1):
        h = after + ahead
        ok = h < len(hits)
        frame = hits[np.minimum(h, len(hits) - 1)] if len(hits) else index
        columns.append(np.where(ok, touch[frame] * np.exp(-(frame - index) / SEQUENCE_FADE), 0.0))
    for k in SEQUENCE_WINDOWS:
        before, later = window_sums(touch, k)
        columns += [before / 3.0, later / 3.0]
    return columns


def window_sums(x, k):
    """Sum over the k frames before each frame, and over the k frames after it."""
    total = np.concatenate(([0.0], np.cumsum(x)))
    n = len(x)
    index = np.arange(n)
    before = total[index] - total[np.maximum(index - k, 0)]
    after = total[np.minimum(index + 1 + k, n)] - total[index + 1]
    return before, after


def last_and_next(signal, tau):
    """Sign of the last and of the next non-zero frame, faded by exp(-frames away / tau)."""
    n = len(signal)
    index = np.arange(n)
    hit = signal != 0
    last = np.maximum.accumulate(np.where(hit, index, -1))
    following = np.minimum.accumulate(np.where(hit, index, n)[::-1])[::-1]
    before = np.where(last >= 0, signal[np.maximum(last, 0)] * np.exp(-(index - last) / tau), 0.0)
    after = np.where(following < n, signal[np.minimum(following, n - 1)]
                     * np.exp(-(following - index) / tau), 0.0)
    return before, after


# -- decoding -------------------------------------------------------------------------------------

def viterbi_two_state(score, switch_cost):
    """Best H/A labelling of one stretch: a path earns each frame's score where it says H (and
    nothing where it says A) and pays switch_cost per change. Only the gap between the best path
    ending H and the best ending A matters, so the forward pass keeps that one number."""
    values = score.tolist()
    n = len(values)
    stay_home = np.ones(n, dtype=bool)   # the best path ending H at i was H at i - 1
    stay_away = np.ones(n, dtype=bool)
    lead = values[0]
    for i in range(1, n):
        stay_home[i] = lead >= -switch_cost
        stay_away[i] = lead <= switch_cost
        lead = min(max(lead, -switch_cost), switch_cost) + values[i]
    states = np.empty(n, dtype=np.int8)
    state = HOME if lead >= 0 else AWAY
    states[-1] = state
    for i in range(n - 1, 0, -1):
        if state == HOME:
            state = HOME if stay_home[i] else AWAY
        else:
            state = AWAY if stay_away[i] else HOME
        states[i - 1] = state
    return states


def _runs(mask):
    flag = mask.astype(np.int8)
    edges = np.diff(np.concatenate(([0], flag, [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
