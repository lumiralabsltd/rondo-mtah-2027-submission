"""Export a game for the browser viewer: positions, dead-ball runs and the detector's evidence.

Writes viewer/data/<game>.bin (int16 decimetres, NaN as -32768) and viewer/data/<game>.json.
The prediction comes from exactly the same features / rules / dropout path that run.py scores.

With a homography (BAR-ATH only) the viewer also draws detection boxes on the video:
viewer/data/<game>.hom.bin is float32, 9 values per video frame, mapping normalised pitch
coordinates (0-1, origin top-left) to video pixels.

Binary layout, one record per exported step (every STEP frames), all int16:
    ball_seen_x, ball_seen_y, ball_clean_x, ball_clean_y, then x, y for each player slot
"ball_seen" is the ball the detector actually got (after simulated dropout); "ball_clean" is the
original track, so the viewer can show what the dropout hid.
"""

import json
import os

import numpy as np

from . import evaluate, features, rules

# Params
VIEW_FPS = 5           # playback samples per second; players are interpolated between them
NAN_CODE = -32768

HERE = os.path.dirname(os.path.abspath(__file__))
VIEW_DATA = os.path.join(os.path.dirname(HERE), "viewer", "data")


def runs(mask):
    """[[start, stop), ...] of True runs, in frames."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return [[int(a), int(b)] for a, b in zip(np.flatnonzero(edges == 1),
                                            np.flatnonzero(edges == -1))]


def _int16(values):
    out = np.round(values * 10)
    out[np.isnan(out)] = NAN_CODE
    return np.clip(out, NAN_CODE, 32767).astype("<i2")


def _series(values, step, digits=2):
    sampled = np.asarray(values, dtype=float)[::step]
    return [None if np.isnan(v) else round(float(v), digits) for v in sampled]


def export(name, frames, dead=None, graded=None, dropout=True, seed=0, dead_visible=0.0,
           restarts=None, match=None, video=None, homography=None, video_size=None):
    """Write one game; `dead` is the truth (None when there is none), `graded` where it applies.

    dead_visible > 0 runs the stress test: the ball stays on screen through that share of
    stoppage frames (evaluate.show_dead_ball), as automatic tracking might show it.
    """
    clean = frames
    if dead is not None and dead_visible > 0:
        frames = evaluate.show_dead_ball(frames, dead, visible=dead_visible, seed=seed)
    if dropout and dead is not None:
        frames = evaluate.add_ball_dropout(frames, dead, seed=seed)
    f = features.compute(frames)
    s = rules.score(f)
    pred = rules.detect(f, frames.fps, tags=(match or {}).get("tags"))

    step = max(1, frames.fps // VIEW_FPS)
    os.makedirs(VIEW_DATA, exist_ok=True)
    record = np.concatenate([frames.ball, clean.ball,
                             frames.players.reshape(frames.n, -1)], axis=1)[::step]
    _int16(record).tofile(os.path.join(VIEW_DATA, name + ".bin"))

    graded = np.ones(frames.n, dtype=bool) if graded is None else graded
    meta = {
        "name": name,
        "fps": frames.fps,
        "step": step,
        "n_frames": frames.n,
        "n_steps": len(record),
        "n_slots": len(frames.team),
        "team": frames.team.tolist(),
        "pitch": list(frames.pitch),
        "dropout": bool(dropout and dead is not None),
        "dead_visible": float(dead_visible) if dead is not None else 0.0,
        "pred_dead": runs(pred),
        "true_dead": runs(dead & graded) if dead is not None else None,
        "graded": runs(graded),
        "params": {k: rules.PARAMS[k] for k in ("enter", "exit", "long_gap_s", "slow_speed",
                                                 "walking_share", "ball_still_s")},
        "evidence": {
            "score": _series(np.minimum(s, 6.0), step),
            "player_speed": _series(f["player_speed"], step),
            "walking_share": _series(f["walking_share"], step),
            "ball_gap_s": _series(f["ball_gap_s"], step, 1),
            "ball_still_s": _series(f["ball_still_s"], step, 1),
            "ball_out_m": _series(f["ball_out_m"], step, 1),
            "kickoff_shape": _series(f["kickoff_shape"], step, 0),
            "n_players": _series(f["n_players"], step, 0),
        },
    }
    if dead is not None:
        meta["scores"] = {k: (round(float(v), 3) if isinstance(v, float) else int(v))
                          for k, v in evaluate.scores(dead, pred, graded).items()}
    from . import requirements
    truth = (dead & graded) if dead is not None else None
    meta["requirements"] = requirements.checks(
        truth, pred, frames.fps, restarts=restarts, match=match,
        stress=requirements.stress_summary(os.path.join(VIEW_DATA, "stress.json")))
    meta["video"] = video
    meta["video_size"] = list(video_size) if video_size else None
    meta["homography"] = None
    if homography is not None:
        # Full frame rate (the camera pans), float32, NaN where no matrix exists for that frame.
        homography.reshape(len(homography), 9).astype("<f4").tofile(
            os.path.join(VIEW_DATA, name + ".hom.bin"))
        meta["homography"] = "data/" + name + ".hom.bin"
    with open(os.path.join(VIEW_DATA, name + ".json"), "w") as handle:
        json.dump(meta, handle, separators=(",", ":"))

    index_path = os.path.join(VIEW_DATA, "index.json")
    names = json.load(open(index_path)) if os.path.exists(index_path) else []
    if name not in names:
        names.append(name)
    with open(index_path, "w") as handle:
        json.dump(sorted(names), handle)
    return meta
