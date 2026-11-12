"""ROLE 1 — dead ball: is the ball in play on each frame?

Contract (do not change without telling role 3):

    predict_alive(match) -> np.ndarray of bool, shape (match.n_frames,)
        True  = ball in play
        False = dead ball (out of play, foul, waiting for a restart)  -> becomes D

Only graded frames are scored, so whatever this says at halftime does not matter.

Version 2 — Sukh's detector in deadball/ (see deadball/README.md). Weighted evidence from the
tracking (ball out, ball missing for long, ball lying still, players walking, kick-off shape, an
out-of-play latch that holds until the restart is struck), then hysteresis, minimum run lengths,
edge snapping, and Smart Tagging SET PIECES as anchors for restarts. Every parameter is tuned on
Metrica's open sample games, never on our match.

This file only converts our Match into deadball's Frames, so the pipeline and deadball/run.py
label the match identically. Version 1 (simpler rules) is in git history, commit b908e5a.
"""

import os
import sys

import numpy as np

from rondo.match import ROOT

sys.path.insert(0, os.path.join(ROOT, "deadball"))
from deadball import features, loaders, rules  # noqa: E402


def predict_alive(match):
    frames = frames_from_match(match)
    dead = rules.detect(features.compute(frames), frames.fps, tags=set_piece_tags(match))
    return ~np.asarray(dead, dtype=bool)


def frames_from_match(match):
    """Our Match (0-1 pitch fractions, origin top-left) as deadball Frames (centred metres)."""
    pitch = tuple(float(v) for v in match.pitch)
    ball = match.ball[:, :2].astype(np.float64)
    players = match.players[:, :, :2].astype(np.float64)
    ball[:, 0], ball[:, 1] = loaders._to_metres(ball[:, 0], ball[:, 1], pitch)
    players[..., 0], players[..., 1] = loaders._to_metres(players[..., 0], players[..., 1], pitch)
    return loaders.Frames(ball=ball, players=players, team=match.team_of_slot.copy(),
                          pitch=pitch, fps=int(match.fps))


def set_piece_tags(match):
    """Smart Tagging SET PIECES as [(start_frame, end_frame)] video frames."""
    return [(tag["start_frame"], tag["end_frame"]) for tag in match.tags
            if (tag["code"] or "").strip().upper() == "SET PIECES"]
