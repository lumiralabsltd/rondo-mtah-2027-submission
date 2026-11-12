"""Shared data loading. Everyone reads the match through load_match(), so nobody re-parses files.

The raw tracking file is 199,319 text lines, which takes a minute or two to parse in Python. The
first call parses it and saves numpy arrays to <data dir>/cache/tracking.npz; every later call
loads that cache in about a second. The cache lives inside data/, so git never sees it.

Every per-frame array here is indexed by VIDEO frame (0 … 199318), the same index as a row of the
submission. The ATD counts from 1, so this module subtracts 1 once and nobody else has to.

Do not print rows of these arrays. Aggregates (counts, means, scores) only: the match data is
licensed to participants only.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter.load_smart_tagging import load as load_smart_tagging  # noqa: E402

HOME, AWAY = 0, 1
CACHE_VERSION = 1


class Match(object):
    """One match, as arrays.

    ball          (n, 3) float32   x, y (0–1 pitch fractions) and speed; NaN where not tracked
    players       (n, slots, 3)    same per track slot; NaN where the slot is empty
    team_of_slot  (slots,) int8    HOME (0) or AWAY (1) for each track slot
    half          (n,) int8        1 or 2 on graded frames, 0 elsewhere (pre-match, halftime, after)
    graded        (n,) bool        half > 0 — the only frames the scorer looks at
    tags          list of dicts    Smart Tagging intervals in video frames, see starter/load_smart_tagging.py
    pitch         (length, width)  metres, for converting 0–1 coordinates to distances
    fps           25
    """

    def __init__(self, ball, players, team_of_slot, half, tags, pitch, fps):
        self.ball = ball
        self.players = players
        self.team_of_slot = team_of_slot
        self.half = half
        self.graded = half > 0
        self.tags = tags
        self.pitch = pitch
        self.fps = fps

    @property
    def n_frames(self):
        return len(self.ball)

    @property
    def ball_seen(self):
        """(n,) bool — the tracker located the ball on this frame."""
        return ~np.isnan(self.ball[:, 0])

    def metres(self, xy):
        """Convert (..., 2) pitch fractions to metres, so distances are not stretched."""
        return xy * np.asarray(self.pitch, dtype=np.float32)


def load_match(game=GAME, data_dir=None, use_cache=True):
    """The tactical recording of `game` as a Match."""
    data_dir = data_dir or DATA_DIR
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)

    cache_path = os.path.join(data_dir, "cache", "tracking.npz")
    arrays = _load_cache(cache_path, raw_path) if use_cache else None
    if arrays is None:
        print("parsing tracking (first run only, takes a minute or two) ...")
        arrays = _parse_tracking(raw_path, n_frames)
        if use_cache:
            _save_cache(cache_path, raw_path, arrays)

    half = np.zeros(n_frames, dtype=np.int8)
    for number, first, last in half_video_blocks(game):
        half[first:last + 1] = number

    return Match(
        ball=arrays["ball"],
        players=arrays["players"],
        team_of_slot=_team_of_each_slot(meta_path),
        half=half,
        tags=load_smart_tagging(game, data_dir),
        pitch=GAMES[game]["pitch"],
        fps=GAMES[game]["fps"],
    )


def _team_of_each_slot(meta_path):
    """HOME/AWAY per track slot. First team in the metadata is home (Barcelona) — same rule as
    starter/baseline.py."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in the ATD metadata, found {}".format(len(team_ids)))
    return np.array([HOME if player.get("teamId") == team_ids[0] else AWAY
                     for player in root.iter("Player")], dtype=np.int8)


def _parse_group(text):
    """'x,y,speed' -> three floats. 'NaN' parses as nan, so missing values need no special case."""
    values = text.split(",")
    return [float(v) for v in values[:3]] + [np.nan] * (3 - len(values))


def _parse_tracking(raw_path, n_frames):
    """Raw ATD -> ball (n, 3) and players (n, slots, 3), NaN where nothing was tracked.

    Line format: "<frame_id>:<slot>;<slot>;...:<ball>", each group "x,y,speed".
    """
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    players = None
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            groups = parts[1].split(";")
            if players is None:
                players = np.full((n_frames, len(groups), 3), np.nan, dtype=np.float32)
            players[frame, :len(groups)] = [_parse_group(g) for g in groups[:players.shape[1]]]
            ball[frame] = _parse_group(parts[2])
    if players is None:
        raise ValueError("no tracking lines could be read from {}".format(raw_path))
    return {"ball": ball, "players": players}


def _load_cache(cache_path, raw_path):
    if not os.path.isfile(cache_path):
        return None
    cached = np.load(cache_path)
    stale = (int(cached["version"]) != CACHE_VERSION
             or int(cached["source_size"]) != os.path.getsize(raw_path))
    return None if stale else {"ball": cached["ball"], "players": cached["players"]}


def _save_cache(cache_path, raw_path, arrays):
    folder = os.path.dirname(cache_path)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    np.savez(cache_path, version=CACHE_VERSION, source_size=os.path.getsize(raw_path), **arrays)


def window_sum(values, half_width):
    """Sum of `values` over a centred window of +/- half_width frames, clipped at the ends.

    Cumulative sums, so a wide window costs the same as a narrow one.
    """
    values = np.asarray(values)
    values = values.astype(np.float64 if values.dtype.kind == "f" else np.int64)
    if half_width <= 0:
        return values.copy()
    cumulative = np.concatenate(([0], np.cumsum(values)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(values), index + half_width + 1)
    return cumulative[high] - cumulative[low]
