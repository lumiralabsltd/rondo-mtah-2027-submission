"""Load tracking from every source we use into one shape: `Frames`.

Sources
-------
- Metrica FIFA/EPTS (`load_epts`): the challenge ATD (BAR-ATH) and Metrica Sample Game 3.
- Metrica CSV (`load_metrica_csv`): Metrica Sample Games 1 and 2.
- DFL / Sportec open data (`load_dfl`): via kloppy. Figshare blocks scripted downloads, so the
  XML files have to be fetched in a browser into data/dfl/ first (see README).

Coordinates come out in metres, centred on the halfway line, x along the pitch, y up. Missing
observations are NaN. Array index i is tracking frame i + 1 (all three sources count from 1),
which for BAR-ATH is also video frame / submission row i.
"""

import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
_PARENT = os.path.dirname(ROOT)   # the team repo holds this folder, so the challenge files are here
MTAH_REPO = os.environ.get("MTAH_REPO") or (
    _PARENT if os.path.exists(os.path.join(_PARENT, "games.py"))
    else os.path.join(_PARENT, "more-than-a-hack-2027"))

HOME, AWAY = 0, 1


@dataclass
class Frames:
    ball: np.ndarray      # (n, 2) metres, NaN where the ball is not tracked
    players: np.ndarray   # (n, k, 2) metres, NaN where a slot is empty
    team: np.ndarray      # (k,) HOME / AWAY per slot
    pitch: tuple          # (length, width) metres
    fps: int = 25

    @property
    def n(self):
        return len(self.ball)


def _to_metres(nx, ny, pitch):
    """Normalised 0-1 (origin top-left, y down) to centred metres (y up)."""
    length, width = pitch
    return (nx - 0.5) * length, (0.5 - ny) * width


def mtah_repo():
    """Import path for the challenge repo, so games.py and the scorer can be reused."""
    if not os.path.isdir(MTAH_REPO):
        raise FileNotFoundError("challenge repo not found at {} -- set MTAH_REPO".format(MTAH_REPO))
    if MTAH_REPO not in sys.path:
        sys.path.insert(0, MTAH_REPO)
    return MTAH_REPO


# -- Metrica FIFA/EPTS --------------------------------------------------------------------------

def _epts_layout(meta_path):
    """(team per slot, layouts, fps, pitch) from an EPTS metadata XML.

    A slot is one player id. The raw file's column layout can change mid-match (Sample Game 3
    starts a new DataFormatSpecification at every substitution), so `layouts` is a list of
    (start_frame, end_frame, slot index of each player group, values per group). Without a spec,
    groups follow the <Player> order -- what the challenge baseline assumes for the ATD.
    """
    root = ET.parse(meta_path).getroot()
    home_id = [t.get("id") for t in root.iter("Team")][0]
    team_of_player = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    player_of_channel = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}

    slots, layouts = [], []

    def slot_of(pid):
        if pid not in slots:
            slots.append(pid)
        return slots.index(pid)

    for spec in root.iter("DataFormatSpecification"):
        groups, width = [], []
        for group in spec.iter("SplitRegister"):
            refs = [r.get("playerChannelId") for r in group.findall("PlayerChannelRef")]
            if refs:
                groups.append(slot_of(player_of_channel[refs[0]]))
                width.append(len(refs))
        layouts.append((int(spec.get("startFrame")), int(spec.get("endFrame")), groups, width))
    if not layouts:
        groups = [slot_of(p.get("id")) for p in root.iter("Player")]
        layouts = [(1, 10 ** 9, groups, None)]

    team = np.array([HOME if team_of_player[p] == home_id else AWAY for p in slots], dtype=np.int8)
    fps = int(float(root.findtext(".//FrameRate") or 25))
    length = float(root.findtext(".//FieldSize/Width") or 105)
    width = float(root.findtext(".//FieldSize/Height") or 68)
    return team, layouts, fps, (length, width)


def load_epts(meta_path, raw_path, pitch=None):
    """Raw EPTS text: '<frame>:<x,y[,s]>;<...>:<ball x,y[,z]>', NaN for missing."""
    team, layouts, fps, meta_pitch = _epts_layout(meta_path)
    pitch = pitch or meta_pitch

    with open(raw_path) as handle:
        lines = [line.rstrip("\n").split(":") for line in handle]
    lines = [parts for parts in lines if len(parts) == 3]
    n = max(int(parts[0]) for parts in lines)
    players = np.full((n, len(team), 2), np.nan)
    ball = np.full((n, 2), np.nan)

    for start, end, groups, _ in layouts:
        block = [p for p in lines if start <= int(p[0]) <= end]
        if not block:
            continue
        index = np.array([int(p[0]) for p in block]) - 1
        values = np.array([p[1].replace(";", ",").split(",") for p in block], dtype=float)
        values = values.reshape(len(block), len(groups), -1)[..., :2]
        players[index[:, None], np.array(groups)[None, :]] = values
        ball[index] = np.array([p[2].split(",")[:2] for p in block], dtype=float)

    players[..., 0], players[..., 1] = _to_metres(players[..., 0], players[..., 1], pitch)
    ball[:, 0], ball[:, 1] = _to_metres(ball[:, 0], ball[:, 1], pitch)
    return Frames(ball=ball, players=players, team=team, pitch=pitch, fps=fps)


def load_match(game=None):
    """The challenge match (tactical ATD), or None when the data bundle is not downloaded yet."""
    mtah_repo()
    import games
    game = game or games.GAME
    paths = games.asset_paths(game)
    if not (os.path.exists(paths["atd_meta"]) and os.path.exists(paths["atd_raw"])):
        return None
    return load_epts(paths["atd_meta"], paths["atd_raw"], pitch=tuple(games.GAMES[game]["pitch"]))


# -- Metrica CSV (Sample Games 1 and 2) ---------------------------------------------------------

def load_metrica_csv(home_path, away_path, pitch=(105.0, 68.0), fps=25):
    def read(path):
        table = np.genfromtxt(path, delimiter=",", skip_header=3)
        return table[:, 1].astype(int), table[:, 3:-2], table[:, -2:]

    frames, home_xy, ball = read(home_path)
    _, away_xy, _ = read(away_path)
    n = frames.max()
    xy = np.concatenate([home_xy, away_xy], axis=1).reshape(len(frames), -1, 2)
    team = np.array([HOME] * (home_xy.shape[1] // 2) + [AWAY] * (away_xy.shape[1] // 2),
                    dtype=np.int8)

    players = np.full((n, len(team), 2), np.nan)
    ball_out = np.full((n, 2), np.nan)
    players[frames - 1] = xy
    ball_out[frames - 1] = ball
    players[..., 0], players[..., 1] = _to_metres(players[..., 0], players[..., 1], pitch)
    ball_out[:, 0], ball_out[:, 1] = _to_metres(ball_out[:, 0], ball_out[:, 1], pitch)
    return Frames(ball=ball_out, players=players, team=team, pitch=pitch, fps=fps)


# -- DFL / Sportec open data --------------------------------------------------------------------

def load_dfl(meta_path, raw_path):
    """(Frames, dead) from a DFL open-data match, dead being kloppy's ball_state per frame."""
    frames, dead, _ = load_dfl_full(meta_path, raw_path)
    return frames, dead


def load_dfl_full(meta_path, raw_path):
    """(Frames, dead, owner) from a DFL open-data match. owner is HOME / AWAY per frame from the
    tracking provider's own possession flag (-1 where it has none). Parsing the XML takes minutes,
    so the arrays are cached next to it in cache/<match>.npz (gitignored with the rest of data/)."""
    cache = os.path.join(os.path.dirname(raw_path), "cache",
                         os.path.basename(raw_path).replace("_tracking.xml", ".npz"))
    if os.path.exists(cache):
        z = np.load(cache)
        return Frames(ball=z["ball"], players=z["players"], team=z["team"],
                      pitch=tuple(z["pitch"]), fps=int(z["fps"])), z["dead"], z["owner"]
    frames, dead, owner = _parse_dfl(meta_path, raw_path)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, ball=frames.ball, players=frames.players, team=frames.team,
             pitch=np.array(frames.pitch), fps=frames.fps, dead=dead, owner=owner)
    return frames, dead, owner


def _parse_dfl(meta_path, raw_path):
    from kloppy import sportec
    dataset = sportec.load_tracking(raw_data=raw_path, meta_data=meta_path, only_alive=False)
    meta = dataset.metadata
    pitch = (meta.pitch_dimensions.pitch_length, meta.pitch_dimensions.pitch_width)
    slots = [(p, i) for i, t in enumerate(meta.teams) for p in t.players]
    team = np.array([HOME if i == 0 else AWAY for _, i in slots], dtype=np.int8)

    n = len(dataset.records)
    players = np.full((n, len(slots), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    dead = np.zeros(n, dtype=bool)
    owner = np.full(n, -1, dtype=np.int8)
    home_team = meta.teams[0]
    for f, record in enumerate(dataset.records):
        if record.ball_coordinates is not None:
            ball[f] = record.ball_coordinates.x, record.ball_coordinates.y
        dead[f] = record.ball_state is not None and record.ball_state.value == "dead"
        if record.ball_owning_team is not None:
            owner[f] = HOME if record.ball_owning_team == home_team else AWAY
        for s, (player, _) in enumerate(slots):
            data = record.players_data.get(player)
            if data is not None and data.coordinates is not None:
                players[f, s] = data.coordinates.x, data.coordinates.y

    players[..., 0], players[..., 1] = _to_metres(players[..., 0], players[..., 1], pitch)
    ball[:, 0], ball[:, 1] = _to_metres(ball[:, 0], ball[:, 1], pitch)
    return Frames(ball=ball, players=players, team=team, pitch=pitch,
                  fps=int(meta.frame_rate or 25)), dead, owner
