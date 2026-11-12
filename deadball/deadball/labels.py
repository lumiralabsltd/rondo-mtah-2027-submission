"""Dead-ball labels for other games, derived from Metrica's open event data.

Metrica's sample events have no "ball dead" flag, but every restart is an event: a SET PIECE
(kick-off, throw-in, corner, free kick) or a PASS / BALL LOST tagged GOAL KICK. So a stoppage is
taken to run from the end of the event before a restart to the restart itself. Before the first
kick-off and after the last event are dead too.

What this misses: stoppages that end without a tagged restart (drop balls, some injuries). Treat
these labels as a good approximation, not the truth -- they are only used to measure and tune on
games other than the one we submit for, which the rules allow.
"""

import csv
import json

import numpy as np

RESTART_TYPES = {"SET PIECE"}
RESTART_SUBTYPES = {"GOAL KICK"}
# Logged while play is already stopped, so they must not count as live play.
STOPPAGE_ONLY_TYPES = {"CARD"}
# A ball blank at least this long in the original tracking is a stoppage (live blanks are short).
MIN_BLANK_S = 8.0


def _events_json(path):
    out = []
    for e in json.load(open(path))["data"]:
        subtypes = e.get("subtypes") or []
        if isinstance(subtypes, dict):
            subtypes = [subtypes]
        out.append((e["type"]["name"], {s["name"] for s in subtypes},
                    e["start"]["frame"], e["end"]["frame"] or e["start"]["frame"]))
    return out


def _events_csv(path):
    out = []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            start = int(row["Start Frame"])
            end = int(row["End Frame"]) or start
            subtypes = {s.strip() for s in row["Subtype"].split("-")} | {row["Subtype"].strip()}
            out.append((row["Type"], subtypes, start, max(end, start)))
    return out


RESTART_NAMES = (("THROW IN", "Throw-in"), ("CORNER KICK", "Corner"), ("GOAL KICK", "Goal kick"),
                 ("FREE KICK", "Free kick"), ("KICK OFF", "Kick-off"))


def restarts_from_events(path):
    """[(frame index, restart type)] for every restart in a Metrica event file."""
    events = _events_json(path) if path.endswith(".json") else _events_csv(path)
    out = []
    for kind, subtypes, start, _ in events:
        if kind not in RESTART_TYPES and not subtypes & RESTART_SUBTYPES:
            continue
        name = next((label for key, label in RESTART_NAMES if key in subtypes), "Other restart")
        out.append((start - 1, name))
    return sorted(out)


def dead_from_events(path, n_frames, ball_missing=None, min_blank_s=MIN_BLANK_S, fps=25):
    """Boolean array over tracking frames (index = frame - 1): True where the ball is dead.

    With `ball_missing` (the original, undegraded ball track), long blanks count as dead too.
    Metrica's sample tracking blanks the ball for the whole of a stoppage, so a blank of
    min_blank_s or more is a stoppage the events did not tag (injuries, untagged restarts).
    """
    events = _events_json(path) if path.endswith(".json") else _events_csv(path)
    events.sort(key=lambda e: (e[2], e[3]))

    dead = np.zeros(n_frames, dtype=bool)
    last_live_end = 0  # frame index where play last stopped being observable as live
    for kind, subtypes, start, end in events:
        if kind in STOPPAGE_ONLY_TYPES:
            continue
        restart = kind in RESTART_TYPES or bool(subtypes & RESTART_SUBTYPES)
        if restart and start - 1 > last_live_end:
            dead[last_live_end:start - 1] = True
        last_live_end = max(last_live_end, end - 1)
    dead[last_live_end + 1:] = True

    if ball_missing is not None:
        edges = np.diff(np.concatenate(([0], ball_missing.astype(np.int8), [0])))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if stop - start >= min_blank_s * fps:
                dead[start:stop] = True
    return dead
