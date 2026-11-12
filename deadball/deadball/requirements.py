"""Checks for each part of Role 1 (dead ball), so the viewer can show what is and isn't met.

Each check returns {id, title, what, status, value, detail}. Status is one of:
    met      measured, and good enough
    partial  measured, works but with clear gaps
    todo     not built yet
    na       can't be measured on this game (e.g. Smart Tagging only exists for BAR-ATH)
Thresholds for met/partial live in BANDS. They're judgement calls, not challenge rules.
"""

import json
import os

import numpy as np

from .export import runs

# Params
BANDS = {"recall": (0.8, 0.5), "edge_s": (0.5, 1.5), "stress": (0.7, 0.5)}
FOUND_SHARE = 0.5   # a true stoppage counts as found if predictions cover this much of it
TYPE_TOLERANCE_S = 1.0


def _band(value, good, ok, higher=True):
    if value is None:
        return "na"
    if higher:
        return "met" if value >= good else "partial" if value >= ok else "todo"
    return "met" if value <= good else "partial" if value <= ok else "todo"


def _found(dead_run, pred):
    a, b = dead_run
    return pred[a:b].mean() >= FOUND_SHARE


def edge_errors(dead, pred, fps):
    """Median |start| and |end| error (s) over true stoppages overlapped by exactly one prediction."""
    starts, ends = [], []
    pred_runs = runs(pred)
    for a, b in runs(dead):
        hits = [(c, d) for c, d in pred_runs if c < b and d > a]
        if len(hits) == 1:
            starts.append(abs(hits[0][0] - a) / fps)
            ends.append(abs(hits[0][1] - b) / fps)
    if not starts:
        return None, None
    return float(np.median(starts)), float(np.median(ends))


def type_stoppages(dead, restarts, fps):
    """{restart type: [stoppage runs]}; a stoppage takes the type of the restart that ends it."""
    out = {}
    tol = int(TYPE_TOLERANCE_S * fps)
    frames = np.array([f for f, _ in restarts]) if restarts else np.array([], dtype=int)
    for a, b in runs(dead):
        near = np.flatnonzero(np.abs(frames - b) <= tol) if len(frames) else []
        kind = restarts[near[0]][1] if len(near) else "Untagged (injury, drop ball…)"
        out.setdefault(kind, []).append((a, b))
    return out


def _recall_check(cid, title, what, groups, kinds, pred):
    stoppages = [r for k in kinds for r in groups.get(k, [])]
    if not stoppages:
        return {"id": cid, "title": title, "what": what, "status": "na", "value": None,
                "detail": "no labelled stoppages of this kind in this game"}
    found = sum(_found(r, pred) for r in stoppages)
    recall = found / len(stoppages)
    return {"id": cid, "title": title, "what": what,
            "status": _band(recall, *BANDS["recall"]), "value": recall,
            "detail": "{} of {} found".format(found, len(stoppages))}


def stress_summary(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def checks(dead, pred, fps, restarts=None, stress=None, match=None):
    """All Role 1 checks for one game.

    dead / pred: truth and prediction per frame (truth may be None). restarts: [(frame, type)]
    from events (sample games). match: BAR-ATH extras, {"tags": [...], "pre_kickoff": (a, b),
    "csv": path}.
    """
    out = []
    groups = type_stoppages(dead, restarts or [], fps) if dead is not None and restarts else {}

    out.append(_recall_check(
        "out", "Ball leaving the pitch",
        "Throw-ins, corners and goal kicks are found as stoppages",
        groups, ("Throw-in", "Corner", "Goal kick"), pred))
    out.append(_recall_check(
        "fouls", "Fouls and free kicks",
        "Play stopped by the referee: players stand still before the restart",
        groups, ("Free kick",), pred))
    out.append(_recall_check(
        "kickoff", "Kick-offs",
        "Start of each half and after goals: teams line up in their own halves",
        groups, ("Kick-off",), pred))
    out.append(_recall_check(
        "other", "Other stoppages",
        "Injuries, drop balls and anything without a tagged restart",
        groups, ("Untagged (injury, drop ball…)",), pred))

    if dead is not None:
        start, end = edge_errors(dead, pred, fps)
        worst = None if start is None else max(start, end)
        out.append({"id": "timing", "title": "Start and end timing",
                    "what": "A found stoppage starts and ends on the right frame (median error)",
                    "status": _band(worst, *BANDS["edge_s"], higher=False), "value": worst,
                    "detail": "start {:.1f} s · end {:.1f} s".format(start, end)
                    if start is not None else "no matched stoppages"})

    if stress:
        worst = min(stress["by_visible"].values())
        out.append({"id": "stress", "title": "Robust to the ball being missing or visible",
                    "what": "Works whether tracking blanks the ball in stoppages or keeps it on screen",
                    "status": _band(worst, *BANDS["stress"]), "value": worst,
                    "detail": " · ".join("{:.0f}% visible {:.2f}".format(float(v) * 100, f)
                                         for v, f in stress["by_visible"].items())})

    tags = (match or {}).get("tags")
    if tags is None:
        out.append({"id": "tags", "title": "Smart Tagging set pieces",
                    "what": "Set-piece tags mark restarts, so a stoppage should end where one starts",
                    "status": "na", "value": None,
                    "detail": "only BAR-ATH has tags (sample games: scored with simulated tags, run.py samples --tags)"})
    else:
        from .rules import PARAMS
        search = int(PARAMS["tag_search_s"] * fps)
        ends = np.flatnonzero(np.diff(pred.astype(np.int8)) == -1) + 1
        hit = sum(((ends >= start) & (ends <= start + search)).any() for start, _ in tags)
        share = hit / len(tags) if tags else None
        out.append({"id": "tags", "title": "Smart Tagging set pieces",
                    "what": "Set-piece tags mark restarts, so a stoppage should end where one starts",
                    "status": _band(share, *BANDS["recall"]),
                    "value": share,
                    "detail": "used as input · {} of {} tags have a stoppage ending in their first "
                              "{:.0f} s".format(hit, len(tags), PARAMS["tag_search_s"])})

    gap = (match or {}).get("pre_kickoff")
    if gap is None:
        out.append({"id": "pre_kickoff", "title": "2nd-half pre-kickoff dead time",
                    "what": "The ~2 minutes graded before the 2nd-half kick-off are all dead",
                    "status": "na", "value": None, "detail": "BAR-ATH only"})
    else:
        share = float(pred[gap[0]:gap[1]].mean())
        out.append({"id": "pre_kickoff", "title": "2nd-half pre-kickoff dead time",
                    "what": "The ~2 minutes graded before the 2nd-half kick-off are all dead",
                    "status": _band(share, 0.98, 0.8), "value": share,
                    "detail": "{:.0%} predicted dead, found by the detector, not hard-coded".format(share)})

    out.append({"id": "rules", "title": "Challenge rules",
                "what": "No hand labels or frame constants for BAR-ATH; nothing tuned on its ground truth, tracking or video",
                "status": "met", "value": None,
                "detail": "tuned only on Metrica's open sample games"})

    csv = (match or {}).get("csv")
    out.append({"id": "output", "title": "Per-frame D output",
                "what": "A D/L label for every one of the 199,319 frames, ready to merge with H/A",
                "status": "met" if csv and os.path.exists(csv) else "todo", "value": None,
                "detail": os.path.basename(csv) if csv and os.path.exists(csv)
                else "written by run.py match once BAR-ATH data is in"})
    return out
