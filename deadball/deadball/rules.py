"""Rule-based dead-ball detector: weighted evidence, then hysteresis, then minimum run lengths.

Every threshold lives in PARAMS and is expressed in metres or seconds, never as a frame number of
a particular game -- the challenge rules forbid constants tied to frames of the provided match.
"""

import numpy as np

# Tuned on Metrica Sample Games 1-3 only, never on the challenge match. Two re-tunes on 2026-11-12,
# after the organisers' rules clarification, each with a selection rule fixed before running:
#   1. deadball/tune.py: highest mean stoppage F1 over 10 stress scenarios (changed enter 1.5 -> 2.0)
#   2. rondo/tune_dead_ball.py: highest mean spellF1 of the whole pipeline (this detector, then the
#      v3 possession model) over 60 degraded sample runs, one setting at a time, two passes. It
#      changed w_slow 0.5 -> 1.0, w_ball_still 1.0 -> 0.5, exit 1.0 -> 1.5, min_dead_s 3 -> 5 and the
#      edge snapping (start_fwd 2 -> 1, end_back 3 -> 4, end_fwd 1 -> 0.5): mean 0.576 -> 0.593.
# spellF1 is what we are ranked on, so (2) has the last word; re-running tune.py alone would undo it.
PARAMS = {
    # evidence weights
    "w_out": 2.0,            # last seen ball beyond the lines, now missing
    "w_out_seen": 2.0,       # ball seen lying beyond the lines
    "w_long_gap": 1.0,       # ball missing for a long stretch
    "w_slow": 1.0,           # players barely moving
    "w_walking": 1.0,        # nearly everyone walking
    "w_kickoff": 2.0,        # kick-off shape
    "w_ball_still": 0.5,     # ball seen lying still for a while (placed for a set piece)
    "w_out_latch": 1.0,      # the ball went out and the restart has not been struck yet: = exit, so it holds a stoppage alone
    # evidence thresholds
    "out_margin_m": 0.0,
    "out_seen_margin_m": 0.5, # a seen ball must be this far out, so tracking noise on the line is not a stoppage
    "long_gap_s": 5.0,
    "slow_speed": 1.6,       # m/s, median player speed
    "walking_share": 0.85,
    "ball_still_s": 1.5,     # seconds the ball must lie still
    # out-of-play latch (out_latch): starts once a seen ball has been beyond a line this long...
    "latch_min_out_s": 0.4,
    # ...and ends at the restart: a ball slower than set_speed for set_s, then faster than
    # strike_speed. Out over a goal line, the restart is taken on or inside the lines (goal kick,
    # corner), so a ball thrown back to the keeper from behind the goal does not count.
    "set_speed": 3.5,
    "set_s": 0.6,
    "strike_speed": 5.0,
    "inside_margin_m": 0.5,
    "reappear_gap_s": 1.0,   # a ball back in sight after this long, already moving, is also a restart
    "latch_max_s": 30.0,
    # Smart Tagging set pieces: the restart is the first strike (or moving reappearance) this soon
    # after a tag starts; a stoppage ending within tag_slack_s of the tag is moved to end there,
    # and with no stoppage at all one is added, back to the last moment the ball was in play
    "tag_search_s": 10.0,
    "tag_slack_s": 5.0,
    "tag_insert": True,
    "tag_insert_max_s": 30.0,
    # decision
    "enter": 2.0,            # score to switch live -> dead
    "exit": 1.5,             # score to switch dead -> live
    "min_dead_s": 5.0,
    "min_live_s": 1.0,       # short, so two stoppages either side of a quick restart stay apart
    "no_players": 10,        # fewer tracked players than this = nobody on the pitch = dead
    # edge snapping: move each stoppage's start to where the ball vanished or came to rest, and
    # its end to where it came back or was struck, if that happened within these windows
    "snap_start_back_s": 3.0,
    "snap_start_fwd_s": 1.0,
    "snap_end_back_s": 4.0,
    "snap_end_fwd_s": 0.5,
    "min_seen_s": 0.4,       # a reappearance only counts if the ball then stays seen this long
}


def score(f, p=PARAMS, latch=None):
    s = np.zeros(len(f["ball_missing"]))
    s += p["w_out"] * (f["ball_out_m"] > p["out_margin_m"]) * f["ball_missing"]
    s += p["w_long_gap"] * (f["ball_gap_s"] >= p["long_gap_s"])
    s += p["w_slow"] * np.nan_to_num(f["player_speed"] < p["slow_speed"])
    s += p["w_walking"] * np.nan_to_num(f["walking_share"] > p["walking_share"])
    s += p["w_kickoff"] * f["kickoff_shape"]
    s += p["w_out_seen"] * (f["ball_out_m"] > p["out_seen_margin_m"]) * ~f["ball_missing"]
    s += p["w_ball_still"] * (f["ball_still_s"] >= p["ball_still_s"])
    if "ball_speed_smooth" in f:
        s += p["w_out_latch"] * (out_latch(f, p)[0] if latch is None else latch)
    s[f["n_players"] < p["no_players"]] = 99.0
    return s


def out_latch(f, p=PARAMS, fps=25):
    """(latched, restarts): True from a seen ball going out until the restart is struck, and the
    frames where a latch ended at a restart.

    Without this a stoppage dies as soon as the ball is carried back onto the pitch, because the
    "out" evidence is gone and players are often still jogging back.
    """
    n = len(f["ball_missing"])
    missing, out_m = f["ball_missing"], f["ball_out_m"]
    speed = f["ball_speed_smooth"]
    goal_line = f["over_goal_line"]
    out_seen = ~missing & (out_m > p["out_seen_margin_m"])
    min_out, set_len = int(p["latch_min_out_s"] * fps), int(p["set_s"] * fps)
    max_len, gap_len = int(p["latch_max_s"] * fps), int(p["reappear_gap_s"] * fps)
    look = accel_len = int(0.6 * fps)  # a kick takes this long to get through set -> strike speed

    latched = np.zeros(n, dtype=bool)
    restarts = []
    on, since, over_goal, out_run, slow_run, slow_from, gap = False, 0, False, 0, 0, 0, 0
    for i in range(n):
        out_run = out_run + 1 if out_seen[i] else 0
        if out_run >= min_out:
            if not on:
                on, since, slow_run = True, i - out_run + 1, 0
                latched[since:i] = True
            over_goal = bool(goal_line[i])
        if not on:
            continue
        ended = i - since > max_len
        if missing[i]:
            gap += 1
        else:
            if gap >= gap_len:  # back in sight after a blank: in play already if it is moving
                ahead = speed[i:i + look]
                ahead = ahead[~np.isnan(ahead)]
                inside = out_m[i] <= p["inside_margin_m"] or not over_goal
                if len(ahead) and ahead.max() > p["strike_speed"] and inside:
                    ended = True
                    restarts.append(i)
            gap = 0
            v = speed[i]
            if np.isnan(v):
                pass
            elif v < p["set_speed"]:
                if slow_run == 0 or i - (slow_from + slow_run) > accel_len:
                    slow_from, slow_run = i, 0
                slow_run = i - slow_from + 1
            elif i - (slow_from + slow_run) > accel_len:
                slow_run = 0  # moving for a while: whatever lay set before is gone
            elif v > p["strike_speed"] and slow_run >= set_len and not ended:
                strike = slow_from + slow_run  # first frame after the ball lay set
                if not over_goal or out_m[strike - 1] <= p["inside_margin_m"]:
                    ended = True
                    restarts.append(strike)
                slow_run = 0
        if ended:
            on = False
            end = restarts[-1] if restarts and restarts[-1] >= since else i
            latched[end:i + 1] = False
            continue
        latched[i] = True
    return latched, np.array(restarts, dtype=int)


def hysteresis(s, enter, exit_):
    dead = np.zeros(len(s), dtype=bool)
    state = False
    for i, value in enumerate(s):
        if state and value < exit_:
            state = False
        elif not state and value >= enter:
            state = True
        dead[i] = state
    return dead


def drop_short_runs(mask, value, min_len):
    """Flip runs of `value` shorter than min_len frames to the other value."""
    out = mask.copy()
    edges = np.diff(np.concatenate(([0], (mask == value).astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start < min_len:
            out[start:stop] = not value
    return out


def _nearest(candidates, target, back, fwd):
    """The candidate frame closest to target within [target - back, target + fwd], or target."""
    lo, hi = np.searchsorted(candidates, [target - back, target + fwd + 1])
    if lo == hi:
        return target
    window = candidates[lo:hi]
    return int(window[np.argmin(np.abs(window - target))])


def snap_edges(dead, ball_missing, fps, p=PARAMS, ball_still=None, latch_restarts=None):
    """Move stoppage edges onto ball events: it vanishes or comes to rest (start), it lastingly
    reappears or is struck after resting (end).

    The score needs a few seconds of evidence, so its raw edges lag the real stoppage; the ball
    track is what moves first. Edges with no ball event nearby are left where they are.
    """
    edges = np.diff(np.concatenate(([0], ball_missing.astype(np.int8), [0])))
    vanish = np.flatnonzero(edges == 1)
    back = np.flatnonzero(edges == -1)
    back = back[back < len(ball_missing)]  # a gap running to the last frame never comes back
    seen_run = int(p["min_seen_s"] * fps)
    if seen_run > 1:
        seen = ~ball_missing
        lasting = np.convolve(seen.astype(int), np.ones(seen_run, dtype=int), "full")[seen_run - 1:]
        back = back[lasting[back] >= seen_run]
    if latch_restarts is not None and len(latch_restarts):
        back = np.union1d(back, latch_restarts)
    if ball_still is not None:
        still_edges = np.diff(np.concatenate(([0], ball_still.astype(np.int8), [0])))
        rest, struck = np.flatnonzero(still_edges == 1), np.flatnonzero(still_edges == -1)
        keep = (struck - rest) >= int(p["ball_still_s"] * fps)
        vanish = np.union1d(vanish, rest[keep])
        back = np.union1d(back, struck[keep][struck[keep] < len(ball_missing)])

    out = np.zeros_like(dead)
    d_edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(d_edges == 1), np.flatnonzero(d_edges == -1)):
        new_start = _nearest(vanish, start, int(p["snap_start_back_s"] * fps),
                             int(p["snap_start_fwd_s"] * fps))
        new_stop = _nearest(back, stop, int(p["snap_end_back_s"] * fps),
                            int(p["snap_end_fwd_s"] * fps))
        if new_stop > new_start:
            out[new_start:new_stop] = True
    return out


def detect(f, fps, p=PARAMS, tags=None):
    """Boolean per frame: True where the ball is dead. `tags`: Smart Tagging set pieces as
    [(start_frame, end_frame)], when the match has them."""
    latch, latch_restarts = out_latch(f, p, fps) if "ball_speed_smooth" in f else (None, None)
    dead = hysteresis(score(f, p, latch), p["enter"], p["exit"])
    dead = drop_short_runs(dead, True, int(p["min_dead_s"] * fps))
    dead = drop_short_runs(dead, False, int(p["min_live_s"] * fps))
    dead = snap_edges(dead, f["ball_missing"], fps, p, f.get("ball_still"), latch_restarts)
    if tags:
        dead = apply_tags(dead, f, tags, fps, p)
    return drop_short_runs(dead, True, int(p["min_dead_s"] * fps))


def set_strikes(f, p=PARAMS, fps=25):
    """Frames where a ball that lay set (slower than set_speed for set_s) is struck (faster than
    strike_speed within the next 0.6 s), plus frames where it comes back in sight already moving
    after a blank of reappear_gap_s. Candidate restarts, wherever they happen."""
    speed, missing = f["ball_speed_smooth"], f["ball_missing"]
    set_len, gap_len = int(p["set_s"] * fps), int(p["reappear_gap_s"] * fps)
    accel_len = int(0.6 * fps)
    out, slow_from, slow_run, gap = [], 0, 0, 0
    for i in range(len(speed)):
        if missing[i]:
            gap += 1
            continue
        if gap >= gap_len:
            ahead = speed[i:i + accel_len]
            ahead = ahead[~np.isnan(ahead)]
            if len(ahead) and ahead.max() > p["strike_speed"]:
                out.append(i)
        gap = 0
        v = speed[i]
        if np.isnan(v):
            continue
        if v < p["set_speed"]:
            if slow_run == 0 or i - (slow_from + slow_run) > accel_len:
                slow_from = i
            slow_run = i - slow_from + 1
        elif i - (slow_from + slow_run) > accel_len:
            slow_run = 0
        elif v > p["strike_speed"] and slow_run >= set_len:
            out.append(slow_from + slow_run)
            slow_run = 0
    return np.array(sorted(set(out)), dtype=int)


def apply_tags(dead, f, tags, fps, p=PARAMS):
    """Anchor stoppage ends on Smart Tagging set pieces ([(start_frame, end_frame)]).

    Tags are automatic and noisy (the starter says so), so they only decide *where* a stoppage
    ends, using a restart seen in the tracking; a tag with no strike or reappearance near its start
    changes nothing.
    """
    dead = dead.copy()
    strikes = set_strikes(f, p, fps)
    seen_fast = np.flatnonzero(~f["ball_missing"] & (np.nan_to_num(f["ball_speed_smooth"]) > p["strike_speed"]))
    search, slack = int(p["tag_search_s"] * fps), int(p["tag_slack_s"] * fps)
    min_dead, max_back = int(p["min_dead_s"] * fps), int(p["tag_insert_max_s"] * fps)
    for start, _ in sorted(tags):
        lo, hi = np.searchsorted(strikes, [start, start + search])
        if lo == hi:
            continue
        restart = int(strikes[lo])
        edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        near = np.flatnonzero((stops >= start - slack) & (starts <= restart + slack))
        if len(near):
            k = near[np.argmin(np.abs(stops[near] - restart))]
            if starts[k] < restart:
                dead[starts[k]:restart] = True
                dead[restart:stops[k]] = False
            continue
        if not p["tag_insert"]:
            continue
        before = seen_fast[(seen_fast < restart - min_dead) & (seen_fast >= restart - max_back)]
        # the last moment the ball was seen in play; with none, the stoppage ran the whole time
        begin = int(before[-1]) + 1 if len(before) else restart - max_back
        dead[max(0, begin):restart] = True
    return dead
