"""Per-frame features that say "the ball is dead", computed the same way on every source.

All inputs come from `loaders.Frames` (metres, centred, NaN = missing). Everything is a plain
numpy array of length n, so new features can be added without touching the rules.
"""

import warnings

import numpy as np

# Params
MAX_PLAYER_SPEED = 12.0   # m/s; faster than this is a tracker identity switch, not a sprint
WALK_SPEED = 2.0          # m/s; below this a player is standing or walking
KICKOFF_OWN_HALF = 0.9    # share of each team that must be in its own half for a kick-off shape
CENTRE_RADIUS = 4.0       # m; ball this close to the centre spot counts as "on the spot"
SPEED_WINDOW_S = 1.0      # rolling mean window for player speed
STILL_WINDOW_S = 1.0      # the ball counts as still if it moved less than STILL_M over this window
STILL_M = 1.0
STILL_SMOOTH_S = 0.4      # ball position is averaged over this long first, to ride out jitter
STRIKE_WINDOW_S = 0.4     # window of the smoothed ball speed


def rolling_mean(values, window):
    """Centred NaN-aware rolling mean over `window` frames."""
    if window <= 1:
        return values.astype(float)
    ok = ~np.isnan(values)
    filled = np.where(ok, values, 0.0)
    kernel = np.ones(int(window))
    total = np.convolve(filled, kernel, mode="same")
    count = np.convolve(ok.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def run_lengths(mask):
    """For each frame, the length of the True run it sits in (0 where False)."""
    out = np.zeros(len(mask), dtype=int)
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        out[start:stop] = stop - start
    return out


def fill_forward(values):
    """Last non-NaN row at or before each frame (NaN before the first observation)."""
    ok = ~np.isnan(values[:, 0])
    index = np.maximum.accumulate(np.where(ok, np.arange(len(values)), -1))
    out = np.full_like(values, np.nan)
    out[index >= 0] = values[index[index >= 0]]
    return out


def compute(frames):
    fps = frames.fps
    length, width = frames.pitch
    ball, players = frames.ball, frames.players

    ball_missing = np.isnan(ball[:, 0])
    last_ball = fill_forward(ball)

    # How far the last seen ball is beyond the lines (negative = inside the pitch).
    out_distance = np.maximum(np.abs(last_ball[:, 0]) - length / 2,
                              np.abs(last_ball[:, 1]) - width / 2)

    # A still ball: seen now and a window ago, and barely moved. Live play almost never leaves the
    # ball lying for seconds; set pieces always do. Measured as displacement over a window rather
    # than per-frame speed, so tracker jitter does not break a run.
    lag = int(STILL_WINDOW_S * fps)
    smooth = np.stack([rolling_mean(ball[:, k], int(STILL_SMOOTH_S * fps)) for k in (0, 1)], 1)
    smooth[ball_missing] = np.nan
    moved = np.full(len(ball), np.nan)
    moved[lag:] = np.hypot(smooth[lag:, 0] - smooth[:-lag, 0], smooth[lag:, 1] - smooth[:-lag, 1])
    with np.errstate(invalid="ignore"):
        ball_still = moved < STILL_M

    # Speed of the smoothed ball over a short window: jitter-free enough to tell a ball rolled or
    # carried about from one that is struck. NaN where the ball is not seen at both ends.
    step_lag = int(STRIKE_WINDOW_S * fps)
    smooth_speed = np.full(len(ball), np.nan)
    smooth_speed[step_lag:] = np.hypot(smooth[step_lag:, 0] - smooth[:-step_lag, 0],
                                       smooth[step_lag:, 1] - smooth[:-step_lag, 1]) / STRIKE_WINDOW_S

    ball_velocity = np.diff(ball, axis=0, prepend=ball[:1]) * fps
    ball_speed = rolling_mean(np.hypot(ball_velocity[:, 0], ball_velocity[:, 1]), fps // 5)

    step = np.diff(players, axis=0, prepend=players[:1]) * fps
    speed = np.hypot(step[..., 0], step[..., 1])
    speed[speed > MAX_PLAYER_SPEED] = np.nan
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with nobody tracked -> NaN
        median_speed = np.nanmedian(speed, axis=1)
        walking_share = np.nanmean(np.where(np.isnan(speed), np.nan, speed < WALK_SPEED), axis=1)
    window = int(SPEED_WINDOW_S * fps)

    # Kick-off shape: each team almost entirely in one half, on opposite halves, ball near the
    # spot (or not seen). Which half belongs to whom flips at half-time, so only "opposite" matters.
    x = players[..., 0]
    team_side = []
    for t in (0, 1):
        tx = x[:, frames.team == t]
        seen = (~np.isnan(tx)).sum(1)
        with np.errstate(all="ignore"):
            right = (tx > 0).sum(1) / np.maximum(seen, 1)
        team_side.append(np.where(seen >= 7, right, np.nan))
    home_right, away_right = team_side
    separated = (((home_right >= KICKOFF_OWN_HALF) & (away_right <= 1 - KICKOFF_OWN_HALF)) |
                 ((away_right >= KICKOFF_OWN_HALF) & (home_right <= 1 - KICKOFF_OWN_HALF)))
    ball_on_spot = np.hypot(last_ball[:, 0], last_ball[:, 1]) < CENTRE_RADIUS
    kickoff_shape = separated & (ball_missing | ball_on_spot)

    return {
        "ball_missing": ball_missing,
        "ball_gap_s": run_lengths(ball_missing) / fps,
        "ball_out_m": out_distance,
        # which line the last seen ball is beyond: True = a goal line, False = a touchline
        "over_goal_line": (np.abs(last_ball[:, 0]) - length / 2) >= (np.abs(last_ball[:, 1]) - width / 2),
        "ball_speed_smooth": smooth_speed,
        "ball_speed": ball_speed,
        "ball_still": ball_still,
        "ball_still_s": run_lengths(ball_still) / fps,
        "player_speed": rolling_mean(median_speed, window),
        "walking_share": rolling_mean(walking_share, window),
        "kickoff_shape": kickoff_shape,
        "n_players": (~np.isnan(x)).sum(1),
    }
