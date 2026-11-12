# deadball

Role 1 of team Rondo's More than a Hack 2027 entry. For every frame of the match, is the ball in
play or dead? The output feeds the `D` label of the possession file.

The challenge's ground truth marks `D` wherever Metrica's *corrected tracking* calls the ball dead,
so this works from tracking alone: where the ball was last seen, how long it has been missing,
how fast the players are moving, and whether the teams are lined up for a kick-off.

## Setup

Use the environment from the repo root: Python 3.9+ and
`requirements-rondo.txt` run everything here, and the pipeline imports this folder directly
(`rondo/dead_ball.py`). `requirements.txt` adds `kloppy`, which only the DFL loader needs.

This folder lives inside the team repo, which has the challenge files. The code reuses its
`games.py` and the scorer in `evaluation/possession_metrics.py`.

## Data (none of it is committed)

| folder | what | how to get it |
|---|---|---|
| `data/metrica/` | Metrica Sample Games 1-3: tracking and events | `python get_metrica.py` |
| `data/dfl/` | DFL open data: tracking with a true alive/dead flag | Figshare blocks scripted downloads. Open the dataset in a browser, save `<id>_tracking.xml` and `<id>_meta.xml` here, e.g. `J03WMX` and `J03WN1` |
| challenge `data/` | BAR-ATH bundle (only `run.py match` reads it) | enrol, then follow the challenge's DATA.md |

Metrica's sample data is open ("be responsible with the use of this data" and credit Metrica if
used publicly): https://github.com/metrica-sports/sample-data

## Run

```bash
.venv/bin/python run.py samples    # score on Metrica sample games (labels from events)
.venv/bin/python run.py dfl        # score on DFL open data, once downloaded
.venv/bin/python run.py match      # BAR-ATH -> results/deadball_BAR-ATH.csv (frame,label D/L)
.venv/bin/python tune.py           # re-tune every parameter on the sample games (~15 min)
```

The pipeline (`python -m rondo.pipeline` from the repo root) runs the same detector through
`rondo/dead_ball.py`, so `run.py match` is only needed to look at this part on its own.

## Viewer

```bash
.venv/bin/python run.py view                    # export the sample games, open http://127.0.0.1:8765
.venv/bin/python run.py view --game metrica_3   # one game
.venv/bin/python run.py view --dead-visible 1.0 # stress test: ball kept on screen in stoppages
```

The viewer is for the sample games only. It puts our predictions next to the truth, and the
organisers' rules clarification (2026-11-06) forbids checking our output against the challenge
match's video or ground truth, so `run.py view` no longer exports BAR-ATH or serves its video.
Exports stay in `viewer/data/` on your own machine: they are gitignored and must never be
committed.

It's a single-screen dashboard with no page scrolling on a laptop or bigger (1280×800 and up). On
smaller windows it falls back to a scrolling page.

- **Top bar:** game picker, the scenario being shown, and the headline numbers. Those are stoppage
  F1, dead F1, true and predicted stoppages, median start/end error, and missed / invented / edge.
- **Pitch:** CV-style detection boxes with ID tags. Home is blue (`H01`…), away is white
  (`A01`…), plus the speed in m/s. The ball and the player nearest it are amber
  (`A08 · NEAREST 3.9m`). IDs are per-team track numbers, not shirt numbers, because the challenge
  tracking has no identities. A dashed `BALL · LOST` box is a ball the dropout hid. There's a 5 m
  grid with axis ticks, and hovering shows pitch coordinates in the status bar. While the detector
  says dead, the pitch gets a red hatch and frame and a `DEAD BALL T+…s` banner showing whether
  the truth agrees.
- **Camera:** switched off. The sample games have no video, and the challenge match is not
  shown (see above).
- **Detector state:** truth and prediction at the playhead, plus the evidence behind the score.
- **Timeline:** whole-match truth / predicted / error rows (click to seek), and the score for ±60 s
  around the playhead against the enter/exit thresholds.
- **Role 1 checklist:** each part of the role with a status (met / partly / to do / n/a) measured
  on that game. It covers ball out of play, fouls, kick-offs, other stoppages, timing, the stress
  test, Smart Tagging, the 2nd-half pre-kickoff gap, the rules, and the per-frame output. The
  thresholds are in `deadball/requirements.py`. Hover a row for what it checks.
- **Errors:** every disagreement, filterable by type. Click one to jump to 2 s before it.

`TAGS` turns the labels off. Keys: `space` play/pause, `←` `→` 5 s, `,` `.` one frame, `n` / `p` next or previous error,
`Esc` closes the enlarged camera. Run `scratch/stress.py` once to fill the checklist's stress row.

`samples` and `dfl` add ball dropout in bursts by default, so about 30% of live frames have no ball,
as automatic tracking loses it. `--no-dropout` scores on the clean ball track.

## Where it stands (2026-11-12)

Every parameter is tuned on Metrica's three sample games and nothing else. That's what the
organisers' rules clarification of 2026-11-06 requires: no training, fitting or calibrating on the
challenge match's ground truth or video, and every tuned value from external data.

On the sample games, with 30% live-play dropout (seed 0, `run.py samples`):

| game | stoppage F1 | stoppages true / predicted | baseline rule ("ball missing = dead") |
|---|---|---|---|
| metrica_1 | 0.899 | 100 / 94 | 0.271 |
| metrica_2 | 0.824 | 100 / 87 | 0.277 |
| metrica_3 | 0.856 | 101 / 86 | 0.255 |

Over the 10 stress scenarios `tune.py` uses (below), the mean stoppage F1 is 0.778 and the worst
scenario is 0.737 (settings after both re-tunes below).

Stoppage F1 is the challenge's own spellF1 code, applied to dead-ball runs: stoppages are matched by
temporal IoU and the F1 is averaged over IoU thresholds 0.3-0.7.

### Stress test: the ball stays visible during stoppages

Metrica's sample tracking blanks the ball for the whole of a stoppage, so "ball missing" is nearly
the answer there. Automatic tracking may well keep the ball on screen while it's held for a
throw-in or lying by the flag. `--dead-visible` simulates that. The ball lies where it went dead,
is carried to the restart spot, then sits there, with 30 cm of tracking jitter. The flag gives the
share of stoppage frames that keep the ball, and the rest are blanked in bursts. `--beyond-line`
also puts balls out for throws, goal kicks and corners past the line (0.5-3 m for a throw, 1-8 m
behind the goal) and fetches them back early.

Two pieces of evidence handle a visible ball: a seen ball lying beyond the lines, and a ball lying
still (moved < 1 m in 1 s, after 0.4 s smoothing) for 1.5 s or more. Stoppage edges also snap to the
ball coming to rest and being struck. With the ball visible, more errors are "edge off", because
"ball comes to rest" is a fuzzier moment than "ball vanishes".

**Out-of-play latch** (`rules.out_latch`). Once a seen ball has been past a line for 0.4 s, the
stoppage holds until the restart is struck: the ball lies set (under 3.5 m/s for 0.6 s), then goes
over 5 m/s. Past a goal line, the strike has to come from on or inside the lines, so a ball thrown
back from behind the goal doesn't count. A throw-in may come from outside. It's capped at 30 s, and
a ball that comes back in sight already moving also ends it.

**Smart Tagging anchors** (`rules.apply_tags`). The restart for a `SET PIECES` tag is the first
strike (or moving reappearance) within 10 s of the tag's start. A stoppage ending within 5 s of the
tag is moved to end there. If there's no stoppage, one is added, back to the last moment the ball
was seen in play (at most 30 s). A tag with no strike nearby changes nothing. The sample games
have no tags, so `--tags` simulates them from the restart events (`scratch/tags_insert.py`). With
simulated tags, mean stoppage F1 over 5 visibility scenarios goes from 0.751 to 0.774 (2026-11-12).

The latch and the tag anchors were first designed in late October after looking at the public 5
minutes of the challenge match, before the rules clarification. Looking at the provided data to
understand it is allowed; the values are not taken from it. The stress-test settings that had been
taken from the challenge match were replaced with generic ones, and every value was re-tuned on the
sample games alone (next section).

### Re-tune on the sample games only (2026-11-12)

```bash
.venv/bin/python tune.py
```

10 scenarios on all three sample games: ball visible in stoppages 0% / 50% / 100%, the visible ones
with and without balls lying past the line, times live-play dropout 20% / 40%. Three stages, in the
order they were first tuned: base rules, stoppage evidence, out-of-play latch. The rule was fixed
before running: the highest mean stoppage F1 over the 10 scenarios wins, anything within 0.002 of
the best is a tie, and a tie keeps the current value.

Result: only `enter` changed (1.5 -> 2.0, mean 0.731 -> 0.751). The latch weight stays at 1.0, now
chosen on the sample games alone.

### Re-tune for the ranked metric (2026-11-12, `rondo/tune_dead_ball.py`)

What is ranked is spellF1 of the whole file, not stoppage F1. So the settings were tuned again by
running the whole pipeline (this detector, then the v3 possession model) on 60 degraded sample runs
and scoring spellF1, one setting at a time, two passes, with the same tie rule. It changed `w_slow`
0.5 -> 1.0, `w_ball_still` 1.0 -> 0.5, `exit` 1.0 -> 1.5, `min_dead_s` 3 -> 5 and the edge snapping
(`snap_start_fwd_s` 2 -> 1, `snap_end_back_s` 3 -> 4, `snap_end_fwd_s` 1 -> 0.5). Pipeline spellF1
0.576 -> 0.593, and stoppage F1 over the 10 scenarios rose too: 0.751 -> 0.778, worst 0.636 -> 0.737.
Re-running `tune.py` on its own would undo this, so run `tune_dead_ball` after it. Re-run on top of
the v4 possession model and the official-definition labels, it changed nothing.

### DFL open data (independent check)

`python get_dfl.py` files the hand-downloaded DFL XMLs (Figshare blocks scripted downloads), and
`../.venv/bin/python -m rondo.check_dfl` (from the repo root) scores the whole pipeline on them
against the provider's own ball status. Never tuned on: stoppage F1 0.856 over 2 matches x 16 runs.

## Caveats

- The sample-game labels come from events: a stoppage runs from the last event before a restart
  (set piece or goal kick) to the restart. Any ball blank of 8 s or more in Metrica's original
  track is also dead, because Metrica blanks the ball through a stoppage. That catches injuries and
  untagged restarts.
- `run.py match` prints the public 5 minutes as a check that the code runs. It holds 4 stoppages,
  so it is not a score, and the rules forbid tuning on it.
- No constants tied to frames of BAR-ATH. The 2nd-half pre-kickoff gap should come out as dead from
  the kick-off shape and the empty or slow pitch.
- Edge-snapping windows were tuned with `scratch/sweep_edges.py` on the sample games at 30%
  dropout and were not re-run in the 2026-11-12 re-tune.

## Layout

```
deadball/loaders.py    Metrica EPTS (BAR-ATH + sample game 3), Metrica CSV, DFL -> Frames (metres)
deadball/labels.py     dead-ball labels from Metrica events
deadball/features.py   per-frame features
deadball/rules.py      score -> hysteresis -> minimum run lengths -> snap edges to ball vanish/return
deadball/evaluate.py   dropout simulation, stress scenarios, frame and stoppage scores
deadball/export.py     writes viewer/data/<game>.bin + .json for the viewer
deadball/requirements.py  Role 1 checklist shown in the viewer
viewer/index.html      the viewer (one file, no dependencies)
run.py                 samples / dfl / match / view
tune.py                the re-tune on the sample games
scratch/               edge sweep (sweep_edges.py), stress grid (stress.py) and error breakdowns on
                       the sample games (edges.py, losses.py, invented.py, tags_insert.py)
```

## Next

1. Get the DFL matches and check the parameters on them, especially the latch weight.
2. Replace the hand-weighted score with a small classifier trained on DFL + sample-game labels.
