# Team Rondo — More Than A Hack 2027, Phase 1

Every frame of Barça v Athletic labelled `H` / `A` / `D` / `X` by two models in series: a
**dead-ball detector** (is the ball in play?) and a **possession model** (who has it?). Both read
only the tactical tracking (players and ball) and, for dead ball, the Smart Tagging `SET PIECES`
tags. No video, no homography, nothing labelled by hand.

**Every tuned or learned value comes from external data:** Metrica's open sample games, with two DFL
open-data matches as a second check. Nothing was fitted, tuned or chosen on the challenge match,
its ground truth or its video, as the organisers' rules clarification of 6 November requires.
Section 6 says exactly how, including what we looked at before that clarification.

| | spellF1 | macroF1 |
|---|---|---|
| Metrica sample games, each scored by a model that never saw it, 48 degraded runs (Section 7) | 0.660 | — |
| DFL open data, 2 matches never trained or tuned on, 32 degraded runs (Section 7) | 0.492 | — |
| Public 5 minutes (a code check, not a target) | 0.489 | 0.852 |
| `baseline_smooth50` on the same 5 minutes | 0.326 | 0.621 |

Roles: dead ball, Sukh; who has the ball, David and Nick; pipeline, checks and write-up, Nick.

---

## 1. Data

**From the challenge:** `BAR-ATH_tactical_atd` (players and ball, 25 fps, automatic, 46 track
slots with a team each, ball missing on 41% of graded frames) and `BAR-ATH_tactical_smart_tagging`
(only `SET PIECES`, used as dead-ball evidence). Not used: the videos, the broadcast tracking, the
homography, the other tags (tested in Section 4, not adopted).

**External, for every tuned or learned value:** Metrica's open sample games 1–3
(github.com/metrica-sports/sample-data: tracking and event data for three full matches). Metrica
asks users to "be responsible with the use of this data" and to acknowledge the source: thank you.
From the events we build the two labels we need:

- *dead ball*: a stoppage runs from the last event before a restart (set piece, goal kick) to the
  restart; any ball blank of 8 s or more is also dead (`deadball/labels.py`)
- *possession*: on live frames, the team of the latest on-ball event (pass, carry, shot, set piece,
  recovery, ball lost; challenges are skipped because both teams are in them), **under the official
  definition of a change**: CHALLENGE.md says possession changes only when the other team gains
  *controlled* possession, and that "a simple touch of the ball without control", "a deflection of
  a pass that does not intercept it" and "an aerial duel won without subsequent control" are not
  changes. So a lone pass or lost ball by one team between two events of the other, inside one
  stretch of live play (a clearance, a header, a deflection, a miscontrol), does not change
  possession. That removes 6–10% of the event-based possessions (`rondo/train_possession.py`).

**External, as a second check:** two matches of the DFL / Sportec open data (Bassek, Rein, Weber &
Memmert 2025, *Scientific Data*, CC BY 4.0), whose tracking carries the provider's own ball status
and possession flag on every frame. That is the same kind of label as the challenge's ground truth
("derived from corrected tracking data"), from a different provider. Nothing is fitted on it; it is
used to score, and in one fixed rule to choose between possession variants (Section 6).

**Making the external games look like our match.** Their tracking is clean, ours is not. Every
external game is degraded before anything is tuned, trained or scored on it, in every combination
of:

- live-play ball dropout of 20% and 40%, in bursts
- the ball blanked during stoppages, or kept on screen (50% / 100%), optionally lying past the line
  for throw-ins, goal kicks and corners, with 30 cm of jitter
- tracking: clean; ball jitter of 0.5 m; player jitter of 0.2 m with 15% of player positions lost
  in bursts of about 2 s; both

These levels are generic, not measured on the challenge match.

## 2. Dead ball (`deadball/`, `rondo/dead_ball.py`)

A rule-based detector built by Sukh. Per frame it adds up weighted evidence that play has stopped:

- the ball was last seen past a line and is now missing, or is seen lying past a line
- the ball is missing for a long stretch, or lies still for 1.5 s (placed for a set piece)
- players barely moving, or nearly everyone walking; a kick-off shape; fewer than 10 players tracked
- an **out-of-play latch**: once a seen ball has been past a line for 0.4 s, the stoppage holds until
  the restart is struck (the ball lies set, then moves fast, from on or inside the lines for goal
  kicks and corners), capped at 30 s
- **Smart Tagging anchors**: the restart of a `SET PIECES` tag is the first strike within 10 s of its
  start; a nearby stoppage is moved to end there, or one is added if there is none

The score then goes through hysteresis (enter / exit thresholds), minimum run lengths, and edge
snapping (a stoppage starts where the ball vanished or came to rest and ends where it reappeared or
was struck). Every threshold is in metres or seconds, never a frame number. The settings are tuned
for the ranked metric (spellF1 of the whole file), not just for finding stoppages (Section 6).

## 3. Possession (`rondo/possession.py`, `rondo/train_possession.py`)

1. **Clean the ball track.** Average each seen position with its neighbours (5 frames) to remove
   jitter, and fill gaps of up to 1 s inside live play with a straight line.
2. **Score the scene around the ball.** Each frame falls into one of 540 scenes: distance from the
   ball to the nearest player, how much closer that player is than the nearest opponent, ball speed,
   ball seen or filled in. A table counted on the external games turns each scene into log-odds that
   the nearest player's team has the ball. A slow ball at a player's feet with no opponent close is
   strong evidence; a fast ball flying past someone is almost none.
3. **Add context.** That evidence summed over windows before and after the frame (1 frame to 10 s);
   the team of the last and of the next strong piece of evidence; and the team of the last and next
   **touch** (the ball's velocity changes by 5–10 m/s right next to a player, measured over 3–6
   frames so tracking noise does not fire it). Each fades with time (0.2 s, 1 s, 5 s).
4. **Touch sequence.** The team of the last two and the next two touches, and touches per team in
   the 0.5, 1 and 3 s either side. A lone touch by one team between two of the other's looks like a
   deflection; two in a row looks like control.
5. **Combine.** A logistic regression with no intercept (so neither team is favoured) turns these 81
   features into one number per frame: the log-odds that Barça has the ball. In training each frame
   is weighted 1 / √(length of its possession), because spellF1 counts a 1 s possession exactly as
   much as a 30 s one; with equal frame weights the long possessions drown the short ones out.
6. **Decode.** For each stretch of live play a two-state model (H / A) picks the labelling that
   agrees most with those numbers, paying a fixed cost for every change of possession (Viterbi).
   That is what makes a deflection or a pass flying past an opponent not a change, while a real
   interception is one.
7. **Abstain** (`X`) only on a stretch of play with no ball information at all. Inside a possession
   the scorer carries the previous answer through an `X`, but at a hand-over an `X` carries the old
   team across, so we always guess when there is any evidence. On the match, 260 of 158,857 graded
   frames (0.2%) are `X`.

The scene table and the weights are counted and fitted (there is nothing to choose); the switch
cost is chosen (Section 6). All of it is in `rondo/possession_model.json` (81 weights, a 541-entry
table, one cost).

**Why this design.** On the sample games our dead ball plus a perfect possession model scores 0.908,
so possession was where the points were. Our rule-based model (v2: "nearest player within 3 m is
control" plus the same Viterbi) found only 24% of the real changes of possession within 1 s, and
missed 95% of possessions shorter than 2 s, which are over a fifth of all possessions in the sample
games and do occur in the public 5 minutes. v3 added steps 1–3 (learned scene evidence, context,
touches); v4 added the official definition of a change, the touch sequence and the per-possession
weighting.

### Where we drew the line

The model is learned, but the labels it learns from, and the decoder, encode these rules:

- **Loose balls** stay with the last team in control. The two-state decoder never leaves "who has
  it" undecided, and a change has to pay the switch cost, so a ball nobody controls (in the air,
  bouncing, rolling between players) keeps the previous team.
- **Touches without control** are not changes: a clearance, a header, a deflection of a pass that
  is not an interception, a miscontrol. In the training labels, a lone pass or lost ball by one team
  between two events of the other is given to the team already in possession; in the model, the
  touch-sequence features tell a lone touch from two in a row.
- **Challenges and duels** are not changes by themselves (the labels skip challenge events, since
  both teams are in them). Possession changes when the winner's team then controls the ball.
- **Interceptions and tackles that keep the ball** are changes, placed at the first controlled touch
  of the new team.
- **Restarts**: dead ball is `D` and has no owner. Each stretch of live play is decided afresh, so
  the team taking the restart gets it from its own first touches, not from who had the ball before
  the stoppage.
- **Abstain** only where a stretch of play has no ball information at all (0.2% of graded frames).

## 4. Short possessions, and Smart Tagging (what we tried)

**How short is a possession?** The answer keys differ:

| | possessions per live minute | under 2 s |
|---|---|---|
| Metrica sample games, every on-ball event | 6.2–7.1 | 25–30% |
| Metrica sample games, official "controlled" definition (what we train on) | 5.9–6.7 | 21–24% |
| DFL open data, provider's possession flag | 9.0–11.0 | 34–37% |
| Public 5 minutes (17 possessions, too few to calibrate on, and we did not) | 4.2 | 12% |

So short possessions are common and count fully in spellF1. v3 found only 16% of those under 2 s (at
IoU 0.5). We tested, with the rule fixed before running (Section 6), the official definition alone,
two training weightings (1/√length, 1/length) and the touch-sequence features, alone and together:

| variant (each game scored by a model that never saw it) | Metrica | DFL | 5-game mean |
|---|---|---|---|
| v3 (every touch, equal frame weights) | 0.609* | 0.464 | 0.551 |
| official definition | 0.644 | 0.472 | 0.575 |
| + weight 1/√length | 0.655 | 0.491 | 0.589 |
| + weight 1/length | 0.636 | 0.503 | 0.583 |
| + touch sequence | 0.648 | 0.469 | 0.576 |
| **+ 1/√length + touch sequence (v4)** | **0.661** | **0.492** | **0.593** |
| + 1/length + touch sequence | 0.638 | 0.503 | 0.584 |

\*scored against the every-touch labels; the other Metrica numbers against the official definition.
The DFL labels are the same in every row, so that column compares like with like: +0.028.

**Smart Tagging team phases.** Every phase carries a team (`BUILD UP`, `PROGRESSION`,
`DEFENDING IN MIDDLE THIRD`, ...), which could be evidence of who has the ball. The rules require any
weight to be tuned on external data, and no external game has Smart Tagging, so we simulated it: from
the external labels, a phase tag over each possession of 2 s or more, with a share of possessions
untagged, a share with the wrong team, jittered edges, and one in five covering two possessions. The
tag weight was learned with *poor* simulated tags; the rule, fixed before running, was to adopt the
tags only if they helped on held-out games at every quality:

| simulated tags (wrong team / tagged / edge jitter) | Metrica | DFL |
|---|---|---|
| none (v4) | 0.6605 | 0.4921 |
| good (10% / 80% / 1 s) | 0.6686 | 0.5026 |
| poor (30% / 50% / 3 s) | 0.6605 | 0.4929 |
| very poor (40% / 40% / 4 s) | 0.6602 | 0.4919 |

Tags help only if they are good, and do nothing at "poor", so they are not used. (To read the codes
the right way round we checked, on aggregate only, that "with the ball" codes agree with our
tracking-only answer on 71% of their live frames and "defending" codes on 82%; nothing was tuned on
that.)

## 5. Combining them (`rondo/pipeline.py`)

`D` wherever the ball is dead, otherwise the possession answer; `X` outside the graded halves. The
output is validated with the organisers' `validate_submission.py`. Two runs give a byte-identical
file; a fresh copy of this repository in a fresh environment reproduces it byte for byte; and
retraining the possession model from scratch reproduces `possession_model.json` byte for byte.

## 6. How every value was chosen (and what we did not do)

The rules clarification (6 November) forbids checking output against the video, hand labels on the
match, uploading the match data to AI services, and training, fitting or calibrating on the
ground-truth possession data or the video; anything tuned must be tuned on external data. We
followed that as follows.

- **External data only.** Every tuned or learned value comes from the degraded external games.
- **Selection rules fixed before running.** Each script states its rule in its header, written
  before the run. Anything within 0.002 of the best is a tie.
  - `deadball/tune.py`: the detector's settings for its own stoppage F1, 10 scenarios; a tie keeps
    the current value.
  - `rondo/tune_dead_ball.py`: the same settings for the ranked metric, by running the whole
    pipeline on 60 degraded sample runs and scoring spellF1; one setting at a time, two passes; a
    tie keeps the current value. The first run moved 7 settings
    (mean spellF1 0.576 to 0.593, every game better), and the detector's own stoppage F1 rose with
    it (0.751 to 0.778, worst scenario 0.636 to 0.737). Re-run on top of v4 and the
    official-definition labels, it changed nothing.
  - `rondo/train_possession.py`: the switch cost by leave-one-game-out (fit on two games, score the
    third), so the reported score is for games the model never saw; a tie goes to the higher cost.
  - v4's variant (Section 4): highest mean over 5 games (3 Metrica, each scored by a model fitted on
    the other two; 2 DFL, scored by the model fitted on all three), games weighted equally; a tie
    goes to the simplest.
  - Smart Tagging (Section 4): adopt only if better at every simulated quality. It was not.
  - Training on DFL as well: adopt only if the 5-game mean rises by more than 0.002. It did not
    (0.551 vs 0.551), so nothing is fitted on DFL.
- **The public 5 minutes** were used to check the code runs and the file is right, never to pick a
  value or a version. Twice a change chosen on external data lowered the public score, and we kept
  it both times: the dead-ball re-tune after the clarification (stoppage F1 0.800 to 0.667), and v4
  (spellF1 0.526 to 0.489, about one possession of 17).
- **Honest note on what came before the clarification.** In late October, before the clarification,
  version 1 had two settings partly picked on the public 5 minutes, and the out-of-play latch and the
  Smart Tagging anchors were designed after looking at those minutes. Looking at the data to
  understand it is allowed. The values are not taken from it. Every one was re-tuned on the sample
  games after 6 November, and the stress-test settings that had been measured on the challenge match
  were replaced with generic ones.
- **No match data left our laptops.** The match data was never given to an AI service. Only aggregate
  numbers (scores, counts) were ever printed or shared. This repository has none of it: no
  tracking, no tags, no exports, no submission file.

## 7. Results

**Sample games** (spellF1 of the whole pipeline against the official-definition labels, each game
scored by a model fitted on the other two; 3 games × 4 dead-ball scenarios × 4 tracking conditions
= 48 runs; every column with the same, final dead-ball detector):

| tracking | v2 (rule) | v3 | v4 |
|---|---|---|---|
| clean | 0.533 | 0.664 | **0.701** |
| ball jitter | 0.529 | 0.665 | **0.684** |
| player jitter and loss | 0.475 | 0.624 | **0.638** |
| both | 0.472 | 0.614 | **0.619** |
| **mean** (worst run) | 0.502 (0.428) | 0.642 (0.540) | **0.660** (0.539) |

v2's settings were picked on all three games, so if anything the comparison flatters v2.

**Dead ball on the sample games**: stoppage F1 0.778 (mean of 10 stress scenarios, worst
0.737), against 0.25–0.28 for the challenge baseline's "ball missing = dead" rule.

**DFL open data** (another provider, never trained or tuned on; labels are the provider's own ball
status and possession flag; 2 matches × ball in stoppages as tracked or blanked × dropout 20% / 40%
× the 4 tracking conditions = 32 runs; v3 and v4 fitted on all three sample games):

| tracking | v2 | v3 | v4 |
|---|---|---|---|
| clean | 0.380 | 0.469 | **0.510** |
| ball jitter | 0.370 | 0.482 | **0.508** |
| player jitter and loss | 0.340 | 0.453 | **0.476** |
| both | 0.344 | 0.450 | **0.475** |
| **mean** (worst run) | 0.358 (0.289) | 0.463 (0.406) | **0.492** (0.428) |

With a perfect dead ball v4 would score 0.555 on DFL; the dead-ball detector's stoppage F1 there is
0.856.

The absolute level is lower than on the sample games because the provider's flag changes hands at
every touch (9–11 possessions per live minute), but the ordering is the same, and the dead-ball
detector, never tuned on DFL, holds up.

**Public 5 minutes** (17 possessions, 4 stoppages; a check, not a score):

| | spellF1 | macroF1 | possessions (true 17) |
|---|---|---|---|
| Rondo v4 | 0.489 | 0.852 | 19 |
| Rondo v3 | 0.526 | 0.856 | 18 |
| Rondo v2 | 0.451 | 0.778 | 22 |
| baseline_smooth50 | 0.326 | 0.621 | 26 |

On the full match v4 reports 312 possessions, 5.6 per live minute.

## 8. Limitations and next steps

- **Proxy labels.** The sample games' possession comes from events and their dead ball from restarts,
  while the challenge's ground truth comes from corrected tracking. We follow the official written
  definition, but the details of what Metrica's corrected tracking calls "owning" may differ; that
  is why we also score on DFL, whose labels come from tracking.
- **Simulated noise.** Our degradations are generic guesses at automatic tracking, not measurements.
  Missing players hurt most.
- **Short possessions.** v4 finds far more of them than v3, but the very short ones (under 1 s) are
  still mostly missed.
- **Not used:** Smart Tagging phases (only useful if the tags are good, Section 4); team shape (the
  side in possession spreads out) for the stretches where the ball is lost.

## 9. Reproduce

See `README.md`. In short, with the challenge data in `data/`:

```bash
python3.9 -m venv .venv && .venv/bin/pip install -r requirements-rondo.txt
.venv/bin/python -m rondo.pipeline              # -> evaluation/results/Rondo.csv
```

To retrain or re-tune from the external data:

```bash
.venv/bin/python deadball/get_metrica.py        # Metrica sample games (about 215 MB)
.venv/bin/python -m rondo.train_possession      # possession model, about 4 minutes
.venv/bin/python -m rondo.tune_dead_ball        # dead-ball settings for spellF1, about 30 minutes
.venv/bin/python -m rondo.check_dfl             # DFL check (files fetched by hand, deadball/get_dfl.py)
```

## Credits

Metrica Sports for the challenge, the starter code and scorer (MIT), and the open sample games.
DFL / Sportec Solutions and Bassek et al. for the open DFL dataset (CC BY 4.0). kloppy for reading
the DFL files.
