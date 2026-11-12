# Team Rondo — More Than A Hack 2027, Phase 1

Every frame of Barça v Athletic labelled `H` / `A` / `D` / `X` by two models in series: a
**dead-ball detector** (is the ball in play?) and a **possession model** (who has it?). Both read
only the tactical tracking (players and ball) and, for dead ball, the Smart Tagging `SET PIECES`
tags. No video, no homography, nothing labelled by hand.

**Every tuned or learned value comes from external data:** Metrica's open sample games (and DFL's
open data as an independent check). Nothing was fitted, tuned or chosen on the challenge match, its
ground truth or its video, as the organisers' rules clarification of 6 November requires. Section 5
says exactly how, including what we looked at before that clarification.

| | spellF1 | macroF1 |
|---|---|---|
| Metrica sample games, held-out, 48 degraded runs (Section 6) | 0.609 | — |
| DFL open data, 2 matches never used for anything else (Section 6) | 0.462 | — |
| Public 5 minutes (a code check, not a target) | 0.526 | 0.856 |
| `baseline_smooth50` on the same 5 minutes | 0.326 | 0.621 |

Roles: dead ball, Sukh; who has the ball, David and Nick; pipeline, checks and write-up, Nick.

---

## 1. Data

**From the challenge:** `BAR-ATH_tactical_atd` (players and ball, 25 fps, automatic, 46 track
slots with a team each, ball missing on 41% of graded frames) and `BAR-ATH_tactical_smart_tagging`
(only `SET PIECES`, used as dead-ball evidence). Not used: the videos, the broadcast tracking, the
homography, the other tags.

**External, for every tuned or learned value:** Metrica's open sample games 1–3
(github.com/metrica-sports/sample-data: tracking and event data for three full matches). Metrica
asks users to "be responsible with the use of this data" and to acknowledge the source: thank you.
From the events we build the two labels we need:

- *dead ball*: a stoppage runs from the last event before a restart (set piece, goal kick) to the
  restart; any ball blank of 8 s or more is also dead (`deadball/labels.py`)
- *possession*: on live frames, the team of the latest on-ball event (pass, carry, shot, set piece,
  recovery, ball lost); challenges are skipped because both teams are in them
  (`rondo/train_possession.py`)

**External, as an independent check only:** two matches of the DFL / Sportec open data (Bassek,
Rein, Weber & Memmert 2025, *Scientific Data*, CC BY 4.0), whose tracking carries the provider's
own ball status and possession flag on every frame. That is the same kind of label as the
challenge's ground truth ("derived from corrected tracking data"), from a different provider.

**Making the sample games look like our match.** The sample tracking is clean, ours is not. Every
external game is degraded before anything is tuned or trained on it, in every combination of:

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
was struck). Every threshold is in metres or seconds, never a frame number.

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
4. **Combine.** A logistic regression with no intercept (so neither team is favoured) turns these 61
   features into one number per frame: the log-odds that Barça has the ball.
5. **Decode.** For each stretch of live play a two-state model (H / A) picks the labelling that
   agrees most with those numbers, paying a fixed cost for every change of possession (Viterbi).
   That is what makes a deflection or a pass flying past an opponent not a change, while a real
   interception is one.
6. **Abstain** (`X`) only on a stretch of play with no ball information at all. Inside a possession
   the scorer carries the previous answer through an `X`, but at a hand-over an `X` carries the old
   team across, so we always guess when there is any evidence. On the match, 260 of 158,857 graded frames (0.2%)
   are `X`.

The scene table and the weights are counted and fitted (there is nothing to choose); the switch
cost is chosen (Section 5). All of it is in `rondo/possession_model.json` (61 weights, a 541-entry
table, one cost).

**Why this design.** On the sample games our dead ball plus a perfect possession model scores 0.908,
so possession was where the points were. Our previous possession model (v2: "nearest player within
3 m is control" plus the same Viterbi) found only 24% of the real changes of possession within 1 s,
and missed 95% of possessions shorter than 2 s, which are over a quarter of all possessions in the
sample games and do occur in the public 5 minutes. The context features and touches are what find
those.

## 4. Combining them (`rondo/pipeline.py`)

`D` wherever the ball is dead, otherwise the possession answer; `X` outside the graded halves. The
output is validated with the organisers' `validate_submission.py`. Two runs give a byte-identical
file, and retraining the possession model from scratch reproduces `possession_model.json` byte for
byte.

## 5. How every value was chosen (and what we did not do)

The rules clarification (6 November) forbids checking output against the video, hand labels on the
match, uploading the match data to AI services, and training, fitting or calibrating on the
ground-truth possession data or the video; anything tuned must be tuned on external data. We
followed that as follows.

- **External data only.** Every tuned or learned value comes from the degraded sample games above.
- **Selection rules fixed before running.** Each tuning script states its rule in its header, and the
  rule was written before the run: highest mean score over all the scenarios wins; anything within
  0.002 of the best is a tie; a tie keeps the current value (or, for the switch cost, the higher
  cost).
  - `deadball/tune.py`: the detector's settings for its own stoppage F1, 10 scenarios.
  - `rondo/tune_dead_ball.py`: the same settings for the ranked metric, by running the whole
    pipeline on 60 degraded sample runs and scoring spellF1; one setting at a time, two passes. It
    moved 7 settings (mean spellF1 0.576 to 0.593, every game better), and the detector's own
    stoppage F1 rose with it (0.751 to 0.778, worst scenario 0.636 to 0.737).
  - `rondo/train_possession.py`: the switch cost, by leave-one-game-out (fit on two games, score the
    third), so the reported score is for games the model never saw.
- **The public 5 minutes** were used to check the code runs and the file is right, never to pick a
  value or a version. After the clarification, a re-tune on the sample games lowered the dead-ball
  score on the public window (stoppage F1 0.800 to 0.667), and we kept the re-tuned value.
- **Honest note on what came before the clarification.** In late October, before the clarification,
  version 1 had two settings partly picked on the public 5 minutes, and the out-of-play latch and the
  Smart Tagging anchors were designed after looking at those minutes. Looking at the data to
  understand it is allowed. The values are not taken from it. Every one was re-tuned on the sample
  games after 6 November, and the stress-test settings that had been measured on the challenge match
  were replaced with generic ones.
- **No match data left our laptops.** The match data was never given to an AI service. Only aggregate
  numbers (scores, counts) were ever printed or shared. This repository has none of it: no
  tracking, no tags, no exports, no submission file.

## 6. Results

**Possession on the sample games** (spellF1 of the whole pipeline, each game scored by a model fitted
on the other two, 3 games × 4 dead-ball scenarios × 4 tracking conditions = 48 runs):

| tracking | v2 (rule) | v3 (learned) |
|---|---|---|
| clean | 0.505 | 0.629 |
| ball jitter | 0.501 | 0.631 |
| player jitter and loss | 0.450 | 0.592 |
| both | 0.446 | 0.586 |
| **mean** (worst run) | **0.476** (0.413) | **0.609** (0.521) |

Both with the same, final dead-ball detector. v2's settings were picked on all three games, so if
anything the comparison flatters v2.

**Dead ball on the sample games**: stoppage F1 0.778 (mean of 10 stress scenarios, worst
0.737), against 0.25–0.28 for the challenge baseline's "ball missing = dead" rule.

**DFL open data** (two matches from another provider, never used to tune or train; labels are the
provider's own ball status and possession flag; 2 matches x ball in stoppages as tracked or blanked
x dropout 20% / 40% x clean / noisy tracking = 16 runs):

| | v2 | v3 |
|---|---|---|
| spellF1, whole pipeline (worst run) | 0.359 (0.278) | **0.462** (0.387) |
| spellF1 with a perfect dead ball | 0.422 | 0.523 |
| dead ball, stoppage F1 | 0.856 | 0.856 |

The absolute level is lower than on the sample games because the labels are a different definition
(the provider's possession flag rather than events), but the gain from v2 to v3 is the same size,
and the dead-ball detector, never tuned on DFL, holds up.

We also tested training the possession model on the two DFL matches as well as the sample games,
with the rule fixed before running (each of the 5 games scored by a model that never saw it; adopt
only if the mean beats the sample-games-only model by more than 0.002). It did not help (0.551 vs
0.551), so the model stays trained on the sample games alone and DFL stays a pure check.

**Public 5 minutes** (17 possessions, 4 stoppages; a check, not a score):

| | spellF1 | macroF1 | possessions (true 17) |
|---|---|---|---|
| Rondo v3 | 0.526 | 0.856 | 18 |
| Rondo v2 | 0.451 | 0.778 | 22 |
| baseline_smooth50 | 0.326 | 0.621 | 26 |

## 7. Limitations and next steps

- **Proxy labels.** The sample games' possession comes from events and their dead ball from restarts,
  while the challenge's ground truth comes from corrected tracking. The definitions are close but
  not identical, which is why we also check on DFL, whose labels come from tracking.
- **Simulated noise.** Our degradations are generic guesses at automatic tracking, not measurements.
  Missing players hurt most (spellF1 0.62 clean vs 0.58 with player loss).
- **Still too few possession changes.** On the sample games v3 finds about 270 of 383 possessions per
  game; most of the rest are very short.
- **Not used yet:** Smart Tagging phases carry a team (BUILD UP, PROGRESSION, ...), but the external
  games have no tags to learn a weight from; team shape (the side in possession spreads out) for the
  stretches where the ball is lost.

## 8. Reproduce

See `README.md`. In short, with the challenge data in `data/`:

```bash
python3.9 -m venv .venv && .venv/bin/pip install -r requirements-rondo.txt
.venv/bin/python -m rondo.pipeline              # -> evaluation/results/Rondo.csv
```

To retrain or re-tune from the external data:

```bash
.venv/bin/python deadball/get_metrica.py        # Metrica sample games (about 215 MB)
.venv/bin/python -m rondo.train_possession      # possession model, about 3 minutes
.venv/bin/python -m rondo.tune_dead_ball        # dead-ball settings for spellF1, about 30 minutes
```

## Credits

Metrica Sports for the challenge, the starter code and scorer (MIT), and the open sample games.
DFL / Sportec Solutions and Bassek et al. for the open DFL dataset (CC BY 4.0). kloppy for reading
the DFL files.
