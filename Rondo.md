# Team Rondo — More Than A Hack 2027, Phase 1

Every frame of Barça v Athletic labelled `H` / `A` / `D` / `X` by two models in series: a
**dead-ball detector** (is the ball in play?) and a **possession model** (who has it?). Both read
only the tactical tracking (players and ball) and, for dead ball, the Smart Tagging `SET PIECES`
tags. No video, no homography, nothing labelled by hand.

**Every tuned or learned value comes from external data:** Metrica's open sample games, and seven
DFL open-data matches as a second source. Nothing was fitted, tuned or chosen on the challenge match,
its ground truth or its video, as the organisers' rules clarification of 6 November requires.
Section 6 says exactly how, including what we looked at before that clarification.

| | spellF1 | macroF1 |
|---|---|---|
| Metrica sample games, each scored by a model that never saw it, 48 degraded runs (Section 7) | **0.696** | — |
| DFL open data, 7 matches the possession model never trained on, 112 degraded runs (Section 7) | **0.560** | — |
| Public 5 minutes (a code check, not a target) | 0.627 | 0.888 |
| `baseline_smooth50` on the same 5 minutes | 0.326 | 0.621 |

Team: Nikolaos Efthymiopoulos, David Small, Sukhpinder Ubhi. Roles: dead ball, Sukhpinder Ubhi;
who has the ball, David Small and Nikolaos Efthymiopoulos; pipeline, checks and write-up,
Nikolaos Efthymiopoulos.

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

**External, as a second source:** all seven matches of the DFL / Sportec open data (Bassek, Rein,
Weber & Memmert 2025, *Scientific Data*, CC BY 4.0), whose tracking carries the provider's own ball
status and possession flag on every frame. That is the same kind of label as the challenge's ground
truth ("derived from corrected tracking data"), from a different provider. The possession model is
never trained on DFL. Two matches (J03WMX, J03WN1) helped choose between possession variants under
fixed rules; the other five confirmed the choice before it was adopted; four (J03WMX, J03WN1,
J03WOH, J03WOY) were used, alongside the sample games, to tune the dead-ball settings, and the
other three (J03WPY, J03WQQ, J03WR9) were held out of that tune to confirm it; all seven had to not
get worse for the five-network average to be adopted (Sections 4 and 6).

**Making the external games look like our match.** Their tracking is clean, ours is not. Every
external game is degraded before anything is tuned, trained or scored on it, in every combination
of:

- live-play ball dropout of 20% and 40%, in bursts
- the ball blanked during stoppages, or kept on screen (50% / 100%), optionally lying past the line
  for throw-ins, goal kicks and corners, with 30 cm of jitter
- tracking: clean; ball jitter of 0.5 m; player jitter of 0.2 m with 15% of player positions lost
  in bursts of about 2 s; both

These levels are generic, not measured on the challenge match.

### Training data: every labelled dataset we used

None of it comes from the provided game, and nobody on our team labelled anything by hand.

| dataset | source | who labelled it | how much | used for |
|---|---|---|---|---|
| Metrica Sample Games 1–3: tracking and event data | github.com/metrica-sports/sample-data (open) | the event data as published by Metrica Sports; our possession and dead-ball labels are derived from those events by code (`rondo/train_possession.py`, `deadball/labels.py`) | 3 full matches, about 430,000 frames | training the possession model; choosing the switch cost; tuning the dead-ball settings |
| DFL / Sportec open data, 7 Bundesliga matches: positions with ball status and possession flags | Bassek et al. 2025, Figshare (CC BY 4.0) | the flags as published by DFL / Sportec Solutions, used unchanged | 7 full matches, about 1,000,000 frames | scoring; choosing between possession variants under fixed rules (2 matches) and confirming the choice (5); dead-ball tuning (4) and confirming it (3); a must-not-fall check for the five-network average (7); never used to train the possession model |
| Public 5 minutes of the provided game (`evaluation/gt`) | the challenge repository | the organisers | 7,500 graded frames | checking the pipeline is correct, only; never training, tuning or choosing (Section 6) |

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
4. **Touch sequence and team shape.** The team of the last two and the next two touches, and
   touches per team in the 0.5, 1 and 3 s either side: a lone touch by one team between two of the
   other's looks like a deflection; two in a row looks like control. And the shape of the two teams:
   how spread out each is (the side with the ball spreads out, the other packs in), which way the
   two blocks are moving, and how deep into either half play is, with copies that only count when
   the ball is missing or filled in. On our match about 10.8 of each team's 11 players are tracked
   on a typical live frame, so the shape is well measured even when the ball is not.
5. **Combine.** A small neural network (91 features, one hidden layer of 24 tanh units) turns these
   into one number per frame: the log-odds that Barça has the ball. It is built as g(x) − g(−x), so
   swapping the two teams flips the answer exactly and neither team is favoured. Five copies,
   trained the same way from different random starts, are averaged. Their size, learning rate,
   number of training steps and number of copies were fixed before running, not tuned. In training each frame
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

The scene table and the network are counted and fitted (there is nothing to choose); the switch
cost is chosen (Section 6). All of it is in `rondo/possession_model.json` (a 541-entry table, five
networks of 2,232 weights each, one cost).

**Why this design.** On the sample games our dead ball plus a perfect possession model scores 0.908,
so possession was where the points were. Our rule-based model (v2: "nearest player within 3 m is
control" plus the same Viterbi) found only 24% of the real changes of possession within 1 s, and
missed 95% of possessions shorter than 2 s, which are over a fifth of all possessions in the sample
games and do occur in the public 5 minutes. v3 added steps 1–3 (learned scene evidence, context,
touches); v4 added the official definition of a change, the touch sequence and the per-possession
weighting; v5 added team shape and replaced the logistic regression with the small network (an
average of five).

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

The tables in this section are the numbers each decision was made on, with the dead-ball settings
of the time. Section 7 re-scores every version with the final settings.

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

**Last round: team shape and a small neural network.** Same protocol, rule fixed before running: a
variant replaces v4 only if it beats v4's 5-game mean by more than 0.005 (extra complexity has to
earn more than noise) and does not lower the DFL mean; then it must also not lose on the five DFL
matches that had not been used for anything.

| variant | Metrica | DFL (2) | 5-game mean |
|---|---|---|---|
| v4 (logistic) | 0.661 | 0.492 | 0.593 |
| + team shape | 0.678 | 0.507 | 0.610 |
| neural network | 0.672 | 0.496 | 0.602 |
| **neural network + team shape (v5)** | **0.690** | **0.520** | **0.622** |

| confirmation on the 5 other DFL matches | J03WOH | J03WOY | J03WPY | J03WQQ | J03WR9 | mean |
|---|---|---|---|---|---|---|
| v4 | 0.525 | 0.512 | 0.533 | 0.524 | 0.553 | 0.529 |
| v5 | 0.540 | 0.549 | 0.565 | 0.569 | 0.586 | **0.562** |

Every one of the five matches improved, so v5 was adopted.

**Five networks instead of one.** A network's answer depends a little on its random starting point.
Averaging five copies trained the same way from different starts (the number five fixed before
running, not tuned) evens that out. Rule, fixed before running: adopt only if the held-out
sample-game mean rises by more than 0.002 and the mean over the seven DFL matches does not fall.
Measured with the final dead-ball settings (Section 6):

| | Metrica, 48 runs, each game held out | DFL, 7 matches, 112 runs |
|---|---|---|
| one network | 0.691 | 0.556 |
| **average of five** | **0.696** | **0.560** |

Both rose, and so did every one of the seven DFL matches, so the final model averages five networks.

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
  - v5 (Section 4): must beat v4 on the same 5 games by more than 0.005 without lowering DFL, then
    not lose on the 5 DFL matches not used before. It passed both.
  - Dead ball, last re-tune: `rondo/tune_dead_ball.py` again on top of v5, with four DFL matches
    added (J03WMX, J03WN1, J03WOH, J03WOY) and the score set to half the sample-game mean, half the
    DFL mean; a change is kept only if the three DFL matches it never saw (J03WPY, J03WQQ, J03WR9)
    do not get worse. It changed one setting, the weight of "ball lying still" (0.5 to 1.5; 0.606 to
    0.611, every DFL match better), and the three unseen matches improved too (0.568 to 0.581, each
    one better), so it was kept and the possession model was retrained on top of it.
  - Five networks instead of one (Section 4): adopt only if the held-out sample mean rises by more
    than 0.002 and the seven-match DFL mean does not fall. It passed (+0.004 and +0.004).
  - Smart Tagging (Section 4): adopt only if better at every simulated quality. It was not.
  - Training on DFL as well: adopt only if the 5-game mean rises by more than 0.002. It did not
    (0.551 vs 0.551), so nothing is fitted on DFL.
- **The public 5 minutes** were used to check the code runs and the file is right, never to pick a
  value or a version. Twice a change chosen on external data lowered the public score, and we kept
  it both times: the dead-ball re-tune after the clarification (stoppage F1 0.800 to 0.667), and v4
  (spellF1 0.526 to 0.489, about one possession of 17). v5 happened to raise it again (0.627);
  that played no part in any choice.
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

| tracking | v2 (rule) | v3 | v4 | v5 (final) |
|---|---|---|---|---|
| clean | 0.533 | 0.666 | 0.704 | **0.720** |
| ball jitter | 0.529 | 0.667 | 0.684 | **0.715** |
| player jitter and loss | 0.476 | 0.626 | 0.641 | **0.679** |
| both | 0.472 | 0.614 | 0.619 | **0.668** |
| **mean** (worst run) | 0.502 (0.430) | 0.643 (0.550) | 0.662 (0.548) | **0.696** (0.604) |

v2's settings were picked on all three games, so if anything the comparison flatters v2.

**Dead ball on the sample games**: stoppage F1 0.777 (mean of 10 stress scenarios, worst
0.732), against 0.25–0.28 for the challenge baseline's "ball missing = dead" rule.

**DFL open data** (another provider; the possession model is never trained on it, and four of the
matches helped tune the dead-ball settings, Section 6; labels are the provider's own ball status
and possession flag; 7 matches × ball in stoppages as tracked or blanked × dropout 20% / 40% × the
4 tracking conditions = 112 runs; v3–v5 fitted on all three sample games; same final dead-ball
detector in every column):

| tracking | v2 | v3 | v4 | v5 (final) |
|---|---|---|---|---|
| clean | 0.424 | 0.508 | 0.542 | **0.567** |
| ball jitter | 0.414 | 0.521 | 0.546 | **0.575** |
| player jitter and loss | 0.385 | 0.492 | 0.515 | **0.548** |
| both | 0.382 | 0.495 | 0.509 | **0.548** |
| **mean** (worst run) | 0.401 (0.304) | 0.504 (0.412) | 0.528 (0.438) | **0.560** (0.457) |

By match, v5: J03WMX 0.518, J03WN1 0.525, J03WOH 0.554, J03WOY 0.559, J03WPY 0.569, J03WQQ 0.595,
J03WR9 0.596. With a perfect dead ball v5 would score 0.615; the dead-ball detector's stoppage F1
there is 0.891 (0.857 before the last re-tune).

The absolute level is lower than on the sample games because the provider's flag changes hands at
every touch (9–11 possessions per live minute), but the ordering is the same on both sources, under
every tracking condition.

**Public 5 minutes** (17 possessions, 4 stoppages; a check, not a score; each version with the
dead-ball settings it had at the time):

| | spellF1 | macroF1 | possessions (true 17) |
|---|---|---|---|
| **Rondo v5 (final)** | **0.627** | **0.888** | 13 |
| Rondo v4 | 0.489 | 0.852 | 19 |
| Rondo v3 | 0.526 | 0.856 | 18 |
| Rondo v2 | 0.451 | 0.778 | 22 |
| baseline_smooth50 | 0.326 | 0.621 | 26 |

On the full match v5 reports 253 possessions, 4.5 per live minute (v4: 312).

## 8. Limitations and next steps

- **Proxy labels.** The sample games' possession comes from events and their dead ball from restarts,
  while the challenge's ground truth comes from corrected tracking. We follow the official written
  definition, but the details of what Metrica's corrected tracking calls "owning" may differ; that
  is why we also score on DFL, whose labels come from tracking.
- **Simulated noise.** Our degradations are generic guesses at automatic tracking, not measurements.
  Missing players hurt most.
- **Short possessions.** v4 and v5 find far more of them than v3, but the very short ones (under
  1 s) are still mostly missed. The answer keys disagree on how many there are (Section 4), so we
  follow the official definition rather than chase the count.
- **Dead ball is still worth points.** With a perfect dead ball the final model would score 0.615
  on DFL instead of 0.560. The detector is hand-weighted rules; a classifier learned on external
  labels is the obvious next step.
- **Not used:** Smart Tagging phases (only useful if the tags are good, Section 4).

## 9. Reproduce

See `README.md`. In short, with the challenge data in `data/`:

```bash
python3.9 -m venv .venv && .venv/bin/pip install -r requirements-rondo.txt
.venv/bin/python -m rondo.pipeline              # -> evaluation/results/Rondo.csv
```

To retrain or re-tune from the external data:

```bash
.venv/bin/python deadball/get_metrica.py        # Metrica sample games (about 215 MB)
.venv/bin/python -m rondo.tune_dead_ball --dfl J03WMX J03WN1 J03WOH J03WOY   # dead ball, ~1.5 h
.venv/bin/python -m rondo.train_possession      # possession model, about 1.5 hours
.venv/bin/python -m rondo.check_dfl             # DFL check (files fetched by hand, deadball/get_dfl.py)
```

## Credits

Metrica Sports for the challenge, the starter code and scorer (MIT), and the open sample games.
DFL / Sportec Solutions and Bassek et al. for the open DFL dataset (CC BY 4.0). kloppy for reading
the DFL files.
