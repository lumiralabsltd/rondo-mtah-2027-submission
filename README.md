# Rondo — More Than A Hack 2027, Phase 1

Team Rondo's code for Phase 1 of More Than A Hack 2027 (FC Barcelona, Metrica Sports, MWC): label
every frame of Barça v Athletic `H` (Barça has the ball), `A` (Athletic), `D` (dead ball) or `X`
(abstain), ranked on `spellF1`.

How it works, how every value was chosen, and the results: **[Rondo.md](Rondo.md)**.

In one line: a rule-based dead-ball detector, then a possession model learned on Metrica's open
sample games (a scene table, context and touch features, logistic regression, then a two-state
Viterbi). Every tuned or learned value comes from external data, never from the challenge match,
its ground truth or its video.

## What is here

```
rondo/                    our pipeline
  match.py                loads the challenge tracking into arrays (cached in data/cache/)
  dead_ball.py            role 1: predict_alive(match), a thin wrapper around deadball/
  possession.py           role 2: predict_team(match, alive), the v3 possession model
  possession_model.json   that model's learned values (from train_possession.py)
  train_possession.py     trains the possession model on Metrica's sample games
  tune_dead_ball.py       tunes the dead-ball settings for spellF1 on Metrica's sample games
  pipeline.py             writes evaluation/results/Rondo.csv, validates it, scores the public 5 min
deadball/                 the dead-ball detector (see deadball/README.md), its own tuner, a viewer
                          for the sample games, and loaders for Metrica and DFL open data
tests/smoke_test.py       runs the whole pipeline on fake data, no real data needed
evaluation/, games.py,    the organisers' scorer, validator, public 5-minute ground truth,
starter/, *.md            baselines, helpers and docs (MIT, Metrica Sports; see CHALLENGE_README.md)
```

**No challenge data is in this repository**, and none ever was: no video, tracking, Smart Tagging,
homography, exports or submission file. The match data is licensed to participants only.

## Reproduce Rondo.csv

Needs Python 3.9 and the challenge data bundle, unpacked as `data/` (see `DATA.md`).

```bash
python3.9 -m venv .venv
.venv/bin/pip install -r requirements-rondo.txt
.venv/bin/python tests/smoke_test.py          # optional: checks everything runs, on fake data
.venv/bin/python -m rondo.pipeline            # -> evaluation/results/Rondo.csv, validated and scored
```

The first run parses the tracking (a minute or two) and caches it in `data/cache/`. The pipeline
uses the committed `rondo/possession_model.json`, so it needs no external data and gives the same
file every time.

## Retrain from the external data (optional)

```bash
.venv/bin/python deadball/get_metrica.py      # Metrica sample games 1-3 (about 215 MB)
.venv/bin/python -m rondo.train_possession    # possession model, about 3 minutes
.venv/bin/python -m rondo.tune_dead_ball      # dead-ball settings for spellF1, about 30-60 minutes
(cd deadball && ../.venv/bin/python tune.py)  # dead-ball settings for stoppage F1, about 15 minutes
```

`train_possession.py` rewrites `rondo/possession_model.json` (byte for byte the same on our
machine). The tuners print what they would change; values are set in `deadball/deadball/rules.py`.
For the DFL check, `deadball/get_dfl.py` explains how to fetch those files by hand (Figshare blocks
scripted downloads).

## Credits and licence

External data: Metrica Sports' open sample games (github.com/metrica-sports/sample-data) and the DFL
/ Sportec open data (Bassek, Rein, Weber & Memmert 2025, CC BY 4.0). The organisers' files are MIT
licensed (see `LICENSE`, which also explains that the match data is not covered); our code is
released under the same MIT terms.
