"""Team Rondo — More Than A Hack 2027, Phase 1: who has the ball, every frame.

    match.py       shared: load the tracking + tagging once, as numpy arrays     (everyone)
    dead_ball.py   is the ball in play on each frame?                            (role 1)
    possession.py  while it is, which team has it?                               (role 2)
    pipeline.py    combine the two into Rondo.csv, validate, score              (role 3)

    train_possession.py  train possession_model.json on Metrica's open sample games
    tune_dead_ball.py    tune the dead-ball settings for spellF1 on the same games
    check_dfl.py         independent check on the DFL open data

Run everything with:

    .venv/bin/python -m rondo.pipeline

Never print or share rows of the match data: it is licensed to participants only.
"""
