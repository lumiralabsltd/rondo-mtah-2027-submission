"""Dead-ball detector: score it on other games, or label the challenge match.

    python run.py samples            # Metrica Sample Games 1-3, events -> labels, ball dropout added
    python run.py dfl                # DFL open data in data/dfl/ (true alive/dead flag)
    python run.py match              # BAR-ATH: writes results/deadball_BAR-ATH.csv
    python run.py view               # export the sample games and open the viewer on localhost:8765
"""

import argparse
import glob
import os
import sys

import numpy as np

from deadball import evaluate, features, labels, loaders, rules

HERE = os.path.dirname(os.path.abspath(__file__))
METRICA = os.path.join(loaders.DATA, "metrica")
DFL = os.path.join(loaders.DATA, "dfl")
RESULTS = os.path.join(HERE, "results")


def restart_file(name):
    """Event file of a sample game, for typing its stoppages (throw-in, corner, ...)."""
    g = name.split("_")[-1]
    return os.path.join(METRICA, "Sample_Game_3_events.json" if g == "3"
                        else "Sample_Game_{}_RawEventsData.csv".format(g))


def sample_games():
    """(name, Frames, dead) for each Metrica sample game that is downloaded."""
    m = lambda name: os.path.join(METRICA, name)  # noqa: E731
    for g in (1, 2):
        home = m("Sample_Game_{}_RawTrackingData_Home_Team.csv".format(g))
        if os.path.exists(home):
            frames = loaders.load_metrica_csv(home, home.replace("Home", "Away"))
            yield "metrica_{}".format(g), frames, labels.dead_from_events(
                m("Sample_Game_{}_RawEventsData.csv".format(g)), frames.n,
                np.isnan(frames.ball[:, 0]), fps=frames.fps)
    if os.path.exists(m("Sample_Game_3_tracking.txt")):
        frames = loaders.load_epts(m("Sample_Game_3_metadata.xml"), m("Sample_Game_3_tracking.txt"))
        yield "metrica_3", frames, labels.dead_from_events(
            m("Sample_Game_3_events.json"), frames.n, np.isnan(frames.ball[:, 0]), fps=frames.fps)


def dfl_games():
    for meta in sorted(glob.glob(os.path.join(DFL, "*_meta.xml"))):
        frames, dead = loaders.load_dfl(meta, meta.replace("_meta", "_tracking"))
        yield os.path.basename(meta).split("_")[0], frames, dead


def degrade(frames, dead, dropout, seed=0, dead_visible=0.0, restarts=None):
    """Make a clean sample game look like automatic tracking: ball shown in stoppages, then
    live-play dropout. `restarts` also puts balls out for throws, goal kicks and corners past
    the line."""
    if dead_visible > 0:
        frames = evaluate.show_dead_ball(frames, dead, visible=dead_visible, seed=seed,
                                         restarts=restarts)
    if dropout:
        frames = evaluate.add_ball_dropout(frames, dead, seed=seed)
    return frames


def report(games, dropout, seed=0, dead_visible=0.0, beyond_line=False, tags=False):
    header = "{:<12} {:>7} {:>8} {:>9} {:>6} {:>6}   {}".format(
        "game", "deadF1", "macroF1", "stopF1", "true", "pred", "baseline (ball missing = dead)")
    print(header + "\n" + "-" * len(header))
    for name, frames, dead in games:
        events = labels.restarts_from_events(restart_file(name)) if beyond_line or tags else None
        frames = degrade(frames, dead, dropout, seed, dead_visible, events if beyond_line else None)
        f = features.compute(frames)
        pred = rules.detect(f, frames.fps,
                            tags=evaluate.simulate_tags(events, frames.fps, seed=seed) if tags else None)
        s = evaluate.scores(dead, pred)
        b = evaluate.scores(dead, rules.drop_short_runs(f["ball_missing"], False, frames.fps))
        print("{:<12} {:>7.3f} {:>8.3f} {:>9.3f} {:>6} {:>6}   deadF1 {:.3f} stopF1 {:.3f} pred {}".format(
            name, s["dead_f1"], s["macro_f1"], s["stoppage_f1"], s["n_true"], s["n_pred"],
            b["dead_f1"], b["stoppage_f1"], b["n_pred"]))


def match_tags():
    """BAR-ATH's Smart Tagging set pieces as [(start_frame, end_frame)] video frames."""
    loaders.mtah_repo()
    from starter import load_smart_tagging as st
    return [(e["start_frame"], e["end_frame"]) for e in st.load()
            if (e["code"] or "").strip().upper() == "SET PIECES"]


def match():
    frames = loaders.load_match()
    if frames is None:
        print("BAR-ATH data not found under {}/data -- download the bundle (DATA.md) first."
              .format(loaders.MTAH_REPO))
        return 0
    import games
    from evaluation import possession_metrics as pm

    dead = rules.detect(features.compute(frames), frames.fps, tags=match_tags())
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "deadball_{}.csv".format(games.GAME))
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, "D" if d else "L") for i, d in enumerate(dead))
    print("wrote", path)

    gt_path = os.path.join(loaders.MTAH_REPO, "evaluation", "gt",
                           "{}_possession_gt_public.csv".format(games.GAME))
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    graded = gt != pm.X
    s = evaluate.scores(gt == pm.D, dead, graded)
    print("public 5 min (check only, do not tune on it): deadF1 {:.3f} macroF1 {:.3f} "
          "stopF1 {:.3f} stoppages true {} pred {}".format(
              s["dead_f1"], s["macro_f1"], s["stoppage_f1"], s["n_true"], s["n_pred"]))
    return 0


def viewer_handler():
    """Static files from viewer/. The challenge match and its video are not served: the rules
    (organisers' clarification, 2026-11-06) forbid checking our output against the video, and the
    viewer exists to show predictions next to the truth."""
    import http.server

    root = os.path.join(HERE, "viewer")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=root, **kwargs)

        def log_message(self, *args):
            pass

    return Handler


def view(names, dropout, seed, port, serve, browser, dead_visible=0.0):
    """Export games for the viewer, then serve viewer/ on localhost."""
    from deadball import export

    wanted = set(names or [])
    for name, frames, dead in sample_games():
        if not wanted or name in wanted:
            meta = export.export(name, frames, dead, dropout=dropout, seed=seed,
                                 dead_visible=dead_visible,
                                 restarts=labels.restarts_from_events(restart_file(name)))
            print("exported {} ({} steps)".format(name, meta["n_steps"]))
    if "BAR-ATH" in wanted:
        print("BAR-ATH is not exported: the viewer compares predictions with the truth, and the "
              "rules forbid checking our output against the match's video or ground truth")
    if not serve:
        return 0

    import http.server
    import webbrowser
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), viewer_handler())
    url = "http://127.0.0.1:{}/".format(port)
    print("viewer at", url, "(Ctrl+C to stop)")
    if browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("samples", "dfl", "match", "view"))
    parser.add_argument("--game", action="append",
                        help="view only: sample game to export (repeatable), e.g. metrica_3")
    parser.add_argument("--port", type=int, default=8765, help="view only: localhost port")
    parser.add_argument("--export-only", action="store_true", help="view only: don't serve")
    parser.add_argument("--no-browser", action="store_true", help="view only: don't open a tab")
    parser.add_argument("--no-dropout", action="store_true",
                        help="score on the clean ball track instead of simulating ours")
    parser.add_argument("--seed", type=int, default=0, help="seed of the simulated dropout")
    parser.add_argument("--dead-visible", type=float, default=0.0,
                        help="stress test: share of stoppage frames where the ball stays visible")
    parser.add_argument("--tags", action="store_true",
                        help="samples: anchor on simulated Smart Tagging set pieces (from events)")
    parser.add_argument("--beyond-line", action="store_true",
                        help="stress test: with --dead-visible, balls out for a throw, goal kick or "
                             "corner lie past the line and are carried back")
    args = parser.parse_args()
    if args.what == "match":
        return match()
    if args.what == "view":
        return view(args.game, not args.no_dropout, args.seed, args.port,
                    serve=not args.export_only, browser=not args.no_browser,
                    dead_visible=args.dead_visible)
    games = sample_games() if args.what == "samples" else dfl_games()
    report(games, dropout=not args.no_dropout, seed=args.seed, dead_visible=args.dead_visible,
           beyond_line=args.beyond_line, tags=args.tags)
    return 0


if __name__ == "__main__":
    sys.exit(main())
