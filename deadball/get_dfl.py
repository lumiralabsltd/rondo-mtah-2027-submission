"""Put the DFL open-data files you downloaded in a browser into data/dfl/, named for load_dfl.

    python get_dfl.py               # prints the links, then files whatever it finds in ~/Downloads
    python get_dfl.py --from DIR    # look in DIR instead

Figshare blocks scripted downloads with a bot check, so the files have to be downloaded by hand:
click each link below in a normal browser. Each match is a small "meta" file (match information)
and a big "tracking" file (positions, with the tracking provider's ball status and possession flag
on every frame). Two matches are plenty; all seven is better.

Dataset: Bassek, Rein, Weber, Memmert (2025), "An integrated dataset of spatiotemporal and event
data in elite soccer", Scientific Data. CC BY 4.0, so credit it if used publicly.
"""

import argparse
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "dfl")
LINK = "https://springernature.figshare.com/ndownloader/files/{}"
# match id -> (meta file id, tracking file id), from kloppy's sportec open-data table
MATCHES = {
    "J03WMX": (51643475, 51643514),
    "J03WN1": (51643472, 51643517),
    "J03WOH": (51643478, 51643520),
    "J03WOY": (51643481, 51643523),
    "J03WPY": (51643487, 51643526),
    "J03WQQ": (51643484, 51643529),
    "J03WR9": (51643490, 51643532),
}


def kind_and_match(path):
    """('meta' | 'tracking' | None, match id | None) from the file's name and first few KB."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(8192).decode("utf-8", "ignore")
    except OSError:
        return None, None
    found = re.search(r"J03W[A-Z0-9]{2}", os.path.basename(path)) or re.search(r"J03W[A-Z0-9]{2}", head)
    match = found.group(0) if found else None
    if "<MatchInformation" in head:
        return "meta", match
    if "<Positions" in head or "<FrameSet" in head:
        return "tracking", match
    return None, match


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="source", default=os.path.expanduser("~/Downloads"))
    args = parser.parse_args()

    print("Download links (open in a browser; meta is small, tracking is big):")
    for match, (meta, tracking) in MATCHES.items():
        print("  {}  meta {}   tracking {}".format(match, LINK.format(meta), LINK.format(tracking)))

    os.makedirs(OUT, exist_ok=True)
    moved = 0
    for name in sorted(os.listdir(args.source)):
        path = os.path.join(args.source, name)
        if not os.path.isfile(path) or not name.lower().endswith(".xml"):
            continue
        kind, match = kind_and_match(path)
        if kind is None or match not in MATCHES:
            continue
        target = os.path.join(OUT, "{}_{}.xml".format(match, kind))
        shutil.move(path, target)
        moved += 1
        print("filed  {}  ->  data/dfl/{}  ({:.0f} MB)".format(name, os.path.basename(target),
                                                             os.path.getsize(target) / 1e6))

    print("\nIn data/dfl/ now:")
    complete = 0
    for match in MATCHES:
        have = [k for k in ("meta", "tracking") if os.path.exists(os.path.join(OUT, "{}_{}.xml".format(match, k)))]
        if have:
            print("  {}  {}".format(match, "complete" if len(have) == 2 else "only " + have[0]))
            complete += len(have) == 2
    print("{} complete match(es){}".format(complete, "" if moved else "; nothing new found in " + args.source))
    return 0


if __name__ == "__main__":
    main()
