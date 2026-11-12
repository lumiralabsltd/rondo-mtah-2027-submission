"""Smoke test: run the whole pipeline on a FAKE match and check the output file is valid.

    .venv/bin/python tests/smoke_test.py

No real data needed and none touched — it writes random tracking in the real file format to a
temporary folder, runs rondo.pipeline on it, validates the CSV, and deletes everything. Run it
before you push, so a broken change is caught on your machine instead of in the final file.
"""

import os
import random
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, asset_paths, source_frames  # noqa: E402

N_SLOTS = 6


def fake_group(rng, missing_rate):
    if rng.random() < missing_rate:
        return "NaN,NaN,NaN"
    return "{:.4f},{:.4f},{:.2f}".format(rng.random(), rng.random(), rng.random() * 8)


def make_fake_data(folder):
    paths = asset_paths(GAME, data_dir=folder)
    os.makedirs(os.path.dirname(paths["atd_raw"]))
    rng = random.Random(0)

    with open(paths["atd_meta"], "w") as handle:
        handle.write('<Main><Metadata><Teams>'
                     '<Team id="T1"><Name>Home</Name></Team><Team id="T2"><Name>Away</Name></Team>'
                     '</Teams><Players>')
        for slot in range(N_SLOTS):
            handle.write('<Player id="P{}" teamId="{}"><Name>Track {}</Name></Player>'.format(
                slot, "T1" if slot % 2 == 0 else "T2", slot + 1))
        handle.write('</Players></Metadata></Main>')

    with open(paths["atd_raw"], "w") as handle:
        for frame_id in range(1, source_frames("tactical", GAME) + 1):
            players = ";".join(fake_group(rng, 0.2) for _ in range(N_SLOTS))
            handle.write("{}:{}:{}\n".format(frame_id, players, fake_group(rng, 0.4)))

    with open(paths["smart_tagging"], "w") as handle:
        handle.write('<file><ALL_INSTANCES><instance><ID>1</ID><start>447.0</start><end>452.0</end>'
                     '<code>SET PIECES</code><label><group>Type</group><text>Kick Off (Start)</text>'
                     '</label></instance></ALL_INSTANCES></file>')


def main():
    folder = tempfile.mkdtemp(prefix="rondo-smoke-")
    try:
        make_fake_data(folder)
        out = os.path.join(folder, "smoke.csv")
        subprocess.check_call([sys.executable, "-m", "rondo.pipeline", "--data-dir", folder,
                               "--out", out, "--no-score"], cwd=ROOT)
        subprocess.check_call([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), out])
        print("\nSMOKE TEST PASSED")
    finally:
        shutil.rmtree(folder)


if __name__ == "__main__":
    main()
