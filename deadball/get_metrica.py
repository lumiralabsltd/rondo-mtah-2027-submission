"""Download Metrica's open sample games into data/metrica/ (about 215 MB, gitignored).

    python get_metrica.py

Source: https://github.com/metrica-sports/sample-data ("be responsible with the use of this data",
and credit Metrica if you use it publicly). Only tuning and the viewer need it; the submission
itself does not.
"""

import os
import urllib.request

BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
FILES = [
    "Sample_Game_1/Sample_Game_1_RawEventsData.csv",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
    "Sample_Game_2/Sample_Game_2_RawEventsData.csv",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
    "Sample_Game_3/Sample_Game_3_events.json",
    "Sample_Game_3/Sample_Game_3_metadata.xml",
    "Sample_Game_3/Sample_Game_3_tracking.txt",
]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "metrica")


def main():
    os.makedirs(OUT, exist_ok=True)
    for path in FILES:
        target = os.path.join(OUT, os.path.basename(path))
        if os.path.exists(target):
            print("have", os.path.basename(path))
            continue
        print("downloading", os.path.basename(path))
        urllib.request.urlretrieve(BASE + path, target + ".part")
        os.replace(target + ".part", target)


if __name__ == "__main__":
    main()
