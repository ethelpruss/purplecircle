"""Mean landmark displacement per frame, for every cam02 body file in a folder.

Reads:  <IN_DIR>/<name>_cam02_body.csv     (only files starting with a digit)
Writes: <OUT_DIR>/<name>_cam02_displacement.csv

Units are shoulder-widths per frame. Set the two folders below, then hit run.
"""

import glob
import os
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------
IN_DIR = "/Users/tmullins/Documents/GitHub/TilburgMultiscaleSummerschool2026/Datasets/BalanceCorpus/motiontracking/Output_TimeSeries"
OUT_DIR = "/Users/tmullins/Desktop/motiontest"
# ---------------------------------------------------------------------------


# 15 non-redundant landmarks. The other 18 are face detail (which just tracks
# the head), finger triplets (duplicate the wrist) and heel/foot (duplicate the
# ankle) -- averaging over all 33 weights "head" 11x.
CORE = [
    "NOSE",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_INDEX", "RIGHT_INDEX",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
]


def displacement(df):
    """Per-frame displacement for one pose CSV. Same maths as the single-file
    version -- see comments there for why Z is dropped etc."""
    X = df[["X_" + p for p in CORE]].to_numpy()
    Y = df[["Y_" + p for p in CORE]].to_numpy()
    V = df[["visibility_" + p for p in CORE]].to_numpy()

    # Z is left out on purpose: MediaPipe's depth estimate jitters ~8x more per
    # frame than X/Y, so including it would swamp the real movement.
    d = np.hypot(np.diff(X, axis=0), np.diff(Y, axis=0))

    # Weight each landmark by tracking confidence, using the lower of the two
    # frames so a landmark only counts if it was visible for both.
    w = np.minimum(V[1:], V[:-1])
    w[w < 0.5] = 0.0
    disp = (d * w).sum(axis=1) / w.sum(axis=1)

    # Body units, not image units, so the score is comparable across
    # participants and cameras. One median per clip rather than per frame --
    # projected shoulder width also shrinks when someone rotates.
    shoulder = np.hypot(df.X_LEFT_SHOULDER - df.X_RIGHT_SHOULDER,
                        df.Y_LEFT_SHOULDER - df.Y_RIGHT_SHOULDER).median()
    disp = disp / shoulder

    out = pd.DataFrame({
        "time": df["time"].to_numpy()[1:],
        "frame": np.arange(1, len(df)),
        "displacement": disp,
    })
    # Rolling median (not mean) drops one-frame tracker glitches without
    # smearing them across neighbours. 5 frames is about 165 ms at 30 fps.
    out["displacement_smooth"] = (out["displacement"]
                                  .rolling(5, center=True, min_periods=1).median())
    return out


os.makedirs(OUT_DIR, exist_ok=True)

# cam02 body files whose name starts with a digit -- this is what excludes
# "preparation_*", which are the same format but not task trials.
files = sorted(f for f in glob.glob(os.path.join(IN_DIR, "*cam02_body.csv"))
               if re.match(r"\d", os.path.basename(f)))

print(f"{len(files)} matching files in {IN_DIR}\n")

done, failed = 0, []
for path in files:
    name = os.path.basename(path)
    out_path = os.path.join(OUT_DIR, name.replace("_body.csv", "body_displacement.csv"))
    try:
        df = pd.read_csv(path)
        out = displacement(df)
        out.to_csv(out_path, index=False)
        done += 1
        print(f"{name:70s} {len(out):5d} rows  "
              f"median {out.displacement_smooth.median():.5f}")
    except Exception as e:
        # Keep going -- one malformed file shouldn't kill a batch of 200.
        failed.append((name, e))
        print(f"{name:70s} FAILED: {e}")

print(f"\n{done} written to {OUT_DIR}")
if failed:
    print(f"{len(failed)} failed:")
    for name, e in failed:
        print(f"  {name}: {e}")