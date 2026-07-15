"""Mean landmark displacement per frame from a MediaPipe Pose CSV.

Units are shoulder-widths per frame. Set the two paths below, then hit run.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# EDIT THESE TWO LINES
# ---------------------------------------------------------------------------
IN_FILE = "/Users/tmullins/Documents/GitHub/TilburgMultiscaleSummerschool2026/Datasets/BalanceCorpus/motiontracking/Output_TimeSeries/103_203_12_1_20250113_152455_doughnut_board_guesser_cam02_body.csv"
OUT_FILE = "/Users/tmullins/Desktop/motiontest/103_203_12_1_20250113_152455_doughnut_board_guesser_cam02_displacement.csv"
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


df = pd.read_csv(IN_FILE)

X = df[["X_" + p for p in CORE]].to_numpy()
Y = df[["Y_" + p for p in CORE]].to_numpy()
V = df[["visibility_" + p for p in CORE]].to_numpy()

# How far each landmark moved between consecutive frames.
# I left Z out here because MediaPipe's depth estimate jitters ~8x more per
# frame than X/Y, but it can be added back in.
d = np.hypot(np.diff(X, axis=0), np.diff(Y, axis=0))

# Weight each landmark by how confidently it was tracked, using the lower of
# the two frames so a landmark only counts if it was visible for both.
w = np.minimum(V[1:], V[:-1])
w[w < 0.5] = 0.0
disp = (d * w).sum(axis=1) / w.sum(axis=1)

# Put it in body units instead of image units, so the score means the same
# thing regardless of how far the participant sits from the camera.
# One constant for the whole clip, not per frame
shoulder = np.hypot(df.X_LEFT_SHOULDER - df.X_RIGHT_SHOULDER,
                    df.Y_LEFT_SHOULDER - df.Y_RIGHT_SHOULDER).median()
disp = disp / shoulder

out = pd.DataFrame({
    "time": df["time"].to_numpy()[1:],
    "frame": np.arange(1, len(df)),
    "displacement": disp,
})
# Rolling median (not mean) kills one-frame tracker glitches without smearing
# them across their neighbours. 5 frames is about 165 ms at 30 fps.
out["displacement_smooth"] = (out["displacement"]
                              .rolling(5, center=True, min_periods=1).median())

out.to_csv(OUT_FILE, index=False)

print(f"read  {len(df)} frames from {IN_FILE}")
print(f"wrote {len(out)} rows to {OUT_FILE}")
print(f"displacement_smooth (shoulder-widths/frame): "
      f"median {out.displacement_smooth.median():.5f}   "
      f"p95 {out.displacement_smooth.quantile(.95):.5f}   "
      f"max {out.displacement_smooth.max():.5f}")