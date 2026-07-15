"""One overlay plot per guesser/clueGiver pair.

Reads the *_displacement.csv files written by batch.py, matches each guesser to
its clueGiver by filename (everything except the role must be identical), and
saves one PNG per pair.

Set the folders below, then hit run.
"""

import glob
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")   # write PNGs without opening a window; drop for interactive
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------
IN_DIR = "/Users/tmullins/Desktop/motiontest"
OUT_DIR = "/Users/tmullins/Desktop/motiontest/images"

COLUMN = "displacement"   # or "displacement" for the unsmoothed version
# ---------------------------------------------------------------------------

GUESSER_COLOUR = "#2a78d6"
CLUEGIVER_COLOUR = "#eda100"

# Splits a filename on the role token. Case-insensitive, so clueGiver /
# cluegiver / ClueGiver all match. Whatever sits either side is the pair key --
# that's the dyad, date, timestamp, object, board and cam, all of which must be
# identical for two files to be a pair.
ROLE = re.compile(r"_(guesser|cluegiver)_", re.IGNORECASE)

os.makedirs(OUT_DIR, exist_ok=True)

pairs = defaultdict(dict)
for path in sorted(glob.glob(os.path.join(IN_DIR, "*_displacement.csv"))):
    name = os.path.basename(path)
    m = ROLE.search(name)
    if not m:
        print(f"skipped (no role in name): {name}")
        continue
    key = name[:m.start()] + "_" + name[m.end():]
    role = m.group(1).lower()          # normalise so "clueGiver" == "cluegiver"
    pairs[key][role] = path

print(f"{len(pairs)} potential pairs in {IN_DIR}\n")

made, unpaired = 0, []
for key in sorted(pairs):
    got = pairs[key]
    if "guesser" not in got or "cluegiver" not in got:
        # A dyad with only one role present -- worth knowing about rather than
        # silently plotting half a pair.
        unpaired.append((key, list(got)))
        continue

    g = pd.read_csv(got["guesser"])
    c = pd.read_csv(got["cluegiver"])

    label = os.path.splitext(key)[0].replace("__", "_")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(c.time / 1000, c[COLUMN], color=CLUEGIVER_COLOUR, lw=1.2,
            ls="--", label="clueGiver")
    ax.plot(g.time / 1000, g[COLUMN], color=GUESSER_COLOUR, lw=1.4,
            label="guesser")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("displacement (shoulder-widths/frame)")
    ax.set_title(label, fontsize=9, loc="left")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    png = os.path.join(OUT_DIR, label + ".png")
    fig.savefig(png, dpi=150)
    plt.close(fig)   # without this you leak a figure per pair and matplotlib warns
    made += 1

    # The two roles are filmed separately, so a length mismatch means the clips
    # aren't the same span -- the plot will still draw, but check before you
    # read anything into the alignment.
    flag = "" if len(g) == len(c) else f"   <- length mismatch {len(g)} vs {len(c)}"
    print(f"{label:75s} {len(g):5d} rows{flag}")

print(f"\n{made} plots written to {OUT_DIR}")
if unpaired:
    print(f"\n{len(unpaired)} without a partner:")
    for key, roles in unpaired:
        print(f"  {key}  (only: {', '.join(roles)})")