# purplecircle
MDIG2026 Tilburg Summer School project using a sub-sample of the Balance Corpus dataset (see https://github.com/WimPouw/TilburgMultiscaleSummerschool2026/tree/main/Datasets/BalanceCorpus).

# Dyadic movement coordination pipeline

Turns MediaPipe pose output into a per-frame movement signal for each participant, then uses multidimensional recurrence quantification analysis (mdRQA) on the dyad to locate moments of elevated determinism within a trial. Additional visualization with Windowed Multiscale Synchrony (WMS) plots is included in the last section -- this also uses additional output data from OpenFace.

## Pipeline

### `movement_one_file.py` and `movement_loop`— amount of movement since last frame extraction

These files read a MediaPipe Pose CSV for one participant and computes mean landmark displacement between consecutive frames, using 15 of the 33 landmarks (face, finger and heel points are dropped so they don't silently overweight the head, wrist and ankle) and X/Y only, since MediaPipe's Z estimate is mostly frame-to-frame noise. Displacement is divided by the clip's median shoulder width, so the same physical movement scores the same regardless of how far the participant sits from the camera.

`movement_one_file` allows you to explore one data file. `movement_loop` will loop through files within a folder matching a naming convention. 

Output: `displacement.csv` with `time`, `frame`, `displacement`, and
`displacement_smooth` (5-frame rolling median, ~165 ms at 30 fps for visualization purposes). Units are shoulder-widths per frame.

### `graphs.py` — two-participant line plot

Reads the two participants' displacement CSVs for a trial and plots their time series on shared axes so their movement can be eyeballed against each other before any formal analysis. This file loops through a folder (where the displacement files are saved), matches the participant according to file_name, then creates one plot per trial. 

### `mdrqa_displacement_det_windows.Rmd` — mdRQA and sliding-window DET

Time-matches the clueGiver and guesser displacement streams on an inner join, z-scores each, and treats the dyad as a single 2-dimensional system (one stream per person, no embedding). Runs mdRQA via the `crqa` package with Euclidean distance, setting the radius per window to fix recurrence rate at 5% so that DET reflects the structure of recurrence rather than its density.

The second half slides a 2s window (60 samples) across the trial at a 1s hop, recomputing DET at each step, flagging windows above mean + 2 SD (within-trial), and merging contiguous flagged windows into timestamped events. These values can be modified to change fixed RR, embedding, windows, overlap, etc. etc.

*Note*: These files read in gaze data beacuse our original plan was to use mdRQA with movement and gaze. 

### `Data_exploration.Rmd` — delta scores for gaze and facial expressivity, line plots and WMS plots

Data extraction and compilation over all trials is included in the beginning of the file. You need to specify the correct path for this (to the OpenFace output data that is used in this script) or include that BalanceData folder in the same folder as this script (your working directory). After doing this once, the dataframe is extracted to csv and can be imported to save time (skipping the first code blocks). The code then goes through computing delta scores, initial visualization with line plots and finally, visualization with WMS plots.

---
See the presentation slides for an overview of the research plan, plot outputs, references, and authors.
