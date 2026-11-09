# Possession viewer

A local viewer for checking frame-by-frame possession predictions against the tactical video.

## Run

```bash
cd possession-viewer
python3 -m http.server 8080
```

Open <http://localhost:8080>. Choose **Open video** and **Predictions CSV**.
Use ↶ to remove the video, or hover over the prediction letter to remove the CSV. Files stay in your browser.

The CSV needs `frame,label` columns with frames 0–199318 and labels H (home),
A (away), D (dead ball), or X (unknown). The viewer uses 25 fps.
A local copy of the challenge baseline is at `data/baseline_smooth50.csv`, ignored by Git.

A warning symbol appears only when frame counts differ or the CSV has missing or duplicate frames. Hover over it for details. Video frames are estimated from duration × 25 fps.
For an exact check, with FFmpeg installed:

```bash
python3 verify_frames.py ~/Downloads/BAR-ATH_tactical.mp4 data/baseline_smooth50.csv
```

This checks counts, frame IDs, order, and labels. Matching counts do not establish prediction accuracy.

## Controls

- » / «: increase / decrease playback speed through 1×, 2×, 4×, and 8×
- Space: play or pause
- Left/right: step one frame
- Shift + left/right: step one second
- Scrubber: seek
- Frame number: jump to a frame

Use the tactical video: the broadcast feed has a different timeline.
Do not commit licensed match video or datasets.

## Compare pipelines

See [pipeline setup and commands](../pipeline/README.md) to generate `lenny-model.csv`,
`mall-model.csv`, or a model-free `mall-events.csv` locally. Load one using **Predictions CSV**,
then choose another CSV to compare at the same video frame. Outputs use the same frame timeline.

## Public ground-truth overlay

The public five-minute reference loads automatically: frames **11185–18684** (25 fps).
Click **Ground truth** beneath the graph to jump to the start. The current expected label appears
beside the prediction, with a white dashed reference on the local graph. Amber shading marks
disagreements; an amber border highlights the video while its current prediction differs.
The full-match timeline shows predictions above, the reference window below, and amber mismatch
marks between them. These marks include short errors that would otherwise disappear at this scale.

Outside the public window, no expected label or match/mismatch is shown. A prediction of X or a
missing prediction counts as a disagreement when a reference label exists. Removing predictions
retains the reference. Reference labels are for visual evaluation only and do not affect pipelines.

`reference/public-ground-truth.json` stores the official public CSV as 21 half-open segments
(start included, end excluded), with the original source URL and SHA-256. No match video is included.

Browser regression checks (with `playwright-cli` installed and the viewer served on port 8080):

```bash
playwright-cli open http://localhost:8080
playwright-cli run-code --filename=possession-viewer/tests/ground-truth.js
```

Run those commands from the repository root. The two tiny CSV fixtures are synthetic.
