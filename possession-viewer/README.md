# Possession viewer

A local viewer for checking frame-by-frame possession predictions against the tactical video.

## Run

```bash
cd possession-viewer
python3 -m http.server 8080
```

Open <http://localhost:8080>. Choose **Open video**, then select a CSV from the
**Predictions CSV** dropdown. It lists CSV files in `data/`, including generated pipeline outputs.
Use **Choose file…** to load a CSV from elsewhere on your computer. The dropdown stays visible
so you can switch predictions at the same video frame. Its file list refreshes when focused.
Use ↶ to remove the video, or hover over the prediction letter to remove the CSV.
Manually chosen files stay in your browser; files in `data/` are read from your local server.

The automatic list uses the directory listing from `python3 -m http.server`. If the server
does not provide directory listings, **Choose file…** still works.

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

- « / »: decrease / increase playback speed by one step
- Speed dropdown: choose a signed speed directly; negative values play in reverse
- Space: pause or resume in the current direction
- Left/right: step one frame
- Shift + left/right: step one second
- Scrubber: seek
- Frame number: jump to a frame

The speed steps are −8, −4, −2, −1, −0.5, −0.25, +0.25, +0.5, +1, +2, +4, +8×.
Changing speed preserves play/pause state. For example, « from +1× selects +0.5×; another
click selects +0.25×, then −0.25×. Play/pause is separate from the speed steps.

Reverse stops at the start of the video. Seeking or stepping a frame pauses continuous playback.
Reverse uses decoded frame seeks rather than negative browser playback rates; it has no reverse
audio, and smoothness depends on the browser, video encoding and speed. Use 0.25× or 0.5× to
inspect touches in either direction.

## Zoom and pan

After loading a video, the controls at the bottom left offer 1–8× zoom. The slider supports
0.01× adjustments; −/+ buttons move by 0.25×. Scroll over the footage to zoom toward the cursor
(hold Shift for smaller changes), then drag to pan. The view is constrained to the footage bounds.
Click the zoom value or double-click the footage to reset the full view. Zoom persists while
playing, reversing, seeking and switching predictions; removing or replacing the video resets it.
The prediction/reference graph stays fixed and independent of zoom.

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
playwright-cli run-code --filename=possession-viewer/tests/csv-picker.js
```

Run those commands from the repository root. The two tiny CSV fixtures are synthetic.

Playback and zoom regression checks use a generated six-second synthetic video (FFmpeg required):

```bash
ffmpeg -hide_banner -loglevel error -f lavfi -i testsrc2=size=640x360:rate=25 -t 6 \
  -c:v libvpx-vp9 -deadline realtime -cpu-used 8 -threads 2 -y \
  possession-viewer/tests/fixtures/inspection.webm
playwright-cli run-code --filename=possession-viewer/tests/inspection.js
```

The generated clip is ignored by Git and contains no match footage.
