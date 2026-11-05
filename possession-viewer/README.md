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
