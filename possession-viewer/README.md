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
The viewer displays **BAR** for H and **ATH** for A in predictions, ground truth and graph labels.
Team names and abbreviations live in `match.js`; CSV codes stay H/A/D/X. Changing these names
alone does not add a new match timeline: other games still need their frame/FPS settings and
reference data configured. Reference labels are checked against the configured match ID.

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
Reverse now displays a rolling cache of decoded frames; it does not seek the video element
for each frame. Forward playback stays on the normal video element. Use 0.25× or 0.5× to inspect
touches in either direction. Reverse audio is not played.

## Reverse-frame cache

[Mediabunny](vendor/mediabunny/README.md) reads short sections directly from the chosen local
file and decodes them into independent frame bitmaps. The current section and two preceding
sections are retained, with older bitmaps explicitly closed. The cache adapts section size to
resolution and caps stored RGBA frame data at **256 MiB**; decoder and display buffers add some
memory beyond that. Full 1920×1080 detail is preserved. Larger sources are scaled to a maximum
1920-pixel edge for cached reverse; normal forward playback keeps the original resolution.

Paused positions are warmed in the background. Reverse prefetches earlier sections while
showing cached frames. A short **Preparing reverse…** notice appears on a cold start or if
prefetch cannot keep up, especially at high reverse speeds. While refilling, the image and
prediction clock stay on the same frame. Pausing synchronizes the native video once; switching
forward resumes from the cached position. Seeking, replacing or removing a video cancels stale
playback work, and replacing/removing it closes the cache's bitmaps and decoder input.

The dependency is version-pinned and served locally, with its license and source link in
`vendor/mediabunny/`. There is no CDN, video upload, full-match decode, or reversed video file.
If the browser or codec cannot supply decoded frames, the viewer labels its slower compatibility
mode **Basic rewind · smooth reverse unavailable**. Forward playback still works.

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
  -c:v libvpx-vp9 -deadline realtime -cpu-used 8 -threads 2 \
  -colorspace bt709 -color_trc bt709 -color_primaries bt709 -color_range tv -y \
  possession-viewer/tests/fixtures/inspection.webm
playwright-cli run-code --filename=possession-viewer/tests/inspection.js
playwright-cli run-code --filename=possession-viewer/tests/reverse-cache.js
```

The generated clip is ignored by Git and contains no match footage.

The cache checks use a deliberately small 8 MiB budget to force section changes. They compare
cached/native frame pixels and timestamps and check bounded memory, bitmap disposal, zero
native seeks during cached reverse, forward handoff, cancelled preparation, file replacement,
and unsupported-decoder fallback. A manual Chromium check on the 1920×1080 H.264 tactical file
at frame 12000 displayed 106 distinct frames in 4.2 seconds at −1×, with zero per-frame native
video seeks and a maximum sampled inter-frame gap of 64 ms. This is a local observation, not a
performance guarantee on other machines or codecs.
