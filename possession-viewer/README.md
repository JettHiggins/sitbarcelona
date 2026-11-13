# Possession viewer

A local viewer for checking frame-by-frame possession predictions against the tactical video.

## Run

```bash
cd possession-viewer
python3 server.py
```

Open <http://127.0.0.1:8081/>. The dedicated port avoids cached apps previously served on
8080, and this server disables HTTP caching and lists predictions through `/api/predictions`.
Choose **Open video**, then select a CSV from the
**Predictions CSV** dropdown. It lists CSV files in `data/`, including generated pipeline outputs.
Use **Choose file…** to load a CSV from elsewhere on your computer. The dropdown stays visible
so you can switch predictions at the same video frame. Its file list refreshes when focused.
Use ↶ to remove the video, or hover over the prediction letter to remove the CSV.
Manually chosen files stay in your browser; files in `data/` are read from your local server.

The dedicated server lists predictions directly. On ordinary static servers, the automatic
list falls back to the `data/` directory listing. **Choose file…** also works without either.

The dedicated-server check is `playwright-cli run-code --filename=possession-viewer/tests/prediction-server.js`.
Existing viewer regression scripts use port 8080; start `python3 server.py --port 8080` for those.

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
- Scrubber: live video previews while dragging; exact frame on release
- Frame number: jump to a frame
- Scroll over the bottom timeline: zoom around the cursor, down to a one-second/25-frame view
- Shift+scroll (or horizontal trackpad scroll) over the timeline: pan the visible window

Timeline zoom adds no buttons. Scroll back out for the full match. The existing **Ground truth**
link also fits the public five-minute window. Endpoint labels show exact frame bounds, hover
shows the frame and labels under the cursor, and clicking/dragging seeks within the visible
window. Playback or a frame jump keeps the current frame visible at the selected zoom.
Video zoom and timeline zoom are independent.

The speed steps are −8, −4, −2, −1, −0.5, −0.25, +0.25, +0.5, +1, +2, +4, +8×.
Changing speed preserves play/pause state. For example, « from +1× selects +0.5×; another
click selects +0.25×, then −0.25×. Play/pause is separate from the speed steps.

Reverse stops at the start of the video. Seeking or stepping a frame pauses continuous playback.
Reverse now displays a rolling cache of decoded frames; it does not seek the video element
for each frame. Forward playback stays on the normal video element. Use 0.25× or 0.5× to inspect
touches in either direction. Reverse audio is not played.

## Reverse-frame cache

[Mediabunny](vendor/mediabunny/README.md) reads short sections directly from the chosen local
file through WebCodecs. A module worker handles decoding and frame preparation, transferring
ready frames to the viewer. If workers cannot start, the same decoder runs on the main thread.
Copyable native video planes are retained as independent `VideoFrame` objects, avoiding an
RGBA canvas conversion for every decoded frame. Decoder surfaces are released immediately.
Rotated, scaled, or opaque sources use the canvas/bitmap compatibility path.

The current section and two preceding sections are retained, with older frames explicitly closed.
The cache adapts section size to resolution and pixel format and caps stored frame data at
**256 MiB**; decoder, transfer, temporary-copy, and display buffers add memory beyond that.
For the 1080p H.264 tactical video, packed frames use 3,110,400 bytes instead of 8,294,400 RGBA
bytes, allowing three one-second sections in the same budget. Full 1920×1080 detail is preserved.
Larger sources are scaled to a maximum 1920-pixel edge for cached reverse; normal forward
playback keeps the original resolution.

Paused positions are warmed in the background. Reverse primes the current and two preceding
sections before starting, then prefetches earlier sections while
showing cached frames. Cold starts or refills may briefly hold the image, especially at high
reverse speeds, without displaying a preparation notice. While refilling, the image and
prediction clock stay on the same frame. Switching forward from inspection or reverse
synchronizes the native video once. Ordinary forward pause/resume keeps the browser's
current position without seeking or requesting an exact-frame image. Seeking, replacing or removing a video cancels stale
playback work, and replacing/removing it closes the cached frames, decoder input, and worker.

Dragging the scrubber uses native video previews, with one seek in flight and only the latest
destination queued. Reverse-cache decoding is suspended during the drag. Releasing the scrubber
restores exact inspection; keyboard scrubbing settles after a brief idle. Browsers with fast seeking
may show nearby keyframes during dragging. Exact inspection waits only for the requested cache
section; earlier sections prefetch in the background.

Frame stepping and paused seeks display the exact decoded frame with its matching possession
label. Quick repeated steps accumulate their requested destination while decoding; the old
image keeps its old label until the new image is ready. Forward playback takes the frame
number from the browser's displayed-frame timestamp rather than its playback clock, and
ordinary pause holds the native displayed image. Forward playback cancels background
reverse preparation. Browsers without decoded-frame support retain native seeking.

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
playwright-cli run-code --filename=possession-viewer/tests/timeline-zoom.js
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
playwright-cli run-code --filename=possession-viewer/tests/reverse-worker.js
playwright-cli run-code --filename=possession-viewer/tests/frame-sync.js
playwright-cli run-code --filename=possession-viewer/tests/native-playback.js
playwright-cli run-code --filename=possession-viewer/tests/seek-playback.js
playwright-cli run-code --filename=possession-viewer/tests/scrubbing.js
```

The generated clip is ignored by Git and contains no match footage.

The cache checks use a deliberately small 8 MiB budget to force section changes. They compare
cached/native frame pixels and timestamps and check bounded memory, bitmap disposal, zero
native seeks during cached reverse, forward handoff, cancelled preparation, file replacement,
and unsupported-decoder fallback. A local Chromium comparison on the 1920×1080 H.264 tactical file at frame 12000 measured
a maximum display gap of 203 ms before these changes. With worker decoding, packed planes,
and priming, a ten-second −1× run advanced 249 frames with zero cache misses and a maximum
display gap of 54 ms; priming took 282 ms. This is a local observation, not a performance
guarantee on other machines or codecs.
