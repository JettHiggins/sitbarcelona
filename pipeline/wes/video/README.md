# Other-game video gate

The tracking-only segmentation variants are evaluated before visual modeling.
`sample-data` supplies tracking and events; it does not establish that a video is
available and synchronized to those games. Audit any separately obtained, permitted
other-game videos with:

```bash
python -m pipeline.wes.video.audit --video-root /path/to/other-game-videos
```

An optional JSON alignment manifest may give each `bench1`, `bench2`, `bench3` a
`sync_verified` and `labels_usable` boolean, with a source and method for each.
The audit reports metadata but does not infer either boolean from filenames or duration.
`ffprobe` (from FFmpeg) is optional; it adds fps and duration metadata. It does not
prove alignment. The report stays under the Git-ignored `pipeline/reports/wes/` directory.

Supervised video training waits for verified synchronization and usable other-game
labels. If those are unavailable, the next experiment should use locally run,
inspectable pretrained detectors and appearance features only in short uncertain
windows, with a tracking-only output preserved for comparison. The challenge match
must never be hand-labeled or hand-corrected. The public five-minute labels are not
used to select a model or setting.
