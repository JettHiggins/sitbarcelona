// Shared by the decoding worker and the compatibility path on the main thread.
export class ReverseDecoder {
  async initialize(file, fps) {
    const { Input, ALL_FORMATS, BlobSource, CanvasSink, VideoSampleSink } = await import('./vendor/mediabunny/mediabunny.min.mjs');
    this.input = new Input({ formats: ALL_FORMATS, source: new BlobSource(file) });
    const track = await this.input.getPrimaryVideoTrack();
    if (!track || !await track.canDecode()) throw new Error('This browser cannot decode this video into cached frames.');
    const sourceWidth = await track.getDisplayWidth();
    const sourceHeight = await track.getDisplayHeight();
    const firstFrame = Math.max(0, Math.ceil(await track.getFirstTimestamp() * fps - 0.00001));
    const scale = Math.min(1, 1920 / Math.max(sourceWidth, sourceHeight));
    const width = Math.max(1, Math.round(sourceWidth * scale));
    const height = Math.max(1, Math.round(sourceHeight * scale));
    let frameBytes = width * height * 4;
    // Packed native planes take much less space than RGBA bitmaps. Copy them
    // out of decoder surfaces so retaining a reverse window cannot stall it.
    if (scale === 1 && await track.getRotation() === 0 && !await track.getFlip()) {
      const sink = new VideoSampleSink(track);
      const sample = await sink.getSample((firstFrame + 0.001) / fps);
      try {
        if (sample) {
          const entry = await this.packSample(sample);
          frameBytes = entry.bytes;
          entry.bitmap.close();
          this.sampleSink = sink;
        }
      } catch {
        // Opaque or non-copyable surfaces retain the canvas compatibility path.
      } finally { sample?.close(); }
    }
    if (!this.sampleSink) this.sink = new CanvasSink(track, { width, height, fit: 'fill', poolSize: 1 });
    return { width, height, firstFrame, frameBytes };
  }

  async packSample(sample) {
    const source = sample.toVideoFrame();
    try {
      const data = new Uint8Array(source.allocationSize());
      const layout = await source.copyTo(data);
      const bitmap = new VideoFrame(data, {
        format: source.format, codedWidth: source.visibleRect.width,
        codedHeight: source.visibleRect.height, layout,
        displayWidth: source.displayWidth, displayHeight: source.displayHeight,
        timestamp: source.timestamp, duration: source.duration ?? undefined,
        colorSpace: source.colorSpace,
      });
      return { bitmap, bytes: data.byteLength, timestamp: sample.timestamp };
    } finally { source.close(); }
  }

  async *decode(times, cancelled) {
    if (this.sampleSink) {
      for await (const sample of this.sampleSink.samplesAtTimestamps(times)) {
        if (!sample) { yield null; continue; }
        let entry;
        try {
          if (cancelled()) return;
          entry = await this.packSample(sample);
        } finally { sample.close(); }
        if (cancelled()) { entry.bitmap.close(); return; }
        yield entry;
      }
      return;
    }
    for await (const sample of this.sink.canvasesAtTimestamps(times)) {
      if (cancelled()) return;
      if (!sample) { yield null; continue; }
      const bitmap = await createImageBitmap(sample.canvas);
      if (cancelled()) { bitmap.close(); return; }
      yield { bitmap, bytes: bitmap.width * bitmap.height * 4, timestamp: sample.timestamp };
    }
  }

  dispose() { this.input?.dispose(); }
}

if (typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope) {
  const decoder = new ReverseDecoder();
  let generation = 0;
  self.onmessage = async ({ data }) => {
    if (data.type === 'cancel') { ++generation; return; }
    const { id, type } = data;
    try {
      if (type === 'initialize') {
        const metadata = await decoder.initialize(data.file, data.fps);
        self.postMessage({ id, result: metadata });
      } else if (type === 'decode') {
        const version = generation;
        let index = 0;
        for await (const entry of decoder.decode(data.times, () => version !== generation)) {
          if (entry) self.postMessage({ id, index, entry }, [entry.bitmap]);
          ++index;
        }
        self.postMessage({ id, result: null });
      }
    } catch (error) {
      self.postMessage({ id, error: error.message });
    }
  };
}
