/* Short, bounded windows of decoded frames. Nothing here seeks the HTML video element. */
class ReverseFrameCache {
  constructor(file, { fps = 25, budget = 256 * 1024 * 1024 } = {}) {
    this.fps = fps;
    this.budget = budget;
    this.frames = new Map();
    this.chunks = new Set();
    this.targetChunk = null;
    this.worker = null;
    this.disposed = false;
    this.error = null;
    this.bytes = 0;
    this.decodedFrames = 0;
    this.chunkDecodes = 0;
    this.ready = this.initialize(file);
    // Initialization can finish before the first reverse request.
    this.ready.catch(() => {});
  }

  async initialize(file) {
    if (typeof VideoDecoder === 'undefined' || typeof createImageBitmap === 'undefined') {
      throw new Error('This browser does not support decoded-frame playback.');
    }
    const { Input, ALL_FORMATS, BlobSource, CanvasSink } = await import('./vendor/mediabunny/mediabunny.min.mjs');
    if (this.disposed) return;
    this.input = new Input({ formats: ALL_FORMATS, source: new BlobSource(file) });
    const track = await this.input.getPrimaryVideoTrack();
    if (!track || !await track.canDecode()) throw new Error('This browser cannot decode this video into cached frames.');
    if (this.disposed) return;
    const sourceWidth = await track.getDisplayWidth();
    const sourceHeight = await track.getDisplayHeight();
    this.firstFrame = Math.max(0, Math.ceil(await track.getFirstTimestamp() * this.fps - 0.00001));
    if (this.disposed) return;
    // Preserve 1080p details; cap larger footage to keep a useful bounded window.
    const scale = Math.min(1, 1920 / Math.max(sourceWidth, sourceHeight));
    this.width = Math.max(1, Math.round(sourceWidth * scale));
    this.height = Math.max(1, Math.round(sourceHeight * scale));
    this.frameBytes = this.width * this.height * 4;
    this.capacity = Math.floor(this.budget / this.frameBytes);
    if (this.capacity < 3) throw new Error('Video frames exceed the reverse-cache budget.');
    this.chunkFrames = Math.max(1, Math.min(this.fps, Math.floor(this.capacity / 3)));
    this.sink = new CanvasSink(track, { width: this.width, height: this.height, fit: 'fill', poolSize: 1 });
  }

  wanted(chunk) {
    return chunk >= 0 && this.targetChunk !== null && chunk <= this.targetChunk && chunk >= this.targetChunk - 2;
  }

  get(frame) { return this.frames.get(frame); }

  trim() {
    for (const [frame, entry] of this.frames) {
      if (!this.wanted(Math.floor(frame / this.chunkFrames))) {
        entry.bitmap.close();
        this.frames.delete(frame);
        this.bytes -= this.frameBytes;
      }
    }
    for (const chunk of this.chunks) if (!this.wanted(chunk)) this.chunks.delete(chunk);
  }

  async warm(frame) {
    try {
      await this.ready;
      if (this.disposed) return;
      if (this.error) throw this.error;
      this.targetChunk = Math.floor(Math.max(0, frame) / this.chunkFrames);
      this.trim();
      const target = this.targetChunk;
      do { await this.work(); } while (!this.disposed && target === this.targetChunk && !this.chunks.has(target));
      if (!this.disposed && this.error) throw this.error;
    } catch (error) {
      if (!this.disposed) { this.error = error; throw error; }
    }
  }

  work() {
    if (this.worker) return this.worker;
    this.worker = (async () => {
      while (!this.disposed) {
        const chunk = [this.targetChunk, this.targetChunk - 1, this.targetChunk - 2]
          .find(value => value >= 0 && !this.chunks.has(value));
        if (chunk === undefined) return;
        await this.decode(chunk);
      }
    })().finally(() => { this.worker = null; });
    return this.worker;
  }

  async decode(chunk) {
    const start = chunk * this.chunkFrames;
    const times = Array.from({ length: this.chunkFrames }, (_, i) => (start + i + 0.001) / this.fps);
    let i = 0;
    this.chunkDecodes++;
    // A single pooled canvas is copied into independent bitmaps before it is reused.
    for await (const sample of this.sink.canvasesAtTimestamps(times)) {
      if (this.disposed || !this.wanted(chunk)) break;
      const frame = start + i++;
      if (!sample) continue;
      const bitmap = await createImageBitmap(sample.canvas);
      if (this.disposed || !this.wanted(chunk)) { bitmap.close(); break; }
      const previous = this.frames.get(frame);
      if (previous) { previous.bitmap.close(); this.bytes -= this.frameBytes; }
      this.frames.set(frame, { bitmap, timestamp: sample.timestamp });
      this.bytes += this.frameBytes;
      this.decodedFrames++;
      if (this.bytes > this.budget) throw new Error('Reverse cache exceeded its frame budget.');
    }
    if (!this.disposed && this.wanted(chunk)) this.chunks.add(chunk);
  }

  dispose() {
    this.disposed = true;
    this.targetChunk = null;
    for (const entry of this.frames.values()) entry.bitmap.close();
    this.frames.clear();
    this.chunks.clear();
    this.bytes = 0;
    this.input?.dispose();
  }
}
