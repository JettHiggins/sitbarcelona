/* Short, bounded windows of decoded frames. Nothing here seeks the HTML video element. */
class ReverseFrameCache {
  constructor(file, { fps = 25, budget = 256 * 1024 * 1024 } = {}) {
    this.fps = fps;
    this.budget = budget;
    this.frames = new Map();
    this.chunks = new Set();
    this.targetChunk = null;
    this.worker = null;
    this.waiters = new Set();
    this.warmVersion = 0;
    this.decodeVersion = 0;
    this.decoderRequests = new Map();
    this.decoderSequence = 0;
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
    let metadata;
    if (typeof Worker !== 'undefined' && typeof OffscreenCanvas !== 'undefined') {
      try {
        this.decoderWorker = new Worker(new URL('./reverse-decoder.js', document.baseURI), { type: 'module' });
        this.decoderWorker.onmessage = ({ data }) => {
          const request = this.decoderRequests.get(data.id);
          if (data.entry) {
            if (!request || this.disposed) data.entry.bitmap.close();
            else {
              try { request.onFrame(data.index, data.entry); }
              catch (error) {
                this.decoderRequests.delete(data.id);
                request.reject(error);
              }
            }
            return;
          }
          if (!request) return;
          this.decoderRequests.delete(data.id);
          if (data.error) request.reject(new Error(data.error));
          else request.resolve(data.result);
        };
        this.decoderWorker.onerror = event => {
          event.preventDefault();
          this.stopDecoderWorker(new Error('Reverse decoding worker failed.'));
        };
        metadata = await this.requestDecoder('initialize', { file, fps: this.fps });
      } catch {
        this.stopDecoderWorker(new Error('Worker decoding unavailable.'));
      }
    }
    if (this.disposed) return;
    if (!metadata) {
      const { ReverseDecoder } = await import('./reverse-decoder.js');
      if (this.disposed) return;
      this.decoder = new ReverseDecoder();
      metadata = await this.decoder.initialize(file, this.fps);
      if (this.disposed) { this.decoder.dispose(); return; }
    }
    Object.assign(this, metadata);
    this.capacity = Math.floor(this.budget / this.frameBytes);
    if (this.capacity < 3) throw new Error('Video frames exceed the reverse-cache budget.');
    this.chunkFrames = Math.max(1, Math.min(this.fps, Math.floor(this.capacity / 3)));
  }

  requestDecoder(type, data, onFrame) {
    const id = ++this.decoderSequence;
    return new Promise((resolve, reject) => {
      this.decoderRequests.set(id, { resolve, reject, onFrame });
      try { this.decoderWorker.postMessage({ id, type, ...data }); }
      catch (error) { this.decoderRequests.delete(id); reject(error); }
    });
  }

  stopDecoderWorker(error) {
    this.decoderWorker?.terminate();
    this.decoderWorker = null;
    for (const request of this.decoderRequests.values()) request.reject(error);
    this.decoderRequests.clear();
  }

  wanted(chunk) {
    return chunk >= 0 && this.targetChunk !== null && chunk <= this.targetChunk && chunk >= this.targetChunk - 2;
  }

  get(frame) { return this.frames.get(frame); }

  notifyProgress() {
    for (const resolve of this.waiters) resolve();
    this.waiters.clear();
  }

  cancelWarm() {
    ++this.warmVersion;
    ++this.decodeVersion;
    this.decoderWorker?.postMessage({ type: 'cancel' });
    this.targetChunk = null;
    this.notifyProgress();
  }

  trim() {
    for (const [frame, entry] of this.frames) {
      if (!this.wanted(Math.floor(frame / this.chunkFrames))) {
        entry.bitmap.close();
        this.frames.delete(frame);
        this.bytes -= entry.bytes;
      }
    }
    for (const chunk of this.chunks) if (!this.wanted(chunk)) this.chunks.delete(chunk);
  }

  async warm(frame, { prefetch = false } = {}) {
    const version = ++this.warmVersion;
    try {
      await this.ready;
      if (this.disposed || version !== this.warmVersion) return;
      if (this.error) throw this.error;
      this.targetChunk = Math.floor(Math.max(0, frame) / this.chunkFrames);
      this.trim();
      const target = this.targetChunk;
      const required = (prefetch ? [target, target - 1, target - 2] : [target]).filter(chunk => chunk >= 0);
      // Return as soon as the requested chunk is ready. Earlier reverse chunks
      // continue in the background instead of delaying the displayed image.
      while (!this.disposed && version === this.warmVersion && !required.every(chunk => this.chunks.has(chunk))) {
        const progress = new Promise(resolve => this.waiters.add(resolve));
        this.work().catch(() => {});
        await progress;
        if (this.error) throw this.error;
      }
      if (!this.disposed && version === this.warmVersion) this.work().catch(() => {});
      if (!this.disposed && this.error) throw this.error;
    } catch (error) {
      if (!this.disposed) { this.error = error; throw error; }
    }
  }

  work() {
    if (this.worker) return this.worker;
    this.worker = (async () => {
      while (!this.disposed && this.targetChunk !== null) {
        const chunk = [this.targetChunk, this.targetChunk - 1, this.targetChunk - 2]
          .find(value => value >= 0 && !this.chunks.has(value));
        if (chunk === undefined) return;
        await this.decode(chunk);
        this.notifyProgress();
      }
    })().catch(error => {
      if (!this.disposed) this.error = error;
      throw error;
    }).finally(() => { this.worker = null; this.notifyProgress(); });
    return this.worker;
  }

  async decode(chunk) {
    const version = this.decodeVersion;
    const start = chunk * this.chunkFrames;
    const times = Array.from({ length: this.chunkFrames }, (_, i) => (start + i + 0.001) / this.fps);
    this.chunkDecodes++;
    const store = (index, entry) => {
      if (this.disposed || version !== this.decodeVersion || !this.wanted(chunk)) { entry.bitmap.close(); return; }
      const frame = start + index;
      const previous = this.frames.get(frame);
      if (previous) { previous.bitmap.close(); this.bytes -= previous.bytes; }
      if (this.bytes + entry.bytes > this.budget) {
        entry.bitmap.close();
        throw new Error('Reverse cache exceeded its frame budget.');
      }
      this.frames.set(frame, entry);
      this.bytes += entry.bytes;
      this.decodedFrames++;
    };
    if (this.decoderWorker) {
      await this.requestDecoder('decode', { times }, store);
    } else {
      let index = 0;
      for await (const entry of this.decoder.decode(times, () => this.disposed || version !== this.decodeVersion || !this.wanted(chunk))) {
        if (entry) store(index, entry);
        ++index;
      }
    }
    if (this.bytes > this.budget) throw new Error('Reverse cache exceeded its frame budget.');
    if (!this.disposed && version === this.decodeVersion && this.wanted(chunk)) this.chunks.add(chunk);
  }

  dispose() {
    this.disposed = true;
    this.cancelWarm();
    for (const entry of this.frames.values()) entry.bitmap.close();
    this.frames.clear();
    this.chunks.clear();
    this.bytes = 0;
    this.stopDecoderWorker(new Error('Reverse cache disposed.'));
    this.decoder?.dispose();
  }
}
