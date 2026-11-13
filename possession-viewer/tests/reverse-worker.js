// Local server and synthetic fixture setup: README.md.
async page => {
  const check = (value, message) => { if (!value) throw new Error(message); };
  await page.route('**/*', route => route.continue());
  await page.goto('http://localhost:8080');
  await page.locator('#video-file').setInputFiles('possession-viewer/tests/fixtures/inspection.webm');
  await page.waitForFunction(() => state.hasVideo && video.readyState >= 2);
  const result = await page.evaluate(async () => {
    clearTimeout(state.warmTimer);
    const cache = state.reverseCache;
    await cache.warm(100, { prefetch: true });
    const chunk = Math.floor(100 / cache.chunkFrames);
    const result = {
      worker: !!cache.decoderWorker,
      packed: cache.frameBytes < cache.width * cache.height * 4,
      primed: [chunk, chunk - 1, chunk - 2].filter(c => c >= 0).every(c => cache.chunks.has(c)),
      bytes: cache.bytes, budget: cache.budget,
    };
    // Cancel and immediately revisit the same section, including partially decoded work.
    cache.cancelWarm();
    const pending = cache.warm(20);
    await new Promise(resolve => setTimeout(resolve, 10));
    cache.cancelWarm();
    await cache.warm(20, { prefetch: true });
    await pending;
    const target = Math.floor(20 / cache.chunkFrames);
    result.complete = Array.from({ length: cache.chunkFrames }, (_, i) => target * cache.chunkFrames + i)
      .every(frame => !!cache.get(frame));
    result.requests = cache.decoderRequests.size;
    return result;
  });
  check(result.worker && result.packed, 'Worker decoding retains compact native frames');
  check(result.primed && result.bytes <= result.budget, 'Reverse primes its look-behind buffer within the memory budget');
  check(result.complete && result.requests === 0, 'Cancelled decoding cannot mark a partial section as complete');
  // Worker restrictions retain the same decoded-frame controls on the main thread.
  await page.evaluate(async () => {
    state.reverseCache.dispose();
    const WorkerClass = window.Worker;
    window.Worker = class { constructor() { throw new Error('Worker blocked by test'); } };
    try {
      state.reverseCache = new ReverseFrameCache(state.videoFile, { fps: FPS });
      await state.reverseCache.warm(50);
    } finally { window.Worker = WorkerClass; }
  });
  check(await page.evaluate(() => !state.reverseCache.decoderWorker && !!state.reverseCache.get(50)),
    'Main-thread decoding works when workers cannot start');
  const bitmapFallback = await page.evaluate(async () => {
    const { ReverseDecoder } = await import('/reverse-decoder.js');
    const decoder = new ReverseDecoder();
    decoder.packSample = async () => { throw new Error('Non-copyable surface'); };
    try {
      const metadata = await decoder.initialize(state.videoFile, FPS);
      for await (const entry of decoder.decode([2], () => false)) {
        if (!entry) return false;
        const valid = entry.bitmap instanceof ImageBitmap && Math.abs(entry.timestamp - 2) < 0.001 &&
          entry.bytes === metadata.frameBytes;
        entry.bitmap.close();
        return valid;
      }
      return false;
    } finally { decoder.dispose(); }
  });
  check(bitmapFallback, 'Non-copyable surfaces retain timestamped canvas/bitmap decoding');
  await page.evaluate(() => state.reverseCache.dispose());
  check(await page.evaluate(() => state.reverseCache.frames.size === 0 && state.reverseCache.decoderRequests.size === 0),
    'Disposal releases frames and outstanding worker requests');
  return { result: 'PASS: worker decoding, compact storage, priming, cancel/revisit, worker fallback, disposal', ...result };
}
