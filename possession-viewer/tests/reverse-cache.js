// Uses the generated inspection.webm fixture documented in README.md.
// playwright-cli run-code --filename=possession-viewer/tests/reverse-cache.js
async page => {
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  const fixture = 'possession-viewer/tests/fixtures/inspection.webm';
  const seek = async frame => {
    await page.locator('#frame-input').fill(String(frame));
    await page.locator('#frame-input').press('Enter');
    await page.waitForFunction(() => !video.seeking);
  };
  await page.goto('http://localhost:8080');
  await page.locator('#video-file').setInputFiles(fixture);
  await page.waitForFunction(() => state.hasVideo && video.readyState >= 2);
  await seek(100);
  const initial = await page.evaluate(async () => {
    clearTimeout(state.warmTimer);
    state.reverseCache.dispose();
    // Deliberately small budget forces frequent chunk changes and evictions.
    state.reverseCache = new ReverseFrameCache(state.videoFile, { fps: FPS, budget: 8 * 1024 * 1024 });
    const cache = state.reverseCache;
    await cache.warm(100);
    const entry = cache.get(100);
    const comparison = document.createElement('canvas');
    comparison.width = video.videoWidth; comparison.height = video.videoHeight;
    const context = comparison.getContext('2d', { willReadFrequently: true });
    context.drawImage(video, 0, 0);
    const native = context.getImageData(0, 0, comparison.width, comparison.height).data;
    context.drawImage(entry.bitmap, 0, 0);
    const cached = context.getImageData(0, 0, comparison.width, comparison.height).data;
    let difference = 0;
    for (let i = 0; i < native.length; i++) difference += Math.abs(native[i] - cached[i]);
    window.cacheProbe = { seeks: 0, cache, initialBitmap: entry.bitmap };
    video.addEventListener('seeking', () => window.cacheProbe.seeks++);
    return { frames: cache.frames.size, capacity: cache.capacity, bytes: cache.bytes,
      budget: cache.budget, timestamp: entry.timestamp, pixelDifference: difference / native.length };
  });
  check(initial.bytes <= initial.budget && initial.frames <= initial.capacity, 'Decoded cache obeys its budget');
  check(Math.abs(initial.timestamp - 4) < 0.001, 'Cached frame timestamp matches the requested video frame');
  check(initial.pixelDifference < 1, 'Cached pixels agree with the native video at the same timestamp');
  await page.locator('#playback-rate').selectOption('-1');
  await page.locator('#play').click();
  await page.waitForFunction(() => state.reverseMode === 'cache' && state.frame < 75);
  const running = await page.evaluate(() => ({
    frame: state.frame, seeks: window.cacheProbe.seeks,
    bounded: state.reverseCache.bytes <= state.reverseCache.budget,
    visible: state.cacheVisible, nativePaused: video.paused,
  }));
  check(running.seeks === 0 && running.nativePaused, 'Cached reverse does not seek or play the native video per frame');
  check(running.bounded && running.visible, 'Cache stays bounded across reverse chunk boundaries');
  await page.locator('#play').click();
  await page.waitForFunction(() => !video.seeking);
  check(await page.evaluate(() => window.cacheProbe.seeks === 0), 'Pausing reverse holds its cached image without seeking the native video');
  check(await page.evaluate(() => !state.reversing && state.cacheVisible), 'Paused cached frame stays visible');

  await page.locator('#play').click();
  await page.waitForFunction(() => state.reversing);
  await page.locator('#playback-rate').selectOption('0.5');
  await page.waitForFunction(() => !state.starting && !state.reversing && !video.paused && !state.cacheVisible);
  check(await page.evaluate(() => video.currentTime < 4), 'Forward playback resumes from the cached position without jumping to the old native clock');
  await page.locator('#play').click();

  // A pending preparation must not take over after the user changes direction.
  await seek(90);
  await page.evaluate(() => {
    const ready = state.reverseCache.ready;
    state.reverseCache.ready = new Promise(resolve => { window.releaseCacheReady = () => ready.then(resolve); });
  });
  await page.locator('#playback-rate').selectOption('-1');
  await page.locator('#play').click();
  await page.waitForFunction(() => state.reversing);
  await page.locator('#playback-rate').selectOption('1');
  await page.evaluate(() => window.releaseCacheReady());
  await page.waitForTimeout(300);
  check(await page.evaluate(() => !video.paused && !state.reversing && !state.cacheVisible), 'Cancelled reverse preparation cannot replace forward playback');
  await page.locator('#play').click();

  await page.evaluate(async () => {
    await state.reverseCache.warm(10);
  });
  check(await page.evaluate(() => !state.reverseCache.get(100) && window.cacheProbe.initialBitmap.width === 0), 'Distant windows evict and close old bitmaps');
  const previousCache = await page.evaluateHandle(() => state.reverseCache);
  await page.locator('#video-file').setInputFiles([]);
  await page.locator('#video-file').setInputFiles(fixture);
  await page.waitForFunction(old => state.hasVideo && state.reverseCache !== old, previousCache);
  check(await previousCache.evaluate(cache => cache.disposed && cache.frames.size === 0 && cache.bytes === 0), 'Replacing the video frees the old cache');
  await page.waitForTimeout(250);
  check(await page.evaluate(() => state.frame === 0 && !state.cacheVisible), 'Old decoding work cannot paint the replacement video');
  await previousCache.dispose();

  // Basic playback remains usable if the browser cannot provide WebCodecs frames.
  await page.evaluate(() => { window.savedVideoDecoder = window.VideoDecoder; window.VideoDecoder = undefined; });
  await page.locator('#video-file').setInputFiles([]);
  await page.locator('#video-file').setInputFiles(fixture);
  await page.waitForFunction(() => state.hasVideo && video.readyState >= 2);
  await seek(100);
  await page.locator('#playback-rate').selectOption('-1');
  await page.locator('#play').click();
  await page.waitForFunction(() => state.reverseMode === 'seeking' && state.frame < 95);
  check(await page.locator('#reverse-status').isVisible(), 'Unsupported decoding clearly shows the basic-rewind fallback');
  await page.locator('#play').click();
  await page.evaluate(() => { window.VideoDecoder = window.savedVideoDecoder; });
  await page.locator('#reset-video').click();
  check(await page.evaluate(() => !state.reverseCache && !state.cacheVisible && !state.reversing && !state.starting), 'Removing video clears cache and pending playback');
  check(await page.locator('#reverse-frame').isHidden(), 'Cached display hides on reset');
  check(await page.locator('#reverse-status').isHidden(), 'Buffering/fallback notice hides on reset');
  return { result: 'PASS: cache pixels/timestamps, memory cap, eviction, zero-seek reverse, pause/forward handoff, cancellation, replacement, fallback and reset', initial, running };
}
