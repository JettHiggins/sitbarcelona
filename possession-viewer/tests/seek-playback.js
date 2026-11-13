// Synthetic fixture and local server setup are documented in README.md.
async page => {
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  await page.route('**/*', route => route.continue());
  await page.goto('http://localhost:8080');
  await page.locator('#video-file').setInputFiles('possession-viewer/tests/fixtures/inspection.webm');
  await page.waitForFunction(() => state.hasVideo && video.readyState >= 2);
  await page.evaluate(() => seekToFrame(100));
  await page.evaluate(() => startPlayback(1));
  // Hold inspection decoding to exercise scrubbing and Play before it finishes.
  await page.evaluate(() => {
    const cache = state.reverseCache;
    const warm = cache.warm.bind(cache);
    const get = cache.get.bind(cache);
    let release;
    const gate = new Promise(resolve => { release = resolve; });
    window.releaseSeek = () => { cache.warm = warm; cache.get = get; release(); };
    cache.get = () => undefined;
    cache.warm = async frame => { await gate; return warm(frame); };
    for (const frame of [80, 40, 60, 25]) {
      const slider = document.getElementById('seek');
      slider.value = String(frame);
      slider.dispatchEvent(new Event('input'));
      if (slider.value !== String(frame)) throw new Error('Scrubber snapped back while decoding');
    }
  });
  check(await page.evaluate(() => state.pendingFrame === 25 && document.getElementById('seek').value === '25'),
    'Rapid scrubbing keeps the latest requested thumb position');
  await page.locator('#play').click();
  await page.waitForFunction(() => !video.paused && !state.starting && state.frame > 28 && !state.cacheVisible);
  await page.evaluate(() => window.releaseSeek());
  await page.waitForTimeout(150);
  check(await page.evaluate(() => !video.paused && !state.cacheVisible && state.pendingFrame === null),
    'Stale decoded seeks cannot cover or pause resumed playback');
  for (const frame of [80, 20, 90, 10]) {
    await page.evaluate(frame => seekToFrame(frame), frame);
    await page.locator('#play').click();
    await page.waitForFunction(frame => !video.paused && state.frame > frame + 2 && !state.cacheVisible, frame);
  }
  await page.evaluate(() => { pausePlayback(false); void seekToFrame(70); pausePlayback(false); });
  check(await page.evaluate(() => !state.starting && video.paused && state.cancelNativeSeek === null),
    'Cancelling an in-flight seek releases its listener and playback state');
  return 'PASS: rapid scrubbing, immediate Play, stale decode cancellation, repeated backward seek/play, and seek cleanup';
}
