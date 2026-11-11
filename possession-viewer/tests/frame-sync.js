// Uses the ignored inspection.webm fixture documented in README.md.
// playwright-cli run-code --filename=possession-viewer/tests/frame-sync.js
async page => {
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  await page.goto('http://localhost:8080');
  await page.locator('#video-file').setInputFiles('possession-viewer/tests/fixtures/inspection.webm');
  await page.waitForFunction(() => state.hasVideo && video.readyState >= 2);
  const inspect = async frame => {
    await page.evaluate(frame => seekToFrame(frame), frame);
    await page.waitForFunction(frame => state.pendingFrame === null && state.frame === frame && state.cacheVisible, frame);
    return page.evaluate(() => {
      const entry = state.reverseCache.get(state.frame);
      const actual = document.getElementById('reverse-frame');
      const expected = document.createElement('canvas');
      expected.width = actual.width; expected.height = actual.height;
      expected.getContext('2d').drawImage(entry.bitmap, 0, 0);
      const a = actual.getContext('2d').getImageData(0, 0, actual.width, actual.height).data;
      const b = expected.getContext('2d').getImageData(0, 0, actual.width, actual.height).data;
      return { frame: state.frame, timestamp: entry.timestamp,
        pixelsMatch: a.every((value, index) => value === b[index]) };
    });
  };
  for (const frame of [100, 99, 98, 99, 100, 101, 24, 25, 26, 0, 149]) {
    const result = await inspect(frame);
    check(result.pixelsMatch && Math.round(result.timestamp * 25) === frame,
      `Displayed pixels and possession frame agree at ${frame}`);
  }
  await inspect(100);
  // Force decoding to wait so rapid steps exercise the pending target, not the old image.
  await page.evaluate(() => {
    const cache = state.reverseCache;
    const warm = cache.warm.bind(cache);
    let release;
    const gate = new Promise(resolve => { release = resolve; });
    window.releaseInspection = () => { cache.warm = warm; release(); };
    cache.warm = async frame => { await gate; return warm(frame); };
    document.getElementById('back-frame').click();
    document.getElementById('back-frame').click();
    document.getElementById('back-frame').click();
  });
  check(await page.evaluate(() => state.frame === 100 && state.pendingFrame === 97),
    'Rapid backward steps accumulate while the old image keeps its matching label');
  await page.evaluate(() => window.releaseInspection());
  await page.waitForFunction(() => state.pendingFrame === null && state.frame === 97);
  await page.waitForTimeout(150);
  check(await page.evaluate(() => state.frame === 97 && state.cacheVisible), 'Stale seek callbacks do not overwrite the latest inspection');
  for (let i = 0; i < 6; i++) {
    await page.evaluate(() => startPlayback(1));
    await page.waitForTimeout(45 + i * 9);
    await page.evaluate(() => pausePlayback());
    await page.waitForFunction(() => video.paused && !video.seeking && state.pendingFrame === null && !state.cacheVisible);
    // Ordinary pause holds the native displayed image without another seek.
    await page.waitForTimeout(40);
    const frame = await page.evaluate(() => state.frame);
    const result = await inspect(frame);
    check(result.pixelsMatch && Math.round(result.timestamp * 25) === frame, 'Quick play/pause freezes a matching image and label');
  }
  await page.evaluate(() => {
    state.reverseCache.warm = () => Promise.reject(new Error('Simulated unsupported decoder'));
  });
  await page.evaluate(() => seekToFrame(50));
  check(await page.evaluate(() => state.frame === 50 && state.pendingFrame === null && !state.cacheVisible),
    'Native seeking remains available when decoded inspection fails');
  return 'PASS: exact inspection pixels/timestamps, adjacent backward seeks, rapid pending steps, quick play/pause, and native fallback';
}
