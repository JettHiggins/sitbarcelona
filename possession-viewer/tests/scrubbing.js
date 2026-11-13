// Run with the local server and synthetic fixture documented in README.md.
async page => {
  const check = (value, message) => { if (!value) throw new Error(message); };
  await page.route('**/*', route => route.continue());
  await page.goto('http://localhost:8080');
  await page.locator('#video-file').setInputFiles('possession-viewer/tests/fixtures/inspection.webm');
  await page.waitForFunction(() => state.hasVideo && video.readyState >= 2);
  await page.evaluate(async () => {
    await seekToFrame(100);
    setTimelineWindow(0, 150);
    clearTimeout(state.warmTimer);
    window.scrubWarmCalls = 0;
    const warm = state.reverseCache.warm.bind(state.reverseCache);
    state.reverseCache.warm = frame => { ++window.scrubWarmCalls; return warm(frame); };
    document.getElementById('seek').dispatchEvent(new PointerEvent('pointerdown'));
  });
  const previews = [];
  for (const frame of [20, 80, 35, 110, 50]) {
    await page.evaluate(frame => {
      const slider = document.getElementById('seek');
      slider.value = String(frame);
      slider.dispatchEvent(new Event('input'));
      const expected = frame / 149 * 100;
      const actual = parseFloat(document.getElementById('timeline-playhead').style.left);
      if (Math.abs(actual - expected) > 0.01) throw new Error('Playhead must follow the pointer immediately');
    }, frame);
    await page.waitForFunction(frame => state.scrubbing && !video.seeking && state.scrubRequest === frame && state.frame === frame, frame);
    previews.push(await page.evaluate(() => state.frame));
    check(await page.evaluate(() => !state.cacheVisible && window.scrubWarmCalls === 0),
      'Dragging displays native previews without exact-frame decoding or an image cover');
  }
  // Burst requests must coalesce instead of repeatedly restarting the decoder.
  await page.evaluate(() => {
    window.scrubSeeks = 0;
    video.addEventListener('seeking', () => ++window.scrubSeeks);
    for (const frame of [10, 90, 25, 100, 40]) {
      const slider = document.getElementById('seek');
      slider.value = String(frame);
      slider.dispatchEvent(new Event('input'));
    }
  });
  await page.waitForFunction(() => !video.seeking && state.frame === 40);
  check(await page.evaluate(() => window.scrubSeeks <= 2), 'A drag burst decodes only the in-flight and latest destinations');
  await page.locator('#seek').dispatchEvent('pointerup');
  await page.waitForFunction(() => !state.scrubbing && state.pendingFrame === null && state.cacheVisible && state.frame === 40);
  check(await page.evaluate(() => window.scrubWarmCalls > 0), 'Release restores exact-frame inspection');
  await page.locator('#play').click();
  await page.waitForFunction(() => !video.paused && state.frame > 43 && !state.cacheVisible);
  await page.evaluate(() => pausePlayback(false));
  // No-pointer input covers keyboard/accessibility controls and settles on idle.
  await page.evaluate(() => {
    const slider = document.getElementById('seek');
    slider.value = '15'; slider.dispatchEvent(new Event('input'));
  });
  await page.waitForFunction(() => !state.scrubbing && state.pendingFrame === null && state.cacheVisible && state.frame === 15);
  return { result: 'PASS: visible drag previews, no cache decoding during drag, coalesced seeks, exact release, immediate playback, keyboard settle', previews };
}
