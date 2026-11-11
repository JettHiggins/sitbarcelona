// Ordinary Play/Pause must not seek, snap to a frame, or decode inspection frames.
// Run against the dedicated server on port 8081 with the synthetic inspection clip.
async page => {
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  await page.goto('http://127.0.0.1:8081/');
  await page.locator('#video-file').setInputFiles('possession-viewer/tests/fixtures/inspection.webm');
  await page.waitForFunction(() => state.hasVideo && video.readyState >= 2);
  await page.evaluate(async () => {
    await state.reverseCache.ready;
    const pendingWarm = state.reverseCache.warm(100);
    state.reverseCache.cancelWarm();
    await pendingWarm;
    if (state.reverseCache.targetChunk !== null) throw new Error('Cancelled preparation restarted after Play');
    const clock = Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, 'currentTime');
    window.nativeSeeks = [];
    Object.defineProperty(video, 'currentTime', {
      get() { return clock.get.call(this); },
      set(value) { window.nativeSeeks.push(value); clock.set.call(this, value); },
    });
    window.inspectionRequests = 0;
    const warm = state.reverseCache.warm.bind(state.reverseCache);
    state.reverseCache.warm = (...args) => { ++window.inspectionRequests; return warm(...args); };
    await startPlayback(1);
  });
  await page.waitForTimeout(200);
  for (let i = 0; i < 4; i++) {
    await page.evaluate(() => pausePlayback());
    await page.waitForTimeout(30);
    const before = await page.evaluate(() => video.currentTime);
    await page.evaluate(() => startPlayback(1));
    await page.waitForTimeout(200);
    check(await page.evaluate(before => !video.paused && !video.seeking && !state.cacheVisible &&
      video.currentTime > before + .10, before), 'Native playback advances after quick pause/resume');
  }
  const result = await page.evaluate(() => ({ seeks: window.nativeSeeks, inspections: window.inspectionRequests }));
  check(result.seeks.length === 0, `Ordinary Play/Pause must not seek: ${JSON.stringify(result)}`);
  check(result.inspections === 0, 'Ordinary playback must not request inspection decoding');
  // Explicit frame inspection still gets exactly one native handoff before Play.
  await page.evaluate(() => seekToFrame(75));
  await page.evaluate(() => { window.nativeSeeks = []; });
  await page.evaluate(() => startPlayback(1));
  await page.waitForFunction(() => !video.paused && state.frame > 78 && !state.cacheVisible);
  check(await page.evaluate(() => window.nativeSeeks.length <= 1), 'Inspection resumes with at most one native seek');
  return 'PASS: repeated native pause/resume without seeking or inspection decoding; explicit inspection handoff';
}
