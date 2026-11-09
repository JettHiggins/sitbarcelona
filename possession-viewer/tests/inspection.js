// Generate the ignored synthetic clip described in README.md, serve on port 8080, then:
// playwright-cli run-code --filename=possession-viewer/tests/inspection.js
async page => {
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  await page.goto('http://localhost:8080');
  await page.locator('#video-file').setInputFiles('possession-viewer/tests/fixtures/inspection.webm');
  await page.waitForFunction(() => state.hasVideo && video.readyState >= 2);
  const seek = async frame => {
    await page.locator('#frame-input').fill(String(frame));
    await page.locator('#frame-input').press('Enter');
    await page.waitForFunction(() => !video.seeking);
  };
  await seek(100);
  for (const rate of [0.5, 0.25, -0.25, -0.5, -1]) {
    await page.locator('#slower').click();
    check(await page.locator('#playback-rate').inputValue() === String(rate), 'Decrease moves exactly one signed speed step');
    check(await page.evaluate(() => video.paused && !state.reversing), 'Adjusting speed while paused does not start playback');
  }
  await page.locator('#faster').click();
  check(await page.locator('#playback-rate').inputValue() === '-0.5', 'Increase slows negative playback toward zero');
  await page.locator('#slower').click();
  check(await page.evaluate(() => state.zoom) === 1, 'Speed changes do not affect zoom');
  await page.locator('#play').click();
  await page.waitForFunction(() => state.reversing && state.frame < 90);
  check(await page.locator('#play').getAttribute('aria-label') === 'Pause', 'Reverse playback shows pause');
  await page.locator('#play').click();
  await page.waitForFunction(() => !video.seeking);
  const pausedTime = await page.evaluate(() => video.currentTime);
  await page.waitForTimeout(200);
  check(await page.evaluate(() => !state.reversing && video.paused), 'Pause stops reverse loop');
  check(Math.abs(await page.evaluate(() => video.currentTime) - pausedTime) < 0.001, 'Paused reverse stays at the same time');
  await page.locator('#play').click();
  await page.waitForFunction(time => state.frame / FPS < time - 0.12, pausedTime);
  await page.locator('#back-frame').click();
  check(await page.evaluate(() => !state.reversing && video.paused), 'Frame stepping stops continuous reverse');
  await seek(90);
  await page.locator('#playback-rate').selectOption('-0.25');
  await page.locator('#play').click();
  const slowStart = await page.evaluate(() => state.frame / FPS);
  await page.waitForTimeout(850);
  const slowEnd = await page.evaluate(() => state.frame / FPS);
  check(slowStart > slowEnd && slowStart - slowEnd < 0.5, 'Quarter-speed reverse moves slowly');
  await page.locator('#faster').click();
  await page.waitForFunction(() => !video.paused && state.direction === 1 && !state.reversing);
  check(await page.evaluate(() => video.playbackRate) === 0.25, 'Forward playback retains chosen slow speed');
  await seek(25);
  await page.locator('#playback-rate').selectOption('-8');
  check(await page.locator('#slower').isDisabled(), 'Minimum speed is bounded');
  await page.locator('#play').click();
  await page.waitForFunction(() => !state.reversing && video.currentTime === 0);
  check(await page.locator('#play').getAttribute('aria-label') === 'Play in reverse', 'Reverse stops cleanly at frame zero');

  const graphBefore = await page.locator('#trace').boundingBox();
  await page.locator('#zoom').fill('2.35');
  check(await page.locator('#reset-zoom').textContent() === '2.35×', 'Fine zoom value is shown');
  await page.locator('#zoom-in').click();
  check(await page.locator('#reset-zoom').textContent() === '2.60×', 'Zoom buttons use quarter steps');
  const stage = await page.locator('#stage').boundingBox();
  const point = { x: stage.x + stage.width / 2 - 100, y: stage.y + stage.height / 2 };
  const contentX = await page.evaluate(x => (x - document.getElementById('stage').clientWidth / 2 - state.panX) / state.zoom, point.x);
  await page.mouse.move(point.x, point.y);
  await page.mouse.wheel(0, -100);
  await page.waitForFunction(() => state.zoom > 2.6);
  const contentAfter = await page.evaluate(x => (x - document.getElementById('stage').clientWidth / 2 - state.panX) / state.zoom, point.x);
  check(Math.abs(contentX - contentAfter) < 0.1, 'Scroll zoom keeps the point under the cursor fixed');
  const beforePan = await page.evaluate(() => ({ x: state.panX, y: state.panY }));
  await page.mouse.down();
  await page.mouse.move(point.x + 70, point.y + 40, { steps: 5 });
  await page.mouse.up();
  check(await page.evaluate(before => state.panX !== before.x || state.panY !== before.y, beforePan), 'Dragging pans the zoomed footage');
  check(JSON.stringify(await page.locator('#trace').boundingBox()) === JSON.stringify(graphBefore), 'Video zoom does not move or scale the comparison graph');
  await page.mouse.move(stage.width / 2, stage.height / 2);
  await page.mouse.down();
  await page.mouse.move(stage.width * 5, stage.height * 5);
  await page.mouse.up();
  check(await page.evaluate(() => {
    const stage = document.getElementById('stage');
    const fit = Math.min(stage.clientWidth / video.videoWidth, stage.clientHeight / video.videoHeight);
    return Math.abs(state.panX) <= Math.max(0, (video.videoWidth * fit * state.zoom - stage.clientWidth) / 2) + 0.01 &&
      Math.abs(state.panY) <= Math.max(0, (video.videoHeight * fit * state.zoom - stage.clientHeight) / 2) + 0.01;
  }), 'Panning stops at footage bounds');
  await page.locator('#reset-zoom').click();
  check(await page.evaluate(() => state.zoom === 1 && state.panX === 0 && state.panY === 0), 'Reset restores full-frame view');
  await page.locator('#zoom').fill('8');
  check(await page.locator('#zoom-in').isDisabled(), 'Zoom is bounded at 8 times');
  await page.locator('#stage').dblclick({ position: { x: stage.width / 2, y: stage.height / 2 } });
  check(await page.evaluate(() => state.zoom) === 1, 'Double-click resets zoom');
  await page.setViewportSize({ width: 390, height: 844 });
  check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Inspection controls fit mobile width');
  check(await page.locator('#zoom-controls').isVisible(), 'Zoom controls remain available on mobile');
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.locator('#zoom').fill('3');
  await seek(75);
  await page.locator('#play').click();
  await seek(50);
  check(await page.evaluate(() => !state.reversing && video.paused && state.zoom === 3), 'Seeking stops reverse while retaining zoom');
  await page.locator('#play').click();
  await page.locator('#reset-video').click();
  check(await page.evaluate(() => !state.reversing && !state.hasVideo && state.zoom === 1 && state.direction === 1 && state.speed === 1), 'Removing video clears reverse playback and zoom');
  check(await page.locator('#zoom-controls').isHidden(), 'Zoom controls hide without video');
  return 'PASS: reverse, pause/resume, frame step, slow motion, direction switching, start boundary, anchored zoom, pan, reset, mobile, and video removal';
}
