// Serve the viewer on port 8080, then:
// playwright-cli run-code --filename=possession-viewer/tests/csv-picker.js
async page => {
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  let releaseSlow;
  const listing = '<a href="a.csv">A</a><a href="b%20model.csv">B</a>' +
    '<a href="slow.csv">Slow</a><a href="missing.csv">Missing</a>' +
    '<a href="notes.txt">Notes</a><a href="../outside.csv">Outside</a>' +
    '<a href="nested/ignored.csv">Nested</a><a href="https://example.com/foreign.csv">External</a>';
  await page.route('**/data/', route => route.fulfill({ contentType: 'text/html', body: listing }));
  await page.route('**/data/a.csv', route => route.fulfill({ body: 'frame,label\n11185,A\n' }));
  await page.route('**/data/b%20model.csv', route => route.fulfill({ body: 'frame,label\n11185,H\n' }));
  await page.route('**/data/missing.csv', route => route.fulfill({ status: 404, body: 'Missing' }));
  await page.route('**/data/slow.csv', async route => {
    await new Promise(resolve => { releaseSlow = resolve; });
    await route.fulfill({ body: 'frame,label\n11185,D\n' });
  });
  await page.goto('http://localhost:8080');
  await page.waitForFunction(() => document.querySelectorAll('#prediction-select option').length === 5);
  const names = await page.locator('#prediction-select option').allTextContents();
  check(names.join('|') === 'Predictions CSV|a.csv|b model.csv|missing.csv|slow.csv', 'List only direct CSV files, sorted and decoded');
  await page.locator('#jump-reference').click();
  await page.locator('#prediction-select').selectOption('data:a.csv');
  await page.waitForFunction(() => state.predictions?.filename === 'a.csv');
  check(await page.locator('#comparison-status').textContent() === 'Matches ground truth', 'Dropdown updates comparison');
  await page.locator('#prediction-select').selectOption('data:b model.csv');
  await page.waitForFunction(() => state.predictions?.filename === 'b model.csv');
  check(await page.locator('#comparison-status').textContent() === 'Mismatch', 'Switching files updates reference comparison');
  check(await page.locator('#frame-input').inputValue() === '11185', 'Switching keeps current frame');
  await page.locator('#prediction-select').selectOption('data:missing.csv');
  await page.waitForFunction(() => !document.getElementById('source-message').hidden);
  check(await page.locator('#prediction-select').inputValue() === 'data:b model.csv', 'Failed load restores prior selection');
  check(await page.evaluate(() => state.predictions.filename) === 'b model.csv', 'Failed load retains prior predictions');

  await page.locator('#prediction-select').selectOption('data:slow.csv');
  while (!releaseSlow) await page.waitForTimeout(20);
  await page.locator('#prediction-select').selectOption('data:a.csv');
  await page.waitForFunction(() => state.predictions?.filename === 'a.csv');
  releaseSlow();
  await page.waitForResponse('**/data/slow.csv');
  await page.waitForTimeout(100);
  check(await page.evaluate(() => state.predictions.filename) === 'a.csv', 'A late response must not overwrite the latest selection');

  // Observe the native picker request without leaving the CLI paused in an OS dialog.
  await page.evaluate(() => {
    const input = document.getElementById('prediction-file');
    input.click = () => { input.dataset.pickerRequested = 'yes'; };
  });
  await page.locator('#choose-predictions').click();
  check(await page.locator('#prediction-file').getAttribute('data-picker-requested') === 'yes', 'Choose file invokes the native file input');
  await page.evaluate(() => { delete document.getElementById('prediction-file').click; });
  await page.locator('#prediction-file').setInputFiles('possession-viewer/tests/fixtures/replacement.csv');
  await page.waitForFunction(() => state.predictions?.filename === 'replacement.csv');
  check(await page.locator('#prediction-select').inputValue() === 'local', 'Chosen file appears in selector');
  await page.locator('.prediction-control').hover();
  await page.locator('#reset-predictions').click();
  check(await page.locator('#prediction-select').inputValue() === '', 'Reset clears selection');
  check(await page.locator('#prediction-select option[value="local"]').count() === 0, 'Reset removes custom-file option');
  await page.setViewportSize({ width: 390, height: 844 });
  check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'CSV picker fits mobile width');
  await page.setViewportSize({ width: 1280, height: 720 });

  await page.unrouteAll();
  await page.route('**/data/', route => route.fulfill({ status: 404, body: 'No listing' }));
  await page.reload();
  await page.locator('#prediction-file').setInputFiles('possession-viewer/tests/fixtures/replacement.csv');
  await page.waitForFunction(() => state.predictions?.filename === 'replacement.csv');
  check(await page.locator('#choose-predictions').isEnabled(), 'File chooser remains usable without a directory listing');
  await page.unrouteAll();
  await page.reload();
  return 'PASS: CSV discovery, switching, current-frame retention, failed loads, request races, file chooser, reset, mobile, and no-listing fallback';
}
