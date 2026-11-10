// Start python3 possession-viewer/server.py, then run with playwright-cli run-code.
async page => {
  const origin = 'http://127.0.0.1:8081';
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  for (const path of ['/', '/app.js', '/styles.css', '/api/predictions']) {
    const response = await page.request.get(origin + path);
    check(response.ok(), `${path} is served`);
    check(response.headers()['cache-control'].includes('no-store'), `${path} is not cached`);
  }
  const response = await page.request.get(origin + '/api/predictions');
  const files = await response.json();
  check(Array.isArray(files) && files.every(name => name.endsWith('.csv')), 'API lists CSV filenames');
  await page.route('**/data/', route => route.fulfill({ status: 404, body: 'Listing disabled' }));
  await page.goto(origin);
  await page.waitForFunction(count => document.querySelectorAll('#prediction-select option').length === count + 1, files.length);
  check((await page.locator('#prediction-select option').allTextContents()).slice(1).sort().join('|') === files.slice().sort().join('|'),
    'Dropdown uses the API without a directory listing');
  if (files.length) {
    const filename = files.includes('event-first.csv') ? 'event-first.csv' : files[0];
    await page.locator('#prediction-select').selectOption('data:' + filename);
    await page.waitForFunction(name => state.predictions?.filename === name, filename);
    check(await page.evaluate(() => state.predictions.rows) > 0, 'Listed CSV loads');
  }
  for (let i = 0; i < 3; i++) {
    await page.reload();
    check(await page.title() === 'Possession viewer', 'Normal reload remains in the viewer');
    await page.waitForFunction(count => document.querySelectorAll('#prediction-select option').length === count + 1, files.length);
  }
  await page.unrouteAll();
  return { result: 'PASS: uncached assets, explicit CSV list without directory listing, CSV loading and normal reloads', files };
}
