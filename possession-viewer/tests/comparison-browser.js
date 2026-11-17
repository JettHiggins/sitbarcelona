// Synthetic complete CSVs exercise comparison controls without match data.
async page => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const csv = wes => 'frame,label\n' + Array.from({ length: 199319 }, (_, frame) => {
    let label = frame >= 11185 && frame < 11285 ? 'H' : 'X';
    if (wes && frame >= 11200 && frame < 11210) label = 'D';
    if (wes && frame >= 11230 && frame < 11240) label = 'A';
    return `${frame},${label}\n`;
  }).join('');
  await page.route('**/api/predictions', route => route.fulfill({
    json: ['mall-model.csv', 'wes-model.csv'],
  }));
  await page.route('**/data/mall-model.csv', route => route.fulfill({ body: csv(false) }));
  await page.route('**/data/wes-model.csv', route => route.fulfill({ body: csv(true) }));
  await page.route('**/data/wes-model.segmentation.json', route => route.fulfill({
    json: { windows: [], dead_revisions: [], change_candidates: [] },
  }));
  await page.goto('http://localhost:8080');
  await page.locator('#experiment-select option[value="data:wes-model.csv"]').waitFor({ state: 'attached' });
  await page.locator('#experiment-select').selectOption('data:wes-model.csv');
  await page.waitForFunction(() => state.experiment?.differenceFrames === 20);
  await page.locator('#next-difference').click();
  await page.waitForFunction(() => state.frame === 11200);
  await page.locator('#difference-filter').selectOption('team');
  await page.locator('#next-difference').click();
  await page.waitForFunction(() => state.frame === 11230);
  if (await page.locator('#experiment-span-list li').count() !== 1) {
    throw new Error('Team filter must retain exactly one synthetic disagreement');
  }
  await page.locator('#experiment-select').selectOption('');
  await page.waitForFunction(() => state.predictions?.filename === 'mall-model.csv' && !state.experiment);
  if (errors.length) throw new Error(errors.join('\n'));
  await page.unrouteAll();
  return 'PASS: full CSV comparison, exact spans, navigation, filters, return to Mall';
}
