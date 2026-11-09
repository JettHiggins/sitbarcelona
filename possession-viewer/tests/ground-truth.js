// Run after serving possession-viewer at localhost:8080:
// playwright-cli open http://localhost:8080
// playwright-cli run-code --filename=possession-viewer/tests/ground-truth.js
async page => {
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  await page.goto('http://localhost:8080');
  await page.waitForFunction(() => Boolean(state.reference));
  const reference = await page.evaluate(() => ({
    count: state.reference.labels.filter(Boolean).length,
    before: referenceAt(11184), first: referenceAt(11185),
    transitionBefore: referenceAt(11350), transitionAfter: referenceAt(11351),
    last: referenceAt(18684), after: referenceAt(18685),
  }));
  check(reference.count === 7500, 'Reference must cover exactly five minutes');
  check(reference.before === null && reference.after === null, 'No reference outside the public window');
  check(reference.first === 'A' && reference.last === 'H', 'Reference boundary labels');
  check(reference.transitionBefore === 'A' && reference.transitionAfter === 'H', 'Reference transition must be frame-exact');
  await page.locator('#jump-reference').click();
  check(await page.locator('#expected-symbol').textContent() === 'ATH', 'Jump shows the first reference label');
  check(await page.locator('#comparison-status').isHidden(), 'No comparison without predictions');

  // Sparse input deliberately exercises missing predictions and explicit abstention.
  await page.locator('#prediction-file').setInputFiles('possession-viewer/tests/fixtures/comparison-fixture.csv');
  await page.waitForFunction(() => state.predictions?.filename === 'comparison-fixture.csv');
  const jump = async frame => {
    await page.locator('#frame-input').fill(String(frame));
    await page.locator('#frame-input').press('Enter');
  };
  const status = () => page.evaluate(() => ({
    mismatch: document.getElementById('stage').classList.contains('prediction-mismatch'),
    text: document.getElementById('comparison-status').textContent,
    expected: document.getElementById('expected-symbol').textContent,
  }));
  check((await status()).text === 'Matches ground truth', 'Correct prediction is a match');
  await jump(11351);
  check((await status()).mismatch && (await status()).expected === 'BAR', 'Wrong team highlights video and expected label');
  await jump(12851);
  check(!(await status()).mismatch && (await status()).expected === 'D', 'Dead ball is a real reference label');
  await jump(18684);
  check((await status()).mismatch, 'Prediction X is an error when reference is known');
  await jump(12000);
  check((await status()).mismatch, 'Missing prediction is an error when reference is known');
  for (const frame of [11184, 18685]) {
    await jump(frame);
    check(await page.locator('#reference-decision').isHidden(), 'Expected label hidden outside reference');
    check(await page.locator('#comparison-status').isHidden(), 'No false match outside reference');
    check(!(await status()).mismatch, 'No video error highlight outside reference');
  }
  await jump(11185);
  await page.locator('#prediction-file').setInputFiles('possession-viewer/tests/fixtures/replacement.csv');
  await page.waitForFunction(() => state.predictions?.filename === 'replacement.csv');
  check((await status()).mismatch, 'Replacing CSV immediately recomputes comparison');
  await page.locator('.prediction-control').hover();
  await page.locator('#reset-predictions').click();
  check(await page.locator('#comparison-status').isHidden(), 'Reset clears comparison');
  check(!(await status()).mismatch, 'Reset clears video error highlight');
  check(await page.locator('#expected-symbol').isVisible(), 'Reset retains independent ground truth');
  await page.setViewportSize({ width: 390, height: 844 });
  check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Mobile layout must not overflow');
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.route('**/reference/public-ground-truth.json', route => route.fulfill({ status: 404, body: 'Missing' }));
  await page.reload();
  await page.waitForFunction(() => Boolean(state.referenceError));
  await page.locator('#prediction-file').setInputFiles('possession-viewer/tests/fixtures/replacement.csv');
  await page.waitForFunction(() => Boolean(state.predictions));
  check(await page.locator('#jump-reference').textContent() === 'Ground truth unavailable', 'Reference failure remains visible after CSV loading');
  check(await page.locator('#comparison-status').isHidden(), 'Reference failure never implies a match');
  await page.unroute('**/reference/public-ground-truth.json');
  await page.reload();
  await page.waitForFunction(() => Boolean(state.reference));
  return 'PASS: reference boundaries, match/mismatch, dead ball, abstention, CSV replacement/reset, mobile layout, and reference-load failure';
}
