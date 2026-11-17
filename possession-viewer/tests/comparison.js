const assert = require('node:assert/strict');
const compare = require('../comparison.js');

const data = compare.buildSpans(
  ['X', 'H', 'H', 'D', 'D', 'A', 'A', 'H', 'X'],
  ['X', 'H', 'A', 'H', 'H', 'H', 'A', 'A', 'X'],
);
assert.equal(data.comparedFrames, 7);
assert.equal(data.differenceFrames, 5);
assert.deepEqual(data.framesByKind, { 'dead-live': 2, team: 3 });
assert.deepEqual(data.spans.map(s => [s.start, s.end, s.mall, s.experiment]), [
  [2, 3, 'H', 'A'], [3, 5, 'D', 'H'], [5, 6, 'A', 'H'], [7, 8, 'H', 'A'],
]);
assert.equal(compare.containing(data.spans, 4).start, 3);
assert.equal(compare.containing(data.spans, 6), null);
assert.equal(compare.adjacent(data.spans, 4, 1).start, 5);
assert.equal(compare.adjacent(data.spans, 4, -1).start, 2);
assert.equal(compare.adjacent(data.spans, 7, 1).start, 2);
console.log('PASS: exact comparison spans, filtering types, lookup and navigation');
