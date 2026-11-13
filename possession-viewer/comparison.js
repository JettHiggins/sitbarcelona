// Exact, label-pair disagreement spans. These are model differences, not errors.
const PossessionCompare = (() => {
  const TEAM = new Set(['H', 'A']);

  function kind(mall, experiment) {
    if (!mall || !experiment || mall === 'X' || experiment === 'X' || mall === experiment) return null;
    if (mall === 'D' || experiment === 'D') return 'dead-live';
    if (TEAM.has(mall) && TEAM.has(experiment)) return 'team';
    return null;
  }

  function buildSpans(mallLabels, experimentLabels) {
    const length = Math.min(mallLabels.length, experimentLabels.length);
    const spans = [];
    let current = null, comparedFrames = 0, differenceFrames = 0;
    const framesByKind = { 'dead-live': 0, team: 0 };
    for (let frame = 0; frame < length; frame++) {
      const mall = mallLabels[frame], experiment = experimentLabels[frame];
      if (mall && experiment && mall !== 'X' && experiment !== 'X') comparedFrames++;
      const type = kind(mall, experiment);
      if (!type) { current = null; continue; }
      differenceFrames++;
      framesByKind[type]++;
      if (current && current.end === frame && current.mall === mall && current.experiment === experiment) {
        current.end++;
      } else {
        current = { start: frame, end: frame + 1, mall, experiment, kind: type };
        spans.push(current);
      }
    }
    return { spans, comparedFrames, differenceFrames, framesByKind };
  }

  function containing(spans, frame) {
    let lo = 0, hi = spans.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (spans[mid].end <= frame) lo = mid + 1;
      else hi = mid;
    }
    return lo < spans.length && spans[lo].start <= frame ? spans[lo] : null;
  }

  function adjacent(spans, frame, direction) {
    if (!spans.length) return null;
    const inside = containing(spans, frame);
    if (direction > 0) return spans.find(span => span.start > (inside ? inside.start : frame)) || spans[0];
    for (let i = spans.length - 1; i >= 0; i--) {
      if (spans[i].start < (inside ? inside.start : frame)) return spans[i];
    }
    return spans.at(-1);
  }

  return { kind, buildSpans, containing, adjacent };
})();

if (typeof module !== 'undefined') module.exports = PossessionCompare;
