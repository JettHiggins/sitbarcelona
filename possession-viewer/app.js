const FPS = 25;
const FRAME_COUNT = 199319;
const LAST_FRAME = FRAME_COUNT - 1;
const PLAYBACK_RATES = [-8, -4, -2, -1, -0.5, -0.25, 0.25, 0.5, 1, 2, 4, 8];
const LABELS = {
  H: { name: `${MATCH_CONFIG.home.name} possession`, display: MATCH_CONFIG.home.abbreviation, color: '#568eff', level: 2 },
  A: { name: `${MATCH_CONFIG.away.name} possession`, display: MATCH_CONFIG.away.abbreviation, color: '#f34d4d', level: 0 },
  D: { name: 'Dead ball', display: 'D', color: '#e4bc46', level: 1 },
  X: { name: 'Unknown', display: 'X', color: '#aaa', level: 3 },
};

const $ = id => document.getElementById(id);
const video = $('video');
const state = {
  frame: 0, predictions: null, videoUrl: null, hasVideo: false,
  reference: null, referenceError: null, predictionSource: '', csvLoadVersion: 0,
  direction: 1, speed: 1, reversing: false, reverseTimer: null, playVersion: 0,
  zoom: 1, panX: 0, panY: 0,
};

function setMessage(message, error = false) {
  $('source-message').textContent = message;
  $('source-message').classList.toggle('error', error);
  $('source-message').hidden = !error;
}

function formatTime(seconds) {
  seconds = Math.max(0, Math.floor(seconds));
  const h = Math.floor(seconds / 3600), m = Math.floor(seconds % 3600 / 60), s = seconds % 60;
  return [h, m, s].map(n => String(n).padStart(2, '0')).join(':');
}

function updateFrameCheck() {
  const csv = state.predictions;
  const check = $('frame-check');
  const issues = [];
  if (csv && (csv.rows !== FRAME_COUNT || csv.duplicates)) {
    issues.push(`CSV: ${csv.rows.toLocaleString()} unique frames; expected ${FRAME_COUNT.toLocaleString()}. ${csv.duplicates} duplicate rows.`);
  }
  if (state.hasVideo && Number.isFinite(video.duration)) {
    const estimated = Math.round(video.duration * FPS);
    const expected = csv ? csv.rows : FRAME_COUNT;
    if (estimated !== expected) issues.push(`Video: approximately ${estimated.toLocaleString()} frames; ${csv ? 'CSV' : 'expected'}: ${expected.toLocaleString()}. Video count estimated at 25 fps.`);
  }
  check.hidden = issues.length === 0;
  check.title = issues.join(' ');
  check.setAttribute('aria-label', issues.length ? issues.join(' ') : 'Frame count warning');
}

function setFrame(frame) {
  frame = Math.max(0, Math.min(LAST_FRAME, Math.floor(frame) || 0));
  state.frame = frame;
  $('timecode').textContent = formatTime(frame / FPS);
  $('seek').value = String(frame);
  $('timeline-playhead').style.left = `${frame / LAST_FRAME * 100}%`;
  $('seek-progress').style.width = `${frame / LAST_FRAME * 100}%`;
  if (document.activeElement !== $('frame-input')) $('frame-input').value = String(frame);

  const code = state.predictions?.labels[frame] || 'X';
  $('decision-symbol').textContent = state.predictions ? LABELS[code].display : '—';
  $('decision-symbol').className = `decision-symbol ${code.toLowerCase()}`;
  $('decision-symbol').setAttribute('aria-label', state.predictions ? LABELS[code].name : 'No predictions loaded');
  updateComparison();
  drawTrace();
}

function seekToFrame(frame, keepPlaying = false) {
  if (!keepPlaying) pausePlayback();
  frame = Math.max(0, Math.min(LAST_FRAME, Math.floor(Number(frame)) || 0));
  if (state.hasVideo) video.currentTime = frame / FPS;
  setFrame(frame);
}

function parseCsv(text, filename) {
  const lines = text.replace(/^\uFEFF/, '').split(/\r?\n/);
  const header = lines.shift()?.trim().split(',').map(s => s.trim().replace(/^"|"$/g, '').toLowerCase()) || [];
  const frameCol = header.indexOf('frame'), labelCol = header.indexOf('label');
  if (frameCol < 0 || labelCol < 0) throw new Error(`${filename}: expected frame,label columns.`);
  const labels = new Array(FRAME_COUNT);
  let rows = 0, duplicates = 0;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const cells = line.split(',');
    const frameText = cells[frameCol]?.trim().replace(/^"|"$/g, '');
    const code = cells[labelCol]?.trim().replace(/^"|"$/g, '').toUpperCase();
    const frame = Number(frameText);
    if (!Number.isInteger(frame) || frame < 0 || frame > LAST_FRAME || !LABELS[code]) {
      throw new Error(`${filename}: invalid frame or label on CSV line ${i + 2}.`);
    }
    if (labels[frame] !== undefined) duplicates++;
    else rows++;
    labels[frame] = code;
  }
  if (!rows) throw new Error(`${filename}: no prediction rows found.`);
  return { labels, rows, duplicates, filename };
}

function updateCsvSelection() {
  const select = $('prediction-select');
  select.querySelector('option[value="local"]')?.remove();
  if (state.predictionSource === 'local' && state.predictions) {
    select.add(new Option(state.predictions.filename, 'local'));
  }
  select.value = state.predictionSource;
  select.title = state.predictions?.filename || 'CSV files in data/';
}

async function loadCsvOptions() {
  const select = $('prediction-select');
  try {
    // The existing python http.server exposes data/ as an HTML directory listing.
    const directory = new URL('data/', location.href);
    const response = await fetch(directory, { cache: 'no-store' });
    if (!response.ok) throw new Error('CSV list unavailable');
    const listing = new DOMParser().parseFromString(await response.text(), 'text/html');
    const names = new Set();
    for (const link of listing.querySelectorAll('a[href]')) {
      try {
        const url = new URL(link.getAttribute('href'), directory);
        if (url.origin !== directory.origin || !url.pathname.startsWith(directory.pathname)) continue;
        const name = decodeURIComponent(url.pathname.slice(directory.pathname.length));
        if (name && !name.includes('/') && !name.includes('\\') && /\.csv$/i.test(name)) names.add(name);
      } catch { /* Ignore malformed directory entries. */ }
    }
    const selected = state.predictionSource;
    select.replaceChildren(new Option(names.size ? 'Predictions CSV' : 'No CSVs in data/', ''));
    select.options[0].disabled = true;
    for (const name of [...names].sort((a, b) => a.localeCompare(b))) select.add(new Option(name, `data:${name}`));
    // Keep the selected file visible even if it was removed from the directory meanwhile.
    if (selected.startsWith('data:') && ![...select.options].some(option => option.value === selected)) {
      select.add(new Option(state.predictions.filename, selected));
    }
    updateCsvSelection();
  } catch {
    // Manual uploads remain available on hosts without directory listings.
    if (select.options.length === 1) select.options[0].textContent = 'Predictions CSV';
  }
}

async function loadCsv(file, source = 'local') {
  if (!file) return;
  const loadVersion = ++state.csvLoadVersion;
  try {
    const parsed = parseCsv(await file.text(), file.name);
    if (loadVersion !== state.csvLoadVersion) return;
    state.predictions = parsed;
    state.predictionSource = source;
    updateCsvSelection();
    $('decision-symbol').hidden = false;
    $('prediction-value').hidden = false;
    $('reset-predictions').hidden = false;
    updateFrameCheck();
    const missing = FRAME_COUNT - parsed.rows;
    const notes = [`${file.name}: ${parsed.rows.toLocaleString()} frames loaded`];
    if (missing) notes.push(`${missing.toLocaleString()} missing`);
    if (parsed.duplicates) notes.push(`${parsed.duplicates.toLocaleString()} duplicate rows`);
    setMessage(notes.join(' · ') + '.');
    setFrame(state.frame);
    drawTimeline();
  } catch (error) {
    if (loadVersion !== state.csvLoadVersion) return;
    updateCsvSelection();
    setMessage(error.message, true);
  }
}

function loadVideo(source, label, objectUrl = false) {
  if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
  state.videoUrl = objectUrl ? source : null;
  state.hasVideo = false;
  updateFrameCheck();
  pausePlayback();
  state.direction = 1;
  setPlaybackRate(1);
  resetZoom();
  video.src = source;
  video.load();
  setMessage(`Loading ${label}…`);
}

function prepareCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(canvas.clientWidth));
  const height = Math.max(1, Math.floor(canvas.clientHeight));
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  const ctx = canvas.getContext('2d');
  ctx.scale(ratio, ratio);
  return { ctx, width, height };
}

function referenceAt(frame) {
  return state.reference?.labels[frame] || null;
}

function updateComparison() {
  const expected = referenceAt(state.frame);
  const prediction = state.predictions?.labels[state.frame];
  const comparable = Boolean(expected && state.predictions);
  const mismatch = comparable && prediction !== expected;
  $('reference-decision').hidden = !expected;
  $('expected-symbol').textContent = expected ? LABELS[expected].display : '—';
  $('expected-symbol').classList.toggle('team', expected === 'H' || expected === 'A');
  $('expected-symbol').style.color = expected ? LABELS[expected].color : '';
  $('expected-symbol').setAttribute('aria-label', expected ? `Ground truth: ${LABELS[expected].name}` : 'No ground truth for this frame');
  $('stage').classList.toggle('prediction-mismatch', mismatch);
  $('comparison-status').hidden = !comparable;
  $('comparison-status').classList.toggle('mismatch', mismatch);
  const statusText = mismatch ? 'Mismatch' : 'Matches ground truth';
  if ($('comparison-status').textContent !== statusText) $('comparison-status').textContent = statusText;
  $('trace-legend').hidden = !state.reference;
  $('jump-reference').hidden = !state.reference && !state.referenceError;
}

async function loadReference() {
  try {
    const response = await fetch('reference/public-ground-truth.json');
    if (!response.ok) throw new Error('Could not load public ground truth');
    const data = await response.json();
    if (data.game !== MATCH_CONFIG.id || data.fps !== FPS || data.frame_count !== FRAME_COUNT || !Array.isArray(data.segments) || !data.segments.length) {
      throw new Error('Ground-truth timeline does not match this video');
    }
    const labels = new Array(FRAME_COUNT);
    let lastEnd = 0;
    for (const [start, end, code] of data.segments) {
      if (!Number.isInteger(start) || !Number.isInteger(end) || start < lastEnd || end <= start || end > FRAME_COUNT || !['H', 'A', 'D'].includes(code)) {
        throw new Error('Invalid ground-truth segment');
      }
      labels.fill(code, start, end);
      lastEnd = end;
    }
    state.reference = { labels, segments: data.segments, first: data.segments[0][0] };
    const first = data.segments[0][0], last = data.segments.at(-1)[1] - 1;
    $('jump-reference').textContent = `Ground truth · ${formatTime(first / FPS)}–${formatTime((last + 1) / FPS)} ↗`;
    $('jump-reference').title = 'Jump to the public five-minute reference window';
    setFrame(state.frame);
    drawTimeline();
  } catch (error) {
    state.referenceError = error.message;
    $('jump-reference').hidden = false;
    $('jump-reference').textContent = 'Ground truth unavailable';
    $('jump-reference').disabled = true;
    $('jump-reference').title = error.message;
  }
}

function drawTimeline() {
  const { ctx, width, height } = prepareCanvas($('timeline'));
  ctx.fillStyle = '#333'; ctx.fillRect(0, 0, width, 4);
  // Predictions above; the public reference occupies only its actual labeled interval below.
  if (state.predictions) {
    for (let x = 0; x < width; x++) {
      const frame = Math.min(LAST_FRAME, Math.floor((x + .5) / width * FRAME_COUNT));
      ctx.fillStyle = LABELS[state.predictions.labels[frame] || 'X'].color;
      ctx.fillRect(x, 0, 1, 4);
    }
  }
  if (!state.reference) return;
  for (const [start, end, code] of state.reference.segments) {
    ctx.fillStyle = LABELS[code].color;
    ctx.fillRect(start / FRAME_COUNT * width, 8, Math.max(1, (end - start) / FRAME_COUNT * width), height - 8);
  }
  // Aggregate each pixel's frames so even a short mismatch remains visible at full-match scale.
  if (state.predictions) {
    for (let x = 0; x < width; x++) {
      const start = Math.floor(x / width * FRAME_COUNT);
      const end = Math.min(FRAME_COUNT, Math.ceil((x + 1) / width * FRAME_COUNT));
      for (let frame = start; frame < end; frame++) {
        const truth = referenceAt(frame);
        if (truth && truth !== state.predictions.labels[frame]) {
          ctx.fillStyle = '#ffb36b'; ctx.fillRect(x, 5, 1, 2); break;
        }
      }
    }
  }
}

function drawTrace() {
  const { ctx, width, height } = prepareCanvas($('trace'));
  const left = 30;
  const pad = 13, innerHeight = height - pad * 2;
  ctx.strokeStyle = '#ffffff12'; ctx.lineWidth = 1;
  for (let row = 0; row < 4; row++) {
    const y = pad + row * innerHeight / 3;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
  }
  ctx.fillStyle = '#aaa'; ctx.font = '9px monospace';
  ['A','D','H','X'].forEach((code, index) => { ctx.fillStyle = LABELS[code].color; ctx.fillText(LABELS[code].display, 4, pad + index * innerHeight / 3 + 3); });
  const frameAtX = x => state.frame + Math.round((x - left) / Math.max(1, width - left - 1) * 500) - 250;
  if (state.predictions && state.reference) {
    for (let x = left; x < width; x++) {
      const frame = frameAtX(x), truth = referenceAt(frame);
      if (truth && state.predictions.labels[frame] !== truth) {
        ctx.fillStyle = '#ffb36b22'; ctx.fillRect(x, 0, 1, height);
      }
    }
  }
  const labels = state.predictions?.labels;
  if (labels) {
    let previous = null;
    for (let x = left; x < width; x++) {
      const frame = Math.max(0, Math.min(LAST_FRAME, frameAtX(x)));
      const code = labels[frame] || 'X';
      const y = pad + LABELS[code].level * innerHeight / 3;
      ctx.strokeStyle = LABELS[code].color; ctx.lineWidth = 2;
      if (previous) {
        ctx.beginPath(); ctx.moveTo(x - 1, previous.y); ctx.lineTo(x, previous.y);
        if (previous.y !== y) ctx.lineTo(x, y);
        ctx.stroke();
      }
      previous = { y };
    }
  }
  // White dashed reference line: gaps stay blank, including outside the public window.
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
  ctx.beginPath();
  let previousTruth = null;
  for (let x = left; x < width; x++) {
    const truth = referenceAt(frameAtX(x));
    if (!truth) { previousTruth = null; continue; }
    const y = pad + LABELS[truth].level * innerHeight / 3;
    if (previousTruth) { ctx.lineTo(x, previousTruth.y); ctx.lineTo(x, y); }
    else ctx.moveTo(x, y);
    previousTruth = { y };
  }
  ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = '#ffffffcc'; ctx.lineWidth = 1;
  const nowX = left + (width - left - 1) / 2;
  ctx.beginPath(); ctx.moveTo(nowX, 0); ctx.lineTo(nowX, height); ctx.stroke();
}

function updateFromVideo() {
  if (!state.hasVideo) return;
  setFrame(Math.floor(video.currentTime * FPS + 0.00001));
}

video.addEventListener('loadedmetadata', () => {
  state.hasVideo = true;
  updateFrameCheck();
  $('stage').classList.add('has-video');
  $('open-video').hidden = true;
  $('reset-video').hidden = false;
  $('zoom-controls').hidden = false;
  applyZoom();
  setMessage('');
  updateFromVideo();
});
video.addEventListener('error', () => {
  pausePlayback();
  $('zoom-controls').hidden = true;
  state.hasVideo = false;
  $('stage').classList.remove('has-video');
  setMessage('Video could not play. Choose a browser-compatible video file.', true);
});
video.addEventListener('timeupdate', updateFromVideo);
video.addEventListener('seeked', updateFromVideo);
video.addEventListener('play', updatePlaybackControls);
video.addEventListener('pause', updatePlaybackControls);
video.addEventListener('ended', updatePlaybackControls);
if (video.requestVideoFrameCallback) {
  const onVideoFrame = (_, metadata) => {
    if (state.hasVideo) setFrame(Math.floor(metadata.mediaTime * FPS + 0.00001));
    video.requestVideoFrameCallback(onVideoFrame);
  };
  video.requestVideoFrameCallback(onVideoFrame);
}

$('choose-predictions').addEventListener('click', () => {
  $('prediction-file').value = '';
  $('prediction-file').click();
});
$('prediction-select').addEventListener('focus', loadCsvOptions);
$('prediction-select').addEventListener('change', event => {
  const source = event.target.value;
  if (!source.startsWith('data:')) return;
  const name = source.slice(5);
  loadCsv({ name, text: async () => {
    const response = await fetch(`data/${encodeURIComponent(name)}`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`${name}: could not load CSV.`);
    return response.text();
  } }, source);
});
$('open-video').addEventListener('click', () => $('video-file').click());
$('video-file').addEventListener('change', event => {
  const file = event.target.files[0];
  if (file) { $('open-video').title = file.name; loadVideo(URL.createObjectURL(file), file.name, true); }
});
$('prediction-file').addEventListener('change', event => loadCsv(event.target.files[0]));
function updatePlaybackControls() {
  const playing = state.reversing || !video.paused;
  $('play').textContent = playing ? 'Ⅱ' : '▶';
  $('play').setAttribute('aria-label', playing ? 'Pause' : state.direction < 0 ? 'Play in reverse' : 'Play');
  const rate = state.direction * state.speed;
  $('slower').disabled = rate === PLAYBACK_RATES[0];
  $('faster').disabled = rate === PLAYBACK_RATES.at(-1);
  $('playback-rate').value = String(rate);
}

function pausePlayback() {
  ++state.playVersion;
  state.reversing = false;
  if (state.reverseTimer !== null) cancelAnimationFrame(state.reverseTimer);
  state.reverseTimer = null;
  video.pause();
  updatePlaybackControls();
}

function startPlayback(direction = state.direction) {
  if (!state.hasVideo) { setMessage('Load a tactical video to play it.', true); return; }
  pausePlayback();
  state.direction = direction;
  if (direction > 0) {
    video.playbackRate = state.speed;
    const playVersion = state.playVersion;
    video.play().catch(error => {
      if (playVersion !== state.playVersion || error.name === 'AbortError') return;
      pausePlayback();
      setMessage('Browser could not start playback.', true);
    });
  } else if (video.currentTime > 0) {
    // Negative playbackRate is not portable. Seek backwards only after each seek completes.
    state.reversing = true;
    let last = performance.now();
    const tick = now => {
      if (!state.reversing) return;
      const elapsed = now - last;
      if (!video.seeking && elapsed >= 1000 / (FPS * state.speed)) {
        const frames = Math.floor(Math.min(elapsed, 250) * FPS * state.speed / 1000);
        const target = Math.max(0, Math.round(video.currentTime * FPS) - frames);
        last = now;
        seekToFrame(target, true);
        if (target === 0) { pausePlayback(); return; }
      }
      state.reverseTimer = requestAnimationFrame(tick);
    };
    state.reverseTimer = requestAnimationFrame(tick);
  }
  updatePlaybackControls();
}

function setPlaybackRate(rate) {
  if (!PLAYBACK_RATES.includes(rate)) return;
  const playing = state.reversing || !video.paused;
  const direction = Math.sign(rate);
  const changedDirection = direction !== state.direction;
  state.direction = direction;
  state.speed = Math.abs(rate);
  video.playbackRate = state.speed;
  if (playing && changedDirection) startPlayback(direction);
  else updatePlaybackControls();
}

function stepPlaybackRate(delta) {
  const index = PLAYBACK_RATES.indexOf(state.direction * state.speed);
  const next = Math.max(0, Math.min(PLAYBACK_RATES.length - 1, index + delta));
  setPlaybackRate(PLAYBACK_RATES[next]);
}

$('play').addEventListener('click', () => {
  if (state.reversing || !video.paused) pausePlayback();
  else startPlayback();
});
for (const [id, delta] of [['back-frame', -1], ['forward-frame', 1]]) {
  $(id).addEventListener('click', () => seekToFrame(state.frame + delta));
}
$('slower').addEventListener('click', () => stepPlaybackRate(-1));
$('faster').addEventListener('click', () => stepPlaybackRate(1));
$('playback-rate').addEventListener('change', event => setPlaybackRate(Number(event.target.value)));
document.addEventListener('visibilitychange', () => { if (document.hidden && state.reversing) pausePlayback(); });

function applyZoom() {
  const stage = $('stage');
  const fit = state.hasVideo ? Math.min(stage.clientWidth / video.videoWidth, stage.clientHeight / video.videoHeight) : 0;
  const maxX = Math.max(0, (video.videoWidth * fit * state.zoom - stage.clientWidth) / 2) || 0;
  const maxY = Math.max(0, (video.videoHeight * fit * state.zoom - stage.clientHeight) / 2) || 0;
  state.panX = Math.max(-maxX, Math.min(maxX, state.panX));
  state.panY = Math.max(-maxY, Math.min(maxY, state.panY));
  video.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
  $('zoom').value = String(state.zoom);
  $('reset-zoom').textContent = `${state.zoom.toFixed(2)}×`;
  $('zoom').setAttribute('aria-valuetext', `${state.zoom.toFixed(2)} times`);
  $('zoom-out').disabled = state.zoom === 1;
  $('zoom-in').disabled = state.zoom === 8;
  stage.classList.toggle('zoomed', state.zoom > 1);
}

function zoomAt(value, clientX, clientY) {
  if (!state.hasVideo) return;
  const bounds = $('stage').getBoundingClientRect();
  const x = (clientX ?? bounds.left + bounds.width / 2) - bounds.left - bounds.width / 2;
  const y = (clientY ?? bounds.top + bounds.height / 2) - bounds.top - bounds.height / 2;
  const next = Math.round(Math.max(1, Math.min(8, value)) * 100) / 100;
  const factor = next / state.zoom;
  state.panX = x - (x - state.panX) * factor;
  state.panY = y - (y - state.panY) * factor;
  state.zoom = next;
  applyZoom();
}

function resetZoom() {
  state.zoom = 1; state.panX = 0; state.panY = 0;
  applyZoom();
}

const overVideoControls = target => Boolean(target.closest('button,input,select,aside,.upload-actions,.zoom-controls'));
$('zoom').addEventListener('input', event => zoomAt(Number(event.target.value)));
$('zoom-in').addEventListener('click', () => zoomAt(state.zoom + 0.25));
$('zoom-out').addEventListener('click', () => zoomAt(state.zoom - 0.25));
$('reset-zoom').addEventListener('click', resetZoom);
$('stage').addEventListener('wheel', event => {
  if (!state.hasVideo || overVideoControls(event.target)) return;
  event.preventDefault();
  const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? $('stage').clientHeight : 1);
  zoomAt(state.zoom * Math.exp(-delta * (event.shiftKey ? 0.0005 : 0.002)), event.clientX, event.clientY);
}, { passive: false });
let drag = null;
$('stage').addEventListener('pointerdown', event => {
  if (!state.hasVideo || state.zoom === 1 || event.button !== 0 || overVideoControls(event.target)) return;
  event.preventDefault();
  drag = { id: event.pointerId, x: event.clientX, y: event.clientY, panX: state.panX, panY: state.panY };
  $('stage').setPointerCapture(event.pointerId);
  $('stage').classList.add('panning');
});
$('stage').addEventListener('pointermove', event => {
  if (!drag || event.pointerId !== drag.id) return;
  state.panX = drag.panX + event.clientX - drag.x;
  state.panY = drag.panY + event.clientY - drag.y;
  applyZoom();
});
function endPan() { drag = null; $('stage').classList.remove('panning'); }
$('stage').addEventListener('lostpointercapture', endPan);
$('stage').addEventListener('pointerup', event => {
  if (drag?.id === event.pointerId) { $('stage').releasePointerCapture(event.pointerId); endPan(); }
});
$('stage').addEventListener('pointercancel', endPan);
$('stage').addEventListener('dblclick', event => {
  if (!overVideoControls(event.target)) resetZoom();
});

$('reset-video').addEventListener('click', () => {
  pausePlayback();
  video.removeAttribute('src');
  video.load();
  if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
  state.videoUrl = null;
  state.hasVideo = false;
  $('video-file').value = '';
  $('open-video').hidden = false;
  $('open-video').removeAttribute('title');
  $('reset-video').hidden = true;
  $('stage').classList.remove('has-video');
  $('zoom-controls').hidden = true;
  state.direction = 1;
  resetZoom();
  setPlaybackRate(1);
  setFrame(0);
  updateFrameCheck();
  setMessage('');
});
$('reset-predictions').addEventListener('click', () => {
  ++state.csvLoadVersion;
  state.predictions = null;
  state.predictionSource = '';
  $('prediction-file').value = '';
  updateCsvSelection();
  $('decision-symbol').hidden = true;
  $('prediction-value').hidden = true;
  $('reset-predictions').hidden = true;
  updateFrameCheck();
  setFrame(state.frame);
  drawTimeline();
  setMessage('');
});
setPlaybackRate(1);
$('seek').addEventListener('input', event => seekToFrame(event.target.value));
$('frame-input').addEventListener('change', event => seekToFrame(event.target.value));
document.addEventListener('keydown', event => {
  if (['INPUT', 'TEXTAREA', 'BUTTON', 'SELECT'].includes(document.activeElement?.tagName)) return;
  if (event.code === 'Space') { event.preventDefault(); $('play').click(); }
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
    event.preventDefault();
    seekToFrame(state.frame + (event.key === 'ArrowLeft' ? -1 : 1) * (event.shiftKey ? FPS : 1));
  }
});
window.addEventListener('resize', () => { applyZoom(); drawTimeline(); drawTrace(); });
setFrame(0); drawTimeline();

$('jump-reference').addEventListener('click', () => {
  if (state.reference) seekToFrame(state.reference.first);
});
loadReference();

loadCsvOptions();
