const FPS = 25;
const FRAME_COUNT = 199319;
const LAST_FRAME = FRAME_COUNT - 1;
const LABELS = {
  H: { name: 'Home possession', short: 'HOME · BARCELONA', color: '#568eff', level: 2 },
  A: { name: 'Away possession', short: 'AWAY · ATHLETIC', color: '#f34d4d', level: 0 },
  D: { name: 'Dead ball', short: 'DEAD BALL', color: '#e4bc46', level: 1 },
  X: { name: 'Unknown', short: 'UNKNOWN', color: '#aaa', level: 3 },
};

const $ = id => document.getElementById(id);
const video = $('video');
const state = { frame: 0, predictions: null, videoUrl: null, hasVideo: false };

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
  $('decision-symbol').textContent = state.predictions ? code : '—';
  $('decision-symbol').className = `decision-symbol ${code.toLowerCase()}`;
  $('decision-symbol').setAttribute('aria-label', state.predictions ? LABELS[code].name : 'No predictions loaded');
  drawTrace();
}

function seekToFrame(frame) {
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

async function loadCsv(file) {
  if (!file) return;
  try {
    const parsed = parseCsv(await file.text(), file.name);
    state.predictions = parsed;
    $('open-predictions').hidden = true;
    $('decision-symbol').hidden = false;
    $('reset-predictions').hidden = false;
    updateFrameCheck();
    $('open-predictions').title = file.name;
    const missing = FRAME_COUNT - parsed.rows;
    const notes = [`${file.name}: ${parsed.rows.toLocaleString()} frames loaded`];
    if (missing) notes.push(`${missing.toLocaleString()} missing`);
    if (parsed.duplicates) notes.push(`${parsed.duplicates.toLocaleString()} duplicate rows`);
    setMessage(notes.join(' · ') + '.');
    setFrame(state.frame);
    drawTimeline();
  } catch (error) {
    setMessage(error.message, true);
  }
}

function loadVideo(source, label, objectUrl = false) {
  if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
  state.videoUrl = objectUrl ? source : null;
  state.hasVideo = false;
  updateFrameCheck();
  video.pause();
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

function drawTimeline() {
  const { ctx, width, height } = prepareCanvas($('timeline'));
  ctx.fillStyle = '#333'; ctx.fillRect(0, 0, width, height);
  if (!state.predictions) return;
  // Sampling at each screen pixel keeps rendering fast for the full 199k-frame match.
  for (let x = 0; x < width; x++) {
    const frame = Math.min(LAST_FRAME, Math.floor((x + .5) / width * FRAME_COUNT));
    ctx.fillStyle = LABELS[state.predictions.labels[frame] || 'X'].color;
    ctx.fillRect(x, 0, 1, height);
  }
  // The public graded halves are marked without implying a ground-truth label.
  ctx.fillStyle = '#0005';
  for (const [start, end] of [[0, 11185], [90652, 110808], [190198, FRAME_COUNT]]) {
    ctx.fillRect(start / FRAME_COUNT * width, 0, (end - start) / FRAME_COUNT * width, height);
  }
}

function drawTrace() {
  const { ctx, width, height } = prepareCanvas($('trace'));
  const pad = 13, innerHeight = height - pad * 2;
  ctx.strokeStyle = '#ffffff12'; ctx.lineWidth = 1;
  for (let row = 0; row < 4; row++) {
    const y = pad + row * innerHeight / 3;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
  }
  ctx.fillStyle = '#aaa'; ctx.font = '9px monospace';
  ['A','D','H','X'].forEach((code, index) => { ctx.fillStyle = LABELS[code].color; ctx.fillText(code, 4, pad + index * innerHeight / 3 + 3); });
  const labels = state.predictions?.labels;
  if (labels) {
    let previous = null;
    for (let x = 16; x < width; x++) {
      const offset = Math.round((x - 16) / Math.max(1, width - 17) * 500) - 250;
      const frame = Math.max(0, Math.min(LAST_FRAME, state.frame + offset));
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
  ctx.strokeStyle = '#ffffffcc'; ctx.lineWidth = 1;
  const nowX = 16 + (width - 17) / 2;
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
  setMessage('');
  updateFromVideo();
});
video.addEventListener('error', () => {
  state.hasVideo = false;
  $('stage').classList.remove('has-video');
  setMessage('Video could not play. Choose a browser-compatible video file.', true);
});
video.addEventListener('timeupdate', updateFromVideo);
video.addEventListener('seeked', updateFromVideo);
video.addEventListener('play', () => { $('play').textContent = 'Ⅱ'; $('play').setAttribute('aria-label', 'Pause'); });
video.addEventListener('pause', () => { $('play').textContent = '▶'; $('play').setAttribute('aria-label', 'Play'); });
if (video.requestVideoFrameCallback) {
  const onVideoFrame = (_, metadata) => {
    if (state.hasVideo) setFrame(Math.floor(metadata.mediaTime * FPS + 0.00001));
    video.requestVideoFrameCallback(onVideoFrame);
  };
  video.requestVideoFrameCallback(onVideoFrame);
}

$('open-predictions').addEventListener('click', () => $('prediction-file').click());
$('open-video').addEventListener('click', () => $('video-file').click());
$('video-file').addEventListener('change', event => {
  const file = event.target.files[0];
  if (file) { $('open-video').title = file.name; loadVideo(URL.createObjectURL(file), file.name, true); }
});
$('prediction-file').addEventListener('change', event => loadCsv(event.target.files[0]));
$('play').addEventListener('click', () => {
  if (!state.hasVideo) { setMessage('Load a tactical video to play it.', true); return; }
  if (video.paused) video.play().catch(() => setMessage('Browser could not start playback.', true));
  else video.pause();
});
for (const [id, delta] of [['back-frame', -1], ['forward-frame', 1]]) {
  $(id).addEventListener('click', () => { video.pause(); seekToFrame(state.frame + delta); });
}
function setPlaybackRate(rate) {
  video.playbackRate = rate;
  $('playback-rate').textContent = `${rate}×`;
  $('slower').disabled = rate === 1;
  $('faster').disabled = rate === 8;
}
$('slower').addEventListener('click', () => setPlaybackRate(Math.max(1, video.playbackRate / 2)));
$('faster').addEventListener('click', () => setPlaybackRate(Math.min(8, video.playbackRate * 2)));
$('reset-video').addEventListener('click', () => {
  video.pause();
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
  setPlaybackRate(1);
  setFrame(0);
  updateFrameCheck();
  setMessage('');
});
$('reset-predictions').addEventListener('click', () => {
  state.predictions = null;
  $('prediction-file').value = '';
  $('open-predictions').hidden = false;
  $('open-predictions').removeAttribute('title');
  $('decision-symbol').hidden = true;
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
  if (['INPUT', 'TEXTAREA', 'BUTTON'].includes(document.activeElement?.tagName)) return;
  if (event.code === 'Space') { event.preventDefault(); $('play').click(); }
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
    event.preventDefault(); video.pause();
    seekToFrame(state.frame + (event.key === 'ArrowLeft' ? -1 : 1) * (event.shiftKey ? FPS : 1));
  }
});
window.addEventListener('resize', () => { drawTimeline(); drawTrace(); });
setFrame(0); drawTimeline();
