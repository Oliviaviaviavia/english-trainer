const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script);

function functionSource(name) {
  const start = script.indexOf(`function ${name}(`);
  assert.notEqual(start, -1);
  let end = script.indexOf('{', start) + 1;
  let depth = 1;
  while (depth) {
    if (script[end] === '{') depth++;
    if (script[end] === '}') depth--;
    end++;
  }
  return script.slice(start, end);
}

function harness(paused = false) {
  const frames = new Map();
  let nextId = 0;
  const spans = Array.from({ length: 5 }, () => {
    const classes = new Set();
    return { classList: { add: c => classes.add(c), remove: c => classes.delete(c), contains: c => classes.has(c) } };
  });
  const seek = { value: 0 };
  const player = { currentTime: 12, duration: 50, paused, ended: false, _flowKey: 'material' };
  const context = vm.createContext({
    player, spans, state: { mode: 'flow', source: 'video', timing: [{ start: 0 }] },
    document: { getElementById: () => seek },
    flowPlaybackKey: () => 'material', updateFlowTimeline() {},
    requestAnimationFrame(callback) { const id = ++nextId; frames.set(id, callback); return id; },
    cancelAnimationFrame(id) { frames.delete(id); },
  });
  vm.runInContext(`let hlSpans=spans, hlTimes=[0,10,20,30,40], rafId=null, flowSpeech=null;\n`
    + functionSource('startTick') + '\n' + functionSource('seekFlow'), context);
  const run = code => vm.runInContext(code, context);
  return {
    player, frames,
    start() { run('startTick()'); },
    seekTo(seconds) { seek.value = seconds / player.duration * 1000; run('seekFlow()'); },
    frame() { const pending = [...frames.values()]; frames.clear(); pending.forEach(callback => callback()); },
    highlighted() { return spans.flatMap((span, i) => span.classList.contains('cur') ? [i] : []); },
  };
}

test('seeking backward and forward leaves only the current sentence highlighted', () => {
  const h = harness();
  h.start(); h.frame();
  assert.deepEqual(h.highlighted(), [1]);
  for (const [seconds, index] of [[1, 0], [35, 3], [21, 2], [2, 0], [42, 4]]) {
    h.seekTo(seconds); h.frame();
    assert.deepEqual(h.highlighted(), [index]);
  }
});

test('seeking while paused updates the highlight immediately without starting playback', () => {
  const h = harness(true);
  h.start(); h.frame();
  h.seekTo(25);
  assert.deepEqual(h.highlighted(), [2]);
  assert.equal(h.player.paused, true);
  assert.equal(h.frames.size, 0);
  h.seekTo(5);
  assert.deepEqual(h.highlighted(), [0]);
});

test('rapid seeks keep one animation loop and playback continues highlighting normally', () => {
  const h = harness();
  h.start(); h.frame();
  for (const seconds of [31, 2, 24, 11, 41, 3]) h.seekTo(seconds);
  assert.deepEqual(h.highlighted(), [0]);
  assert.equal(h.frames.size, 1);
  h.player.currentTime = 11; h.frame();
  assert.deepEqual(h.highlighted(), [1]);
  h.player.currentTime = 23; h.frame();
  assert.deepEqual(h.highlighted(), [2]);
});
