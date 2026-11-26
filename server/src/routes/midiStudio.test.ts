import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';
import { runOnMidiLane, startStallTimer } from './midiStudio.js';
import { runOnGpuLane } from '../services/generation/gpuLane.js';
import { exportArgs, readGrid, sourceTrackPrograms } from '../services/muscriptor.js';

test('MIDI rewrite flags accept signed downbeats and zero-based GM programs', () => {
  assert.deepEqual(exportArgs({
    firstDownbeat: -0.25,
    fullVelocity: true,
    programOverrides: { 26: 30, 33: 0, [-1]: 50, 40: 128, 128: 1 },
  }), ['--first-downbeat', '-0.25', '--full-velocity', '--program-override', '26:30', '--program-override', '33:0']);
  assert.deepEqual(exportArgs({ fullVelocity: false, firstDownbeat: Infinity }), []);
});

test('beat data preserves a signed downbeat and reads old audio-clock bar markers', () => {
  const grid = { bpm: 120, onset_delay: 0.125, bars: [1, 3] };
  assert.equal(readGrid(JSON.stringify({ grid: { ...grid, first_downbeat: -0.25 } }))?.firstDownbeat, -0.25);
  assert.equal(readGrid(JSON.stringify({ grid }))?.firstDownbeat, 0.875);
});

test('two MIDI tracks remapped to the same patch retain separate source programs', () => {
  const track = (channel: number, name: string) => {
    const events = Buffer.concat([
      Buffer.from([0, 0xff, 3, name.length]),
      Buffer.from(name),
      Buffer.from([0, 0xc0 | channel, 30, 0, 0xff, 0x2f, 0]),
    ]);
    const header = Buffer.alloc(8);
    header.write('MTrk');
    header.writeUInt32BE(events.length, 4);
    return Buffer.concat([header, events]);
  };
  const header = Buffer.from('4d546864000000060001000301e0', 'hex');
  const midi = Buffer.concat([header, track(0, 'clean electric guitar'), track(1, 'program 30'), track(9, 'drums')]);
  assert.deepEqual([...sourceTrackPrograms(midi)], [[0, 26], [1, 30]]);
});

test('a MIDI job waits behind a running GPU job (#227)', async () => {
  let releaseRender!: () => void;
  const order: string[] = [];
  const render = runOnGpuLane(() => new Promise<void>(resolve => {
    order.push('render start');
    releaseRender = () => { order.push('render end'); resolve(); };
  }), { label: 'render', family: 'yue2' });
  const midi = runOnMidiLane('job', async () => { order.push('midi'); });

  await new Promise(resolve => setTimeout(resolve, 20));
  assert.deepEqual(order, ['render start']);
  releaseRender();
  await Promise.all([render, midi]);
  assert.deepEqual(order, ['render start', 'render end', 'midi']);
});

function runFake(args: string[], idleMs: number, capMs: number): Promise<{ code: number | null; lines: number; reason?: string }> {
  const fixture = fileURLToPath(new URL('./fixtures/slowMidiExe.mjs', import.meta.url));
  const child = spawn(process.execPath, [fixture, ...args], { windowsHide: true });
  let lines = 0;
  let reason: string | undefined;
  const timer = startStallTimer((why) => { reason = why; child.kill(); }, idleMs, capMs);
  readline.createInterface({ input: child.stdout! }).on('line', () => { lines++; timer.touch(); });
  return new Promise(resolve => child.on('close', (c, signal) => {
    timer.clear();
    resolve({ code: signal ? null : c, lines, reason });
  }));
}

test('ace-midi is not killed while chunks keep coming (#227)', async () => {
  // Leave room for process startup under the full suite's parallel load.
  // Eight chunks take 3.2 seconds, past the two-second idle window.
  const r = await runFake(['8', '400'], 2_000, 30_000);
  console.log('steady run:', r);
  assert.equal(r.code, 0);
  assert.equal(r.lines, 8);
});

test('ace-midi is killed when its output stalls (#227)', async () => {
  const r = await runFake(['8', '100', '2'], 2_000, 30_000);
  console.log('stalled run:', r);
  assert.equal(r.code, null);
  assert.equal(r.lines, 2);
  assert.equal(r.reason, 'idle');
});

test('the overall cap still kills a run that never stalls', async () => {
  const r = await runFake(['200', '50'], 2_000, 4_000);
  console.log('capped run:', r);
  assert.equal(r.code, null);
  assert.equal(r.reason, 'cap');
  assert.ok(r.lines >= 5 && r.lines < 200, `lines=${r.lines}`);
});
