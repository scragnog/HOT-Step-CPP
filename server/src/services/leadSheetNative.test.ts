import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { chordLabelPitches, nativeMissingMessage, nativeSheetFromJson, runNativeLeadSheet, NATIVE_GRID_IDENTITY,
  type LeadSheetJson, type NativeDeps, type NativeFiles } from './leadSheetNative.js';
import { coverRenderRolls } from './coverLeadSheet.js';
import { leadSheetRoll } from './leadSheetRoll.js';

const ABC = 'X:1\nT:t\nM:4/4\nL:1/4\nQ:1/4=120\nK:C\nV:Vocal\n"C"CDEF|\n';
const json = (over: Partial<LeadSheetJson> = {}): LeadSheetJson => ({ version: 1, bpm: 120, bpm_rejection: null, vocal_rejection: null,
  vocal: [{ onset: 0, offset: 0.5, pitch: 60 }, { onset: 0.5, offset: 1, pitch: 62 }],
  drums: [{ onset: 0, offset: 0.1, pitch: 36 }],
  chords: [['N', 0, 0.5], ['C:maj', 0.5, 1.5], ['A:min7', 1.5, 2], ['weird', 2, 3]], ...over });

test('chord labels follow upstream: exact quality, then first prefix; N and unknowns give nothing', () => {
  assert.deepEqual(chordLabelPitches('C:maj'), [0, 4, 7]);
  assert.deepEqual(chordLabelPitches('A:min7'), [9, 0, 4, 7]);
  assert.deepEqual(chordLabelPitches('Bb:maj9'), [10, 2, 5]); // prefix "maj"
  assert.deepEqual(chordLabelPitches('G:7(b9)'), [7, 11, 2, 5]); // first prefix hit is "7"
  for (const label of ['N', 'C', 'H:maj', 'C:zzz']) assert.deepEqual(chordLabelPitches(label), []);
});

test('JSON maps to a native sheet from raw seconds with the JSON bpm', () => {
  const { sheet, notice } = nativeSheetFromJson(ABC, json());
  assert.equal(notice, undefined);
  assert.equal(sheet!.timing.path, 'native');
  assert.equal(sheet!.timing.bpm, 120);
  assert.equal(sheet!.timing.grid!.identity, NATIVE_GRID_IDENTITY);
  assert.equal(sheet!.chordQuality, 'transcribed');
  assert.equal(sheet!.leadSheetSource, 'yourmt3-chordnet');
  assert.equal(sheet!.timing.rawEvents.melody.length, 2);
  assert.equal(sheet!.timing.rawEvents.drums!.length, 1);
  assert.equal(sheet!.timing.rawEvents.chords.length, 7); // C:maj 3 + A:min7 4; N and "weird" add nothing
  const draft = { approvedAbc: ABC, leadSheet: sheet!, leadSheetSource: 'yourmt3-chordnet' } as any;
  assert.equal(coverRenderRolls(draft).path, 'native');
});

test('both upstream rejections return the ABC without native timing and say why', () => {
  for (const [over, text] of [[{ bpm: 0, bpm_rejection: 'Could not determine a positive BPM' }, /positive BPM/],
    [{ vocal: [], vocal_rejection: 'Melody must contain at least one note.' }, /at least one note/]] as const) {
    const { sheet, notice } = nativeSheetFromJson(ABC, json(over as Partial<LeadSheetJson>));
    assert.equal(sheet!.timing.path, 'abc');
    assert.match(notice!, text);
    assert.equal(sheet!.abc, ABC);
  }
});

test('an unreadable JSON is an error', () => assert.throws(() => nativeSheetFromJson(ABC, { version: 2 } as any), /unreadable/));

function setup(script: string) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ls-native-'));
  const touch = (n: string) => { const p = path.join(root, n); fs.writeFileSync(p, 'x'); return p; };
  const files: NativeFiles = { exe: process.execPath, yourmt3: touch('y.gguf'), chordnet: touch('c.gguf'), templates: touch('t.tsv'),
    beatDir: root, beatReady: true };
  const audio = touch('a.mp3');
  const stub = path.join(root, 'stub.cjs');
  fs.writeFileSync(stub, script);
  const spawned: string[][] = [];
  const deps: NativeDeps = { files, dir: path.join(root, 'work'), idleMs: 60_000, capMs: 60_000,
    stall: () => ({ touch() {}, clear() {} }),
    spawn: (_exe, args) => { spawned.push(args); return spawn(process.execPath, [stub, ...args], { windowsHide: true }); } };
  return { root, files, audio, deps, spawned };
}
const writes = (j: object, abc = `${ABC}\n`) => `const a=process.argv; const v=n=>a[a.indexOf(n)+1]; const fs=require('fs');
fs.writeFileSync(v('--out'), ${JSON.stringify(JSON.stringify(j))}); fs.writeFileSync(v('--abc-out'), ${JSON.stringify(abc)});`;

test('a run passes the flags and returns the mapped sheet', async () => {
  const t = setup(writes(json()));
  const r = await runNativeLeadSheet(t.audio, new AbortController().signal, t.deps);
  assert.equal(r.leadSheet!.timing.path, 'native');
  assert.equal(r.abc, ABC.trim());
  const a = t.spawned[0];
  for (const flag of ['--lead-sheet', '--yourmt3-model', '--chord-model', '--chord-vocab', '--out', '--beat-model', '--abc-out', '--abc-chordnet']) assert.ok(a.includes(flag), flag);
  assert.equal(fs.existsSync(t.deps.dir), false, 'work dir removed');
});

test('a missing model names the file and spawns nothing', async () => {
  const t = setup('');
  fs.rmSync(t.files.chordnet);
  await assert.rejects(runNativeLeadSheet(t.audio, new AbortController().signal, t.deps), /missing .*chordnet-F32\.gguf.*convert-yourmt3\.py and convert-chordnet\.py/);
  assert.equal(t.spawned.length, 0);
  assert.match(nativeMissingMessage({ ...t.files, exe: null, beatReady: false })!, /ace-midi executable.*Beat This!/);
});

test('a non-zero exit carries the stderr tail', async () => {
  const t = setup(`console.error('boom: no cuda'); process.exit(3);`);
  await assert.rejects(runNativeLeadSheet(t.audio, new AbortController().signal, t.deps), /code 3: boom: no cuda/);
});

test('abort kills the child', async () => {
  const t = setup(`require('fs').writeFileSync(process.argv[process.argv.indexOf('--out')+1]+'.pid', String(process.pid)); setInterval(()=>{},1000);`);
  const ac = new AbortController();
  const run = runNativeLeadSheet(t.audio, ac.signal, t.deps);
  const pidFile = path.join(t.deps.dir, 'lead-sheet.json.pid');
  for (let i = 0; i < 100 && !fs.existsSync(pidFile); i++) await new Promise(r => setTimeout(r, 50));
  const pid = Number(fs.readFileSync(pidFile, 'utf8'));
  ac.abort();
  await assert.rejects(run, /Cancelled/);
  assert.throws(() => process.kill(pid, 0), 'child is gone');
});

// Roll parity: the port's own lead-sheet.json for each capture's clip (ace-midi --lead-sheet --device cpu,
// written to _experiments/lead-sheet-port/parity-json-<capture>/) against the upstream capture's rolls.
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const captures = path.join(repo, '_experiments', 'lead-sheet-reference', 'captures');
const port = path.join(repo, '_experiments', 'lead-sheet-port');
for (const name of ['v2-vocal', 'v2-silent', 'v2-instrumental']) {
  const file = path.join(port, `parity-json-${name}`, 'lead-sheet.json');
  const absent = !fs.existsSync(file) || !fs.existsSync(path.join(captures, name, 'roll.pianoroll.f32'));
  test(`native roll parity: ${name}`, { skip: absent && 'captures or port lead-sheet.json absent' }, () => {
    const { sheet } = nativeSheetFromJson('', JSON.parse(fs.readFileSync(file, 'utf8')));
    assert.equal(sheet?.timing.path, 'native');
    const r = leadSheetRoll({ ...sheet!.timing.rawEvents, bpm: sheet!.timing.bpm, chordTiming: 'seconds' });
    for (const [key, data] of [['pianoroll', r.pianoroll], ['chord', r.chord], ['drum_pianoroll', r.drum_pianoroll], ['context_mask', r.context_mask]] as const) {
      const want = fs.readFileSync(path.join(captures, name, `roll.${key}.f32`));
      const got = Buffer.from(data.buffer, data.byteOffset, data.byteLength);
      if (!got.equals(want)) {
        let i = 0; while (i < Math.min(got.length, want.length) && got[i] === want[i]) i++;
        assert.fail(`${name} roll.${key} differs: first byte ${i} (cell ${Math.floor(i / 4)}), got ${got.length} bytes, want ${want.length}`);
      }
    }
  });
}
