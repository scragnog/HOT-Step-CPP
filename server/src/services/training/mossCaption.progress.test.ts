import assert from 'node:assert/strict';
import test from 'node:test';
import { PassThrough } from 'node:stream';
import { setImmediate } from 'node:timers/promises';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire, syncBuiltinESMExports } from 'node:module';
import { config } from '../../config.js';
import { parseMossProgressLine, runMossCaption, watchMossProgress } from './mossCaption.js';

// MOSS encode/write lines from a one-file label run; output paths anonymized.
// Batch and Flamingo fixtures follow ace-caption.cpp's corresponding status formats.
test('MOSS file, encoded-audio and written-caption lines map to caption stages', () => {
  assert.deepEqual(parseMossProgressLine('[MOSS] (1/2) C:\\audio folder\\sample.wav'), {
    phase: 'caption-encoding', audioPath: 'C:\\audio folder\\sample.wav',
  });
  assert.deepEqual(parseMossProgressLine('[MOSS] 16.0 s audio -> 1600 mel frames -> 200 audio tokens (12.50/s)'), {
    phase: 'caption-writing',
  });
  assert.deepEqual(parseMossProgressLine('[MOSS] prose   -> C:\\output folder\\out.prose.txt (1.8s)\r'), {
    phase: 'caption-writing', mode: 'prose', outputPath: 'C:\\output folder\\out.prose.txt',
  });
  assert.deepEqual(parseMossProgressLine('[MOSS] mm3     -> C:\\output folder\\out.mm3.txt (7.9s)'), {
    phase: 'caption-writing', mode: 'mm3', outputPath: 'C:\\output folder\\out.mm3.txt',
  });
});

test('Flamingo shares the parser and reports the end of loading', () => {
  assert.deepEqual(parseMossProgressLine('[Flamingo] loaded in 4.6s: LM 32 layers, H 4096, vocab 151936'), {
    phase: 'caption-encoding',
  });
  assert.deepEqual(parseMossProgressLine('[Flamingo] (2/2) /audio/sample.wav'), {
    phase: 'caption-encoding', audioPath: '/audio/sample.wav',
  });
  assert.deepEqual(parseMossProgressLine('[Flamingo] 30.0 s audio -> 375 audio tokens, encoded in 1.25s'), {
    phase: 'caption-writing',
  });
  assert.deepEqual(parseMossProgressLine('[Flamingo] lyrics  -> /output/out (6.1s)'), {
    phase: 'caption-writing', mode: 'lyrics', outputPath: '/output/out',
  });
});

test('unrelated output, diagnostics and failed writes do not claim progress', () => {
  for (const line of [
    'ggml_cuda_init: found 1 CUDA devices:',
    'ace-caption: cannot write /output/out.txt',
    '[MOSS] could not open /output/out.txt for writing',
    '[MOSS] dumped encoder_out [375,4096] mean=0.0 rms=1.0 -> /output/embeds',
    '[MOSS] mm3 -> /output/out.mm3.txt (failed)',
    '',
  ]) assert.equal(parseMossProgressLine(line), undefined, line);
});

test('single-file progress is emitted before EOF and split lines and the final line survive', async () => {
  const stderr = new PassThrough();
  const events: unknown[] = [];
  const logs: string[] = [];
  const written: unknown[] = [];
  watchMossProgress(stderr, {
    onProgress: (phase, audioPath) => events.push({ phase, audioPath }),
    log: (_, line) => logs.push(line),
  }, '/audio/single.wav', (outputPath, mode) => written.push({ outputPath, mode }));
  stderr.write('[MOSS] 30.0 s audio -> 3000 mel frames -> ');
  assert.equal(events.length, 0);
  stderr.write('375 audio tokens (12.50/s)\r\n');
  assert.deepEqual(events, [{ phase: 'caption-writing', audioPath: '/audio/single.wav' }]);
  assert.equal(stderr.readableEnded, false, 'progress must arrive while the process is running');
  stderr.end('[MOSS] mm3     -> /output/out.mm3.txt (6.1s)');
  await setImmediate();
  assert.equal(events.length, 2);
  assert.equal(logs.length, 2);
  assert.deepEqual(written, [{ outputPath: '/output/out.mm3.txt', mode: 'mm3' }]);
});

test('batch progress keeps stage and filename paired when the next track starts', async () => {
  const stderr = new PassThrough();
  const events: unknown[] = [];
  watchMossProgress(stderr, {
    onProgress: (phase, audioPath) => events.push({ phase, audioPath }),
  }, '/audio/first.wav');
  stderr.end('[MOSS] (2/2) /audio/second.wav\n'
    + '[MOSS] 30.0 s audio -> 3000 mel frames -> 375 audio tokens (12.50/s)\n');
  await setImmediate();
  assert.deepEqual(events, [
    { phase: 'caption-encoding', audioPath: '/audio/second.wav' },
    { phase: 'caption-writing', audioPath: '/audio/second.wav' },
  ]);
});

test('single-file execution publishes stages before execFile completes', async t => {
  const work = fs.mkdtempSync(path.join(process.cwd(), '.test-moss-progress-'));
  const previousExe = config.aceServer.exe;
  const previousModels = config.aceServer.models;
  const childProcess = createRequire(import.meta.url)('node:child_process');
  const stderr = new PassThrough();
  let finish: (() => void) | undefined;
  t.after(() => {
    t.mock.restoreAll();
    syncBuiltinESMExports();
    config.aceServer.exe = previousExe;
    config.aceServer.models = previousModels;
    fs.rmSync(work, { recursive: true, force: true });
  });
  config.aceServer.exe = path.join(work, 'ace-server.exe');
  config.aceServer.models = work;
  fs.mkdirSync(path.join(work, 'moss'));
  for (const name of ['moss-lm-q8_0.gguf', 'moss-aud-f16.gguf']) {
    fs.writeFileSync(path.join(work, 'moss', name), 'fixture');
  }
  fs.writeFileSync(path.join(work, process.platform === 'win32' ? 'ace-caption.exe' : 'ace-caption'), 'fixture');
  t.mock.method(childProcess, 'execFile', (_exe: string, args: string[], _options: unknown, callback: (...args: unknown[]) => void) => {
    finish = () => {
      fs.writeFileSync(args[args.indexOf('-o') + 1], 'A caption from audio');
      stderr.end();
      callback(null, '', '');
    };
    return { stderr };
  });
  syncBuiltinESMExports();
  const phases: string[] = [];
  let complete = false;
  const run = runMossCaption('/audio/single.wav', {
    modes: ['prose'], onProgress: phase => phases.push(phase),
  }).then(result => { complete = true; return result; });
  await setImmediate();
  assert.deepEqual(phases, ['caption-loading']);
  stderr.write('[MOSS] 30.0 s audio -> 3000 mel frames -> 375 audio tokens (12.50/s)\n');
  assert.deepEqual(phases, ['caption-loading', 'caption-writing']);
  assert.equal(complete, false, 'the card must update before the process callback');
  assert.ok(finish);
  finish();
  assert.deepEqual(await run, { prose: 'A caption from audio' });
});
