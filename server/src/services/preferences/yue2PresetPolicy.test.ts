import assert from 'node:assert/strict';
import test from 'node:test';
import express from 'express';
import { resolveYue2PresetSchema } from '../../contracts/preferences.js';
import preferenceRoutes from '../../routes/preferences.js';
import { resolveYue2Preset } from './yue2PresetPolicy.js';

// Expected values capture Yue2AitkTrainCard's previous loadPreset spread order.
const currentForm = { adapterType: 'lokr', targetKl: 0.9, plannerLrScale: 0.6, narLrScale: 0.2,
  cautious: true, dataset: 'current-dataset', checkpoint: 'current-checkpoint', output: 'current-output',
  resume: 'current-resume', unsavedFutureField: 'keep' };
const defaults = { ...currentForm, adapterType: 'lora', targetKl: 1.4, plannerLrScale: 0.3,
  narLrScale: undefined, cautious: false };

for (const version of [undefined, 1, 2] as const) {
  for (const timing of [undefined, true, false] as const) {
    test(`legacy preset version ${version ?? 'missing'}, lyricTiming ${String(timing)}`, () => {
      const settings = { rank: 64, ...(timing === undefined ? {} : { lyricTiming: timing }) };
      const preset = { name: 'Fixture', ...(version === undefined ? {} : { version }), settings };
      const input = resolveYue2PresetSchema.parse({ preset, currentForm, lyricTiming: true });
      const before = structuredClone(preset);
      assert.deepEqual(resolveYue2Preset(input), {
        effectiveForm: { ...defaults, ...settings },
        lyricTiming: version === 2 && timing !== undefined ? timing : true,
      });
      assert.deepEqual(preset, before);
    });
  }
}

test('stored adapter, stop and cautious values override compatibility defaults', () => {
  const preset = { name: 'Override', version: 2 as const, settings: {
    adapterType: 'lokr', targetKl: 2, plannerLrScale: 0.7, narLrScale: 0.8, cautious: true,
  } };
  const result = resolveYue2Preset({ preset, currentForm, lyricTiming: false });
  assert.deepEqual(result.effectiveForm, { ...defaults, ...preset.settings });
  assert.equal(result.lyricTiming, false);
});

test('invalid application request fails validation', () => {
  assert.equal(resolveYue2PresetSchema.safeParse({ preset: { name: 'Bad', settings: [] }, currentForm, lyricTiming: true }).success, false);
});

test('HTTP resolver returns an effective result and validation errors without writing a preset', async () => {
  const app = express();
  app.use(express.json());
  app.use('/api/preferences', preferenceRoutes);
  const server = app.listen(0, '127.0.0.1');
  await new Promise<void>(resolve => server.once('listening', resolve));
  try {
    const address = server.address();
    assert(address && typeof address !== 'string');
    const url = `http://127.0.0.1:${address.port}/api/preferences/presets/yue2-joint/resolve`;
    const preset = { name: 'Raw fixture', version: 2, settings: { rank: 32, cautious: true, lyricTiming: false } };
    const response = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preset, currentForm, lyricTiming: true }) });
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), JSON.parse(JSON.stringify({ result: {
      effectiveForm: { ...defaults, ...preset.settings }, lyricTiming: false,
    } })));
    const invalid = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preset, currentForm: null, lyricTiming: true }) });
    assert.equal(invalid.status, 400);
    assert.match((await invalid.json() as { error: string }).error, /Invalid YuE2 preset application/);
  } finally {
    await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  }
});
