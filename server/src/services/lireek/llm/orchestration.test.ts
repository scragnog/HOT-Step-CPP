import test from 'node:test';
import assert from 'node:assert/strict';
import { getProvider } from './registry.js';
import { generateLyricsStreaming } from './orchestration.js';
import { buildProfile, type LyricsProfile } from '../profilerService.js';
import { MM3_CAPTION_FIELDS, TITLE_DERIVATION_PROMPT } from '../prompts.js';

const mm3 = MM3_CAPTION_FIELDS.map(field => 'heading' in field ? field.heading :
  `${field.label}: ${'The piano and sustained strings build slowly with room ambience and gentle motion. '.repeat(4)}`).join('\n');

test('instrumental generation plans from all source formats and skips every lyric step', async ctx => {
  const profile = await buildProfile('Fixture', 'Collection', [
    { title: 'A', lyrics: '', instrumental: true, bpm: 70, key: 'D Minor', duration: 120,
      caption: 'ACE source piano', mm3Caption: 'MM3 source strings', yue2Caption: 'YuE2 source ambient', genre: 'Ambient', signature: '4/4' },
    { title: 'B', lyrics: '', instrumental: true, bpm: 90, key: 'A Minor', duration: 180, caption: 'ACE source strings' },
  ], 'not-a-provider') as LyricsProfile;
  const calls: Array<{ system: string; user: string }> = [];
  const phases: string[] = [];
  ctx.mock.method(getProvider('ollama'), 'call', async (system: string, user: string) => {
    calls.push({ system, user });
    if (system.startsWith('Plan an instrumental')) return JSON.stringify({ subject: 'A valley', bpm: 72, key: 'D Minor', duration: 999,
      caption: 'Piano and strings rise through a long crescendo.', structure: 'Piano opening, strings build, quiet ending' });
    if (system.startsWith('Choose a short title')) return 'Valley Path';
    if (system.includes('MiniMax-Music3')) return mm3.replace('Vocal Gender & Timbre: ', 'Vocal Gender & Timbre: Singer A (Instrumental). ');
    if (system.includes('YuE2')) return 'Instrumental ambient orchestral music with a soft piano lead, sustained strings, gentle woodwind swells and light percussion, a reflective and quietly tense mood building into a gradual lift, warm natural room ambience and spacious acoustic production, 72 BPM.';
    throw new Error('Unexpected lyric call: '+system.slice(0, 80));
  });
  const result = await generateLyricsStreaming(profile, 'ollama', 'fixture', 'tense, slow build',
    ['old subject'], [70], ['A minor'], ['Old title'], [120], undefined, phase => phases.push(phase), 'New scene');
  assert.equal(result.lyrics, '[Instrumental]');
  assert.equal(result.subject, 'New scene');
  assert.equal(result.duration, 180);
  assert.equal(result.key, 'D minor');
  assert.ok(result.title.startsWith('Valley Path'));
  assert.ok(result.caption && result.caption_mm3 && result.caption_yue2);
  assert.ok(!result.caption_mm3.includes('Singer A'));
  for (const evidence of ['ACE source piano', 'MM3 source strings', 'YuE2 source ambient', 'distributions', 'old subject', 'tense, slow build', 'New scene'])
    assert.ok(calls[0].user.includes(evidence), evidence);
  assert.ok(calls[1].user.includes('Old title'));
  assert.ok(calls.find(call => call.system.includes('MiniMax-Music3'))?.user.includes('Piano opening, strings build, quiet ending'));
  assert.equal(phases.some(phase => /writing lyrics|refin|slop/i.test(phase)), false);
  assert.equal(calls.length, 4);
});

test('vocal generation still writes lyrics and retains its lyric system prompt', async ctx => {
  const calls: string[] = [];
  ctx.mock.method(getProvider('ollama'), 'call', async (system: string) => {
    calls.push(system);
    if (system.includes('MiniMax-Music3')) return mm3;
    if (system.includes('YuE2')) return 'English acoustic folk with a gentle male voice, softly picked guitar, sustained bass, brushed percussion and light piano accents, a reflective mood that grows into a hopeful chorus, warm room ambience and clear natural acoustic production, 80 BPM.';
    if (system.startsWith(TITLE_DERIVATION_PROMPT)) return 'Road Home';
    return '[Verse 1]\nWe walk the road.\nWe carry home.\n[Chorus]\nThe light is near.\nWe stay right here.';
  });
  const result = await generateLyricsStreaming({ artist: 'Fixture' } as LyricsProfile, 'ollama', 'fixture');
  assert.notEqual(result.lyrics, '[Instrumental]');
  assert.ok(result.lyrics.includes('We walk'));
  assert.ok(calls[0].includes('lyrics'));
  assert.equal(calls.some(call => call.startsWith('Plan an instrumental')), false);
});
