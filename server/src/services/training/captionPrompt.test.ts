// What this guards: the two cleanups that keep a local model's reasoning out of
// dataset sidecars. #189: a closing </think> with no opener (the template opened
// the block, the model reasoned in plain text) left the reasoning in .yue2.txt.
// #190: text-only rewrites that echo the template or narrate their plan were
// written over good captions.

import assert from 'node:assert/strict';
import test from 'node:test';
import { captionProblem } from './captionPrompt.js';
import { stripThinkingBlocks } from '../lireek/llm/postprocess.js';

test('stripThinkingBlocks drops reasoning ended by an orphan closer', () => {
  assert.equal(stripThinkingBlocks('<think>\n\n</think>\n\nLet me analyze...\nstuff\n</think>\nEnglish, worship ballad'),
    'English, worship ballad');
  assert.equal(stripThinkingBlocks('Reasoning here </think> Final'), 'Final');
  assert.equal(stripThinkingBlocks('<think>x</think>Clean answer'), 'Clean answer');
  assert.equal(stripThinkingBlocks('Plain answer'), 'Plain answer');
});

test('captionProblem rejects the failures seen in #190 and keeps a real caption', () => {
  for (const bad of [
    '** I need to adapt the existing caption while following all rules: the tempo and',
    'No audio file was attached to this request, so no audible content can be described here at all',
    '-',
    'Based on local analysis notes (no audio attached): a slow ballad with piano. Wait, I',
    '<2 to 4 sentences on one line>',
    'pop ballad in the modern worship tradition")',
    "I still need to write a caption as it's required by the template, so here it goes now",
  ]) assert.ok(captionProblem(bad), bad);
  assert.equal(captionProblem('A slow contemporary worship ballad led by a warm female mezzo-soprano over '
    + 'sustained piano chords, swelling strings and a restrained kick that builds into a full band chorus.'), null);
});
