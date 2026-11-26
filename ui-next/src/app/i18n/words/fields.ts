// The prompt card's and the lyric sheet's own words (package 4). The core's sentences about them are in
// src/derive/fields.ts, loaded here so the translation table holds them from the start.

import '../../../derive/fields.ts';

export const WORDS: { [id: string]: string } = {
  'ui.prompt.songInfo': 'Song info',
  'ui.prompt.latent': 'Latent',
  'ui.prompt.latent.tip': 'The next ACE-Step render continues from this latent',
  'ui.prompt.wildAuto': 'Expand on Generate',
  'ui.prompt.wildAuto.tip': 'Expand wildcards when you press Generate',
  'ui.prompt.trigger.aria': 'Adapter trigger word, up to 24 characters',
  'ui.lyrics.wildAuto': 'Expand on Generate',
  'ui.lyrics.builtIn': 'built in',
};
