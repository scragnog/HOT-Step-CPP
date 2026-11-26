// Package 3c's words. Its instruments' words are the core's (src/derive/build.ts, recording.ts, score.ts add theirs to
// the core's table as they load: imported here so the table holds them from the start); these are the score
// preview's chrome.

import '../../../derive/build.ts';
import '../../../derive/recording.ts';
import '../../../derive/score.ts';

export const WORDS: { [id: string]: string } = {
  'ui.score.dialog': 'Lead sheet preview',
  'ui.score.close': 'Close',
  'ui.score.play': 'Play',
  'ui.score.stop': 'Stop',
  'ui.score.download': 'Download .abc',
  'ui.score.cancel': 'Cancel',
  'ui.score.retry': 'Retry, new seed',
  'ui.score.go': 'Continue with this score',
  'ui.score.paper': 'The lead sheet as written',
};
