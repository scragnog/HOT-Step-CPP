// abcMidi.ts — a YuE2 ABC score as a Standard MIDI File, through abcjs's own
// MIDI writer (the same one its in-app playback uses): one track per voice
// (Vocal, Ins), the chord symbols as a comping track, the Q: tempo and M:
// meter, so the score opens in a DAW on its bars.

import abcjs from 'abcjs';
import { safeFileStem, saveBlob } from '@/utils/storage/saveFile';

/** The first tune in `abc` as MIDI bytes; null when abcjs finds no notes. */
export function abcToMidi(abc: string): Uint8Array | null {
  const out = abcjs.synth.getMidiFile(abc, { midiOutputType: 'binary' });
  const first = Array.isArray(out) ? out[0] : out;
  // a header and an empty track is all abcjs writes for a tune with no notes
  return first instanceof Uint8Array && first.length > 64 ? first : null;
}

/** Save the score as `<name>.mid`. False when there was nothing to save. */
export function downloadAbcMidi(abc: string, name: string): boolean {
  const midi = abcToMidi(abc);
  if (!midi) return false;
  saveBlob(new Blob([midi as BlobPart], { type: 'audio/midi' }), `${safeFileStem(name, 'score')}.mid`);
  return true;
}
