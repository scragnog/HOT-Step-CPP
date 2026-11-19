import { isPresent } from '@/utils/guards';
// General MIDI 53, Choir Aahs (0-based 52).
const CHOIR_AAHS = 52;

/** The ABC as the preview plays it: the Vocal voice sung on Choir Aahs so it
 *  stands apart from the piano Ins line. Every `V:` line naming Vocal (the
 *  server's isVocalVoice test) gets a `%%MIDI program` line after it; a score
 *  with no voices is all melody, so it gets one after its K: line. Only the
 *  preview uses this, never the saved or approved score. Section offsets
 *  must be read from this string too, since the inserted lines shift them. */
export function withChoirVocal(abc: string): string {
  const program = `%%MIDI program ${CHOIR_AAHS}`;
  const lines = abc.split('\n');
  const isVoice = (line: string) => /^V:/.test(line.trim());
  const isVocal = (line: string) => {
    const text = line.trim();
    return /vocal/i.test(text.slice(2).split(/\s+/).filter(isPresent)[0] ?? '') || /name="Vocal/i.test(text);
  };
  if (!lines.some(isVoice)) {
    const key = lines.findIndex(line => /^K:/.test(line.trim()));
    return key < 0 ? abc : [...lines.slice(0, key + 1), program, ...lines.slice(key + 1)].join('\n');
  }
  // Only body V: lines: a %%MIDI line in the header would set every voice.
  let inBody = false;
  return lines.flatMap(line => {
    if (/^K:/.test(line.trim())) inBody = true;
    return inBody && isVoice(line) && isVocal(line) ? [line, program] : [line];
  }).join('\n');
}
