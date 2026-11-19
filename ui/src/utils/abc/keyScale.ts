/**
 * keyScale.ts — canonical key spelling for generation requests.
 *
 * The engine's metadata FSM builds its keyscale vocabulary from
 * `modes[] = { "major", "minor" }` (engine/src/metadata-fsm.h) — 70 values, all
 * lower-case. "D Major" is OUT OF VOCABULARY, not merely untidy.
 *
 * This exists because the two Lyric Studio paths that send a generation to the
 * engine had already drifted: Send-to-Create normalised the casing inline while
 * the audio queue passed `gen.key` through raw. The dataset sidecars spell the
 * mode capitalised ("E Minor", "C# Major"), so 99.6% of stored generations
 * carry a capitalised mode and only one of the two paths was correcting it.
 * Both now call this.
 *
 * Server-side counterpart: `normalizeKeyScale` in
 * server/src/services/lireek/prompts.ts, which canonicalises on write.
 */
export function normalizeKeyScale(key?: string | null): string {
  const raw = String(key ?? '').trim();
  const m = raw.match(/^([A-Ga-g])\s*([#b♯♭])?\s+([Mm]ajor|[Mm]inor)$/);
  if (!m) return raw;
  return `${m[1].toUpperCase()}${m[2] ?? ''} ${m[3].toLowerCase()}`;
}
