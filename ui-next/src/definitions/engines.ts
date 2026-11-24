// The engines, in one place. Every engine id a definition, the validator or a tool may use comes from this list;
// `EngineId` is derived from it, so an id that is not here fails the type check where it is typed.
//
// Adding an engine here gives it nothing else: no definition is on it until its `engines` list names it, and a
// parameter that differs per engine must say what it is on the new one (docs/06-how-to.md, "Add an engine").

export const ENGINES = [
  /** `id` is the server's (GET /api/capabilities); `name` is how the page says it, in strips, notes and toasts;
   *  `tag` how a card or a parked chip tags it, in a few letters; `short` is how the generated tables write it; `backend` is its folder in server/src/services/backends, and
   *  `prefix` the prefix of the request fields its code reads (mm3Steps): tools/make-fixtures.ts reads both;
   *  `stepSeconds` about how long one sampler step takes, for a step count's estimate ("≈ 1.3 s");
   *  `khz` its output's sample rate, `tagline` the engine panel's two lines about it (words, translated under
   *  engine.<id>.tag1 and tag2), `testing` while it is in integration testing (its card's badge). */
  { id: 'ace', name: 'ACE-Step 1.5', short: 'ace', tag: 'ACE-Step', backend: 'ace', prefix: '', stepSeconds: 0.11, khz: 48, tagline: ['Lyrics and fields', 'Repaint'] },
  /** `createContent`: what the existing createContentParams() is told about this engine. */
  { id: 'minimax-m3', name: 'MiniMax-Music3', short: 'mm3', tag: 'MiniMax', backend: 'minimax', prefix: 'mm3', stepSeconds: 0.11, khz: 44.1, tagline: ['Structured caption', '8B planner'], createContent: { mm3Mode: true } },
  { id: 'yue2', name: 'YuE2', short: 'yue2', tag: 'YuE2', backend: 'yue2', prefix: 'yue2', stepSeconds: 0.16, khz: 48, tagline: ['Lead-sheet composer', 'long songs'], createContent: { yue2Active: true } },
  { id: 'sa3', name: 'Stable Audio 3', short: 'sa3', tag: 'SA3', backend: 'sa3', prefix: 'sa3', stepSeconds: 0.05, khz: 44.1, tagline: ['Prompt only', 'exact length'] },
  /** Its jobs run as a per-job override (backendOverride), not on the active backend. */
  { id: 'heartmula', name: 'HeartMuLa', short: 'hm', tag: 'HeartMuLa', backend: 'heartmula', prefix: 'heartmula', stepSeconds: 0.11, khz: 48, tagline: ['Tags + lyrics', 'in testing'], testing: true, jobOverride: true },
] as const;

export type EngineId = (typeof ENGINES)[number]['id'];

export const ENGINE_IDS: readonly EngineId[] = ENGINES.map(e => e.id);

export const engineName = (id: EngineId): string => ENGINES.find(e => e.id === id)!.name;
/** How a card or a parked chip tags an engine. */
export const engineTag = (id: EngineId): string => ENGINES.find(e => e.id === id)!.tag;
export const engineShort = (id: EngineId): string => ENGINES.find(e => e.id === id)!.short;
/** Every engine's field prefix, for telling a backend's own request field (mm3Steps) from a shared one. */
export const FIELD_PREFIXES: readonly string[] = ENGINES.map(e => e.prefix).filter(Boolean);
