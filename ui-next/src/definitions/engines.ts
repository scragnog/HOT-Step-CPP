// The engines, in one place. Every engine id a definition, the validator or a tool may use comes from this list;
// `EngineId` is derived from it, so an id that is not here fails the type check where it is typed.
//
// Adding an engine here gives it nothing else: no definition is on it until its `engines` list names it, and a
// parameter that differs per engine must say what it is on the new one (docs/06-how-to.md, "Add an engine").

export const ENGINES = [
  /** `id` is the server's (GET /api/capabilities); `name` is how the page says it, in strips, notes and toasts;
   *  `backend` is its folder in server/src/services/backends, which tools/make-fixtures.ts reads. */
  { id: 'ace', name: 'ACE-Step 1.5', backend: 'ace' },
  { id: 'minimax-m3', name: 'MiniMax-Music3', backend: 'minimax' },
  { id: 'yue2', name: 'YuE2', backend: 'yue2' },
  { id: 'sa3', name: 'Stable Audio 3', backend: 'sa3' },
  /** Its jobs run as a per-job override (backendOverride), not on the active backend. */
  { id: 'heartmula', name: 'HeartMuLa', backend: 'heartmula', jobOverride: true },
] as const;

export type EngineId = (typeof ENGINES)[number]['id'];

export const ENGINE_IDS: readonly EngineId[] = ENGINES.map(e => e.id);

export const engineName = (id: EngineId): string => ENGINES.find(e => e.id === id)!.name;
