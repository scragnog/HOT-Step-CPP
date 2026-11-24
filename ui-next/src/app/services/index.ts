// The one boundary between the page and the server. Every call the page makes to the server is a method of one of these
// services, reached by name through `service(name)` (register/services.ts), never by importing an implementation.
// main.tsx registers one implementation of every service, once, before anything asks for one: today the simulated
// server (services/mock/index.ts, the concept's simulation); package 10 registers the real adapters over the server's
// routes in its place, which is that one line of main.tsx. Each interface is in its own file beside this one, grouped
// as the existing frontend's API layer groups the routes (ui/src/services/api.ts).
//
// What is sent is built by the core, never here: generate.submit takes the run the core worked out (derive/generate.ts
// runOf: its intent is pageIntent, the desk in the real application's names).

import type { AdaptersService } from './adapters.ts';
import type { BuildService } from './build.ts';
import type { GenerateService } from './generate.ts';
import type { ListsService } from './lists.ts';
import type { LogsService } from './logs.ts';
import type { ModelsService } from './models.ts';
import type { RecordingService } from './recording.ts';
import type { ReferencesService } from './references.ts';
import type { ScoreService } from './score.ts';

/** Every service, by the name it is registered and asked for under. */
export interface Services {
  /** The lists server-listed choices draw from (parts.ts OPTION_SOURCES), and the plugin host's chain. */
  lists: ListsService;
  /** Each engine's model catalogue and the card's VRAM figures (GET /api/backends/models). */
  models: ModelsService;
  /** The songs the server holds and the runs it renders (/api/generate, /api/songs). */
  generate: GenerateService;
  /** The adapters installed for each engine (the server's adapter folders). */
  adapters: AdaptersService;
  /** The reference tracks the page has been given (useReferenceTracks). */
  references: ReferencesService;
  /** What the server says about itself, for the engine log (GET /api/backends, the health route). */
  logs: LogsService;
  /** A recording's length (Continue from recording). */
  recording: RecordingService;
  /** The planner's score for the lead-sheet preview. */
  score: ScoreService;
  /** The build session (Build by section: yue2CoverApi build*). */
  build: BuildService;
}
