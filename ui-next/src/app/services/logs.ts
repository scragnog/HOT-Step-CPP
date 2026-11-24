// The engine's status (docs/09, package 11): what the server says about itself for the engine log, beside the page's
// Engine (services/engine.ts): whether it answers and which engines it has registered (GET /api/backends, and the
// health route). The log's feature asks once when the page loads (features/logs.tsx); the core says the log's lines
// from it (src/derive/logs.ts engineLogModel).
//
// The mock below is the concept's simulation: the server up, every engine registered. It registers itself here; the
// real adapter (package 10) registers the same interface over the server's routes, and can feed the log the server's
// own lines (the /api/logs stream) beside it.

import { ENGINE_IDS } from '../../definitions/engines.ts';
import type { EngineStatus } from '../../derive.ts';
import { defineService } from '../register/services.ts';

export interface LogsService {
  status(): Promise<EngineStatus>;
}

defineService<LogsService>('logs', {
  status: async () => ({ up: true, engines: [...ENGINE_IDS] }),
});
