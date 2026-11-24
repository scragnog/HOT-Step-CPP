// The engine's status (docs/09, package 11): what the server says about itself for the engine log: whether it answers
// and which engines it has registered (GET /api/backends, and the health route). The log's feature asks once when the
// page loads (features/logs.tsx); the core says the log's lines from it (src/derive/logs.ts engineLogModel). The real
// adapter can feed the log the server's own lines (the /api/logs stream) beside it. The simulated server's:
// services/mock/logs.ts.

import type { EngineStatus } from '../../derive.ts';

export interface LogsService {
  status(): Promise<EngineStatus>;
}
