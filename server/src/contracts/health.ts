// contracts/health.ts — wire shape for GET /api/health (server/src/routes/health.ts).
// GET /presence is a raw SSE stream with no JSON frames (a client keepalive
// signal, not a data feed) and has no response type here.

export interface HealthResponse {
  status: string;
  version: string;
  commit: string;
  dirty: boolean;
  /** ace-server.exe's mtime, or null if the file can't be stat'd. */
  engineBuiltAt: string | null;
  aceServer: {
    status: string;
    url: string;
    version: string;
  };
  server: {
    port: number;
    /** Seconds since this Node process started. */
    uptime: number;
  };
  /** Open SSE-holding browser tabs (health.ts's countPresence). */
  clients: number;
  engine: {
    ready: boolean;
    bootStatus: string;
  };
}
