// contracts/health.ts — wire shape for GET /api/health (server/src/routes/health.ts).
// GET /presence is a raw SSE stream with no JSON frames (a client keepalive
// signal, not a data feed) and has no response type here.

export interface HealthResponse {
  status: string;
  version: string;
  commit: string;
  dirty: boolean;
  /** Package identity from build-info.json (config.ts BUILD_INFO). Without one,
   *  sha falls back to the checkout's HEAD and the rest is null: never guessed. */
  build: {
    tag: string | null;
    sha: string | null;
    /** Exact asset variant, e.g. "cuda12-volta"; engine/.variant stays coarse. */
    variant: string | null;
    channel: 'release' | 'master' | null;
  };
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
    /** True when the engine cannot start without help; bootStatus gives the reason. */
    failed: boolean;
    bootStatus: string;
    /** True while the engine is down only because no model is installed yet. */
    noModels: boolean;
    /** Compute backend the engine reported loading on; null until its first model load. */
    backend: 'cuda' | 'vulkan' | 'metal' | 'hip' | 'cpu' | null;
    /** Raw ggml device name (e.g. "CUDA0"), or null. */
    device: string | null;
  };
}
