// The one boundary between the page and "the engine": the server's API, as the existing frontend's services
// (ui/src/services/api.ts) group it. The page talks to an Engine and nothing else; main.tsx chooses which one. The mock
// (mockEngine.ts) is the concept's simulation (invented songs, a queue that renders on a clock, the model catalogue);
// the real one is a thin adapter over ui/src/services/api.ts (generateApi, songApi, modelApi, vstApi) and is a later
// work package (docs/09-architecture.md, "The service boundary").
//
// What is sent is built by the core, never here: generate.preview and generate.submit take the Create intent that
// createIntent() (src/derive/request.ts) builds from the desk.

import type { CreateIntent } from '../../derive/request.ts';
import type { EngineId, Json } from '../../definitions/types.ts';

/** A song as the Latest list and the player show it (ui/src/types Song, narrowed to what the page reads). */
export interface Song {
  id: string;
  title: string;
  style: string;
  /** Seconds. */
  duration: number;
  /** The colour of its art, as a hue. */
  hue: number;
  engine: EngineId;
  createdAt: number;
  error?: string;
}

/** A generation job as GET /api/generate/status/:id reports it. */
export interface Job {
  id: string;
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';
  /** 0 to 1. */
  progress: number;
  /** What it is doing, in the server's words ("Step 4 of 12"). */
  stage: string;
  engine: EngineId;
  songIds: string[];
}

/** One model file of a bucket (GET /api/backends/models). */
export interface ModelFile { value: string; label: string; gb: number }

export interface Engine {
  songs: {
    list(): Promise<{ songs: Song[] }>;
    delete(id: string): Promise<{ success: boolean }>;
  };
  generate: {
    /** POST /api/resolve/preview: what the server would run for this intent. */
    preview(intent: CreateIntent): Promise<{ params: Record<string, unknown> }>;
    /** Submits the intent; the job renders on the server. */
    submit(intent: CreateIntent): Promise<{ jobId: string }>;
    status(jobId: string): Promise<Job>;
    cancel(jobId: string): Promise<{ success: boolean }>;
    cancelAll(): Promise<{ cancelled: number }>;
    queue(): Promise<{ depth: number; running: boolean }>;
  };
  models: {
    /** Each bucket's installed files and the one the server has selected (`<engine>.<bucket>`). */
    list(engine: EngineId): Promise<{ buckets: { [bucket: string]: ModelFile[] }; selection: { [bucket: string]: string } }>;
  };
  /** The lists server-listed choices draw from (parts.ts OPTION_SOURCES), with their words. */
  lists(): Promise<{ [source: string]: { value: string | number; label: string; data?: Json }[] }>;
  vst: {
    /** The VST3 chain: how many plugins it holds (the desk fact `vstPlugins`). */
    chain(): Promise<{ plugins: { name: string; enabled: boolean }[] }>;
  };
}
