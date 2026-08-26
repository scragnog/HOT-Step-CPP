// backends/ace/index.ts — ACE-Step 1.5 backend
//
// Phase 1 scaffolding: implements EngineBackend by DELEGATING to the existing
// aceClient / aceEngineProcess / engineState modules — no logic moves, no
// behavior changes. This is the proof-of-shape backend; translateParams and
// the LM-echo generation path stay where they are (routes/generate.ts) until
// a later phase migrates them behind this interface (plan §4.1, §5 Phase 1).

import { aceClient } from '../../aceClient.js';
import { config } from '../../../config.js';
import { isEngineSuspended, restartAceServer, stopAceServer } from '../../aceEngineProcess.js';
import { engineReady } from '../../../engineState.js';
import type {
  EngineBackend,
  BackendCapabilities,
  BackendModels,
  BackendLifecycleStatus,
} from '../types.js';

function status(): BackendLifecycleStatus {
  if (isEngineSuspended()) return 'suspended';
  if (engineReady) return 'ready';
  return 'down';
}

async function capabilities(): Promise<BackendCapabilities> {
  const up = engineReady && !isEngineSuspended();
  return {
    backend: 'ace',
    up,
    core: {
      // The LM's constrained-decoding FSM builds a duration prefix tree over
      // 10..600 s (metadata-fsm.h) and clamps anything above 600, so 600 is
      // the engine's real ceiling. 240 was a UI number with nothing behind it
      // (issue #101). The slider max is capability-driven, so this value IS
      // the user-visible limit. Quality past a few minutes is the user's call.
      duration: { max: 600 },
      bpm: true,
      keyscale: true,
      negativePrompt: true,
      batch: { max: 8 },
      seed: true,
    },
    features: {
      models: true,
      lm: true,
      plugins: true,
      // ACE is where the Lua sampler plugins came from; its own Generation
      // dropdown already renders the controls, so this flag changes nothing
      // here. It exists so MM3 can claim the capability without also claiming
      // ACE's dropdown (see the field docs in ../types.ts).
      samplerPlugins: true,
      adapters: true,
      // ACE's planner-LM LoRAs live inside its own LM/Thinking cluster
      // (lmAdapter/lmAdapterScale globals), not the generic picker this flag
      // gates — so false here means "not that UI", not "no LM adapters".
      lmAdapters: false,
      postProcess: true,
      stableStep: true,
      whisper: true,
      lyricTimestamps: true,
      cover: true,
      repaint: true,
      lego: true,
      extract: true,
      streaming: true,
      training: true,
      midi: true,
      stems: true,
      understand: true,
      conceptSteering: true,
    },
    // ACE's solver/scheduler/guidance/adapter knobs are still surfaced via
    // the existing /api/plugins registry, not through this manifest yet —
    // Phase 1 leaves this empty rather than half-duplicating that route.
    extensions: [],
  };
}

async function models(): Promise<BackendModels> {
  try {
    const props = await aceClient.props();
    return {
      buckets: {
        lm: props.models.lm,
        dit: props.models.dit,
        vae: props.models.vae,
        embedding: props.models.embedding,
      },
      adapters: props.adapters,
      lmAdapters: props.lm_adapters ?? [],
      defaults: props.default,
    };
  } catch {
    // ace-server unreachable — same degrade-empty contract as routes/models.ts
    return {
      buckets: { lm: [], dit: [], vae: [], embedding: [] },
      adapters: [],
      lmAdapters: [],
      defaults: {},
    };
  }
}

export const aceBackend: EngineBackend = {
  id: 'ace',
  displayName: 'ACE-Step 1.5',
  resourcePool: 'gpu',
  lifecycle: {
    // Phase 1: the engine process is bootstrapped and managed by
    // index.ts/aceEngineProcess.ts directly (respawn, crash budget,
    // suspension). Routing start/stop through here is a later-phase move —
    // for now these delegate to the same restart/stop primitives so the
    // interface shape is exercised without duplicating lifecycle ownership.
    async start() {
      await restartAceServer();
    },
    async stop() {
      await stopAceServer('Stopped via backend abstraction');
    },
    status,
  },
  capabilities,
  models,
  /** Model-residency arbitration (plan §4.4). Evicts every resident, not
   *  in-use ACE module so the other family isn't fighting it for VRAM. Uses
   *  the same GET /models/loaded + POST /models/unload pair the VRAM
   *  indicator's manual unload uses (routes/logs.ts:120-145) — one label per
   *  call is all the engine's endpoint accepts. Best-effort throughout: this
   *  runs fire-and-forget behind a backend switch and must never throw. */
  async releaseVram() {
    let loaded: Array<{ label: string; in_use?: boolean }> = [];
    try {
      const res = await fetch(`${config.aceServer.url}/models/loaded`, {
        signal: AbortSignal.timeout(3000),
      });
      if (!res.ok) return;
      const body = await res.json() as { loaded?: Array<{ label: string; in_use?: boolean }> };
      loaded = Array.isArray(body.loaded) ? body.loaded : [];
    } catch {
      return;   // engine unreachable — nothing resident that we can free anyway
    }

    const freed: string[] = [];
    for (const m of loaded) {
      if (!m?.label || m.in_use) continue;   // in-use modules are skipped engine-side too
      if (await aceClient.unloadLabel(m.label)) freed.push(m.label);
    }
    if (freed.length) {
      console.log(`[Backends] ACE-Step residency released: ${freed.join(', ')}`);
    }
  },
};
