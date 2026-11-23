// The simulated engine: the concept's invented songs, its model catalogue (the models service's, services/models.ts) and its
// queue, which renders each job on a clock as the concept's go bar does (35-gobar.js): planning for the first fifth,
// then the steps, then decoding. No request leaves the page; the single-file build uses this one.

import type { EngineId } from '../../definitions/types.ts';
import type { CreateIntent } from '../../derive/request.ts';
import type { Engine, Job, Song } from './engine.ts';
import { MODEL_FILES } from './models.ts';

const SONGS: Song[] = [
  { id: '1', title: 'Studio sketch 01', style: 'Lo-fi hip hop, mellow electric piano', duration: 171, hue: 215, engine: 'ace', createdAt: 6 },
  { id: '2', title: 'Evening texture', style: 'Ambient, warm pads, slow evolving textures', duration: 236, hue: 330, engine: 'sa3', createdAt: 5 },
  { id: '3', title: 'Untitled take', style: 'Folk rock, acoustic guitar, brushed drums', duration: 183, hue: 150, engine: 'yue2', createdAt: 4 },
  { id: '4', title: 'Late bus', style: 'Synth-pop, driving bass, bright lead', duration: 221, hue: 265, engine: 'ace', createdAt: 3 },
  { id: '5', title: 'Field recording', style: 'Post-rock, swelling guitars, big drums', duration: 300, hue: 30, engine: 'minimax-m3', createdAt: 2 },
  { id: '6', title: 'Rhodes at 2am', style: 'Soul, warm Rhodes, tight groove', duration: 198, hue: 5, engine: 'ace', createdAt: 1 },
];

/** Each engine's model buckets: the models service's catalogue (services/models.ts), one copy of the concept's. */
const MODELS = MODEL_FILES;

/* the concept's lists (25-module-sampler.js PLUG, 03-controls.js LANGS), so a picker draws what the concept draws */
const LISTS: { [source: string]: { value: string | number; label: string }[] } = {
  'plugins.solver': [{ value: 'euler', label: 'Euler' }, { value: 'heun', label: 'Heun' }, { value: 'sde', label: 'SDE' }, { value: 'rfsolver', label: 'RF-Solver' }, { value: 'dpm2m', label: 'DPM++ 2M' }, { value: 'dpm2m_ada', label: 'DPM++ 2M Adaptive' }, { value: 'dpm3m', label: 'DPM++ 3M' }, { value: 'unipc', label: 'UniPC' }, { value: 'unipc_p', label: 'UniPC Predictor' }, { value: 'rk4', label: 'RK4' }, { value: 'rk5', label: 'RK5' }, { value: 'gl2s', label: 'GL2S' }, { value: 'dopri5', label: 'DOPRI5' }, { value: 'dop853', label: 'DOP853' }, { value: 'stork2', label: 'STORK 2' }, { value: 'stork4', label: 'STORK 4' }, { value: 'storm_sampler_core', label: 'STORM' }, { value: 'aflops', label: 'A-FloPS' }, { value: 'aflops2', label: 'A-FloPS Midpoint' }, { value: 'jkass_fast', label: 'JKASS Fast' }, { value: 'jkass_quality', label: 'JKASS Quality' }, { value: 'md_confluence_v3', label: 'MD Confluence V3' }, { value: 'md_confluence_v4', label: 'MD Confluence V4' }, { value: 'md_confluence_v5', label: 'MD Confluence V5' }, { value: 'md_dual_time_v2', label: 'MD Dual-Time V2' }, { value: 'md_dual_time_v3', label: 'MD Dual-Time V3' }, { value: 'md_eigenflow_v1', label: 'MD Eigenflow V1' }, { value: 'md_eigenflow_v2', label: 'MD Eigenflow V2' }, { value: 'md_fractional_solver', label: 'MD Fractional Solver' }, { value: 'md_granular_collider', label: 'MD Granular Collider' }, { value: 'md_hamiltonian_v2', label: 'MD Hamiltonian V2' }, { value: 'md_hamiltonian_v3', label: 'MD Hamiltonian V3' }, { value: 'md_l_vy_flight', label: 'MD Lévy Flight' }, { value: 'md_omnirelational_v3', label: 'MD OmniRelational V3' }, { value: 'md_omnirelational_v4', label: 'MD OmniRelational V4' }, { value: 'md_p_adic_solver', label: 'MD p-Adic Solver' }, { value: 'md_pingpong_blend', label: 'MD Pingpong Blend' }, { value: 'md_pingpong_simple', label: 'MD PingPong Simple' }, { value: 'md_storm_v4', label: 'MD STORM V4' }, { value: 'md_trajectory_anchor', label: 'MD Trajectory Anchor' }, { value: 'md_trajectory_anchor_v3', label: 'MD Trajectory Anchor V3' }, { value: 'md_trajectory_anchor_v5', label: 'MD Trajectory Anchor V5' }, { value: 'md_vortex_v1', label: 'MD Vortex V1' }, { value: 'md_vortex_v2', label: 'MD Vortex V2' }, { value: 'md_wasserstein_flow_v1_0', label: 'MD Wasserstein Flow v1.0' }, { value: 'md_wasserstein_flow_v1_2_1', label: 'MD Wasserstein Flow v1.2.1' }],
  'plugins.scheduler': [{ value: 'linear', label: 'Linear' }, { value: 'cosine', label: 'Cosine' }, { value: 'beta57', label: 'Beta57' }, { value: 'beta', label: 'Beta (Custom)' }, { value: 'power', label: 'Power' }, { value: 'composite', label: 'Composite (2-Stage)' }, { value: 'bong_tangent', label: 'Tangent' }, { value: 'sgm_uniform', label: 'SGM-Uniform (Karras)' }, { value: 'ddim_uniform', label: 'DDIM Uniform' }, { value: 'linear_quadratic', label: 'Linear-Quadratic' }, { value: 'md_causal', label: 'MD Causal' }, { value: 'md_hap', label: 'MD HAP' }, { value: 'md_ht_scheduler_v3', label: 'MD HT Scheduler V3' }, { value: 'md_ht_scheduler_v5_1', label: 'MD HT Scheduler v5.1' }, { value: 'md_noise_decay_v2', label: 'MD Noise Decay V2' }, { value: 'md_psychoacoustic', label: 'MD Psychoacoustic' }, { value: 'md_qrd_scatter', label: 'MD QRD Scatter' }],
  'plugins.guidance': [{ value: 'apg', label: 'APG' }, { value: 'adg', label: 'ADG (ACE-Step)' }, { value: 'cfg_pp', label: 'CFG++' }, { value: 'cfg_mp', label: 'CFG-MP' }, { value: 'cfg_zero_star', label: 'CFG-Zero⋆' }, { value: 'dynamic_cfg', label: 'Dynamic CFG' }, { value: 'rescaled_cfg', label: 'Rescaled CFG' }, { value: 'smc_cfg', label: 'SMC-CFG' }, { value: 'md_adg', label: 'MD ADG' }, { value: 'md_apg_v4', label: 'MD APG V4' }, { value: 'md_apt', label: 'MD APT' }, { value: 'md_bucket_brigade', label: 'MD Bucket Brigade' }, { value: 'md_clarity_v1', label: 'MD Clarity V1' }, { value: 'md_custom_noise_v3', label: 'MD Custom Noise V3' }, { value: 'md_firefly_ringdown_v2', label: 'MD Firefly Ringdown V2' }, { value: 'md_liquid_pro', label: 'MD Liquid PRO' }, { value: 'md_phase_inversion', label: 'MD Phase Inversion' }, { value: 'md_storm_guidance_v2', label: 'MD STORM Guidance V2' }],
  'plugins.postprocess': [{ value: 'md_audio_tiled V2.1', label: 'MD Audio Tiled Decoder V2.1' }, { value: 'md_audio_tiled V2', label: 'MD Audio Tiled Decoder V2' }],
  languages: [{ value: 'en', label: 'English' }, { value: 'zh', label: 'Chinese' }, { value: 'ja', label: 'Japanese' }, { value: 'ko', label: 'Korean' }, { value: 'es', label: 'Spanish' }, { value: 'fr', label: 'French' }, { value: 'de', label: 'German' }, { value: 'it', label: 'Italian' }, { value: 'pt', label: 'Portuguese' }, { value: 'ru', label: 'Russian' }, { value: 'ar', label: 'Arabic' }, { value: 'hi', label: 'Hindi' }, { value: 'tr', label: 'Turkish' }, { value: 'vi', label: 'Vietnamese' }, { value: 'th', label: 'Thai' }, { value: 'sv', label: 'Swedish' }, { value: 'pl', label: 'Polish' }, { value: 'nl', label: 'Dutch' }, { value: 'yue', label: 'Cantonese' }, { value: 'az', label: 'Azerbaijani' }, { value: 'bg', label: 'Bulgarian' }, { value: 'bn', label: 'Bengali' }, { value: 'ca', label: 'Catalan' }, { value: 'cs', label: 'Czech' }, { value: 'da', label: 'Danish' }, { value: 'el', label: 'Greek' }, { value: 'fa', label: 'Persian' }, { value: 'fi', label: 'Finnish' }, { value: 'he', label: 'Hebrew' }, { value: 'hr', label: 'Croatian' }, { value: 'ht', label: 'Haitian Creole' }, { value: 'hu', label: 'Hungarian' }, { value: 'id', label: 'Indonesian' }, { value: 'is', label: 'Icelandic' }, { value: 'kr', label: 'Kanuri' }, { value: 'la', label: 'Latin' }, { value: 'lt', label: 'Lithuanian' }, { value: 'ms', label: 'Malay' }, { value: 'ne', label: 'Nepali' }, { value: 'no', label: 'Norwegian' }, { value: 'pa', label: 'Punjabi' }, { value: 'ro', label: 'Romanian' }, { value: 'sa', label: 'Sanskrit' }, { value: 'sk', label: 'Slovak' }, { value: 'sr', label: 'Serbian' }, { value: 'sw', label: 'Swahili' }, { value: 'ta', label: 'Tamil' }, { value: 'te', label: 'Telugu' }, { value: 'tl', label: 'Tagalog' }, { value: 'uk', label: 'Ukrainian' }, { value: 'ur', label: 'Urdu' }],
};

/** How long a run takes in the simulation, and what it says along the way (35-gobar.js stage()). */
const RUN_MS = 36000;
function stageOf(f: number, e: EngineId, steps = 12): string {
  if (f < 0.2) return e === 'yue2' ? 'Composing the lead sheet' : e === 'sa3' ? 'Encoding the prompt' : e === 'heartmula' ? 'Writing' : 'Planning';
  if (f < 0.8) return `${e === 'yue2' ? 'NAR step ' : e === 'minimax-m3' ? 'Flow step ' : 'Step '}${Math.max(1, Math.round((f - 0.2) / 0.6 * steps))} of ${steps}`;
  return 'Decoding audio';
}

export function mockEngine(now: () => number = () => Date.now()): Engine {
  const songs = [...SONGS];
  const jobs = new Map<string, { job: Job; at: number; intent: CreateIntent }>();
  const order: string[] = [];
  let next = 100;
  /* the queue renders one job at a time: a job starts when the one before it ends */
  const startOf = (id: string): number => {
    const i = order.indexOf(id); let t = jobs.get(order[0]!)!.at;
    for (let j = 0; j <= i; j++) { const x = jobs.get(order[j]!)!; t = Math.max(t, x.at); if (j < i && x.job.status !== 'cancelled') t += RUN_MS; }
    return t;
  };
  const tick = (id: string): Job => {
    const x = jobs.get(id)!; if (x.job.status === 'cancelled' || x.job.status === 'succeeded') return x.job;
    const f = Math.min(1, Math.max(0, (now() - startOf(id)) / RUN_MS));
    x.job = { ...x.job, progress: f, status: f >= 1 ? 'succeeded' : f > 0 ? 'running' : 'queued', stage: f >= 1 ? 'Done' : f > 0 ? stageOf(f, x.job.engine) : 'Queued' };
    if (f >= 1 && !x.job.songIds.length) {
      const id2 = String(next++), p = x.intent.params as { caption?: string; duration?: number };
      songs.unshift({ id: id2, title: `Take ${id2}`, style: String(p.caption ?? ''), duration: typeof p.duration === 'number' && p.duration > 0 ? p.duration : 180, hue: (next * 47) % 360, engine: x.job.engine, createdAt: now() });
      x.job = { ...x.job, songIds: [id2] };
    }
    return x.job;
  };
  return {
    songs: {
      list: async () => { order.forEach(tick); return { songs: [...songs] }; },
      delete: async id => { const i = songs.findIndex(s => s.id === id); if (i >= 0) songs.splice(i, 1); return { success: i >= 0 }; },
    },
    generate: {
      preview: async intent => ({ params: { ...intent.params } }),
      submit: async intent => {
        const id = `job-${next++}`, engine = (intent.params.backendOverride ?? intent.params.backend ?? 'ace') as EngineId;
        jobs.set(id, { job: { id, status: 'queued', progress: 0, stage: 'Queued', engine, songIds: [] }, at: now(), intent }); order.push(id);
        return { jobId: id };
      },
      status: async id => { if (!jobs.has(id)) throw new Error(`no job ${id}`); return tick(id); },
      cancel: async id => { const x = jobs.get(id); if (!x || x.job.status === 'succeeded') return { success: false }; x.job = { ...x.job, status: 'cancelled', stage: 'Cancelled' }; return { success: true }; },
      cancelAll: async () => { let n = 0; for (const id of order) { const x = jobs.get(id)!; tick(id); if (x.job.status === 'queued' || x.job.status === 'running') { x.job = { ...x.job, status: 'cancelled', stage: 'Cancelled' }; n++; } } return { cancelled: n }; },
      queue: async () => { const live = order.map(tick).filter(j => j.status === 'queued' || j.status === 'running'); return { depth: live.length, running: live.some(j => j.status === 'running') }; },
    },
    models: {
      list: async e => ({ buckets: MODELS[e], selection: Object.fromEntries(Object.entries(MODELS[e]).map(([b, files]) => [`${e}.${b}`, files[0]!.value])) }),
    },
    lists: async () => LISTS,
    vst: { chain: async () => ({ plugins: [] }) },
  };
}
