// The simulated engine: the concept's invented songs, its model catalogue (34-engine-panel.js MODELSETS) and its
// queue, which renders each job on a clock as the concept's go bar does (35-gobar.js): planning for the first fifth,
// then the steps, then decoding. No request leaves the page; the single-file build uses this one.

import type { EngineId } from '../../definitions/types.ts';
import type { CreateIntent } from '../../derive/request.ts';
import type { Engine, Job, ModelFile, Song } from './engine.ts';

const SONGS: Song[] = [
  { id: '1', title: 'Studio sketch 01', style: 'Lo-fi hip hop, mellow electric piano', duration: 171, hue: 215, engine: 'ace', createdAt: 6 },
  { id: '2', title: 'Evening texture', style: 'Ambient, warm pads, slow evolving textures', duration: 236, hue: 330, engine: 'sa3', createdAt: 5 },
  { id: '3', title: 'Untitled take', style: 'Folk rock, acoustic guitar, brushed drums', duration: 183, hue: 150, engine: 'yue2', createdAt: 4 },
  { id: '4', title: 'Late bus', style: 'Synth-pop, driving bass, bright lead', duration: 221, hue: 265, engine: 'ace', createdAt: 3 },
  { id: '5', title: 'Field recording', style: 'Post-rock, swelling guitars, big drums', duration: 300, hue: 30, engine: 'minimax-m3', createdAt: 2 },
  { id: '6', title: 'Rhodes at 2am', style: 'Soul, warm Rhodes, tight groove', duration: 198, hue: 5, engine: 'ace', createdAt: 1 },
];

const m = (label: string, gb: number): ModelFile => ({ value: label, label, gb });
/** Each engine's model buckets, as the concept lists them. */
const MODELS: { [e in EngineId]: { [bucket: string]: ModelFile[] } } = {
  ace: {
    dit: [m('XL Turbo · Q8_0', 5.3), m('XL Turbo · MXFP4', 2.7), m('XL Turbo · Q4_K_M', 3.0), m('Turbo · Q8_0', 2.5), m('Turbo · Q4_K_M', 1.4), m('SFT · Q8_0', 2.5), m('Base · Q8_0', 2.5)],
    lm: [m('4B · Q8_0', 4.5), m('4B · Q6_K', 3.4), m('4B · BF16', 8.4), m('1.7B · Q8_0', 2.0), m('0.6B · Q8_0', 0.71)],
    vae: [m('ScragVAE', 0.34), m('Standard VAE', 0.34), m('Regrind V10b', 0.34)],
  },
  'minimax-m3': {
    dit: [m('Flow DiT · Q8_0', 2.6), m('Flow DiT · Q6_K', 2.0), m('Flow DiT · Q4_K_M', 1.5), m('Flow DiT · F16', 4.9)],
    lm: [m('8B planner · Q8_0', 9.1), m('8B planner · Q6_K', 7.1), m('8B planner · F16', 17.2)],
    x: [m('Depth decoder · Q8_0', 0.69), m('Depth decoder · F16', 1.3)],
  },
  yue2: { dit: [m('YuE2 · Q8_0', 3.9), m('YuE2 · Q6_K', 3.1), m('YuE2 · BF16', 7.3)], vae: [m('Standard VAE', 0.3), m('Legacy VAE', 0.3)] },
  sa3: { dit: [m('SA3 DiT · BF16', 2.9), m('SA3 DiT · NVFP4', 0.83), m('SA3 DiT · MXFP4', 0.79)] },
  heartmula: { dit: [m('Global · Q8_0', 3.7), m('Global · F16', 6.9)], lm: [m('Local · Q8_0', 0.53), m('Local · F16', 0.99)], x: [m('Codec flow · Q8_0', 1.6)] },
};

const LISTS: { [source: string]: { value: string | number; label: string }[] } = {
  'plugins.solver': [{ value: 'euler', label: 'Euler' }, { value: 'heun', label: 'Heun' }, { value: 'dpm2m', label: 'DPM++ 2M' }, { value: 'dpm3m', label: 'DPM++ 3M' }],
  'plugins.scheduler': [{ value: 'linear', label: 'Linear' }, { value: 'cosine', label: 'Cosine' }, { value: 'beta57', label: 'Beta57' }],
  'plugins.guidance': [{ value: 'apg', label: 'APG' }, { value: 'cfg_pp', label: 'CFG++' }],
  'plugins.postprocess': [{ value: 'md_audio_tiled V2.1', label: 'MD Audio Tiled Decoder V2.1' }, { value: 'md_audio_tiled V2', label: 'MD Audio Tiled Decoder V2' }],
  languages: [{ value: 'en', label: 'English' }, { value: 'ja', label: 'Japanese' }, { value: 'zh', label: 'Chinese' }, { value: 'ko', label: 'Korean' }, { value: 'ru', label: 'Russian' }],
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
