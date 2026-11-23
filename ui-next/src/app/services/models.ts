// The models service (docs/09, package 5): what the server says about the model files and the card, beside the page's
// Engine (services/engine.ts). Each engine's model catalogue (GET /api/backends/models?backend=<id>: each bucket's
// installed files with their size on disk, and the file selected in each), and the VRAM figures the meter adds up
// (the card, and what each engine loads besides its model files and costs per take). The engine panel's feature asks
// for both and tells the page (features/engine-panel.tsx); the core works out everything shown from them
// (src/derive/vram.ts).
//
// The mock below is the concept's catalogue and figures (34-engine-panel.js MODELSETS, FIXEDGB; 35-gobar.js CARD_GB,
// TAKE_GB), with the buckets named as the definitions name them (src/definitions/modules/models.ts). It registers
// itself here; the real adapter (package 10) registers the same interface over the server's routes instead.

import type { EngineId } from '../../definitions/types.ts';
import type { VramFigures } from '../../derive.ts';
import { defineService } from '../register/services.ts';
import type { ModelFile } from './engine.ts';

export interface ModelCatalogue {
  /** Each bucket's installed files, in the server's order, each with its size in GB. */
  buckets: { [bucket: string]: ModelFile[] };
  /** The file the server has selected in each bucket: what a model choice left at "" stands for. */
  selection: { [bucket: string]: string };
}
export interface ModelsService {
  catalogue(engine: EngineId): Promise<ModelCatalogue>;
  vram(): Promise<VramFigures>;
}

const m = (label: string, gb: number): ModelFile => ({ value: label, label, gb });
const quants = (name: string, q: [string, number][]) => q.map(([x, gb]) => m(`${name} · ${x}`, gb));

/** The concept's model files per engine, with their size on disk in GB (docs/user/models.md, as the concept lists them). */
export const MODEL_FILES: { [e in EngineId]: { [bucket: string]: ModelFile[] } } = {
  ace: {
    dit: [...quants('XL Turbo', [['Q8_0', 5.3], ['MXFP4', 2.7], ['Q4_K_M', 3.0]]), ...quants('Turbo', [['Q8_0', 2.5], ['Q4_K_M', 1.4]]), m('Turbo Continuous · Q8_0', 2.5), m('SFT · Q8_0', 2.5), m('Base · Q8_0', 2.5), m('SFT+Turbo 50% · Q8_0', 2.5), m('XL Merge SFT+Turbo λ0.3 · Q8_0', 5.3), m('XL Merge Base+Turbo λ0.5 · Q8_0', 5.3)],
    lm: [...quants('4B', [['Q8_0', 4.5], ['Q6_K', 3.4], ['Q5_K_M', 3.0], ['BF16', 8.4]]), ...quants('1.7B', [['Q8_0', 2.0], ['BF16', 3.7]]), ...quants('0.6B', [['Q8_0', 0.71], ['BF16', 1.3]])],
    vae: [m('ScragVAE', 0.34), m('Standard VAE', 0.34), m('Regrind V10b', 0.34), m('Regrind V10b Blend50', 0.34), m('Regrind V9b', 0.34)],
  },
  'minimax-m3': {
    dit: quants('Flow DiT', [['Q8_0', 2.6], ['Q6_K', 2.0], ['Q4_K_M', 1.5], ['NVFP4', 1.4], ['F16', 4.9]]),
    lm: quants('8B planner', [['Q8_0', 9.1], ['Q6_K', 7.1], ['Q4_K_M-imat', 5.5], ['NVFP4', 5.7], ['F16', 17.2]]),
    depth: quants('Depth decoder', [['Q8_0', 0.69], ['Q4_K_M', 0.40], ['F16', 1.3]]),
  },
  yue2: {
    lm: [...quants('YuE2', [['Q8_0', 3.9], ['Q6_K', 3.1], ['Q4_K_M-imat', 2.4], ['NVFP4', 2.5], ['BF16', 7.3]]), m('YuE2-RS 4.0x0.7 · Q8_0', 3.9), m('YuE2-RS Tsunami Wave 2.0x0.75 · Q8_0', 3.9)],
    vae: [m('Standard VAE', 0.3), m('Legacy VAE', 0.3), m('rs-x3 · Ruach Studio', 1.1)],
  },
  sa3: { dit: quants('SA3 DiT', [['BF16', 2.9], ['NVFP4', 0.83], ['MXFP4', 0.79], ['IQ4_XS-imat', 0.79]]) },
  heartmula: {
    variant: [m('HeartMuLa 3B', 0), m('HeartMuLa 3B happy-new-year', 0), m('HeartMuLa 3B RL', 0)],
    global: quants('Global', [['Q8_0', 3.7], ['Q6_K', 2.8], ['Q4_K_M', 2.2], ['NVFP4', 2.2], ['F16', 6.9]]),
    local: quants('Local', [['Q8_0', 0.53], ['Q4_K_M', 0.32], ['F16', 0.99]]),
    flow: quants('Codec flow', [['Q8_0', 1.6], ['F16', 3.0]]),
  },
};

/** The concept's figures: a 31.8 GB card; what each engine always loads; an adapter's 0.15 GB and how many each engine
 *  loads; ACE-Step's planner track per take while it plans (and plans in passes when they do not fit), MiniMax-Music3's
 *  3.4 GB per take after the first (refused before loading when they do not fit), YuE2's 1.8 GB. */
const FIGURES: VramFigures = {
  card: 31.8,
  engines: {
    ace: { fixed: [{ name: 'Text encoder', gb: 1.2 }, { name: 'Working memory', gb: 2.0 }], adapters: { gb: 0.15, most: 3 }, take: { gb: 2.3, when: 'lm.on', first: true }, over: 'passes' },
    'minimax-m3': { fixed: [{ name: 'Encoders and vocoder', gb: 0.8 }, { name: 'Working memory', gb: 3.0 }], adapters: { gb: 0.15, most: 1 }, take: { gb: 3.4, first: false }, over: 'refuse' },
    yue2: { fixed: [{ name: 'Working memory', gb: 1.5 }], adapters: { gb: 0.15, most: 2 }, take: { gb: 1.8, first: false }, over: 'tight' },
    sa3: { fixed: [{ name: 'T5Gemma and SAME', gb: 1.6 }, { name: 'Working memory', gb: 1.2 }], take: { gb: 0, first: false }, over: 'tight' },
    heartmula: { fixed: [{ name: 'Scalar decoder and tokenizer', gb: 0.3 }, { name: 'Working memory', gb: 1.5 }], adapters: { gb: 0.15, most: 1 }, take: { gb: 0, first: false }, over: 'tight' },
  },
};

/** The simulated server: every engine's files, the first of each bucket selected. */
export function mockModels(): ModelsService {
  return {
    catalogue: async e => ({ buckets: MODEL_FILES[e], selection: Object.fromEntries(Object.entries(MODEL_FILES[e]).map(([b, files]) => [b, files[0]!.value])) }),
    vram: async () => FIGURES,
  };
}

defineService<ModelsService>('models', mockModels());
