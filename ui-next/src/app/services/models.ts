// The models service (docs/09, package 5): what the server says about the model files and the card. Each engine's
// model catalogue (GET /api/backends/models?backend=<id>: each bucket's installed files with their size on disk, and the
// file selected in each), and the VRAM figures the meter adds up (the card, and what each engine loads besides its model
// files and costs per take). The engine panel's feature asks for both and tells the page (features/engine-panel.tsx);
// the core works out everything shown from them (src/derive/vram.ts). The simulated server's: services/mock/models.ts.

import type { EngineId } from '../../definitions/types.ts';
import type { VramFigures } from '../../derive.ts';

/** One model file of a bucket. */
export interface ModelFile { value: string; label: string; gb: number }
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
