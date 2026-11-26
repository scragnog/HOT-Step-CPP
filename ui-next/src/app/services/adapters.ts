// The adapters installed for each engine, by what they were trained for (B-02): an ACE-Step DiT adapter is not a
// planner LM adapter, so each slot offers only the files of its own kind; YuE2's two slots are the AR composer and the
// NAR renderer. Server-owned: the page is told them (features/adapters.tsx), as the existing frontend reads its adapter
// list from the server. The simulated server's: services/mock/adapters.ts.

import type { EngineId } from '../../definitions/types.ts';

export interface AdapterFile { file: string; type: 'LoRA' | 'LoKr' | 'DoRA' }
export interface AdaptersService {
  /** Each slot's installed files on an engine, in slot order. */
  slots(engine: EngineId): Promise<AdapterFile[][]>;
  /** The planner LM adapters (ACE-Step's planner adapter). */
  planner(): Promise<AdapterFile[]>;
}
