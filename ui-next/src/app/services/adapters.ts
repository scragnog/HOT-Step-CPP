// The adapters installed for each engine, by what they were trained for (B-02): an ACE-Step DiT adapter is not a
// planner LM adapter, so each slot offers only the files of its own kind; YuE2's two slots are the AR composer and the
// NAR renderer. Server-owned: the page is told them, as the existing frontend reads its adapter list from the server.
//
// This is the simulated server's list, the concept's (25-module-adapters.js ADAPTER_FILES and ad.planner's options);
// the real adapter, over the server's adapter folders, is package 10's.

import type { EngineId } from '../../definitions/types.ts';
import { defineService } from '../register/services.ts';

export interface AdapterFile { file: string; type: 'LoRA' | 'LoKr' | 'DoRA' }
export interface AdaptersService {
  /** Each slot's installed files on an engine, in slot order. */
  slots(engine: EngineId): Promise<AdapterFile[][]>;
  /** The planner LM adapters (ACE-Step's planner adapter). */
  planner(): Promise<AdapterFile[]>;
}

const ACE: AdapterFile[] = [{ file: 'tape-lofi-v2', type: 'LoRA' }, { file: 'folk-voice', type: 'LoKr' }, { file: 'synthwave-80s', type: 'DoRA' }, { file: 'my-band-ep', type: 'LoRA' }, { file: 'choir-room', type: 'LoKr' }, { file: 'grit-mastering', type: 'DoRA' }];
const FILES: { [e in EngineId]?: AdapterFile[][] } = {
  ace: [ACE, ACE, ACE],
  'minimax-m3': [[{ file: 'folk-voice-planner', type: 'LoRA' }, { file: 'my-band-ep-mm3', type: 'LoRA' }, { file: 'city-pop-plans', type: 'LoKr' }]],
  yue2: [[{ file: 'folk-voice-ar', type: 'LoRA' }, { file: 'my-band-ep-ar', type: 'LoRA' }, { file: 'cantopop-ar', type: 'LoKr' }], [{ file: 'tape-warmth-nar', type: 'LoRA' }, { file: 'bright-room-nar', type: 'LoKr' }]],
  heartmula: [[{ file: 'heart-folk-lm', type: 'LoRA' }, { file: 'my-band-ep-hm', type: 'LoRA' }]],
};
const PLANNER: AdapterFile[] = [{ file: 'folk-voice-planner', type: 'LoRA' }, { file: 'my-band-ep-lm', type: 'LoKr' }];

export const adapters = defineService<AdaptersService>('adapters', {
  slots: async e => FILES[e] ?? [],
  planner: async () => PLANNER,
});
