// yue2Tokenizer.ts — which real-audio tokenizer turns audio into YuE2 codes.
//
// v9 (Mothersuperior's head, the default) or ManyOtherFunctions'
// WannabeTokenizer V1/V2. Stored on the server (preferences family
// `yue2-tokenizer`): `training` is what the tokenize stage uses for a dataset's
// codes, `cover` what a cover's Follow-source guide uses. Adapters trained on
// WannabeTokenizer codes train and render without the v9 companion decoder;
// the engine handles that from the adapter itself.

import { useQuery } from '@tanstack/react-query';
import type { Yue2TokenizerBody, Yue2TokenizerId } from '../../../../server/src/contracts/preferences';
import { queriedSettingsApi, settingOptions } from '../../queries/shell-queue/preferences';

export type { Yue2TokenizerId };

const DEFAULT: Yue2TokenizerBody = { training: 'v9', cover: 'v9' };

export const YUE2_TOKENIZER_LABELS: Record<Yue2TokenizerId, string> = {
  v9: 'v9 (Mothersuperior)',
  'wannabe-v1': 'WannabeTokenizer V1',
  'wannabe-v2': 'WannabeTokenizer V2',
};

export function useYue2Tokenizer(): { choice: Yue2TokenizerBody; set: (next: Partial<Yue2TokenizerBody>) => Promise<void> } {
  const { data } = useQuery(settingOptions<Yue2TokenizerBody>('yue2-tokenizer'));
  const doc = data?.document;
  const choice = doc?.body ?? DEFAULT;
  const set = async (next: Partial<Yue2TokenizerBody>) => {
    await queriedSettingsApi.upsert('yue2-tokenizer', doc?.revision, { ...choice, ...next });
  };
  return { choice, set };
}
