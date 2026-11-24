// The lists server-listed choices draw from (parts.ts OPTION_SOURCES: the plugins, the languages, the caption datasets,
// MiniMax-Music3's saved plans), with their words, and the plugin host's VST3 chain (the desk fact `vstPlugins`). Told
// to the desk once when the page loads (App.tsx). The simulated server's: services/mock/lists.ts.

import type { Json } from '../../definitions/types.ts';

export interface ListsService {
  lists(): Promise<{ [source: string]: { value: string | number; label: string; data?: Json }[] }>;
  /** The VST3 chain: the plugins it holds. */
  vstChain(): Promise<{ plugins: { name: string; enabled: boolean }[] }>;
}
