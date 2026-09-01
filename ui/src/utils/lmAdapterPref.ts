// lmAdapterPref.ts — the Lyric Studio "Use LM Adapter" toggle.
//
// Album presets can carry a planner-LM adapter alongside the DiT adapter, and
// for a long while both were applied unconditionally. In practice the DiT
// adapter on the STOCK planner has been sounding better than the pair, so the
// LM half is now opt-in and off by default.
//
// The toggle lives in the Artist page sidebar but is read from the audio queue
// and the Send-to-Create path, so the key and the reader live here rather than
// in the component (a store importing a component would be a nasty cycle).

export const USE_LM_ADAPTER_KEY = 'lireek-useLmAdapter';

/** True only when the user has explicitly ticked the box — default is OFF. */
export function useLmAdapterEnabled(): boolean {
  try {
    const raw = localStorage.getItem(USE_LM_ADAPTER_KEY);
    return raw !== null && JSON.parse(raw) === true;
  } catch {
    return false;
  }
}
