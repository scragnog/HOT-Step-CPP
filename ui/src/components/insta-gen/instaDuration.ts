// instaDuration.ts — the Custom-Gen duration Insta-Gen's captureInput reads
// at submission time. Split out from InstaGenPanel.tsx so it can be
// node:test'd without dragging in that panel's full (CSS-importing) React
// tree. See InstaGenPanel.tsx's isMm3Render for the sibling MM3 check.

/** Reads the persisted Custom-Gen duration ('hs-duration', the same
 *  usePersistedState key CreatePanel writes). Insta-Gen has no duration
 *  control of its own — it rode along on globalParams before, which never
 *  carried duration, so a user's Custom-Gen duration was silently dropped
 *  and the LM's own estimate always won. -1 is the app's "auto" sentinel;
 *  MM3 forces it regardless, mirroring CreatePanel's own
 *  `mm3Mode ? -1 : duration` (MM3 has no length input). */
export function resolveInstaDuration(mm3: boolean, readStored: () => string | null = () => localStorage.getItem('hs-duration')): number {
  if (mm3) return -1;
  try {
    const raw = readStored();
    if (raw === null) return -1;
    const value = JSON.parse(raw);
    return typeof value === 'number' ? value : -1;
  } catch { return -1; }
}
