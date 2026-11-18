/** Only valid audio measurements may replace saved fields. Missing estimates
 *  are omitted so the metadata update preserves the user's existing value. */
export function measuredMetadata(raw: { bpm?: number; key?: string; scale?: string }): { bpm?: number; key_scale?: string } | null {
  const patch: { bpm?: number; key_scale?: string } = {};
  if (typeof raw.bpm === 'number' && Number.isFinite(raw.bpm) && Math.round(raw.bpm) > 0) patch.bpm = Math.round(raw.bpm);
  const key = typeof raw.key === 'string' ? raw.key.trim() : '';
  const scale = typeof raw.scale === 'string' ? raw.scale.trim().toLowerCase() : '';
  if (/^[A-G](?:#|b)?$/.test(key) && (scale === 'major' || scale === 'minor')) {
    patch.key_scale = `${key} ${scale.charAt(0).toUpperCase()}${scale.slice(1)}`;
  }
  return Object.keys(patch).length ? patch : null;
}
