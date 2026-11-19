/**
 * format.ts — the display formatting rules the studios share.
 *
 * Durations were re-derived in a dozen components, each with its own idea of
 * rounding, padding and what happens past an hour. One implementation keeps a
 * song's length reading the same in the library, the playlist, the mixer and
 * the lyric editor.
 */

/** `m:ss`, or `h:mm:ss` once the duration reaches an hour. Negative, NaN and
 *  Infinity collapse to `0:00`, which is what an unloaded duration should show. */
export function formatClock(seconds: number, opts: { round?: boolean } = {}): string {
  const safe = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
  const total = opts.round ? Math.round(safe) : Math.floor(safe);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const ss = String(s).padStart(2, '0');
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`;
}

const BYTE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB'] as const;

/** Human-readable byte size: `512 KB`, `1.5 MB`, `2.3 GB`. One decimal below ten
 *  units, where it carries information; none at or above a hundred, so a 150 MB
 *  file does not read `150.0 MB`. `zero` replaces the answer for a missing or
 *  non-positive size when a caller wants a dash or a blank there. */
export function formatBytes(bytes: number | null | undefined, opts: { zero?: string } = {}): string {
  if (bytes == null || !Number.isFinite(bytes) || bytes <= 0) return opts.zero ?? '0 B';
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < BYTE_UNITS.length - 1) { value /= 1024; unit += 1; }
  const decimals = unit === 0 ? 0 : value < 10 ? 1 : value < 100 ? 1 : 0;
  return `${value.toFixed(decimals)} ${BYTE_UNITS[unit]}`;
}

/** A transfer rate, as bytes per second. */
export function formatBytesPerSecond(bytesPerSecond: number): string {
  return `${formatBytes(bytesPerSecond)}/s`;
}

/** `m:ss.d` — a tenth is the resolution that matters when placing a syllable or
 *  dragging an edit handle. */
export function formatClockFine(seconds: number): string {
  const safe = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
  const m = Math.floor(safe / 60);
  const s = Math.floor(safe % 60);
  const tenths = Math.floor((safe * 10) % 10);
  return `${m}:${String(s).padStart(2, '0')}.${tenths}`;
}

/** A 0..1 ratio as a whole-percent label: `42%`. A non-finite or out-of-range
 *  ratio clamps, so a broken progress value cannot render `NaN%` or `-3%`. */
export function formatPercent(ratio: number, decimals = 0): string {
  if (!Number.isFinite(ratio)) return '0%';
  const pct = Math.min(100, Math.max(0, ratio * 100));
  return decimals === 0 ? `${Math.round(pct)}%` : `${pct.toFixed(decimals)}%`;
}

/**
 * A loss or metric readout: four decimals, em dash when there is no number yet.
 *
 * Training surfaces had this ternary in fourteen places, each with its own
 * truthiness test — so a loss that reached exactly 0 showed as missing.
 */
export function formatLoss(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : '—';
}

/**
 * A number for display at a fixed precision: one decimal for gains, ratios and
 * rates, two where a knob is finer. A missing or non-finite value becomes an em
 * dash instead of `NaN`, which is what the call sites used to print.
 */
export function formatDecimal(value: number | null | undefined, digits = 1): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
}

/**
 * A checkpoint or run folder's size: GB above a gigabyte, whole MB below it.
 * Three cards each carried this identical function, which is why one of them
 * still rounded a 1.4 GB adapter to "1433 MB".
 */
export function formatTrainingSize(bytes: number | null | undefined): string {
  if (typeof bytes !== 'number' || !Number.isFinite(bytes) || bytes <= 0) return '—';
  return bytes >= 1073741824
    ? `${formatDecimal(bytes / 1073741824, 1)} GB`
    : `${Math.round(bytes / 1048576)} MB`;
}
