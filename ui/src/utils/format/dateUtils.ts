// dateUtils.ts — shared display formatting for server timestamps.
//
// The server sends ISO strings (SQLite's datetime('now') shape has no 'Z'
// suffix) and epoch milliseconds, and each list/panel used to format them with
// its own local helper — several of them byte-identical copies of the same
// options object. One named function per distinct display format; all take the
// raw server value and return `fallback` (default '') when it does not parse,
// so a bad timestamp never renders as "Invalid Date".

export type RawDate = string | number | Date | null | undefined;

/** Parse the server's timestamp shapes. A space-separated string without T/Z
 *  is SQLite's UTC-without-suffix form and is read as UTC; ISO strings with
 *  T/Z and epoch milliseconds pass through as-is. */
export function toDisplayDate(value: RawDate): Date | null {
  if (value === null || value === undefined || value === '') return null;
  const input = typeof value === 'string' && !value.includes('T') && !value.includes('Z')
    ? value.replace(' ', 'T') + 'Z'
    : value;
  const d = new Date(input);
  return Number.isNaN(d.getTime()) ? null : d;
}

/** "Oct 21" — seed list, song list (older than a week). */
export function formatDay(value: RawDate, fallback = ''): string {
  const d = toDisplayDate(value);
  return d ? d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) : fallback;
}

/** "Oct 21, 2026" — dataset cards and asset chips. */
export function formatDayYear(value: RawDate, fallback = ''): string {
  const d = toDisplayDate(value);
  return d ? d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' }) : fallback;
}

/** "Oct 21, 2026, 3:42 PM" — training run rows. */
export function formatDateTime(value: RawDate, fallback = ''): string {
  const d = toDisplayDate(value);
  return d ? d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : fallback;
}

/** "Oct 21, 03:42" — pipeline monitor rows. */
export function formatDayTime(value: RawDate, fallback = ''): string {
  const d = toDisplayDate(value);
  return d ? d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : fallback;
}

/** "Just now" / "3h ago" / "5d ago", then "Oct 21" beyond a week — the song
 *  list's time column. */
export function formatRelativeDay(value: RawDate, now = new Date()): string {
  const d = toDisplayDate(value);
  if (!d) return '';
  const diffH = Math.floor((now.getTime() - d.getTime()) / 3600000);
  if (diffH < 1) return 'Just now';
  if (diffH < 24) return `${diffH}h ago`;
  const diffD = Math.floor(diffH / 24);
  if (diffD < 7) return `${diffD}d ago`;
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

/** "3:42:07 PM" — a log line or a moment inside a run, where seconds matter. */
export function formatTimeOfDay(value: RawDate, fallback = ''): string {
  const d = toDisplayDate(value);
  return d ? d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : fallback;
}
