/**
 * `plural(1, 'song')` → "song", `plural(3, 'song')` → "songs".
 *
 * The UI spelled this out inline about forty times as
 * `{n} song{n !== 1 ? 's' : ''}` — and inconsistently: some used `> 1`, which
 * counts zero as singular. One function keeps the count boundary honest.
 */
export function plural(count: number, one: string, many = `${one}s`): string {
  return count === 1 ? one : many;
}
