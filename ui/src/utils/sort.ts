/**
 * Comparators for the lists the UI sorts.
 *
 * Thirty-odd call sites wrote `(a, b) => a.field.localeCompare(b.field)` or
 * `(a, b) => b.step - a.step` by hand. That spelling throws on a missing
 * string and orders a missing number as if it were 0, so a row with no value
 * jumped to the top. These put missing values last in both directions.
 */
export function byText<T>(
  get: (item: T) => string | null | undefined,
  dir: 'asc' | 'desc' = 'asc',
): (a: T, b: T) => number {
  const sign = dir === 'desc' ? -1 : 1;
  return (a, b) => {
    const x = get(a);
    const y = get(b);
    if (x == null && y == null) return 0;
    if (x == null) return 1;
    if (y == null) return -1;
    return sign * x.localeCompare(y);
  };
}

export function byNumber<T>(
  get: (item: T) => number | null | undefined,
  dir: 'asc' | 'desc' = 'asc',
): (a: T, b: T) => number {
  const sign = dir === 'desc' ? -1 : 1;
  return (a, b) => {
    const x = get(a);
    const y = get(b);
    const xn = typeof x === 'number' && Number.isFinite(x);
    const yn = typeof y === 'number' && Number.isFinite(y);
    if (!xn && !yn) return 0;
    if (!xn) return 1;
    if (!yn) return -1;
    return sign * (x - y);
  };
}
