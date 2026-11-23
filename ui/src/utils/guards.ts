/**
 * Drops the empty slots out of an array while narrowing the element type.
 * `filter(Boolean)` also drops falsy values but leaves `null` in the type, so
 * every consumer re-narrows by hand.
 */
export function isPresent<T>(value: T): value is Exclude<T, null | undefined | false | 0 | ''> {
  return Boolean(value);
}
