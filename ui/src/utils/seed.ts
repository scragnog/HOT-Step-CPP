// Generation seeds are uint64 in the engines. A JS number holds them exactly
// only up to 2^53, so anything larger is carried as its decimal string.

export type Seed = number | string;

export const U64_MAX = (1n << 64n) - 1n;

/** Safe integers stay numbers; larger integers become their exact decimal string. */
export function toSeed(n: bigint): Seed {
  return n <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(n) : n.toString();
}

/** Store form of a seed. A number already past 2^53 has lost its low digits;
 *  this at least spells the value it holds exactly, not as "...570000". */
export function normalizeSeed(v: Seed): Seed {
  if (typeof v === 'number') return Number.isSafeInteger(v) || !Number.isInteger(v) ? v : toSeed(BigInt(v));
  return /^-?\d+$/.test(v.trim()) ? toSeed(BigInt(v.trim())) : v;
}

/** Typed seed text, -1 (random) to 2^64-1; undefined when it is not one. */
export function parseSeed(text: string): Seed | undefined {
  const s = text.trim();
  if (!/^-?\d+$/.test(s)) return undefined;
  const n = BigInt(s);
  return n < -1n || n > U64_MAX ? undefined : toSeed(n);
}

/** Exact decimal spelling of a seed for display. */
export function seedText(v: Seed): string {
  return typeof v === 'number' && Number.isInteger(v) ? BigInt(v).toString() : String(v);
}
