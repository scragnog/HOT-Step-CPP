// generation/adapterSections.ts — Per-section adapter masking (regional LoRA)
//
// Parses inline per-section adapter-influence directives from the lyrics, e.g.
//
//   [Intro]{greenday_idiot=1; blink_selftitled=0}
//   [Verse 1]{greenday_idiot=0.5; blink_selftitled=0.5}
//   ...lines...
//   [Chorus]{#1=0; #2=1}   (positional #N / bare N also accepted)
//
// and turns them into a per-section weight table indexed to the loaded adapter
// stack, plus the lyrics with the {…} directives stripped (the model must never
// see them). Keyed by trigger word (adapter filename stem), or positional #N.
//
// Sum/Blend (issue #72) is reused per section; directive-less sections fall back
// to the stack's normal effective scales ("uniform blend of the stack").
// See docs/plans/per-section-adapter-masking.md.

export interface AdapterSection {
  weights: number[]; // effective per-adapter scale, indexed to the stack
  size: number;      // relative frame-allocation hint (section char count)
}

export interface ParsedAdapterSections {
  lyrics: string;                 // directives stripped
  sections?: AdapterSection[];    // undefined when the feature is inactive
}

/** filename stem (trigger word) for an adapter path */
function triggerOf(p: string): string {
  return (p.split(/[\\/]/).pop() || p).replace(/\.safetensors$/i, '');
}

/** Resolve a directive key (trigger word = filename stem, or positional "#2"/"2") to a stack index, or -1. */
function resolveKey(key: string, triggers: string[]): number {
  const k = key.trim().toLowerCase();
  const byTrigger = triggers.findIndex(t => t.toLowerCase() === k);
  if (byTrigger >= 0) return byTrigger;
  const m = k.match(/^#?(\d+)$/);
  if (m) {
    const idx = parseInt(m[1], 10) - 1; // 1-based
    if (idx >= 0 && idx < triggers.length) return idx;
  }
  return -1;
}

interface DirectiveParse {
  raw: number[];        // per-adapter weights (unmentioned → 0)
  pairs: number;        // `key=val` pairs found (0 → not a directive at all)
  resolved: number;     // pairs whose key matched a stacked adapter
  unresolved: string[]; // keys that parsed but matched nothing (typos)
}

/** Parse a `key=val; key=val` directive body into raw per-adapter weights. */
function parseDirective(body: string, triggers: string[]): DirectiveParse {
  const raw = new Array(triggers.length).fill(0);
  let pairs = 0, resolved = 0;
  const unresolved: string[] = [];
  for (const part of body.split(/[;,]/)) {
    const eq = part.indexOf('=');
    if (eq < 0) continue;
    const key = part.slice(0, eq).trim();
    const val = parseFloat(part.slice(eq + 1).trim());
    if (!key || !Number.isFinite(val)) continue;
    pairs++;
    const idx = resolveKey(key, triggers);
    if (idx >= 0) { raw[idx] = Math.max(0, val); resolved++; }
    else unresolved.push(key);
  }
  return { raw, pairs, resolved, unresolved };
}

/** True when a `{…}` body is directive-shaped (≥1 key=val pair), regardless of key resolution. */
function isDirectiveShaped(body: string): boolean {
  for (const part of body.split(/[;,]/)) {
    const eq = part.indexOf('=');
    if (eq < 0) continue;
    const key = part.slice(0, eq).trim();
    const val = parseFloat(part.slice(eq + 1).trim());
    if (key && Number.isFinite(val)) return true;
  }
  return false;
}

/**
 * Strip directive-shaped `[Header]{…}` blocks from lyrics WITHOUT applying them.
 * Used when the ≥2-adapter gate is not met, so stray directives never reach the
 * LM/encoder as garbage tokens. Non-directive `{…}` (no key=val pair, e.g. a
 * stylistic `{softly}`) is left untouched.
 */
export function stripAdapterDirectives(lyrics: string): string {
  if (!lyrics) return lyrics;
  return lyrics.replace(/(\[[^\]\n]+\])[ \t]*\{([^}]*)\}/g, (full, header, body) =>
    isDirectiveShaped(body) ? header : full);
}

/** Apply the #72 Sum/Blend transform to a section's raw weights. */
function applyMode(raw: number[], mode: string, budget: number): number[] {
  if (mode === 'blend') {
    const sum = raw.reduce((a, b) => a + (b || 0), 0);
    if (sum > 0) return raw.map(w => +(budget * (w || 0) / sum).toFixed(4));
    return raw.map(() => 0); // explicit all-zero directive → base only
  }
  return raw.map(w => w || 0); // sum: raw as-is
}

/**
 * Parse per-section adapter directives from lyrics.
 * @param lyrics    raw lyrics (may contain `[Section]{…}` directives)
 * @param stack     loaded adapter stack (effective scales), order matches the engine
 * @param mode      'sum' | 'blend'
 * @param budget    combined-strength budget (blend)
 */
export function parseAdapterSections(
  lyrics: string,
  stack: { path: string; scale: number }[],
  mode: string,
  budget: number,
): ParsedAdapterSections {
  if (!lyrics || !Array.isArray(stack) || stack.length < 2) return { lyrics };
  // Fast bail-out: no directive syntax at all.
  if (!/\]\s*\{[^}]*\}/.test(lyrics) && !/^\s*\{[^}]*\}/.test(lyrics)) return { lyrics };

  const triggers = stack.map(s => triggerOf(s.path));
  const defaultWeights = stack.map(s => s.scale); // uniform blend of the stack

  // Split into sections at [Header] lines, capturing an optional {…} directive
  // that follows the header. Content before the first header is an implicit
  // directive-less section.
  const headerRe = /\[[^\]\n]+\]/g;
  const sections: AdapterSection[] = [];
  let cleaned = '';
  let lastIndex = 0;

  // Helper to push a section given its body text and directive (raw weights or null).
  const pushSection = (body: string, raw: number[] | null) => {
    const size = Math.max(1, body.replace(/\s+/g, ' ').trim().length);
    const weights = raw ? applyMode(raw, mode, budget) : defaultWeights.slice();
    sections.push({ weights, size });
  };

  const matches = [...lyrics.matchAll(headerRe)];
  if (matches.length === 0) return { lyrics };

  // Preamble before the first header (rare) → implicit default section.
  const firstStart = matches[0].index ?? 0;
  if (firstStart > 0 && lyrics.slice(0, firstStart).trim().length > 0) {
    pushSection(lyrics.slice(0, firstStart), null);
  }
  cleaned += lyrics.slice(0, firstStart);
  lastIndex = firstStart;

  for (let mi = 0; mi < matches.length; mi++) {
    const h = matches[mi];
    const hStart = h.index ?? 0;
    const header = h[0];
    let cursor = hStart + header.length;

    // Optional directive immediately after the header (allowing whitespace).
    // A `{…}` block is only treated as a directive when it contains at least
    // one key=val pair — `[Verse] {softly}` is lyric text, not an all-zero
    // directive, and must stay in the lyrics untouched.
    let raw: number[] | null = null;
    const after = lyrics.slice(cursor);
    const dm = after.match(/^[ \t]*\{([^}]*)\}/);
    const headerOut = header;
    if (dm) {
      const p = parseDirective(dm[1], triggers);
      if (p.pairs > 0) {
        cursor += dm[0].length; // directive-shaped → strip from the output
        if (p.unresolved.length) {
          console.warn(`[AdapterSections] ${header} directive: unknown adapter key(s) ${p.unresolved.map(k => `"${k}"`).join(', ')} — loaded triggers: ${triggers.join(', ')}`);
        }
        if (p.resolved > 0) {
          raw = p.raw;
        } else {
          // Every key was a typo — fall back to the stack defaults rather
          // than silently disabling all adapters for this section.
          console.warn(`[AdapterSections] ${header} directive: no keys resolved, using stack default weights`);
          raw = null;
        }
      }
      // p.pairs === 0 → not a directive: leave the `{…}` in the body/lyrics.
    }

    // Body runs until the next header (or end).
    const bodyEnd = (mi + 1 < matches.length) ? (matches[mi + 1].index ?? lyrics.length) : lyrics.length;
    const body = lyrics.slice(cursor, bodyEnd);

    pushSection(body, raw);
    cleaned += headerOut + body;
    lastIndex = bodyEnd;
  }
  cleaned += lyrics.slice(lastIndex);

  if (sections.length === 0) return { lyrics };
  return { lyrics: cleaned, sections };
}
