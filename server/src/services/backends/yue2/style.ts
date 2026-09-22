// yue2/style.ts — the YuE2 style string, one implementation on this side of the wire.
//
// A faithful port of `yue2_style_string()` (engine/src/train/yue2-sidecar.h),
// which is what BOTH YuE2 trainers feed to the model as the style prompt of
// every artist row:
//
//     <trigger>, in the style of <trigger>. <caption> <genre>, <bpm> BPM, key of <key>.
//
// An adapter only ever saw its trigger inside that sentence, so a generation
// that sends the caption box verbatim is off-distribution — measured
// 2026-09-14 (_LISTENING/2026-09-14/RESULTS.md, "the template x adapter x seed
// grid"): the same adapter sounds like the artist under this template and caps
// under the bare "<trigger>, <caption>". The template is a control, not
// decoration, so the app has to compose the identical string.
//
// `bare` reproduces the pre-template behaviour, for adapters trained before it
// existed; the trainer's sidecar records which one a run used
// (`style_template` in the adapter metadata).
//
// The MM3 equivalent is minimax/trigger.ts. They are not shared: MM3 injects a
// trigger token at the front of a Structured Caption, YuE2 wraps the whole
// caption in a sentence with a metadata tail.

/** `upstream` is the training default; `bare` is the legacy "<trigger>, <caption>". */
export type Yue2StyleTemplate = 'upstream' | 'bare';

export interface Yue2StyleParts {
  /** Adapter trigger word, '' for a base-model render. */
  trigger?: string;
  /** The user's caption / style description. */
  caption?: string;
  /** Sidecar tail fields — each omitted from the tail when blank. */
  genre?: string;
  bpm?: string | number;
  key?: string;
  /** Defaults to 'upstream', matching the trainer's own default. */
  template?: Yue2StyleTemplate;
}

/** The C++ squash: collapse runs of ' ' \t \r \n to one space and drop them at
 *  both ends. Deliberately NOT `\s`/`trim()` — those also eat \v, \f and
 *  U+00A0, which the engine keeps, and a caption pasted from a web page is
 *  exactly where a non-breaking space turns up. */
function squash(s: string): string {
  return s.replace(/[ \t\r\n]+/g, ' ').replace(/^ /, '').replace(/ $/, '');
}

function str(v: string | number | undefined): string {
  if (typeof v === 'number') return Number.isFinite(v) ? String(v) : '';
  return v ?? '';
}

/** The "<trigger>, in the style of <trigger>. " opener, '' for no trigger. */
function head(trigger: string, upstream: boolean): string {
  if (!trigger) return '';
  return upstream ? `${trigger}, in the style of ${trigger}. ` : `${trigger}, `;
}

/** The ", "-joined "<genre>, <bpm> BPM, key of <key>." tail, '' when all three
 *  are blank. */
function tail(parts: Yue2StyleParts): string {
  const bits: string[] = [];
  const genre = squash(str(parts.genre));
  const bpm = squash(str(parts.bpm));
  const key = squash(str(parts.key));
  if (genre) bits.push(genre);
  if (bpm) bits.push(`${bpm} BPM`);
  if (key) bits.push(`key of ${key}`);
  return bits.length ? `${bits.join(', ')}.` : '';
}

/**
 * Compose the style string exactly as the trainer did.
 *
 * Note there is no period inserted after the caption: the engine assembles
 * `head + caption + " " + tail`, so whatever punctuation the caption ends with
 * is what reaches the model.
 */
export function yue2StyleString(parts: Yue2StyleParts): string {
  const upstream = (parts.template ?? 'upstream') !== 'bare';
  const trigger = str(parts.trigger);
  const cap = squash(str(parts.caption));

  // Upstream train.py:192's rule, shared by both templates: no caption means
  // the trigger stands alone, unsquashed.
  if (!cap) return trigger;
  if (!upstream) return trigger ? `${trigger}, ${cap}` : cap;

  return squash(`${head(trigger, true)}${cap} ${tail(parts)}`);
}

/** The trailing "…, 168 BPM" and "…, key of D minor." segments, matched one at
 *  a time from the end so either order and either alone both come apart. The
 *  leading [\s,]* eats the separator with them, so nothing is left dangling. */
const TAIL_BPM_RE = /[\s,]*(\d+(?:\.\d+)?)\s*BPM\s*\.?\s*$/i;
const TAIL_KEY_RE = /[\s,]*key of\s+([^,.]+?)\s*\.?\s*$/i;

/**
 * Peel a trained tail off a caption that already carries one.
 *
 * A dataset caption ends with the same "<bpm> BPM, key of <key>." the template
 * composes, because the labeller wrote it in that format. Appending a second
 * tail produced "…, 178 BPM 168 BPM, key of D minor." — the model gets two
 * tempos and believes neither.
 *
 * applyYue2StyleTemplate's own idempotency strip cannot catch this: it removes
 * only a tail IDENTICAL to the one it is about to add, and a caption saying
 * 178 BPM does not match a request saying 168. So pull the fields out
 * generically, and let the caller decide which value wins.
 *
 * Genre is deliberately NOT parsed. It has no marker word, so anything that
 * claimed to find it would be guessing at the end of the user's own prose.
 */
export function splitYue2Tail(caption: string): { caption: string; bpm: string; key: string } {
  let rest = squash(caption);
  let bpm = '';
  let key = '';
  // Two passes: the tail may be written in either order, and each match
  // exposes the next one at the end of the string.
  for (let pass = 0; pass < 2; pass++) {
    const k = rest.match(TAIL_KEY_RE);
    if (k) { key = key || squash(k[1]); rest = rest.slice(0, k.index); }
    const b = rest.match(TAIL_BPM_RE);
    if (b) { bpm = bpm || squash(b[1]); rest = rest.slice(0, b.index); }
    if (!k && !b) break;
  }
  return { caption: squash(rest), bpm, key };
}

/**
 * Compose, but tolerate a caption that is already composed.
 *
 * Same job as minimax/trigger.ts's applyMm3Trigger: the caption box may already
 * hold a hand-written "<trigger>, in the style of <trigger>. …" from an earlier
 * render (or a copy-pasted one from the listening notes), and prefixing that a
 * second time is a sequence the adapter never saw. Strips the opener and the
 * metadata tail this call would produce — case-insensitively, as applyMm3Trigger
 * matches — then composes once, so f(f(x)) === f(x).
 *
 * This is the function the generation path should call; yue2StyleString stays a
 * literal port for anything that needs to reproduce a training row.
 */
export function applyYue2StyleTemplate(parts: Yue2StyleParts): string {
  const upstream = (parts.template ?? 'upstream') !== 'bare';
  const trigger = str(parts.trigger);

  let cap = squash(str(parts.caption));
  const opener = squash(head(trigger, upstream));
  if (opener && cap.toLowerCase().startsWith(opener.toLowerCase())) {
    cap = squash(cap.slice(opener.length));
  }
  if (upstream) {
    const end = tail(parts);
    if (end && cap.toLowerCase().endsWith(end.toLowerCase())) {
      cap = squash(cap.slice(0, cap.length - end.length));
    }
  }

  return yue2StyleString({ ...parts, caption: cap });
}

/**
 * Branch table, one case per early return in the C++. Run it with
 * `npx tsx -e "import('./src/services/backends/yue2/style.js').then(m => console.log(m.yue2StyleSelfCheck()))"`
 * from server/ after touching anything above; [] means every branch still
 * matches the engine.
 */
export function yue2StyleSelfCheck(): string[] {
  const t = 'albumA2';
  const full: Yue2StyleParts = { trigger: t, caption: 'sludgy doom riffs', genre: 'doom metal', bpm: '72', key: 'D minor' };
  const composed = `${t}, in the style of ${t}. sludgy doom riffs doom metal, 72 BPM, key of D minor.`;

  const cases: Array<[string, string, string]> = [
    // [name, actual, expected]
    ['upstream: all fields', yue2StyleString(full), composed],
    ['upstream: no tail fields', yue2StyleString({ trigger: t, caption: 'sludgy doom riffs' }),
      `${t}, in the style of ${t}. sludgy doom riffs`],
    ['upstream: bpm only', yue2StyleString({ trigger: t, caption: 'a caption', bpm: 128 }),
      `${t}, in the style of ${t}. a caption 128 BPM.`],
    ['upstream: key only', yue2StyleString({ trigger: t, caption: 'a caption', key: 'C' }),
      `${t}, in the style of ${t}. a caption key of C.`],
    ['upstream: empty caption -> trigger alone', yue2StyleString({ trigger: t, caption: '  ', genre: 'doom' }), t],
    ['upstream: empty trigger -> no opener', yue2StyleString({ caption: 'a caption', genre: 'doom' }), 'a caption doom.'],
    ['upstream: whitespace squashed', yue2StyleString({ trigger: t, caption: '  two\n\nlines  ', genre: ' doom  metal ' }),
      `${t}, in the style of ${t}. two lines doom metal.`],
    ['bare: trigger + caption', yue2StyleString({ ...full, template: 'bare' }), `${t}, sludgy doom riffs`],
    ['bare: empty trigger', yue2StyleString({ caption: 'a caption', template: 'bare' }), 'a caption'],
    ['bare: empty caption', yue2StyleString({ trigger: t, caption: '', template: 'bare' }), t],
    ['idempotent: already composed', applyYue2StyleTemplate({ ...full, caption: composed }), composed],
    ['idempotent: first pass equals port', applyYue2StyleTemplate(full), composed],
    ['idempotent: bare', applyYue2StyleTemplate({ ...full, caption: `${t}, sludgy doom riffs`, template: 'bare' }),
      `${t}, sludgy doom riffs`],
  ];

  return cases
    .filter(([, actual, expected]) => actual !== expected)
    .map(([name, actual, expected]) => `${name}\n  got:  ${JSON.stringify(actual)}\n  want: ${JSON.stringify(expected)}`);
}
