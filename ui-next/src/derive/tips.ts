// What a tooltip says (the concept's 19-tooltip.js): a control's tip is its title, which the page has already written
// in its language; these are the rules for cutting it to what the moment needs. The sentences of a tip (a full stop
// ends one only before white space, so "1.05", "31.8 GB" and "v1.5" keep every digit: F-01, Q-01, E-04); the
// sentences that teach how to work a control, which a session hears once (L-06); and the words a finger reads instead
// of a mouse's (S-31). The last two are the English table's rules; a locale states its own beside them.

/** The sentences of a tip, every word and digit kept. */
export const tipSentences = (text: string): string[] => text.split(/(?<=[.!?])\s+/).filter(Boolean);

/** A sentence that says how to work a control ("Drag sideways…"), as opposed to what the control means. */
const HOW: { [lang: string]: RegExp } = { en: /^(Drag|Click|Scroll|Double-click|Press|Hold|Use the arrow|Type or|Or use)/ };
export const isHowTo = (sentence: string, lang = 'en'): boolean => (HOW[lang] ?? HOW.en!).test(sentence.trim());

/** A tip with the sentences already taught left out, and the ones it teaches now: what to show, and what to add to
 *  what the session has been taught. */
export function tipLesson(text: string, taught: ReadonlySet<string>, lang = 'en'): { text: string; teaches: string[] } {
  const teaches: string[] = [];
  const keep = tipSentences(text).filter(p => { const s = p.trim(); if (!isHowTo(s, lang)) return true; if (taught.has(s) || teaches.includes(s)) return false; teaches.push(s); return true; });
  return { text: keep.join(' ').trim(), teaches };
}

/** Under a finger: no sentence about keys, Shift, the wheel or F9, which a phone has not got; tap for click; and a
 *  knob's way of working (`knobHow`, its title's last sentences) becomes a finger's (`knobTouch`). */
const TOUCH: { [lang: string]: { drop: RegExp; swap: [RegExp, string][] } } = {
  en: { drop: /arrow key|Shift|scroll|\bF9\b|\bEnter\b|keyboard|Ctrl/i, swap: [[/Double-click/g, 'Double-tap'], [/\bClick\b/g, 'Tap'], [/\bclick\b/g, 'tap']] },
};
export function touchTip(text: string, knobHow: string, knobTouch: string, lang = 'en'): string {
  const t = TOUCH[lang] ?? TOUCH.en!;
  const s = knobHow && text.endsWith(knobHow) ? text.slice(0, -knobHow.length) + knobTouch : text;
  return t.swap.reduce((a, [re, w]) => a.replace(re, w), tipSentences(s).filter(p => !t.drop.test(p)).join(' ')).trim();
}

/** A tip that says no more than a line already printed on the control (its label over its value, Q-11) does not
 *  open: `lines` are the control's own lines of text. */
export const sameAsLabel = (lines: readonly string[], text: string): boolean => {
  const said = text.trim().toLowerCase().replace(/\.$/, '');
  return !!said && lines.some(l => l.trim().toLowerCase() === said);
};
