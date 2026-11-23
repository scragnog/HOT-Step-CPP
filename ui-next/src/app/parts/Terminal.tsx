// The terminal (the concept's .term, 25-module-planner.js and 25-module-repetition.js): dark, monospaced, a command
// bar on top and lines below, each "$ command" and its controls, each control printed after its label as a flag
// ("--temp 0.80"). What each child is, its label and whether it stands back are the core's (the container's kids).
//
//   terminal  holds a bar, the plan, lines and look-back strips. The lines a higher tier adds stand together under a
//             dashed rule (.tparams); with two strips or more, each line stands with the strip after it (.gline). It
//             stands back as the concept's do: with a bar (the planner off) its lines and flags fade and its power
//             switch does not; without one (a repetition guard at 1.00) it is a guard switched off
//   termBar   the bar: its power switch, flags, chips, a button, a note
//   termLine  "$ command" then each control after its flag (a flag prints its own words; a quoted string stands after
//             its flag on a line of its own); a wrapped line hangs under its first flag. A line that can stand back
//             says so after its first control ("guard off"), drawn hidden until it does

import { createElement, Fragment, useContext, type CSSProperties, type ReactNode } from 'react';
import { definePart, useSay, type PartView } from '../kit/index.ts';
import { itemsOf, TermWords } from './kids.ts';

type Kid = NonNullable<PartView['kids']>[number];

export function Terminal(v: PartView) {
  const say = useSay(), kids = v.kids ?? [], items = itemsOf(v.children);
  const bar = kids.some(k => k.shown && k.part === 'termBar'), strips = kids.filter(k => k.shown && k.part === 'lookback').length;
  const out: ReactNode[] = [];
  for (let i = 0; i < kids.length; i++) {
    const k = kids[i]!;
    if (!k.shown) continue;
    /* a line and the strip after it, as one guard (YuE2's plan and composer) */
    if (strips > 1 && k.part === 'termLine' && kids[i + 1]?.part === 'lookback' && kids[i + 1]!.shown) {
      out.push(<div key={i} className={`gline${k.dim ? ' guard-off' : ''}`} data-guard={k.keys[0]}>{items[i]}{items[i + 1]}</div>); i++; continue;
    }
    /* the lines a higher tier adds, together */
    if (k.extra && k.part === 'termLine') {
      const run: ReactNode[] = [];
      for (; i < kids.length && (!kids[i]!.shown || (kids[i]!.extra && kids[i]!.part === 'termLine')); i++) if (kids[i]!.shown) run.push(items[i]);
      i--; out.push(<div key={`x${i}`} className="tparams">{run}</div>); continue;
    }
    out.push(items[i]);
  }
  /* one guard: its strip greys while its line stands back, and the terminal is the guard (data-guard: its penalty) */
  const line = strips === 1 && !bar ? kids.find(k => k.shown && k.part === 'termLine') : undefined;
  const cls = `term${v.dim ? (bar ? ' pw-off' : ' plan-off') : ''}${line?.dim ? ' guard-off' : ''}`;
  return (
    <TermWords.Provider value={v.dimWords && !bar ? say(v.dimWords) : null}>
      <div className={cls} data-guard={line?.keys[0]} title={v.words.title ? say(v.words.title) : v.dim ? say(v.dim) : undefined}>{out}</div>
    </TermWords.Provider>
  );
}

export const TermBar = ({ children }: PartView) => <div className="term-bar">{children}</div>;

/** The flag a control is printed after: its label, unless it prints its own (a flag's [x] --cot). */
const flagOf = (k: Kid) => (k.part === 'flag' || k.label === undefined ? null : k.label);

export function TermLine(v: PartView) {
  const say = useSay(), lent = useContext(TermWords), kids = v.kids ?? [], items = itemsOf(v.children), cmd = v.words.command ? say(v.words.command) : '';
  const shown = kids.map((k, i) => [k, i] as const).filter(([k]) => k.shown);
  /* a quoted string stands after its flag, the line its own (--neg "…") */
  if (shown.length === 1 && shown[0]![0].part === 'textArea') {
    const [k, i] = shown[0]!;
    return <div className="tline neg-line"><span className="k cdot-lcd">{say(flagOf(k) ?? '')}</span>{items[i]}</div>;
  }
  const off = v.dimWords || lent ? <Fragment key="off">{v.dimWords ? <span className="term-off">{say(v.dimWords)}</span> : null}{lent ? <span className="term-plan-off">{lent}</span> : null}</Fragment> : null;
  const opts: ReactNode[] = [];
  /* a guard's words stand after its first control and the chips that go with it (--penalty, --mode) */
  let at = 0; while (shown[at + 1]?.[0].part === 'termChips') at++;
  shown.forEach(([k, i], n) => {
    const flag = flagOf(k);
    if (n > 0 && cmd && !(off && n === at + 1)) opts.push(' ');
    opts.push(createElement('span', { key: i, className: k.part === 'textField' ? 'opt wide' : 'opt' }, flag !== null ? <><span className="k">{say(flag)}</span> {items[i]}</> : items[i]));
    if (n === at && off) { opts.push(off); if (n + 1 < shown.length && cmd) opts.push(' '); }
  });
  if (!cmd) return <div className="tline">{opts}</div>;
  /* a wrapped line hangs under the first flag, past "$ command " (L-24) */
  return (
    <div className="tline hang" style={{ '--hi': `${cmd.length + 3}ch` } as CSSProperties}>
      <span className="p">$</span> <span className="k">{cmd}</span> {opts}
    </div>
  );
}

definePart('terminal', Terminal, { surface: 'term', dims: true });
definePart('termBar', TermBar, { surface: 'term' });
definePart('termLine', TermLine, { surface: 'term', dims: true });
