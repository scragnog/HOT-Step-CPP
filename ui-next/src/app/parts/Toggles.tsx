// The switches that live inside a module's own look (the concept's pillTog, flag and ledLCD, 20-controls.js; the record
// button, 25-module-plans.js; the lock, 25-module-seed.js; a rack group's power chip, 25-module-adhere.js). Each draws
// its markup and sets its switch to the other side; whether it is on, its words and the step are the core's.
//
//   pill        a pill with a lamp (Instrumental, Level match, a rack group's power switch)
//   flag        a terminal flag with its box: [x] --cot
//   lcdButton   a lit word on an LCD that switches (BEAT I/O, AUTO-TRIM)
//   recButton   ● SAVE NEXT, which arms the next run to save; it stands back (is-off) while its node does
//   lockButton  the padlock that keeps a value (Keep this seed)
//   powerChip   a rack group's power chip for a group switched by a number: off is one value, on sets another

import { definePart, useSay, type ParamView, type PartView } from '../kit/index.ts';

const on = (p: ParamView) => p.value === true;
const ch = (p: ParamView) => (p.changed ? ' changed' : '');

export function Pill(v: PartView) {
  const say = useSay(), p = v.params[0];
  if (!p) return null;
  return (
    <button type="button" className={`pill-tog${on(p) ? ' on' : ''}${ch(p)}`} data-ledtog={p.key} data-ctl={p.key} aria-pressed={on(p)} title={v.title ? say(v.title) : undefined} onClick={() => p.set(!on(p))}>
      <i />{say(v.label)}
    </button>
  );
}

export function Flag(v: PartView) {
  const say = useSay(), p = v.params[0];
  if (!p) return null;
  return (
    <button type="button" className={`flag${on(p) ? ' on' : ''}${ch(p)}`} data-ledtog={p.key} data-ctl={p.key} aria-pressed={on(p)} aria-label={say(p.label)} title={v.title ? say(v.title) : undefined} onClick={() => p.set(!on(p))}>
      <span className="fbox" aria-hidden="true" />{say(v.label)}
    </button>
  );
}

export function LcdButton(v: PartView) {
  const say = useSay(), p = v.params[0];
  if (!p) return null;
  return (
    <span className="lcd-btns">
      <button type="button" className={`${on(p) ? 'on' : ''}${ch(p)}`} data-ledtog={p.key} data-ctl={p.key} aria-pressed={on(p)} aria-label={say(p.label)} title={v.title ? say(v.title) : undefined} onClick={() => p.set(!on(p))}>
        {say(v.label, { upper: true })}
      </button>
    </span>
  );
}

export function RecButton(v: PartView) {
  const say = useSay(), p = v.params[0];
  if (!p) return null;
  return (
    <button type="button" className={`rec${on(p) ? ' on' : ''}${v.dim ? ' is-off' : ''}${ch(p)}`} data-ledtog={p.key} data-ctl={p.key} aria-pressed={on(p)} aria-label={say(p.label)}
      title={v.dim ? say(v.dim) : v.title ? say(v.title) : undefined} onClick={() => p.set(!on(p))}>
      {say(v.label)}
    </button>
  );
}

export function LockButton(v: PartView) {
  const say = useSay(), p = v.params[0];
  if (!p) return null;
  return (
    <button type="button" className={`lock${on(p) ? ' on' : ''}${ch(p)}`} data-ledtog={p.key} data-ctl={p.key} aria-pressed={on(p)} aria-label={say(v.label)} title={say(v.title ?? v.label)} onClick={() => p.set(!on(p))}>
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
        <rect x="5" y="11" width="14" height="10" rx="2" /><path className="c" d="M8 11V7a4 4 0 0 1 8 0v4" /><path className="o" d="M8 11V7a4 4 0 0 1 7.5-2" />
      </svg>
    </button>
  );
}

export function PowerChip(v: PartView) {
  const say = useSay(), p = v.params[0], pw = v.control?.power;
  if (!p || !pw) return null;
  return (
    <button type="button" className={`pill-tog${pw.on ? ' on' : ''}${ch(p)}`} data-power={p.key} data-ctl={p.key} aria-pressed={pw.on} aria-label={say(v.label)} title={say(pw.tip)} onClick={() => p.set(pw.to)}>
      <i />{say(pw.words)}
    </button>
  );
}

definePart('pill', Pill);
definePart('flag', Flag);
definePart('lcdButton', LcdButton);
definePart('recButton', RecButton);
definePart('lockButton', LockButton);
definePart('powerChip', PowerChip);
