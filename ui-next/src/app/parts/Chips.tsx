// The choices drawn as chips, one pressed (the concept's rackCtl chip selector and tchips, 20-controls.js; the Beat I/O
// bars, 25-module-tempo.js). The options, their words, their tips and which is pressed are the core's.
//
//   chips      a side unit's short choice at Lab: up to four lit chips under a caps label
//   termChips  a choice in a terminal line, after its flag: --mode Presence Frequency DRY
//   lcdChips   numbered chips on an LCD with a word after them: 1 2 4 8 BARS

import { definePart, useSay, type ParamView, type PartView } from '../kit/index.ts';

type Opt = NonNullable<ParamView['options']>[number];
const ch = (p: ParamView) => (p.changed ? ' changed' : '');

function Buttons({ p, word }: { p: ParamView; word: (o: Opt) => string }) {
  const say = useSay();
  return <>{p.options!.map(o => (
    <button key={String(o.value)} type="button" data-seg={p.key} data-v={String(o.value)} aria-pressed={o.on} title={o.tip ? say(o.tip) : undefined} onClick={() => p.set(o.value)}>{word(o)}</button>
  ))}</>;
}

export function Chips(v: PartView) {
  const say = useSay(), p = v.params[0];
  if (!p?.options) return null;
  return (
    <div className={`chipsw${ch(p)}`} data-ctl={p.key}>
      <span className="kl">{say(v.label)}</span>
      <div>
        {p.options.map(o => (
          <button key={String(o.value)} type="button" data-seg={p.key} data-v={String(o.value)} aria-pressed={o.on} title={say(o.label)} onClick={() => p.set(o.value)}>{say(o.short ?? o.label)}</button>
        ))}
      </div>
    </div>
  );
}

export function TermChips(v: PartView) {
  const say = useSay(), p = v.params[0];
  if (!p?.options) return null;
  return <span className={`tchips${ch(p)}`} data-ctl={p.key} role="group" aria-label={say(p.label)}><Buttons p={p} word={o => say(o.label)} /></span>;
}

export function LcdChips(v: PartView) {
  const say = useSay(), p = v.params[0];
  if (!p?.options) return null;
  return (
    <span className={`lcd-bars${ch(p)}`} data-ctl={p.key} role="group" aria-label={say(p.label)}>
      <Buttons p={p} word={o => say(o.short ?? o.label, { upper: true })} />{v.words.word ? <i>{say(v.words.word, { upper: true })}</i> : null}
    </span>
  );
}

definePart('chips', Chips);
definePart('termChips', TermChips);
definePart('lcdChips', LcdChips);
