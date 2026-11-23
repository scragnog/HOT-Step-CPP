// The racks and rows that hold a module's controls in the concept's housings. What each child is, its label and whether
// it stands back are the core's (the container's kids); none decides anything about a value.
//
//   group   a rack group, by its look:
//           rack    a boxed group (Plan adherence's, Latent & correction's: 15-controls.css .ad-g): its switch in its
//                   header (a pill: the group lines up with its neighbours, .lat-g; a power chip: .ad-g), else its caps
//                   title; its tokens' pairs together (.toks); switched off it is hatched and its knobs go grey
//           strip   an inset strip with its caps title (the Sampler's schedule controls: .schctl)
//           plugin  a plugin's controls (the Sampler's YuE2 plugins: .plugctl): its name, "experimental" and Reset, then
//                   its controls in cells
//   pair    two tokens as a range under one label: from → to (.tk-pair)
//   bank    a fader bank (the adapters' master section: .rig-master): its heading with its note as the tip, the 2-1-0
//           scale, each fader on its tape
//   deck    the dark two-column deck (Plans & endings: .deck); each labelled row stands in its columns
//   lcdRow  one row of an LCD (Tempo's): the top row when it holds the big number (.lcd-top), else a row, spread to its
//           ends (.lcd-ends); a run of trim settings stands together at its end and is held out of sight while it
//           stands back (.lcd-trim.held)

import type { ReactNode } from 'react';
import { definePart, ui, useSay, type PartView } from '../kit/index.ts';
import { itemsOf, LcdSlots } from './kids.ts';

type Kid = NonNullable<PartView['kids']>[number];
/** A token's pair in a rack group (a labelled token, or a range). */
const isPair = (k: Kid) => k.shown && (k.part === 'pair' || (k.part === 'labelled' && k.inner[0] === 'token'));

export function Group(v: PartView) {
  const say = useSay(), kids = v.kids ?? [], items = itemsOf(v.children), look = String(v.opts.look ?? 'rack');
  const title = v.words.title ? <span className="kl">{say(v.words.title)}</span> : null, tip = v.dim ? say(v.dim) : undefined;
  if (look === 'strip') return <div className="schctl" title={tip}>{title}{v.children}</div>;
  if (look === 'plugin') {
    return (
      <div className="plugctl" title={tip}>
        <div className="pc-head">
          <b>{v.words.title ? say(v.words.title) : null}</b><span className="exp">{say(ui('group.experimental'))}</span>
          {v.actions.map((a, i) => <button key={i} type="button" className="small-btn" title={a.tip ? say(a.tip) : undefined} onClick={() => v.run(i)}>{say(a.label)}</button>)}
        </div>
        <div className="pc-grid">{kids.map((k, i) => (k.shown && k.part === 'textField' ? <label key={i} className="pc-i wide"><span className="kl">{k.label !== undefined ? say(k.label) : null}</span>{items[i]}</label> : items[i]))}</div>
      </div>
    );
  }
  /* rack: a switch first is the group's header */
  const first = kids.findIndex(k => k.shown), head = kids[first]?.part === 'pill' || kids[first]?.part === 'powerChip' ? first : -1;
  const lat = head >= 0 && kids[head]!.part === 'pill', out: ReactNode[] = [];
  for (let i = 0; i < kids.length; i++) {
    const k = kids[i]!;
    if (!k.shown) continue;
    if (i === head) { out.push(<span key={i} className="ad-gh">{items[i]}</span>); if (lat && kids[i + 1]?.part === 'row') out.push(<span key="opt" className="lat-opt" aria-hidden="true" />); continue; }
    /* a switched group's option sits on the row under its header */
    if (lat && i === head + 1 && k.part !== 'row') { out.push(k.part === 'segmented' ? items[i] : <span key={i} className="lat-opt">{items[i]}</span>); continue; }
    if (isPair(k)) { const run: ReactNode[] = []; for (; i < kids.length && (isPair(kids[i]!) || !kids[i]!.shown); i++) if (kids[i]!.shown) run.push(items[i]); i--; out.push(<span key={`t${i}`} className="toks">{run}</span>); continue; }
    out.push(items[i]);
  }
  const foot = kids.some(k => k.shown && k.part === 'labelled' && k.inner[0] === 'segmented') && head < 0;
  const cls = lat ? `lat-g${v.dim ? ' off' : ''}` : `ad-g${foot ? ' ad-ps' : ''}${v.dim ? ' dim' : ''}`;
  return <div className={cls}>{head < 0 ? title : null}{out}</div>;
}

export function Pair(v: PartView) {
  const say = useSay(), items = itemsOf(v.children);
  return (
    <span className="tk-pair" title={v.dim ? say(v.dim) : undefined}>
      <span className="kl">{v.label !== undefined ? say(v.label) : null}</span>{items[0]}<span className="tk-arrow" aria-hidden="true">→</span>{items[1]}
    </span>
  );
}

export function Bank(v: PartView) {
  const say = useSay();
  return (
    <div className="rig-master">
      <div className="rig-mh"><span title={v.words.note ? say(v.words.note) : undefined}>{v.words.title ? say(v.words.title) : null}</span></div>
      <div className="faders"><div className="fdr-scale" aria-hidden="true"><span>2</span><span>1</span><span>0</span></div>{v.children}</div>
    </div>
  );
}

export const Deck = ({ children }: PartView) => <div className="deck">{children}</div>;

/** A run of trim settings (labelled tokens) at a row's end. */
const isTrim = (k: Kid) => k.shown && k.part === 'labelled' && k.inner[0] === 'token';

export function LcdRow(v: PartView) {
  const kids = v.kids ?? [], items = itemsOf(v.children), out: ReactNode[] = [];
  const top = kids.some(k => k.shown && k.part === 'lcdNumber');
  /* the top row lends the number its beat lamps and its buttons (LcdSlots), which it draws where the concept does */
  const lend = (part: string) => { const i = top ? kids.findIndex(k => k.shown && k.part === part) : -1; return i < 0 ? undefined : i; };
  const unit = lend('beatPulse'), btns = lend('tapTempo');
  for (let i = 0; i < kids.length; i++) {
    const k = kids[i]!;
    /* a row spread to its ends keeps the place of a switch a higher tier adds, so its others stand where they will */
    if (!k.shown && v.opts.ends && k.extra && k.part === 'lcdButton') { out.push(<span key={i} className="lcd-btns" />); continue; }
    if (!k.shown || i === unit || i === btns) continue;
    if (isTrim(k)) {
      const run: Kid[] = [], els: ReactNode[] = [];
      for (; i < kids.length && (isTrim(kids[i]!) || !kids[i]!.shown); i++) if (kids[i]!.shown) { run.push(kids[i]!); els.push(items[i]); }
      i--; const held = run.every(x => x.dim);
      out.push(<span key={`t${i}`} className={`lcd-trim${held ? ' held' : ''}`} aria-hidden={held || undefined}>{els}</span>); continue;
    }
    out.push(items[i]);
  }
  const row = <div className={top ? 'lcd-top' : `lcd-row${v.opts.ends ? ' lcd-ends' : ''}`}>{out}</div>;
  return top ? <LcdSlots.Provider value={{ unit: unit === undefined ? undefined : items[unit], btns: btns === undefined ? undefined : items[btns] }}>{row}</LcdSlots.Provider> : row;
}

/* a group's look decides what its children stand on: a rack's tokens pair up, a plugin's controls stand in cells; a
   strip holds them as a tile's body does */
definePart('group', Group, { surface: o => (o.look === 'plugin' ? 'plugin' : o.look === 'strip' ? undefined : 'group'), dims: true });
definePart('pair', Pair, { dims: true });
definePart('bank', Bank);
definePart('deck', Deck, { surface: 'deck' });
definePart('lcdRow', LcdRow, { surface: 'lcdrow' });
