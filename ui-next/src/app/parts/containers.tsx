// The containers. Each holds its children, already drawn by the renderer, in the housing of the concept's markup: row
// is the .knobs row (a row of rack groups is the concept's .ad-groups), rack the .pextra.rackrow strip, well the
// recessed strip under an instrument (the Sampler's), lcd an LCD panel, foot a tile's foot row. A container none of
// whose children is drawn is not drawn at all (the core's nodeShown).
//
// labelled pairs a caps label with what it holds, as the surface it stands on pairs them (the core's kids say what it
// holds): on a tile, a foot or a well the label comes first (the concept's foot-ctl), above a strip of switches (the
// plans' Endings) or a row of dice (Re-planning); on hardware (a pedal, a side unit, an LCD) the control comes first and
// the caps label under it (lcdsel); in an LCD's row the label before the control (Tempo's SIG), or a token's
// foot pairing (its trim Buffer); in a rack group a token's pair (its Range) or a choice at the group's foot (Plan
// strength's Curve); in a plugin's controls a cell; in a deck a row of the deck's two columns, a recorder's own lines
// indented under it, and a replay that stands back said on its own line.

import { definePart, useSay, type PartView } from '../kit/index.ts';
import { itemsOf } from './kids.ts';
import { NoteInline } from './inline.ts';

export function Row(v: PartView) {
  const kids = (v.kids ?? []).filter(k => k.shown);
  /* a row of rack groups is the concept's group row; groups switched by a pill line up as Latent & correction's */
  if (kids.length && kids.every(k => k.part === 'group')) return <div className={`ad-groups${kids.every(k => k.inner[0] === 'pill') ? ' lat-groups' : ''}`}>{v.children}</div>;
  /* a pad with a column beside it (Performance trades: the pad, then its gauge over its knob) is the concept's .padrow
     and .padside (25-module-perf.css; package 3a) */
  if (kids.length === 2 && kids[0]!.part === 'xyPad' && kids[1]!.part === 'row') return <div className="padrow">{v.children}</div>;
  if (kids.length === 2 && kids[0]!.part === 'speedGauge' && kids[1]!.part === 'knob') return <div className="padside">{v.children}</div>;
  /* a row of knobs beside a rack of switches shares one line (the Flow renderer's .flow-row, D-11; package 3c) */
  if (kids.length === 2 && kids[0]!.part === 'row' && kids[1]!.part === 'rack') return <div className="flow-row">{v.children}</div>;
  /* a row of plugin pickers is the Sampler's .picks (package 3a) */
  if (kids.length && kids.every(k => k.part === 'pick')) return <div className="picks">{v.children}</div>;
  return <div className="knobs">{v.children}</div>;
}
export const Rack = ({ children }: PartView) => <div className="pextra rackrow">{children}</div>;
/* the Sampler's well: its pickers stand together beside its numbers in .picks (E-20, S-30; package 3a) */
export function Well(v: PartView) {
  const kids = v.kids ?? [], items = itemsOf(v.children), picks = kids.map((k, i) => (k.shown && k.part === 'pick' ? i : -1)).filter(i => i >= 0);
  /* beside one or two numbers (Steps, and Guidance where it applies), as the concept's well holds them */
  const rest = kids.filter(k => k.shown && k.part !== 'pick');
  if (!picks.length || rest.length > 2 || rest.some(k => k.part !== 'numberBox' && k.part !== 'knob')) return <div className="knobs samp-strip">{v.children}</div>;
  return <div className="knobs samp-strip with-picks">{items.filter((_, i) => !picks.includes(i))}<div className="picks">{picks.map(i => items[i])}</div></div>;
}
export const Lcd = ({ children }: PartView) => <div className="lcd">{children}</div>;
export const Foot = ({ children }: PartView) => <div className="tile-foot">{children}</div>;

export function Labelled(v: PartView) {
  const say = useSay(), what = v.kids?.find(k => k.shown)?.part ?? '';
  /* its tip: its own words, else why it (or what it holds) stands back */
  /* a pairing with no label of its own takes the label of what it holds (a plugin's controls) */
  const kid = v.kids?.find(k => k.shown), why = v.dim ?? kid?.dim, label = v.label !== undefined ? say(v.label) : kid?.label !== undefined ? say(kid.label) : '', tip = v.words.tip ? say(v.words.tip) : why ? say(why) : undefined;
  switch (v.surface) {
    case 'pedal': case 'unit': case 'lcd':
      return <div className="lcdsel" title={tip}>{v.children}<span className="kl">{label}</span></div>;
    case 'lcdrow':
      /* a setting that stands back here is held out of sight by its row (lcd-trim held), and says nothing itself */
      return what === 'token' ? <span className="foot-ctl" title={v.words.tip ? say(v.words.tip) : undefined}><span className="kl">{label}</span>{v.children}</span> : <span title={tip}>{label} {v.children}</span>;
    case 'group':
      if (what === 'token') return <span className="tk-pair" title={tip}><span className="kl">{label}</span>{v.children}</span>;
      if (what === 'segmented') return <div className="ad-foot" title={tip}><span className="kl">{label}</span>{v.children}</div>;
      break;
    case 'plugin':
      return <span className="pc-i" title={tip}><span className="kl">{label}</span>{v.children}</span>;
    case 'deck': return <DeckRow v={v} label={label} />;
  }
  if (v.opts.look === 'strip') return <label className="schctl" title={tip}><span className="kl">{label}</span>{v.children}</label>;
  /* a caption above a text field, with what follows the field at its foot (Latent & correction's timesteps) */
  if (what === 'textField') { const [field, ...rest] = itemsOf(v.children); return <label className="ts"><span className="kl" title={tip}>{label}</span>{field}<span className="ts-foot">{rest}</span></label>; }
  /* a caption above what it holds: a strip of switches (the plans' Endings), a row of dice (Re-planning) */
  if (what === 'rack') return <div className="endings" title={tip}><span className="kl">{label}</span>{v.children}</div>;
  if (what === 'dice') return <div className="dice-cap" title={tip}><span className="kl">{label}</span>{v.children}</div>;
  /* a choice and the notes that say what it does and needs: the concept's .ctl-row, its notes one line after it, in
     the warning's ink while a warning is among them (the lead sheet's lyric timing; package 3c) */
  if (what === 'segmented' && (v.kids ?? []).slice(1).length && (v.kids ?? []).slice(1).every(k => k.part === 'note')) {
    const [choice, ...notes] = itemsOf(v.children), kids = v.kids!.slice(1), warn = kids.some(k => k.shown && k.look === 'warn');
    const said = notes.flatMap((n, i) => (kids[i]?.shown ? [n] : [])), line = said.flatMap((n, i) => (i ? [' ', n] : [n]));
    return <div className="ctl-row" title={tip}><span className="kl">{label}</span>{choice}<span className={`lbl${warn ? ' warn' : ''}`}><NoteInline.Provider value={true}>{line}</NoteInline.Provider></span></div>;
  }
  return <span className="foot-ctl" title={tip}><span className="kl">{label}</span>{v.children}</span>;
}

/** A row of a deck (25-module-plans.js): its label in the first column and what it holds in the second; a picker's or
 *  a field's row belongs to the recorder above it and is indented; a picker that stands back is marked and says why on
 *  a line of its own. */
function DeckRow({ v, label }: { v: PartView; label: string }) {
  const say = useSay(), kids = (v.kids ?? []).filter(k => k.shown), what = kids[0]?.part ?? '';
  const sub = what === 'pick' || what === 'textField', off = what === 'pick' && kids[0]?.dim ? kids[0].dim : null;
  /* a row that holds several controls (whether or not each shows now) lays them out as a row */
  const cls = (v.kids ?? []).length > 1 ? ' deck-row' : what === 'pick' ? ' deck-lcd' : '';
  return (
    <>
      <span className={`deck-l${sub ? ' deck-sub' : ''}`} title={v.words.tip ? say(v.words.tip) : undefined}>{label}</span>
      <span className={`deck-c${cls}${off ? ' is-off' : ''}`}>{v.children}</span>
      {off ? <><span /><span className="deck-c deck-why">{say(off)}</span></> : null}
    </>
  );
}

definePart('row', Row);
definePart('rack', Rack, { surface: 'rack' });
definePart('well', Well, { surface: 'well' });
definePart('lcd', Lcd, { surface: 'lcd' });
definePart('foot', Foot, { surface: 'foot' });
definePart('labelled', Labelled, { dims: true, kidsDim: true });
