// The Timbre reference tile's own parts (the concept's 25-module-timbre.js):
//
//   recentRefs   the recent references under their label: the two latest and a way to the rest, never a ragged stack
//                of chips (L-21, Q-28); one loads it as the timbre reference (its file state's `load` step). The list is
//                the server's reference tracks (services/references.ts)
//   timbreBlend  the reference blend as two overlapping circles: the reference dragged into your sound, sideways or
//                with the value contract's keys; it stands back with its reason while there is no track to blend
//
// The blend's percent and every step are the core's.

import { useRef, type CSSProperties, type KeyboardEvent, type PointerEvent } from 'react';
import { definePart, onValueKey, onValueKeyUp, ui, useSay, type PartView } from '../kit/index.ts';
import { useReferences } from '../services/references.ts';

export function RecentRefs(v: PartView) {
  const say = useSay(), refs = useReferences(), file = v.state.find(s => s.kind === 'file');
  if (!refs.length || !file) return null;
  /* the two latest and "N more…" when there are more than three */
  const shown = refs.length > 3 ? refs.slice(0, 2) : refs;
  const use = (r: string) => file.apply('load', r, { file: r });
  return (
    <div className="refs">
      <span className="kl">{say(ui('refs.recent'))}</span>
      {shown.map(r => <button key={r} type="button" className={`pref${file.value === r ? ' on' : ''}`} data-pref={r} title={say(ui('refs.use', { file: r }))} onClick={() => use(r)}>{r}</button>)}
      {refs.length > shown.length ? <button type="button" className="pref refs-more" data-refsmore="" aria-haspopup="dialog">{say(ui('refs.more', { n: refs.length - shown.length }))}</button> : null}
    </div>
  );
}

export function TimbreBlend(v: PartView) {
  const say = useSay(), p = v.params[0], drag = useRef<{ x: number; from: number; shift: boolean } | null>(null);
  if (!p) return null;
  const down = (e: PointerEvent<HTMLDivElement>) => {
    e.preventDefault(); e.currentTarget.setPointerCapture(e.pointerId); e.currentTarget.focus({ preventScroll: true });
    drag.current = { x: e.clientX, from: p.value as number, shift: e.shiftKey }; v.gesture.begin('drag');
  };
  const move = (e: PointerEvent<HTMLDivElement>) => {
    const d = drag.current; if (!d) return;
    /* Shift drags a tenth as far, from where the blend is when it is pressed */
    if (e.shiftKey !== d.shift) { d.from = p.value as number; d.x = e.clientX; d.shift = e.shiftKey; }
    p.scrub(d.from, e.clientX - d.x, e.shiftKey);
  };
  const up = () => { if (!drag.current) return; drag.current = null; v.gesture.end('drag'); };
  const off = v.dim !== null;
  return (
    <>
      <div className={`venn${off ? ' is-off' : ''}`} data-venn={p.key} data-scrub={p.key} role="slider" tabIndex={0} aria-label={say(ui('blend.label'))}
        aria-valuemin={p.range?.[0]} aria-valuemax={p.range?.[1]} aria-valuenow={p.value as number} aria-valuetext={say(ui('blend.valuetext', { pct: v.instrument?.blend?.pct ?? '' }))} style={{ '--p': String(p.at ?? 0) } as CSSProperties} title={say(ui('blend.title'))}
        onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}
        onKeyDown={(e: KeyboardEvent<HTMLDivElement>) => onValueKey(e, p, v.gesture)} onKeyUp={() => onValueKeyUp(v.gesture)} onDoubleClick={() => p.reset()}>
        <span className="venn-scale" aria-hidden="true" />
        <span className="vc b"><b className="venn-pct" data-vennpct="" aria-hidden="true">{say(v.instrument?.blend?.pct)}</b><i className="grip" aria-hidden="true" /></span>
        <span className="vc a" />
        <span className="venn-leg"><span><i className="sw b" />{say(ui('blend.ref'))}</span><span><i className="sw a" />{say(ui('blend.you'))}</span></span>
      </div>
      <div className="venn-cap">
        <span className={`kl cdot${p.changed ? ' changed' : ''}`} data-ctl={p.key}>{say(v.label)}</span>
        {off ? <span className="off-why">{say(v.dim!)}</span> : null}
      </div>
    </>
  );
}

definePart('recentRefs', RecentRefs);
definePart('timbreBlend', TimbreBlend, { dims: true });
