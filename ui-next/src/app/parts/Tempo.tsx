// Tempo's own instruments (the concept's 25-module-tempo.js):
//
//   keyWheel  the circle of fifths beside the LCD it holds (.tempo): majors outside, minors inside, Auto in the middle,
//             a radio group the arrows walk from the key in force (the walk is the core's: act('walk')); the chosen
//             segment fills and nudges 2px outward, then settles (G-D-25 d, G-M-12). With nothing to edit (Quick, an
//             engine without a key) the LCD stands alone (.tempo.solo)
//   tapTempo  TAP: two taps or more within its run set the BPM from their spacing (act('tap')); the first says AGAIN.
//             A run of taps is one step
//   capChips  on engines where tempo and key go in the prompt: chips that put "120 BPM" or "D major" into the text it
//             edits as an item of their group, or take them out (act('chip'))
//
// What each draws (the segments, their spelling, the chips and which are in) is the core's (PartModel.instrument).

import { useEffect, useLayoutEffect, useRef, useState, type FocusEvent, type KeyboardEvent } from 'react';
import { definePart, ui, useSay, type PartView } from '../kit/index.ts';
import { motionVar, mtime, reduced } from './motion.ts';

/* the wheel's drawing (25-module-tempo.js wheelSVG): a 150-unit square, the majors' ring from 50 to 73, the minors'
   from 25 to 50 (25 deep, so their names read at 10px: Q-32), twelve segments from the top */
const C = 75, P = (r: number, a: number): [number, number] => [C + r * Math.cos(a), C + r * Math.sin(a)];
function arc(r1: number, r2: number, a0: number, a1: number): string {
  const [x1, y1] = P(r2, a0), [x2, y2] = P(r2, a1), [x3, y3] = P(r1, a1), [x4, y4] = P(r1, a0);
  return `M${x1} ${y1}A${r2} ${r2} 0 0 1 ${x2} ${y2}L${x3} ${y3}A${r1} ${r1} 0 0 0 ${x4} ${y4}Z`;
}
const angles = (i: number) => { const a0 = (i * 30 - 105) * Math.PI / 180, a1 = (i * 30 - 75) * Math.PI / 180; return [a0, a1, (a0 + a1) / 2] as const; };
const WALK = /^(Arrow(Up|Down|Left|Right)|Home)$/;

export function KeyWheel(v: PartView) {
  const say = useSay(), w = v.instrument?.wheel, p = v.params[0];
  const svg = useRef<SVGSVGElement>(null), was = useRef<string | null>(null);
  const [ring, setRing] = useState('');
  const run = useRef<ReturnType<typeof setTimeout>>(undefined);
  const sel = w ? (w.auto.sel ? w.auto.value : w.segs.find(s => s.sel)?.value ?? null) : null;
  /* the chosen segment takes the focus a key moved it to, and fills in with its nudge */
  useLayoutEffect(() => {
    const el = svg.current, prev = was.current; was.current = sel;
    if (!el || prev === null || prev === sel) return;
    const s = el.querySelector<SVGGraphicsElement>('[aria-checked="true"]');
    if (s && el.contains(document.activeElement) && document.activeElement !== s) s.focus({ preventScroll: true });
    const path = el.querySelector<SVGPathElement>('path.sel');
    if (!path) return;
    if (reduced()) { path.animate([{ opacity: 0.5 }, { opacity: 1 }], { duration: 120 }); return; }
    const b = path.getBBox(), dx = b.x + b.width / 2 - C, dy = b.y + b.height / 2 - C, l = Math.hypot(dx, dy) || 1, o = `translate(${(dx / l * 2).toFixed(2)}px, ${(dy / l * 2).toFixed(2)}px)`;
    path.animate([{ opacity: 0.55, transform: 'none' }, { opacity: 1, transform: o, offset: 0.3 }, { opacity: 1, transform: 'none' }], { duration: mtime('--t-base') + 60, easing: motionVar('--e-out') });
  }, [sel]);
  useEffect(() => () => clearTimeout(run.current), []);
  if (!w || w.solo || !p) return <div className="tempo solo">{v.children}</div>;

  /* a run of keys within the parameter's run is one step (L-25) */
  const press = (fn: () => void) => { v.gesture.begin('run'); fn(); clearTimeout(run.current); run.current = setTimeout(() => v.gesture.end('run'), p.run ?? 600); };
  const key = (e: KeyboardEvent<SVGSVGElement>) => {
    const t = (e.target as Element).closest?.('[data-key]');
    if (!t) return;
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); press(() => p.set(t.getAttribute('data-key') ?? '')); return; }
    if (!WALK.test(e.key)) return;
    e.preventDefault(); press(() => v.act('walk', e.key));
  };
  /* keyboard focus on a segment: a ring drawn last, so its neighbours cannot paint over three of its sides */
  const focus = (e: FocusEvent<SVGSVGElement>) => setRing((e.target as Element).tagName === 'path' ? (e.target as Element).getAttribute('d') ?? '' : '');
  const blur = (e: FocusEvent<SVGSVGElement>) => { if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setRing(''); };
  const choose = (value: string) => p.set(value);
  return (
    <div className="tempo ">
      {v.children}
      <svg ref={svg} className="wheel" viewBox="0 0 150 150" role="radiogroup" aria-label={say(ui('wheel.label'))} onKeyDown={key} onFocus={focus} onBlur={blur}>
        {Array.from({ length: 12 }, (_, i) => {
          const [a0, a1, am] = angles(i), maj = w.segs.find(s => s.ring === 'maj' && s.i === i), min = w.segs.find(s => s.ring === 'min' && s.i === i);
          const [tx, ty] = P(61, am), [ux, uy] = P(40.5, am);
          return [maj ? <Seg key={`M${i}`} s={maj} d={arc(50, 73, a0, a1)} x={tx} y={ty + 3.5} choose={choose} /> : null,
            min ? <Seg key={`m${i}`} s={min} d={arc(25, 50, a0, a1)} x={ux} y={uy + 3.3} choose={choose} /> : null];
        })}
        <circle cx="75" cy="75" r="24" className={w.auto.sel ? 'sel' : ''} data-key={w.auto.value} role="radio" aria-checked={w.auto.sel} tabIndex={w.auto.sel ? 0 : -1} aria-label={say(ui('wheel.auto'))} onClick={() => choose(w.auto.value)} />
        <text x="75" y="79" textAnchor="middle" aria-hidden="true" className={w.auto.sel ? 'sel' : ''} style={{ fontSize: '11px' }}>{say(ui('wheel.autoText'))}</text>
        <path className="wring" aria-hidden="true" d={ring} />
      </svg>
    </div>
  );
}

type SegModel = NonNullable<NonNullable<PartView['instrument']>['wheel']>['segs'][number];
/** One segment and its name: the accidental a smaller raised glyph after the letter, a minor's "m" after it; a minor
 *  with an accidental set a little smaller, so neighbours at the foot of the ring keep 4px between them (Q-26). */
function Seg({ s, d, x, y, choose }: { s: SegModel; d: string; x: number; y: number; choose: (v: string) => void }) {
  const long = s.minor && !!s.acc;
  return (
    <>
      <path className={`${s.ring} ${s.sel ? 'sel' : ''}`} d={d} data-key={s.value} role="radio" aria-checked={s.sel} tabIndex={s.sel ? 0 : -1} aria-label={s.name} onClick={() => choose(s.value)}><title>{s.name}</title></path>
      <text x={x} y={y} textAnchor="middle" aria-hidden="true" className={s.sel ? 'sel' : ''} style={s.minor ? (long ? { fontSize: '8.5px', letterSpacing: '-.02em' } : { fontSize: '9.5px' }) : undefined}>
        {s.letter}{s.acc ? <tspan className="acc" dy="-3">{s.acc}</tspan> : null}{s.minor ? <tspan dy={s.acc ? '3' : undefined}>m</tspan> : null}
      </text>
    </>
  );
}

export function TapTempo(v: PartView) {
  const say = useSay(), p = v.params[0];
  const taps = useRef<number[]>([]), timer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const [armed, setArmed] = useState(false);
  useEffect(() => () => clearTimeout(timer.current), []);
  if (!p) return null;
  const tap = () => {
    const now = performance.now(), win = p.run ?? 2500;
    taps.current = [...taps.current.filter(t => now - t < win), now];
    const n = taps.current.length;
    setArmed(n === 1);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => { setArmed(false); v.gesture.end('tap'); }, win);
    /* the run of taps is one step (L-22) */
    v.gesture.begin('tap');
    if (n > 1) v.act('tap', (taps.current[n - 1]! - taps.current[0]!) / (n - 1));
  };
  return <button type="button" data-tap="" className={armed ? 'armed' : undefined} onClick={tap}>{say(ui(armed ? 'tap.again' : 'tap'))}</button>;
}

export function CapChips(v: PartView) {
  const say = useSay(), chips = v.instrument?.caps?.chips ?? [], p = v.params[0];
  if (!p) return null;
  return (
    <div className="cap-note">
      {`${say(v.label)} `}
      {chips.map(c => <button key={c.text} type="button" className="sbtn" data-capadd={c.text} aria-pressed={c.on} onClick={() => v.act('chip', c.text)}>{c.text}</button>)}
    </div>
  );
}

definePart('keyWheel', KeyWheel);
definePart('tapTempo', TapTempo);
definePart('capChips', CapChips);
