// cartridges: the Adapters' cartridges (the concept's cartHTML, phaseHTML, stripInner and its face, 25-module-adapters.js).
// Each loaded adapter is a cassette: its kind's housing, its label art moving with its strength, its name, its strength
// fader (YuE2's AR adapter beside its Plan and Compose); in a stack its share, its active phase (Lab) and its section
// masks. Quick lays the first flat, its strength a choice row under it. Around them stand, where the concept draws
// them, the node's children: the stack bar above, the fader banks beside or under, the foot, and with one cartridge a
// column beside it with its words and its choices. Everything it shows is the core's (derive/cartridges.ts); a hand on
// a phase handle or a mask chip, a cartridge's ×, is handed to the core (act), and the + buttons run the node's actions
// (the adapter list, features/adapters.tsx).

import { Fragment, useRef, type CSSProperties, type KeyboardEvent, type MouseEvent, type PointerEvent, type ReactNode } from 'react';
import { definePart, Icon, useSay, type PartView } from '../kit/index.ts';
import { Fader } from './Fader.tsx';
import { CartArt } from './CartArt.tsx';
import { itemsOf } from './kids.ts';

type Carts = NonNullable<NonNullable<PartView['instrument']>['carts']>;
type CartM = Carts['carts'][number];

/* the node's children, by place (adapters.ts): 0 Quick's strength row, 1-4 the stack bar, 5-6 the banks, 7 on the foot */
const STACK = [1, 2, 3, 4], BANKS = [5, 6], FOOT = 7;

export function Cartridges(v: PartView) {
  const say = useSay(), m = v.instrument?.carts, items = itemsOf(v.children), kids = v.kids ?? [];
  const paint = useRef<{ w: number; from: string } | null>(null), ph = useRef<{ i: number; h: 0 | 1; from: [number, number] } | null>(null);
  if (!m) return null;
  const shown = (i: number) => !!kids[i]?.shown;
  /* the stack switch, where the node binds it (Quick's note carries it) */
  const stack = v.params.find(p => p.key === v.reads[0]?.key);
  const lyrics = String(v.reads[4]?.value ?? '');
  /* focus, after a cartridge leaves, on the next × or the empty slot (L-07) */
  const remove = (e: MouseEvent<HTMLButtonElement>, i: number) => {
    const body = e.currentTarget.closest('.tile-body');
    v.act('remove', i);
    requestAnimationFrame(() => body?.querySelector<HTMLElement>('[data-cartx], [data-cartadd]')?.focus({ preventScroll: true }));
  };

  /* a cartridge's fader is the fader part, given its own parameter, its place and its name */
  const fader = (f: CartM['faders'][number]) => {
    const p = v.params.find(x => x.key === f.key);
    if (!p) return null;
    const { title: _t, label: _l, ...rest } = v; void _t; void _l;
    return <Fader key={f.key} {...rest} params={[p]} control={{ fader: f.fader }} {...(f.label !== null ? { label: f.label } : {})} aria={f.aria} />;
  };
  const phase = (c: CartM) => {
    const x = c.phase!;
    const place = (e: PointerEvent<HTMLElement>, el: HTMLElement) => { const r = el.getBoundingClientRect(); return (e.clientX - r.left - 3) / (r.width - 6); };
    const down = (e: PointerEvent<HTMLDivElement>) => {
      e.preventDefault(); const el = e.currentTarget, f = place(e, el);
      el.setPointerCapture(e.pointerId);
      const h: 0 | 1 = Math.abs(f * 100 - x.a) < Math.abs(f * 100 - x.b) ? 0 : 1;
      ph.current = { i: c.i, h, from: [x.a, x.b] }; v.gesture.begin('phase');
      v.act('phase', { i: c.i, h, f, from: ph.current.from });
    };
    const move = (e: PointerEvent<HTMLDivElement>) => { const d = ph.current; if (d && e.currentTarget.hasPointerCapture(e.pointerId)) v.act('phase', { i: d.i, h: d.h, f: place(e, e.currentTarget), from: d.from }); };
    const up = () => { if (ph.current) { ph.current = null; v.gesture.end('phase'); } };
    const key = (h: 0 | 1) => (e: KeyboardEvent<HTMLElement>) => {
      if (!/^(Arrow(Left|Right|Up|Down)|Page(Up|Down)|Home|End)$/.test(e.key)) return;
      e.preventDefault(); e.stopPropagation(); v.act('phase', { i: c.i, h, key: e.key });
    };
    return (
      <div className="phase" data-phase={c.i} style={{ '--a': x.a, '--b': x.b } as CSSProperties} role="group" aria-label={say(x.group)} data-tip-side="beside" title={say(x.title)}
        onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}>
        <i />
        <b style={{ '--v': x.a } as CSSProperties} data-ph="0" tabIndex={0} role="slider" aria-label={say(x.start.aria)} aria-valuemin={0} aria-valuemax={x.start.max} aria-valuenow={x.a} aria-valuetext={say(x.start.text)} onKeyDown={key(0)} />
        <b style={{ '--v': x.b } as CSSProperties} data-ph="1" tabIndex={0} role="slider" aria-label={say(x.end.aria)} aria-valuemin={x.end.min} aria-valuemax={100} aria-valuenow={x.b} aria-valuetext={say(x.end.text)} onKeyDown={key(1)} />
        <span><em>{say(x.word)}</em>{x.span}</span>
        {c.eff ? <span className="eff" data-eff={c.i}>{say(c.eff)}</span> : null}
      </div>
    );
  };
  /* a press steps a section's mask; a drag across the strip paints the first chip's new level on the rest (L-09) */
  const masks = (c: CartM) => (
    <div className="secband" data-secband={c.i} role="group" aria-label={say(m.masksLabel)}>
      {c.masks!.map(k => (
        <button key={k.j} type="button" className={k.lv} data-mask={`${k.i}-${k.j}`} style={{ '--c': k.color, '--n': k.n } as CSSProperties} aria-label={say(k.aria)} title={say(k.title)}
          onPointerDown={e => { e.preventDefault(); paint.current = { w: k.next, from: lyrics }; v.gesture.begin('mask'); v.act('mask', { i: k.i, j: k.j, w: k.next, from: lyrics }); const end = () => { paint.current = null; v.gesture.end('mask'); removeEventListener('pointerup', end); }; addEventListener('pointerup', end); }}
          onPointerOver={e => { const p = paint.current; if (p && e.buttons) v.act('mask', { i: k.i, j: k.j, w: p.w, from: p.from }); }}
          onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); v.act('mask', { i: k.i, j: k.j, w: k.next }); } }}>{k.tag}</button>
      ))}
    </div>
  );
  const cart = (c: CartM): ReactNode => {
    const plannerCart = c.i < 0, changed = c.ctl ? v.params.find(p => p.key === c.ctl)?.changed : false;
    const art = c.art, key = c.faders[0]?.key;
    return (
      <div key={c.i} className={`cart t-${c.cls} h-${c.housing}`} title={say(c.title)}>
        <span className={`slot${c.ctl ? ' cdot' : ''}${changed ? ' changed' : ''}`} data-ctl={c.ctl ?? undefined}>{say(c.label)}</span>
        {plannerCart
          ? <button type="button" className="x" data-setv={`${c.ctl}|None`} aria-label={say(c.remove)} onClick={() => v.act('planner', '')}><Icon name="x" size={12} /></button>
          : <button type="button" className="x" data-cartx={c.i} aria-label={say(c.remove)} onClick={e => remove(e, c.i)}><Icon name="x" size={12} /></button>}
        <div className={`label-art${art.still ? ' still' : ''}`} data-adart={key} style={(art.spin ? { '--p': art.p, '--spin': art.spin } : { '--p': art.p }) as unknown as CSSProperties}><CartArt a={art} /></div>
        <span className="lab" title={c.full}><span className="lab-n">{c.name.map((s, i) => <Fragment key={i}>{i ? <wbr /> : null}{s}</Fragment>)}</span><em>{say(c.em)}</em></span>
        {c.split ? <div className="fdr-pair">{c.faders.map(f => fader(f))}</div> : c.faders.length ? fader(c.faders[0]!) : c.note !== null ? <span className="pl-note">{say(c.note)}</span> : null}
        {c.eff && !c.phase ? <span className="eff" data-eff={c.i}>{say(c.eff)}</span> : null}
        {c.phase ? phase(c) : null}
        {c.masks ? masks(c) : null}
      </div>
    );
  };

  /* Quick: the first cartridge lying flat, its strength a choice row under it, the rest said in a line (G-D-36, Q-18) */
  if (m.look === 'flat') {
    const c = m.carts[0];
    return (
      <>
        {c ? (
          <div className={`qcart t-${c.cls}`} title={say(c.title)}>
            <div className={`label-art${c.art.still ? ' still' : ''}`} data-adart={c.faders[0]?.key} style={{ '--p': c.art.p, '--spin': c.art.spin } as CSSProperties}><CartArt a={c.art} /></div>
            <span className="lab" title={c.full}><span className="lab-n">{c.name.map((s, i) => <Fragment key={i}>{i ? <wbr /> : null}{s}</Fragment>)}</span><em>{say(c.em)}</em></span>
            <button type="button" className="x" data-cartx={c.i} aria-label={say(c.remove)} onClick={e => remove(e, c.i)}><Icon name="x" size={12} /></button>
          </div>
        ) : null}
        {items[0]}
        {m.more ? <p className={`q-more${stack?.changed ? ' changed' : ''}`} data-ctl={stack?.key}>{say(m.more)} <button type="button" className="linkbtn" data-tier="studio" onClick={() => v.run(2)}>{say(v.actions[2]?.label ?? '')}</button></p> : null}
      </>
    );
  }

  const add = (bar: boolean) => m.add ? <button type="button" className={`cart empty h-${m.add.housing}${bar ? ' bar' : ''}`} data-cartadd="" onClick={() => v.run(0)}>{say(m.add.label)}</button> : null;
  const planner = m.planner && 'cart' in m.planner ? cart(m.planner.cart) : null;
  const plannerAdd = (bar: boolean) => m.planner && !('cart' in m.planner) ? <button type="button" className={`cart empty h-plan${bar ? ' bar' : ''}`} data-plannerpick="" onClick={() => v.run(1)}>{say(v.actions[1]?.label ?? m.planner.label)}</button> : null;
  const banks = BANKS.map(i => items[i]), hasBank = BANKS.some(shown);
  const unused = m.unused ? (
    <div className="unused"><span className="lbl">{say(m.unused.text)}</span>{m.unused.stack && v.actions[3] ? <button type="button" className="sbtn" onClick={() => v.run(3)}>{say(v.actions[3].label)}</button> : null}</div>
  ) : null;
  const foot = <div className="tile-foot" data-adfoot="">{items.slice(FOOT)}</div>;
  return (
    <>
      {STACK.some(shown) ? <div className="stackbar">{STACK.map(i => items[i])}</div> : null}
      <div className={`rig-wrap${hasBank ? ' has-bank' : ''}`}>
        <div className="rig-left">
          <div className="carts">{m.carts.map(cart)}{m.one ? null : add(m.add?.bar ?? false)}{m.one && !planner ? null : planner ?? plannerAdd(m.planner && !('cart' in m.planner) ? m.planner.bar : false)}</div>
        </div>
        {m.one && m.side ? (
          <div className="rig-right">
            {add(true)}{planner ? null : plannerAdd(true)}
            <div className="rig-side">
              <p>{say(m.side.desc)}</p>
              {m.side.second ? <button type="button" className="sbtn" data-cartadd2="" title={say(m.side.second.tip)} onClick={() => v.run(0)}>{say(m.side.second.label)}</button> : null}
              {m.side.note !== null ? <span className="lbl">{say(m.side.note)}</span> : null}
              {unused}
              {m.sideFoot ? foot : null}
            </div>
            {banks}
          </div>
        ) : banks}
      </div>
      {m.one ? null : unused}
      {m.sideFoot ? null : foot}
      {m.away ? <p className="lbl ad-away">{say(m.away)}</p> : null}
    </>
  );
}

definePart('cartridges', Cartridges, { surface: 'foot' });
