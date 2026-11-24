// buildBoard: Build by section (the concept's 25-module-build.js). Empty, it offers Plan the score; planned, the
// section strip (each section in its colour, kept ones checked, a keep sung to old words marked "!", a section added
// since the plan hatched), the banner while the lyrics differ from the plan, the section being built with its takes
// (each with its waveform, play, length, discard and Keep), the board's settings (its children, under the takes; the
// two a keep locks shown as read-outs with a padlock), and its foot (Re-plan, Start a new build, the count, Play kept,
// Render). What it shows and every word are the core's (instrument.build); the build session is the server's
// (services/build.ts): the board asks it for a plan, takes and a render, and hands each answer to the core
// (act('planned'), act('rendered')), which makes it the record's step. Picking a section and making takes are the
// session's, no steps; Keep, Discard, Re-plan, Keep building, Render and Start a new build are steps.
//
// Hearing a take (B-07): its play button runs a playhead across its waveform for the take's length, and playing one
// stops the other; Play kept (E-46) plays every keep in section order, lighting its section. Under reduced motion the
// playhead steps once a second. Focus stays on the control that had it across a redraw, and goes to making takes when
// a plan lands while it was on Plan (E-09).

import { Children, useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from 'react';
import { definePart, Glyph, Icon, useSay, type PartView } from '../kit/index.ts';
import { buildSession } from '../services/build.ts';
import { reduced } from './motion.ts';

/* the concept's seeded generator (00-util.js rng): a take's waveform is the same on every visit */
const rng = (seed: number) => () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
const bars = (w: number) => { const r = rng(w); return Array.from({ length: 24 }, () => 20 + Math.round(r() * 80)); };

interface Playing { take: string; t0: number; dur: number; queue: { take: string; section: string; dur: number }[]; section: string | null }

export function BuildBoard(v: PartView) {
  const say = useSay(), m = v.instrument?.build, root = useRef<HTMLDivElement>(null);
  const [play0, setPlay] = useState<Playing | null>(null), [pos, setPos] = useState(0), [busy, setBusy] = useState(false);
  const landing = useRef(false), focusKey = useRef<string | null>(null);
  /* the playhead: a frame at a time, a second at a time under reduced motion */
  useEffect(() => {
    const play = play0; if (!play) return;
    let raf = 0;
    const step = () => {
      const f = (performance.now() - play.t0) / 1000 / play.dur;
      if (f >= 1) {
        const [next, ...rest] = play.queue;
        setPos(0); setPlay(next ? { take: next.take, section: next.section, dur: next.dur, t0: performance.now(), queue: rest } : null); return;
      }
      setPos(reduced() ? Math.floor(f * play.dur) / play.dur : f);
      raf = reduced() ? setTimeout(step, 1000) as unknown as number : requestAnimationFrame(step);
    };
    step();
    return () => { cancelAnimationFrame(raf); clearTimeout(raf); };
  }, [play0]);
  /* a take that leaves the build (discarded, a re-plan, a new build) stops playing */
  const here = !!m && (m.takes.some(t => t.id === play0?.take) || m.keptOrder.some(k => k.take === play0?.take)), play = here ? play0 : null;
  /* focus stays where it was across a redraw; a plan that lands while focus was on Plan goes on to making takes */
  useLayoutEffect(() => {
    const box = root.current?.parentElement; if (!box || !m) return;
    if (landing.current && !m.planning && m.planned) {
      landing.current = false;
      const a = document.activeElement;
      if (!a || a === document.body || a.closest?.('.tile') === box.closest('.tile')) box.querySelector<HTMLElement>('[data-btake]')?.focus({ preventScroll: true });
    }
    const k = focusKey.current; focusKey.current = null;
    if (k && !box.contains(document.activeElement)) box.querySelector<HTMLElement>(k)?.focus({ preventScroll: true });
  });
  /* the section shown is the one the session looks at: it stays there after a keep (B-37) */
  const sess = m?.session, curId = m?.cur?.id, looked = m?.looked;
  useEffect(() => { if (sess && curId && looked !== curId) buildSession.look(sess, curId); }, [sess, curId, looked]);
  if (!m) return null;
  const kids = Children.toArray(v.children), keep = (sel: string) => { focusKey.current = sel; };
  const lockedAt = new Map((m.locked ?? []).map(l => [l.at, l]));
  const ctl = m.ctl ? (
    <div className="b-ctl">
      {kids.map((k, i) => { const l = lockedAt.get(i); return l ? (
        <span key={i} className="foot-ctl locked"><span className="kl" title={l.ktip ? say(l.ktip) : undefined}>{say(l.label)}</span>
          <span className="tok-ro" tabIndex={0} role="note" title={say(l.tip)} aria-label={say(l.aria)}>{l.text}<Glyph name="lock" /></span></span>
      ) : k; })}
      {m.locked ? <span className="lbl b-locknote">{say({ id: 'build.locknote' })}</span> : null}
    </div>
  ) : null;
  const doPlan = () => {
    setPlay(null);
    landing.current = true;
    void buildSession.plan(m.plan.sections).then(r => { v.act('planned', r); });
  };
  if (!m.planned || m.planning) {
    return (
      <>
        <div ref={root} className="b-empty">
          <div><b>{say({ id: 'build.empty.title' })}</b><p>{say({ id: 'build.empty.text' })}</p></div>
          <button type="button" className="small-btn primary" data-bplan="" aria-busy={m.planning ? true : undefined} onClick={doPlan}>
            {m.planning ? <><i className="spin" aria-hidden="true" />{say({ id: 'build.planning' })}</> : say({ id: 'build.plan' })}</button>
        </div>
        {ctl}
      </>
    );
  }
  const session = m.session!, cur = m.cur;
  const look = (id: string) => { buildSession.look(session, id); keep(`.b-src [data-bsec="${CSS.escape(id)}"]`); };
  const toggle = (take: string, dur: number) => { setPos(0); setPlay(p => (p?.take === take && !p.queue.length && !p.section ? null : { take, dur, t0: performance.now(), queue: [], section: null })); };
  const playKept = () => { setPos(0); setPlay(p => { if (p?.section) return null; const [a, ...rest] = m.keptOrder; return a ? { take: a.take, section: a.section, dur: a.dur, t0: performance.now(), queue: rest } : null; }); };
  const replan = <button type="button" className="small-btn" data-breplan="" title={say(m.replanTip)} onClick={() => { keep('[data-breplan]'); doPlan(); }}>{say({ id: 'build.replan' })}</button>;
  const render = () => {
    if (!m.render || busy) return;
    setPlay(null); setBusy(true);
    void buildSession.render(session, { title: m.render.title, style: say(m.render.style), dur: m.render.dur, takes: m.render.takes }).then(r => { setBusy(false); v.act('rendered', { song: r.song, title: m.render!.title }); });
  };
  return (
    <>
      <div ref={root} className="b-src" role="group" aria-label={say({ id: 'build.sections' })}>
        {m.sections.map(x => (
          <button key={x.id} type="button" className={[x.kept ? 'kept' : '', x.stale ? 'stale' : '', x.cur ? 'cur' : '', x.fresh ? 'new' : '', play?.section === x.id ? 'playing' : ''].filter(Boolean).join(' ')}
            data-bsec={x.id} style={{ '--c': x.colour, flex: `${x.grow} 1 0` } as CSSProperties} aria-pressed={x.cur} aria-label={say(x.aria)} title={say(x.tip)} onClick={() => look(x.id)}>
            <span className="nm">{x.label}</span><span className="ab" aria-hidden="true">{x.ab}</span>{x.stale ? <b className="b-stale" aria-hidden="true">!</b> : x.kept ? <Icon name="check" size={12} /> : null}
          </button>
        ))}
      </div>
      {m.drift ? (
        <div className="b-drift" role="status"><span>{say(m.drift.text)}</span>{replan}
          {m.drift.goon ? <button type="button" className="small-btn" data-bgoon="" onClick={() => { keep('.b-src > button.cur'); v.act('relyric'); }}>{say({ id: 'build.goon' })}</button> : null}</div>
      ) : null}
      {cur ? (
        <div className="b-cur">
          <b style={{ color: cur.colour }}>{cur.label}</b><span className="lbl">{say(cur.says)}</span>
          <button type="button" className={`small-btn${cur.make.primary ? ' primary' : ''}`} data-btake="" onClick={() => { keep('[data-btake]'); buildSession.takes(session, cur.make.ask.section, cur.make.ask.words, cur.make.ask.n, cur.make.ask.ownChords); }}>{say(cur.make.label)}</button>
          {cur.next ? <button type="button" className="small-btn" data-bsec={cur.next.id} onClick={() => look(cur.next!.id)}>{say(cur.next.label)}</button> : null}
        </div>
      ) : null}
      <div className="b-takes">
        {m.takes.map(t => {
          const on = play?.take === t.id;
          return (
            <div key={t.id} className={['b-take', t.keep ? 'keep' : '', t.ready ? '' : 'busy', on ? 'playing' : ''].filter(Boolean).join(' ')} data-take={t.id} style={on ? { '--pos': `${pos * 100}%` } as CSSProperties : undefined}>
              <span className="b-wave" style={{ '--h': t.hue } as CSSProperties}>
                {t.ready ? <button type="button" className="b-play" data-bplay={t.id} aria-pressed={on} aria-label={say(on ? t.stop : t.play)} title={say({ id: on ? 'build.take.stop.tip' : 'build.take.play.tip' })} onClick={() => { keep(`[data-bplay="${t.id}"]`); toggle(t.id, t.dur); }}><Glyph name={on ? 'stop' : 'play'} size={14} /></button> : null}
                <span className="b-bars">{bars(t.w).map((h, i) => <i key={i} style={{ height: `${h}%` }} />)}<i className="b-head" aria-hidden="true" /></span>
              </span>
              <b>{say(t.name)}</b>
              {t.ready ? <button type="button" className="x b-x" data-bdrop={t.id} aria-label={say(t.discard)} title={say({ id: 'build.take.discard.tip' })} onClick={() => { keep('[data-bdrop], [data-btake]'); v.act('discard', t.id); }}><Icon name="x" size={12} /></button> : null}
              <small title={say(t.tip)}>{say(t.meta)}</small>
              {t.ready ? <button type="button" className={`small-btn${t.keep ? ' primary' : ''}`} data-bkeep={t.id} aria-pressed={t.keep} onClick={() => { keep(`[data-bkeep="${t.id}"]`); v.act('keep', t.id); }}>{say({ id: t.keep ? 'build.kept' : 'build.keep' })}</button> : <span className="b-slot" aria-hidden="true" />}
            </div>
          );
        })}
      </div>
      {ctl}
      <div className="tile-foot b-foot">
        {m.drift ? null : replan}
        {m.foot.reset ? <button type="button" className="small-btn" data-bnew="" onClick={() => { keep('[data-bplan]'); setPlay(null); v.act('reset'); }}>{say({ id: 'build.new' })}</button> : null}
        <span className="lbl b-count">{say(m.foot.count)}</span>
        {m.keptOrder.length ? <button type="button" className="small-btn" data-bplaykept="" aria-pressed={!!play?.section} title={say({ id: 'build.playKept.tip' })} onClick={playKept}><Glyph name={play?.section ? 'stop' : 'play'} />{say({ id: play?.section ? 'build.playKept.stop' : 'build.playKept' })}</button> : null}
        <button type="button" className="small-btn primary" data-brender="" disabled={!m.foot.canRender} aria-busy={busy ? true : undefined} onClick={() => { keep('[data-brender]'); render(); }}>{say(m.foot.render)}</button>
      </div>
    </>
  );
}

definePart('buildBoard', BuildBoard);
