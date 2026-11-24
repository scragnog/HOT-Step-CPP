// The lead-sheet preview (the concept's 34-score-modal.js, #scoreModal): the named action scorePreview, which the ABC
// foot runs. With ABC on the sheet it shows that score as written, read from its own header, with Download and Close;
// without, it asks the planner for one (services/score.ts) and draws it on a staff (the core's scoreModel), with its
// verdict, the sections as a trail that plays from each, Play, Download, Cancel, Retry and Continue. What it shows,
// where every mark goes and every word are the core's; this file draws them, plays the notes, and moves the playhead.
//
// The playhead stands on the system being played, as tall as its five lines, and reaches each note as it sounds; at a
// system's end it moves to the next, and the paper scrolls to keep it in view. The section playing lights in the trail
// and on the staff (E-01). Under reduced motion it steps bar by bar. The dialog is modal: Tab goes round it, Escape and
// a press on the scrim close it, and focus goes back where it was. Package 8's layers may take its opening and closing
// over (as package 7's popover).

import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type KeyboardEvent } from 'react';
import { createPortal } from 'react-dom';
import { SCORE_Q, msg } from '../../derive.ts';
import { close, open, save, useView, type View } from '../score/state.ts';
import { useDesk } from '../store/desk.ts';
import { defineAction } from '../register/actions.ts';
import { fill } from '../register/slots.ts';
import { scoreRuns } from '../services/score.ts';
import { Icon, Glyph, ui, useSay } from '../kit/index.ts';
import { reduced } from '../parts/motion.ts';

defineAction('scorePreview', () => { void open(); });

/** Plays the score from a bar, and moves the playhead; stops when done or asked. */
function usePlayer(v: View) {
  const [at, setAt] = useState<{ x: number; y: number; si: number; row: number } | null>(null);
  const run = useRef<{ ac: AudioContext; t: number } | null>(null);
  const stop = () => { const r = run.current; run.current = null; if (r) { void r.ac.close?.(); cancelAnimationFrame(r.t); clearTimeout(r.t); } setAt(null); };
  useEffect(() => stop, [v]);
  const play = (from: number) => {
    stop();
    if (!v || !('score' in v)) return;
    const AC = (globalThis as unknown as { AudioContext?: typeof AudioContext }).AudioContext;
    if (!AC) { useDesk.getState().showToast({ text: msg('score.noAudio') }); return; }
    const { m } = v, b0 = from * 4, ac = new AC(), t0 = ac.currentTime + 0.05, q = SCORE_Q;
    m.freqs.slice(b0).forEach((f, j) => {
      const o = ac.createOscillator(), g = ac.createGain(); o.type = 'triangle'; o.frequency.value = f;
      g.gain.setValueAtTime(0, t0 + j * q); g.gain.linearRampToValueAtTime(0.18, t0 + j * q + 0.02); g.gain.exponentialRampToValueAtTime(0.001, t0 + j * q + q * 0.95);
      o.connect(g).connect(ac.destination); o.start(t0 + j * q); o.stop(t0 + (j + 1) * q);
    });
    const total = m.play.length, start = performance.now() + 50, r = { ac, t: 0 }; run.current = r;
    const place = (B: number) => {
      const i = Math.min(total - 1, Math.floor(B)), fr = B - i, p = m.play[i]!, n = m.play[i + 1];
      const x = reduced() ? p.x : p.x + ((n && n.row === p.row ? n.x : p.rowEnd) - p.x) * Math.max(0, fr);
      setAt({ x, y: p.y, si: p.si, row: p.row });
    };
    const step = () => {
      if (run.current !== r) return;
      const B = b0 + Math.max(0, (performance.now() - start) / 1000 / q);
      if (B >= total) { stop(); return; }
      place(reduced() ? Math.floor(B / 4) * 4 : B);
      r.t = reduced() ? setTimeout(step, Math.max(16, ((Math.floor(B / 4) + 1) * 4 - B) * q * 1000)) as unknown as number : requestAnimationFrame(step);
    };
    place(b0); step();
  };
  return { at, play, stop, playing: at !== null };
}

export function ScorePreview() {
  const say = useSay(), v = useView(), host = document.getElementById('scoreModal'), card = useRef<HTMLDivElement>(null), paper = useRef<HTMLDivElement>(null);
  const p = usePlayer(v);
  useLayoutEffect(() => { const h = document.getElementById('scoreModal'); if (h) h.hidden = !v; }, [v]);
  /* focus on its way on (Continue; Retry when the plan never closes, B-47; Close for a pasted sheet) */
  const kind = v ? ('score' in v ? `s${v.score.seed}` : 'p') : '';
  useLayoutEffect(() => {
    if (!v || !card.current) return;
    card.current.querySelector<HTMLElement>('score' in v ? (v.m.run ? '[data-scoreretry]' : '[data-scorego]') : '.sc-acts [data-scoreclose]')?.focus({ preventScroll: true });
    /* each rehearsal mark gets a backing as wide as its words */
    card.current.querySelectorAll<SVGTextElement>('.staff text.sc-sec[data-si]').forEach(t => {
      const r = t.previousElementSibling; try { const b = t.getBBox(); if (r && b.width) { r.setAttribute('width', String(b.width + 8)); r.setAttribute('x', String(b.x - 4)); r.setAttribute('y', String(b.y - 2)); r.setAttribute('height', String(b.height + 4)); } } catch { /* not drawn */ }
    });
  }, [kind]); // eslint-disable-line react-hooks/exhaustive-deps
  /* the paper follows the playhead from system to system */
  const row = p.at?.row ?? -1;
  useEffect(() => {
    const pp = paper.current, svg = pp?.querySelector('.staff'); if (!pp || !svg || row < 0 || !v || !('score' in v)) return;
    const k = svg.getBoundingClientRect().height / v.m.staff.H, R = v.m.rows[row]!, y = pp.scrollTop + svg.getBoundingClientRect().top - pp.getBoundingClientRect().top + R.top * k;
    if (y < pp.scrollTop || y + R.h * k > pp.scrollTop + pp.clientHeight) pp.scrollTo({ top: Math.max(0, y - 12), behavior: reduced() ? 'auto' : 'smooth' });
  }, [row, v]);
  useEffect(() => {
    if (!v) return;
    const esc = (e: globalThis.KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close(); } };
    document.addEventListener('keydown', esc, true);
    return () => document.removeEventListener('keydown', esc, true);
  }, [v]);
  if (!host || !v) return null;
  /* Tab goes round the dialog */
  const tab = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== 'Tab' || !card.current) return;
    const f = [...card.current.querySelectorAll<HTMLElement>('button, [tabindex="0"]')].filter(x => !x.hasAttribute('disabled'));
    if (!f.length) return;
    const i = f.indexOf(document.activeElement as HTMLElement), n = e.shiftKey ? (i <= 0 ? f.length - 1 : i - 1) : (i < 0 || i === f.length - 1 ? 0 : i + 1);
    e.preventDefault(); f[n]!.focus();
  };
  const x = <button type="button" className="icon-btn" data-scoreclose="" aria-label={say(ui('score.close'))} onClick={close}><Icon name="x" size={15} /></button>;
  let body;
  if ('pasted' in v) {
    const s = v.pasted;
    body = (
      <>
        <div className="sc-head"><div><h3>{s.title || say(msg('score.untitled'))}</h3><span className="sc-meta">{say(s.meta)}</span></div>{x}</div>
        <div className="sc-paper" tabIndex={0} aria-label={say(ui('score.paper'))}><pre className="sc-abc">{s.abc}</pre></div>
        <div className="sc-acts"><button type="button" className="small-btn" data-scoredlabc="" onClick={() => save(s.file, s.abc)}>{say(ui('score.download'))}</button><span className="sc-gap" />
          <button type="button" className="small-btn primary" data-scoreclose="" onClick={close}>{say(ui('score.close'))}</button></div>
      </>
    );
  } else {
    const { m, score } = v, at = p.at;
    body = (
      <>
        <div className="sc-head"><div><h3>{m.title}</h3><span className="sc-meta">{say(m.meta)}</span></div><span className={`sc-verdict ${m.verdict.cls}`}>{say(m.verdict.text)}</span>{x}</div>
        <p className="sc-why">{say(m.why)}</p>
        <div className="sc-order" role="group" aria-label={say(msg('score.order'))}>
          {m.order.map((o, i) => [i ? <i key={`s${i}`} aria-hidden="true">›</i> : null,
            <button key={i} type="button" className={at?.si === i ? 'cur' : undefined} data-scoresec={i} style={{ '--c': o.colour } as CSSProperties} title={say(o.tip)} onClick={() => { const b = score.meas.findIndex(mm => mm.si === i); if (b >= 0) p.play(b); }}>{o.name}</button>])}
        </div>
        <div ref={paper} className={`sc-paper${at ? ' playing' : ''}`}>
          <svg className="staff" viewBox={`0 0 ${m.staff.W} ${m.staff.H}`} aria-label={say(m.staff.aria)}>
            {m.staff.marks.map((k, i) => {
              const cls = [k.cls, k.si !== undefined && at?.si === k.si ? 'cur' : ''].filter(Boolean).join(' ') || undefined;
              const style = k.colour ? (k.tag === 'rect' ? { fill: k.colour } : { fill: k.colour }) : undefined;
              const props = { key: i, className: cls, ...k.a, ...(k.si !== undefined ? { 'data-si': k.si } : {}), style };
              if (k.tag === 'text') return <text {...props}>{say(k.text)}</text>;
              if (k.tag === 'line') return <line {...props} />;
              if (k.tag === 'rect') return <rect {...props} />;
              if (k.tag === 'circle') return <circle {...props} />;
              return <ellipse {...props} />;
            })}
            <rect className="st-head" x={at ? at.x.toFixed(2) : 0} y={at ? at.y : m.staff.head.y} width={2} height={m.staff.head.h} rx={1} aria-hidden="true" />
          </svg>
        </div>
        <div className="sc-acts">
          <button type="button" className="small-btn" data-scoreplay="" onClick={() => (p.playing ? p.stop() : p.play(0))}><Glyph name={p.playing ? 'stop' : 'play'} />{' '}{say(ui(p.playing ? 'score.stop' : 'score.play'))}</button>
          <button type="button" className="small-btn" data-scoredl="" onClick={() => save(m.file, score.abc)}>{say(ui('score.download'))}</button>
          <span className="sc-gap" />
          <button type="button" className="small-btn" data-scoreclose="" onClick={close}>{say(ui('score.cancel'))}</button>
          <button type="button" className={`small-btn${m.run ? ' primary' : ''}`} data-scoreretry="" onClick={() => { p.stop(); void open(Math.floor(Math.random() * 2147483647)); }}>{say(ui('score.retry'))}</button>
          <button type="button" className={`small-btn${m.run ? '' : ' primary'}`} data-scorego="" onClick={() => { const sd = score.seed; close(); if (scoreRuns.run(sd)) useDesk.getState().showToast({ text: msg('score.continue', { seed: sd }) }); }}>{say(ui('score.go'))}</button>
        </div>
      </>
    );
  }
  return createPortal(
    <div className="sc-card" ref={card} role="dialog" aria-modal="true" aria-label={say(ui('score.dialog'))} onKeyDown={tab}>{body}</div>,
    host,
  );
}

/* a press on the scrim closes it, as it closes the Concept notes (C-39) */
document.getElementById('scoreModal')?.addEventListener('click', e => { if (e.target === e.currentTarget) close(); });
fill('overlays', ScorePreview);
