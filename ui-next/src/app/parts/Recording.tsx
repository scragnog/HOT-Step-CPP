// recordingSpan: the part of a recording, up to 30 seconds, that a new song continues from (the concept's
// 25-module-cont.js). One waveform drawing (G-D-17): 120 bars seeded by the file's name, the chosen part the same bars
// in amber over a 35% base, a grab handle on each edge, a time ruler under it. It is handled like a selection
// (G-B-01): a press inside it drags it whole, a press within 6px of an edge (its handle) moves that edge, a press
// outside draws a new one once the pointer has moved 3px; the cursor says which a press would do. Its keys keep the
// page's one contract (B-28): arrows move it a second, Shift with them moves its end, Page Up and Page Down by its
// length, Home and End to the file's ends; a double-click puts back the first 20 seconds. Where a hand takes it is the
// core's (act('move')); a drag is one step, each key another. Its play button plays the chosen part, the playhead
// running across it (E-47). A recording chosen or replaced while it stands brings focus to it (E-23).

import { useEffect, useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent } from 'react';
import { definePart, Icon, useSay, type PartView } from '../kit/index.ts';
import { reduced } from '../kit/motion.ts';

const rng = (seed: number) => () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
/** The file's waveform (25-module-cont.js fileBars): a random walk of loudness seeded by its name. */
function fileBars(name: string): number[] {
  const r = rng([...name].reduce((h, c) => (h * 31 + c.charCodeAt(0)) | 0, 7)); let env = 0.5;
  return Array.from({ length: 120 }, () => { env = Math.min(1, Math.max(0.25, env + (r() - 0.5) * 0.35)); return Math.round(12 + env * (40 + r() * 48)); });
}
type Mode = 'a' | 'z' | 'move' | 'new';
/* the bars and the selection are drawn 10px in from each end of the strip, so the pointer maps onto that track */
const track = (el: HTMLElement) => { const r = el.getBoundingClientRect(); return { left: r.left + 10, width: r.width - 20 }; };

export function RecordingSpan(v: PartView) {
  const say = useSay(), m = v.instrument?.span, wave = useRef<HTMLDivElement>(null);
  const drag = useRef<{ mode: Mode; x: number; t0: number; a0: number; z0: number; moved: boolean } | null>(null);
  const [grab, setGrab] = useState<Mode | null>(null), [played, setPlay] = useState<{ t0: number; file: string } | null>(null), [pos, setPos] = useState(0);
  const file = m?.file ?? '', was = useRef(file), play = played && played.file === file ? played.t0 : null;
  /* a recording chosen or replaced: focus goes to its waveform (E-23) */
  useEffect(() => { if (file && file !== was.current) wave.current?.focus({ preventScroll: true }); was.current = file; }, [file]);
  /* playing the chosen part: the playhead runs from its start to its end in its own time */
  const len = m ? m.z - m.a : 0;
  useEffect(() => {
    if (play === null) return;
    let t = 0;
    const step = () => { const f = (performance.now() - play) / 1000 / len; if (f >= 1) { setPlay(null); setPos(0); return; } setPos(reduced() ? Math.floor(f * len) / len : f); t = reduced() ? setTimeout(step, 1000) as unknown as number : requestAnimationFrame(step); };
    step();
    return () => { cancelAnimationFrame(t); clearTimeout(t); };
  }, [play, len]);
  if (!m) return null;
  const at = (x: number, el: HTMLElement) => { const r = track(el); return Math.min(1, Math.max(0, (x - r.left) / r.width)); };
  const hit = (x: number, el: HTMLElement): Mode => { const r = track(el), ax = r.left + m.a / m.dur * r.width, zx = r.left + m.z / m.dur * r.width; return Math.abs(x - ax) <= 6 ? 'a' : Math.abs(x - zx) <= 6 ? 'z' : x > ax && x < zx ? 'move' : 'new'; };
  const down = (e: PointerEvent<HTMLDivElement>) => {
    const el = e.currentTarget; e.preventDefault(); el.setPointerCapture(e.pointerId); el.focus({ preventScroll: true });
    const mode = hit(e.clientX, el); drag.current = { mode, x: e.clientX, t0: at(e.clientX, el), a0: m.a, z0: m.z, moved: false }; setGrab(mode);
  };
  const move = (e: PointerEvent<HTMLDivElement>) => {
    const el = e.currentTarget, d = drag.current;
    if (!d) { el.dataset.hit = hit(e.clientX, el); return; }
    if (d.mode === 'new' && Math.abs(e.clientX - d.x) < 3) return;
    if (!d.moved) { d.moved = true; v.gesture.begin('drag'); }
    v.act('move', { mode: d.mode, t: at(e.clientX, el), t0: d.t0, a0: d.a0, z0: d.z0 });
  };
  const up = () => { const d = drag.current; drag.current = null; setGrab(null); if (d?.moved) v.gesture.end('drag'); };
  const key = (e: KeyboardEvent<HTMLDivElement>) => { if (/^(Arrow(Left|Right|Up|Down)|Page(Up|Down)|Home|End)$/.test(e.key) && v.act('move', { key: e.key, shift: e.shiftKey })) e.preventDefault(); };
  const bars = fileBars(m.file), on = play !== null, head = m.a / m.dur + pos * len / m.dur;
  return (
    <>
      <div ref={wave} className={`contwave${grab ? ` grab g-${grab}` : ''}`} data-contwave="" tabIndex={0} role="slider" aria-valuemin={0} aria-valuemax={m.dur} aria-valuenow={m.a} aria-valuetext={say(m.text)}
        aria-label={say(m.aria)} title={say(m.tip)} style={{ '--a': m.pa, '--z': m.pz } as CSSProperties}
        onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up} onKeyDown={key} onDoubleClick={() => v.act('move', { reset: true })}>
        <span className="cw-bars">{bars.map((h, i) => <i key={i} style={{ height: `${h}%` }} />)}</span>
        <span className="cw-bars hi" aria-hidden="true">{bars.map((h, i) => <i key={i} style={{ height: `${h}%` }} />)}</span>
        <i className="cw-sel" />
        {on ? <i className="cw-head" aria-hidden="true" style={{ '--p': (head * 100).toFixed(3) } as CSSProperties} /> : null}
      </div>
      <div className="cw-ruler" aria-hidden="true">{m.ticks.map((t, i) => <span key={i} className={t.end ? 'end' : undefined} style={{ left: `${t.at}%` }}>{t.text}</span>)}</div>
      <div className="cap"><span>{say(m.cap)}</span>
        <button type="button" className="small-btn cw-play" data-contplay="" aria-pressed={on} onClick={() => { setPos(0); setPlay(on ? null : { t0: performance.now(), file }); }}><Icon name={on ? 'stop' : 'play'} size={12} />{say(on ? m.stop : m.play)}</button></div>
    </>
  );
}

definePart('recordingSpan', RecordingSpan);
