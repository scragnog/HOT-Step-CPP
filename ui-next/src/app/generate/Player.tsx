// The player in the go bar (35-player.js): the song's art, play and pause, its title over its waveform (the seek bar,
// B-13), the time, and a quiet close. It rises into the bar from nothing to its width, and folds away the same way,
// faster (G-M-07). The page's skeleton holds its element (#player); it stands in the go bar before the status line,
// as the concept's boot puts it there. Space plays and pauses from anywhere that does not use Space itself (C-27).

import { useEffect, useLayoutEffect, useRef, type CSSProperties, type KeyboardEvent, type PointerEvent } from 'react';
import { createPortal } from 'react-dom';
import { playerSeek, playerWords, songArt, songCard, songWave } from '../../derive.ts';
import { ui, useSay } from '../i18n/index.ts';
import { pausePlayer, seekPlayer, togglePlay, useGen, type Card } from '../store/generate.ts';
import { Icon } from '../kit/index.ts';
import { easing, mtime, reduced } from '../kit/motion.ts';

function Wave({ id }: { id: string }) {
  return <svg viewBox="0 0 288 20" preserveAspectRatio="none" aria-hidden="true"><path d={songWave(id)} /></svg>;
}

function PlayerBody({ c }: { c: Card }) {
  const say = useSay(), pos = useGen(s => s.pos), paused = useGen(s => s.paused), track = useRef<HTMLDivElement>(null), title = useRef<HTMLElement>(null);
  /* a title cut short can still be read, in the tooltip (F-30) */
  useLayoutEffect(() => { const b = title.current; if (b) b.title = b.scrollWidth > b.clientWidth + 1 ? b.textContent?.replace(/^\d+\/\d+/, '') ?? '' : ''; }, [c?.id]);
  if (!c) return null;
  const w = playerWords(pos, c.dur, paused), card = songCard(c), name = say(card.name);
  const seek = (e: PointerEvent) => { const r = track.current!.getBoundingClientRect(); seekPlayer(Math.round(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)) * c.dur)); };
  const down = (e: PointerEvent<HTMLDivElement>) => { e.preventDefault(); e.currentTarget.setPointerCapture(e.pointerId); e.currentTarget.focus({ preventScroll: true }); seek(e); };
  const key = (e: KeyboardEvent) => { const v = playerSeek(e.key, e.shiftKey, pos, c.dur); if (v === null) return; e.preventDefault(); seekPlayer(v); };
  return (
    <>
      <div className="art" style={{ background: songArt(c.id, c.hue) }} />
      <button className="iconbtn" id="ppause" type="button" data-st={paused ? 'play' : 'pause'} aria-label={say(w.act)} title={say(w.tip)} onClick={pausePlayer}><Icon name={paused ? 'play' : 'pause'} /></button>
      <div className="pi">
        <b ref={title}>{card.take ? <span className="ptk">{say(card.take)}</span> : null}{c.title}</b>
        <div className="track" id="ptrack" ref={track} role="slider" tabIndex={0} aria-label={say(ui('gen.position', { name }))} aria-valuemin={0} aria-valuemax={Math.round(c.dur)} aria-valuenow={Math.round(pos)} aria-valuetext={say(w.valueText)} style={{ '--pp': (pos / c.dur).toFixed(4) } as CSSProperties}
          onPointerDown={down} onPointerMove={e => { if (e.currentTarget.hasPointerCapture(e.pointerId)) seek(e); }} onKeyDown={key}>
          <Wave id={c.id} /><i id="trk"><Wave id={c.id} /></i>
        </div>
      </div>
      <span className="mono" id="ptime">{w.time}</span>
      <button className="iconbtn pclose" id="pstop" type="button" aria-label={say(ui('gen.close'))} title={say(ui('gen.closeTip'))} onClick={() => stopFrom(c.id)}><Icon name="x" /></button>
    </>
  );
}
/** Stopping from inside the player puts focus back on the card's Play. */
function stopFrom(id: string): void {
  const inside = document.getElementById('player')?.contains(document.activeElement);
  togglePlay(id);
  if (inside) requestAnimationFrame(() => document.querySelector<HTMLElement>(`#latest .song[data-id="${id}"] [data-play]`)?.focus({ preventScroll: true }));
}

const hostEl = () => document.getElementById('player');
/** Moves the skeleton's #player into the go bar, before the status line, and draws the player into it. */
export function PlayerHost() {
  /* the last song played stays drawn (hidden) after it stops, so what folds away is its last frame */
  const say = useSay(), on = useGen(s => s.playing !== null), last = useGen(s => s.cards.find(c => c.id === s.shown)), was = useRef(false);
  useLayoutEffect(() => {
    const pl = hostEl(), g = document.querySelector('.gobar'), meter = document.getElementById('meter');
    if (pl && g && meter && pl.nextElementSibling !== meter) g.insertBefore(pl, meter);
  });
  useLayoutEffect(() => {
    const pl = hostEl(); if (!pl) return;
    if (on) { pl.setAttribute('role', 'region'); pl.setAttribute('aria-label', say(ui('gen.player'))); }
    if (on === was.current) return; was.current = on;
    pl.getAnimations().forEach(a => a.cancel()); document.querySelector('.gobar > .player-out')?.remove();
    /* the page is laid out once, at the player's new width; the status line beside it slides from where it stood
       (transform) while the player is revealed from its left edge or folds into it (clip-path): nothing is laid out
       again on each frame (package 9, the owner's rule for motion) */
    const line = document.getElementById('meter'), x0 = line?.getBoundingClientRect().left ?? 0;
    const slideLine = (time: string, ease: string) => { const dx = x0 - (line?.getBoundingClientRect().left ?? x0); if (line && Math.abs(dx) > 1) line.animate([{ transform: `translateX(${dx}px)` }, { transform: 'none' }], { duration: mtime(time), easing: easing(ease) }); };
    if (on) {
      pl.hidden = false; if (reduced()) return;
      pl.animate([{ clipPath: 'inset(-8px 100% -8px 0)', opacity: 0 }, { clipPath: 'inset(-8px 0 -8px 0)', opacity: 1 }], { duration: mtime('--t-base'), easing: easing('--e-out') });
      slideLine('--t-base', '--e-out');
      return;
    }
    if (reduced()) { pl.hidden = true; return; }
    /* what folds away is the player's last frame, held in place by a copy for the length of the exit */
    const copy = pl.cloneNode(true) as HTMLElement, r = pl.getBoundingClientRect();
    copy.removeAttribute('id'); copy.querySelectorAll('[id]').forEach(x => x.removeAttribute('id')); copy.classList.add('player-out'); copy.inert = true; copy.setAttribute('aria-hidden', 'true');
    copy.style.cssText += ';position:absolute;margin:0;left:0;top:0;pointer-events:none';
    pl.after(copy); pl.hidden = true;
    const c = copy.getBoundingClientRect(); copy.style.left = `${r.left - c.left}px`; copy.style.top = `${r.top - c.top}px`; copy.style.width = `${r.width}px`;
    copy.animate([{ clipPath: 'inset(-8px 0 -8px 0)', opacity: 1 }, { clipPath: 'inset(-8px 100% -8px 0)', opacity: 0 }], { duration: mtime('--t-fast'), easing: easing('--e-in'), fill: 'forwards' }).finished.then(() => copy.remove(), () => copy.remove());
    slideLine('--t-fast', '--e-in');
  }, [on, say]);
  useEffect(() => () => { hostEl()?.setAttribute('hidden', ''); }, []);
  const host = hostEl();
  return host ? createPortal(last ? <PlayerBody c={last} /> : null, host) : null;
}
