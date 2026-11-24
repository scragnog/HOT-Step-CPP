// Latest (create/20-centre.html .latest, 35-songs.js): the song cards, as many whole rows as fit; a run in progress as
// cards whose colours come into focus as it renders (--f) and that resolve, sharpening with one soft ring, when it
// finishes; a run the server refused as a one-line notice across the top (G-D-09); a long queue folded (C-47). Cards
// already there slide one place along when one arrives, and a new card enters from .96 with a fade (M-12, G-M-06).
// What each card says, and which show, is the core's (derive/generate.ts songCard, latestShown, queueStage).

import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from 'react';
import { latestShown, newSongs, queueStage, refusedSays, songArt, songCard, type Slot } from '../../derive.ts';
import { ui, useSay } from '../i18n/index.ts';
import { useDesk } from '../store/desk.ts';
import { onRun, setMenu, toggleFold, togglePlay, useGen, type Card } from '../store/generate.ts';
import { cancelOne, dismiss, retry } from './actions.ts';
import { GIcon } from './icons.tsx';
import { easing, inView, mtime, places, reduced, slide } from './motion.ts';
import { logoOf } from '../shell/engines.ts';
import { SongMenu } from './SongMenu.tsx';

function Notice({ c }: { c: Card }) {
  const say = useSay(), e = c.error!;
  return (
    <article className="song errored" data-id={c.id} role="alert">
      <span className="err-mark" aria-hidden="true">!</span>
      <p className="err-line" title={say({ id: 'gen.notice.tip', slots: { title: c.title, why: e.why, more: e.more } })}><b>{say(ui('gen.didnt'))}</b> <span className="err-t">{c.title}</span> <span className="err-why">{say(e.why)}</span></p>
      <div className="err-acts">
        {e.retry ? <button className="small-btn primary" data-retry={`${c.id}|${e.retry}`} onClick={() => retry(c.id)}>{say({ id: 'gen.retry', slots: { n: e.retry } })}</button> : null}
        <button className="small-btn" data-dismiss={c.id} onClick={() => dismiss(c.id)}>{say(ui('gen.dismiss'))}</button>
      </div>
    </article>
  );
}

/** The playing card's art carries the player's progress along its foot (B-13). */
const usePlayed = (c: Card): CSSProperties | undefined => { const pp = useGen(s => (s.playing === c.id ? (s.pos / c.dur).toFixed(4) : null)); return pp ? { '--pp': pp } as CSSProperties : undefined; };

function Pending({ c, ahead, folded, hidden }: { c: Card; ahead: number; folded: boolean; hidden: boolean }) {
  const say = useSay(), p = c.pending!, w = songCard(c), name = say(w.name), playing = useGen(s => s.playing === c.id), pp = usePlayed(c);
  const stage: Slot = queueStage(ahead, p.f, p.stage ?? { id: 'gen.queued' });
  return (
    <article className={`song pending${p.stream ? ' streams' : ''}${playing ? ' playing' : ''}${ahead ? ' waiting' : ''}${folded ? ' folded' : ''}`} data-id={c.id} hidden={hidden} style={{ '--f': p.f.toFixed(4), ...pp } as CSSProperties}>
      <div className="art" style={{ '--pbg': songArt(c.id, c.hue) } as CSSProperties}>
        <i className="bar" style={{ width: `${p.f * 100}%` }} />
        {p.stream ? <><i className="bar ready" style={{ width: `${p.ready * 100}%` }} /><button className="play" data-play="" hidden={!p.ready} aria-label={say(ui(playing ? 'gen.stopPart' : 'gen.playPart', { name }))} title={say(ui('gen.streamTip'))} onClick={() => togglePlay(c.id)}><GIcon name={playing ? 'stop' : 'play'} /></button></> : null}
        {w.take ? <span className="take-no" title={say(w.takeTip)}>{say(w.take)}</span> : null}
        <button className="pend-x" data-cancelsong={c.id} title={say(ui('gen.cancelSong'))} aria-label={say(ui('gen.cancelName', { name }))} onClick={() => cancelOne(c.id)}><GIcon name="x" size={13} /></button>
      </div>
      <div className="meta"><div><h3>{c.title}</h3><p className="stage">{say(stage)}</p></div></div>
    </article>
  );
}

function Song({ c, hidden }: { c: Card; hidden: boolean }) {
  const say = useSay(), w = songCard(c), name = say(w.name), playing = useGen(s => s.playing === c.id), menu = useGen(s => s.menuFor === c.id), pp = usePlayed(c);
  return (
    <article className={`song${c.fresh ? ' new' : ''}${c.unheard ? ' unheard' : ''}${playing ? ' playing' : ''}`} data-id={c.id} hidden={hidden} style={pp}>
      <div className="art" style={{ background: songArt(c.id, c.hue) }}>
        <span className="eng-mark" title={w.engine} aria-label={w.engine} dangerouslySetInnerHTML={{ __html: logoOf(c.engine, 14) }} />
        {c.unheard ? <i className="new-dot" title={say(ui('gen.newDot'))}><span className="sr-only">{say(ui('gen.new'))}</span></i> : null}
        {w.take ? <span className="take-no" title={say(w.takeTip)}>{say(w.take)}</span> : null}
        <span className="dur">{w.dur}</span>
        <button className="play" data-play="" aria-label={say(ui(playing ? 'gen.stop' : 'gen.play', { name }))} onClick={() => togglePlay(c.id)}><GIcon name={playing ? 'stop' : 'play'} /></button>
      </div>
      <div className="meta">
        <div><h3>{c.title}</h3><p title={say(w.styleTip)}>{c.style}</p></div>
        <button className="more" data-songmenu="" aria-haspopup="menu" aria-expanded={menu} aria-label={say(ui('gen.actions', { name }))} onClick={() => setMenu(menu ? null : c.id)}><GIcon name="more" size={16} /></button>
      </div>
    </article>
  );
}

/** How many columns the grid lays out (its template's tracks). */
const colsOf = (g: HTMLElement | null) => (g ? Math.max(1, getComputedStyle(g).gridTemplateColumns.split(' ').filter(Boolean).length) : 1);

export function Latest() {
  const say = useSay(), cards = useGen(s => s.cards), runs = useGen(s => s.runs), open = useGen(s => s.qOpen), grid = useRef<HTMLDivElement>(null), [cols, setCols] = useState(4);
  const pos = useRef<Map<string, DOMRect> | null>(null), seen = useRef<Set<string> | null>(null);
  useEffect(() => { const g = grid.current; if (!g) return; const ro = new ResizeObserver(() => setCols(colsOf(g))); ro.observe(g); setCols(colsOf(g)); return () => ro.disconnect(); }, []);
  /* a finished run is said only when its card is out of sight (the card is the message); a refusal likewise */
  useEffect(() => onRun(e => requestAnimationFrame(() => {
    const d = useDesk.getState();
    if (e.kind === 'done') { const el = document.querySelector(`#latest .song[data-id="${e.ids[0]}"]`); if (e.ids.length && (!el || !inView(el))) d.showToast({ text: newSongs(e.ids.length, e.run.spec.engine) }); }
    else if (e.kind === 'refused') { const el = document.querySelector(`#latest .song.errored[data-id="${e.id}"]`), m = refusedSays(e.run.ids.length, e.run.spec.engine); if (!el || !inView(el)) d.showToast({ text: m }); else d.say(m); }
    else { const c = useGen.getState().cards.find(x => x.id === e.id); if (c) d.say({ id: 'gen.stream.ready', slots: { name: songCard(c).name } }); }
  })), []);
  const list = cards.slice(0, Math.max(24, cards.filter(c => c.pending || c.error).length + 24));
  const runAt = (c: Card) => runs.findIndex(r => r.id === c.pending?.run);
  const shown = latestShown(list.map(c => (c.error ? 'error' : c.pending ? 'pending' : 'song')), list.map(c => (c.pending ? (runAt(c) < 0 ? 999 : runAt(c)) * 100 + (c.take ?? 0) : 0)), cols, open);
  /* the cards already there slide one place along; a new one enters (G-M-06, M-12) */
  useLayoutEffect(() => {
    const g = grid.current, key = (el: HTMLElement) => el.dataset.id ?? 'fold', now = new Set(cards.map(c => c.id));
    if (seen.current && pos.current) {
      slide(g, ':scope > .song[data-id]', key, pos.current);
      for (const el of g?.querySelectorAll<HTMLElement>(':scope > .song[data-id]') ?? []) if (el.offsetParent && !seen.current.has(el.dataset.id!)) el.animate(reduced() ? [{ opacity: 0 }, { opacity: 1 }] : [{ opacity: 0, transform: 'scale(.96)' }, { opacity: 1, transform: 'none' }], { duration: reduced() ? 120 : mtime('--t-base'), easing: easing('--e-out') });
    }
    seen.current = now; pos.current = places(g, ':scope > .song[data-id]', key);
    /* a title cut short can still be read, in the tooltip (F-30) */
    for (const h of g?.querySelectorAll<HTMLElement>('.song:not([hidden]) h3') ?? []) h.title = h.scrollWidth > h.clientWidth + 1 ? h.textContent ?? '' : '';
  }, [cards, cols, open]);
  /* a card that has resolved keeps its ring only while it arrives */
  useEffect(() => { if (!cards.some(c => c.fresh)) return; const t = setTimeout(() => useGen.setState(s => ({ cards: s.cards.map(c => (c.fresh ? { ...c, fresh: false } : c)) })), 1400); return () => clearTimeout(t); }, [cards]);
  const items = list.flatMap((c, i) => {
    const hid = shown.hidden[i]!, el = c.error ? <Notice key={c.id} c={c} /> : c.pending ? <Pending key={c.id} c={c} ahead={Math.max(0, runAt(c))} folded={shown.folded[i]!} hidden={hid} /> : <Song key={c.id} c={c} hidden={hid} />;
    return shown.fold && shown.fold.after === i ? [el, <button key="qfold" type="button" className="song qfold" data-qfold="" aria-expanded={shown.fold.open} onClick={toggleFold}><b>{say(shown.fold.title)}</b><small>{say(shown.fold.sub)}</small></button>] : [el];
  });
  return (
    <section className="latest" aria-label={say(ui('latest.songs'))}>
      <div className="res-head"><h2>{say(ui('latest'))}</h2><span className="lbl" id="songCount">{say(shown.count)}</span><button className="linkbtn" data-goto="Library">{say(ui('latest.library'))}</button></div>
      <div className="songgrid" id="latest" ref={grid}>{items}</div>
      <SongMenu />
    </section>
  );
}
