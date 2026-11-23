// The shared dropdown a pick opens (the concept's #dd, 23-dropdown.js), in every variant: a listbox pinned beside its
// button, never taller than the room it has, drawn into the page's one #dd (index.html) as the concept's is: shown
// while open, hidden after, with its variant's class, the choice's name and the button's width. Not a part of the
// inventory: a piece the pick parts share.
//
//   lcd     the LCD list: each option's LCD words and its small second line
//   plugin  a periodic table of symbols in families, a header showing the option under the hand (a long list); a grid
//           of curves (schedules); or cards with their names (six or fewer). A filter from eight options.
//   chip    the page's popover list of words
//   model   one row per model, its quantisations as chips on the row, its size and a bar of the card it takes
//   stamp   ink stamps in the language's own name, the ones used last first, with a filter
//
// The arrow keys walk it (up and down by rows in a grid), Home and End go to its ends, a letter types into its filter
// and Enter there takes the first match, Enter or a click chooses, Escape or a click outside closes it and focus goes
// back to its button, and Tab closes it and carries on from its button (A-24). Under a finger a symbol's first tap shows
// it in the header and a second, or Use it, chooses it (S-02).

import { useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode, type RefObject } from 'react';
import { createPortal } from 'react-dom';
import { LcdFace, ui, useSay, type ParamView, type Slot } from '../kit/index.ts';
import { El, Spark, type Face } from './PickFaces.tsx';
import { nbHyphen, useListFade } from './pickText.ts';

const host = () => document.getElementById('dd');
type Opt = NonNullable<ParamView['options']>[number];
export type Variant = 'lcd' | 'plugin' | 'chip' | 'model' | 'stamp';

/** The languages chosen last, newest first: the page's own memory (deviation 5 of docs/09: the page's own keys). */
const RECENT = 'hotstep.create.langRecent';
const recentLangs = (): string[] => { try { const a: unknown = JSON.parse(localStorage.getItem(RECENT) ?? '[]'); return Array.isArray(a) ? a.filter((x): x is string => typeof x === 'string') : []; } catch { return []; } };
const remember = (v: string) => { try { localStorage.setItem(RECENT, JSON.stringify([v, ...recentLangs().filter(x => x !== v)].slice(0, 6))); } catch { /* storage off: nothing is remembered */ } };

/** The next (or previous) control in the page's tab order from an element (23-dropdown.js tabFrom). */
function tabFrom(el: HTMLElement, dir: number) {
  const all = [...document.querySelectorAll<HTMLElement>('a[href], button, input, select, textarea, [tabindex]')]
    .filter(x => x.tabIndex >= 0 && !(x as HTMLButtonElement).disabled && !x.closest('[hidden], [inert]') && x.getClientRects().length && getComputedStyle(x).visibility !== 'hidden');
  all[all.indexOf(el) + dir]?.focus();
}
/** The option in the next row up or down whose centre is nearest across (23-dropdown.js gridStep). */
function gridStep(opts: HTMLElement[], cur: HTMLElement, dir: number): HTMLElement | null {
  const r = cur.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height / 2;
  const cand = opts.filter(o => o !== cur).map(o => { const q = o.getBoundingClientRect(); return { o, dy: (q.top + q.height / 2 - cy) * dir, dx: Math.abs(q.left + q.width / 2 - cx) }; }).filter(c => c.dy > Math.min(r.height, 12) / 2);
  if (!cand.length) return null;
  const row = Math.min(...cand.map(c => c.dy));
  return cand.filter(c => c.dy < row + 8).sort((a, b) => a.dx - b.dx)[0]!.o;
}

interface Props { anchor: RefObject<HTMLButtonElement | null>; p: ParamView; variant: Variant; faces?: Face[]; title?: Slot; close: (refocus: boolean) => void }

export function Dropdown({ anchor, p, variant, faces, title, close }: Props) {
  const say = useSay(), label = say(p.label), el = host(), opts = useMemo(() => p.options ?? [], [p.options]);
  const [query, setQuery] = useState('');
  const [shown, setShown] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const touch = useRef(false), recent = useMemo(() => (variant === 'stamp' ? recentLangs() : []), [variant]);
  const words = (o: Opt) => say(o.label);
  const faceOf = (o: Opt) => faces?.[opts.indexOf(o)] ?? {};
  /* the plugin list's shape: curves for schedules, a table of symbols for a long list, cards for six or fewer */
  const curves = variant === 'plugin' && !!faces?.some(f => f.curve), table = variant === 'plugin' && !curves && opts.length > 6;
  const filtered = variant === 'stamp' || (variant === 'plugin' && opts.length >= 8);
  /* a filter matches the option's words, value or file, whatever their case */
  const q = query.trim(), find = new RegExp(q.replace(/[\\^$.*+?()[\]{}|]/g, '\\$&'), 'i');
  const matches = (o: Opt) => !q || find.test(`${String(o.value)} ${words(o)} ${faceOf(o).file ?? ''} ${faceOf(o).native ?? ''}`);
  const extra = variant === 'plugin' ? (table ? 'pt' : 'schedw') : '';
  useListFade(host, '.dd-body, .pt-grid', p.key);

  useLayoutEffect(() => {
    const dd = host(), b = anchor.current; if (!dd || !b) return;
    dd.className = `dd dd-${variant} ${extra}`; dd.hidden = false; dd.dataset.k = p.key; dd.setAttribute('aria-label', label);
    dd.style.setProperty('--ddw', b.offsetWidth + 'px'); dd.style.maxHeight = '';
    const h = dd.offsetHeight, centred = variant === 'lcd' || variant === 'stamp';
    let down: boolean | null = null;
    /* pinned to the window beside its button wherever the page scrolls, never taller than the room it has; the side it
       first opened on is kept, and it closes when its button scrolls out of the window (23-dropdown.js placeDD) */
    const place = () => {
      const r = b.getBoundingClientRect(), docW = document.documentElement.clientWidth, w = dd.offsetWidth, gap = 8, edge = 12;
      if (down !== null && (r.bottom < 0 || r.top > innerHeight || !b.isConnected)) { close(false); return; }
      const below = innerHeight - r.bottom - gap - edge, above = r.top - gap - edge - 56;
      if (down === null) down = h <= below || below >= above;
      const room = Math.max(160, down ? below : above), fit = Math.min(h, room);
      dd.style.maxHeight = h > room ? room + 'px' : '';
      dd.style.top = (down ? r.bottom + gap : r.top - gap - fit) + 'px';
      dd.style.transformOrigin = down ? 'top center' : 'bottom center'; dd.classList.toggle('up', !down);
      dd.style.left = Math.min(docW - w - 16, Math.max(16, r.left + (centred ? (r.width - w) / 2 : 0))) + 'px';
      dd.classList.toggle('scrolls', h > room);
    };
    place();
    let raf = 0;
    const scroll = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; place(); }); };
    const resize = () => close(false), pointer = (e: PointerEvent) => { touch.current = e.pointerType === 'touch'; };
    addEventListener('scroll', scroll, { passive: true, capture: true }); addEventListener('resize', resize); dd.addEventListener('pointerdown', pointer);
    const sel = dd.querySelector<HTMLElement>('[aria-selected="true"]') ?? dd.querySelector<HTMLElement>('.opt');
    /* a long list opens with the chosen option in the middle of its box */
    const box = sel?.closest<HTMLElement>('.dd-body, .pt-grid');
    if (variant === 'plugin' && sel && box) box.scrollTop = Math.max(0, sel.offsetTop - box.clientHeight / 2);
    sel?.focus({ preventScroll: variant === 'plugin' });
    return () => { dd.hidden = true; removeEventListener('scroll', scroll, { capture: true }); removeEventListener('resize', resize); dd.removeEventListener('pointerdown', pointer); cancelAnimationFrame(raf); };
  }, [anchor, close, label, p.key, variant, extra]);

  const choose = (o: Opt) => { if (variant === 'stamp') remember(String(o.value)); p.set(o.value); close(true); };
  useEffect(() => {
    const out = (e: PointerEvent) => { if (!host()?.contains(e.target as Node) && !anchor.current?.contains(e.target as Node)) close(false); };
    document.addEventListener('pointerdown', out, true);
    return () => document.removeEventListener('pointerdown', out, true);
  }, [anchor, close]);
  /* the keys, on the list itself (its options and its filter) */
  const now = useRef({ opts, choose, matches }); useEffect(() => { now.current = { opts, choose, matches }; });
  useEffect(() => {
    const dd = host(); if (!dd) return;
    const key = (e: KeyboardEvent) => {
      const list = [...dd.querySelectorAll<HTMLElement>('.opt:not([hidden]):not(.dim)')], i = list.indexOf(document.activeElement as HTMLElement);
      const inFilter = (e.target as HTMLElement).id === 'ddq', filter = dd.querySelector<HTMLInputElement>('#ddq');
      if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close(true); return; }
      if (e.key === 'Tab') { e.preventDefault(); const b = anchor.current; close(true); if (b?.isConnected) tabFrom(b, e.shiftKey ? -1 : 1); return; }
      if (e.key.length === 1 && e.key !== ' ' && !e.ctrlKey && !e.metaKey && !e.altKey && !inFilter) { if (filter) { e.preventDefault(); filter.focus(); filter.value += e.key; setQuery(filter.value); } return; }
      if (e.key === 'Enter' && inFilter) { e.preventDefault(); const o = now.current.opts.find(now.current.matches); if (o) now.current.choose(o); return; }
      const mv = ({ ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 } as Record<string, number>)[e.key];
      if (mv && (e.key === 'ArrowUp' || e.key === 'ArrowDown') && i >= 0) { e.preventDefault(); const n = gridStep(list, list[i]!, mv); if (n) { n.focus(); n.scrollIntoView({ block: 'nearest' }); } else if (mv < 0) filter?.focus(); return; }
      if (mv && !(inFilter && (e.key === 'ArrowLeft' || e.key === 'ArrowRight'))) { e.preventDefault(); const n = list[Math.max(0, Math.min(list.length - 1, (i < 0 ? -1 : i) + mv))]; n?.focus(); n?.scrollIntoView({ block: 'nearest' }); return; }
      if ((e.key === 'Home' || e.key === 'End') && !inFilter) { e.preventDefault(); (e.key === 'Home' ? list[0] : list[list.length - 1])?.focus(); }
    };
    dd.addEventListener('keydown', key);
    return () => dd.removeEventListener('keydown', key);
  }, [anchor, close]);

  if (!el || !p.options) return null;
  const a = (o: Opt) => ({ role: 'option', 'data-opt': String(o.value), 'aria-selected': o.on, tabIndex: -1 }) as const;
  const none = q && !opts.some(matches) ? <p className="dd-none" role="status">{say(ui('dd.none', { q: query }))}</p> : null;
  const filterBox = (placeholder: Slot) => (
    <input id="ddq" placeholder={say(placeholder)} aria-label={say(ui('dd.filter', { what: title ?? p.label }))} onInput={e => setQuery(e.currentTarget.value)} />
  );
  let body: ReactNode;
  if (variant === 'plugin') {
    const head = title ?? p.label;
    if (table) {
      const big = opts.find(o => String(o.value) === (preview ?? shown)) ?? (q ? opts.find(matches) : undefined) ?? opts.find(o => o.on) ?? opts[0]!;
      /* families in the list's order, five symbols a row; a longer family carries on in rows of its own */
      const groups: { name: string; hue: number; os: Opt[] }[] = [];
      for (const o of opts) { const f = faceOf(o); let g = groups.find(x => x.name === f.family); if (!g) groups.push(g = { name: f.family ?? '', hue: f.hue ?? 0, os: [] }); g.os.push(o); }
      const rows = groups.flatMap(g => Array.from({ length: Math.ceil(g.os.length / 5) }, (_, n) => ({ g, n, os: g.os.slice(n * 5, n * 5 + 5) })));
      body = (
        <>
          <div className="pt-head">
            <div className="pt-big" id="ptBig">
              {q && !opts.some(matches) ? <div className="pt-info pt-none"><b>{say(ui('dd.noMatch'))}</b><code>{say(ui('dd.noMatch.more', { q: query }))}</code></div> : <><El f={faceOf(big)} /><div className="pt-info"><span className="fam" style={{ '--hue': faceOf(big).hue } as CSSProperties}>{faceOf(big).family}</span><b>{words(big)}</b><code>{faceOf(big).file}</code></div></>}
              {preview ? <button type="button" className="small-btn pt-use" data-ptuse="" onClick={() => { const o = opts.find(x => String(x.value) === preview); if (o) choose(o); }}>{say(ui('dd.use'))}</button> : null}
            </div>
            {filtered ? filterBox(ui('dd.filter.short')) : null}
          </div>
          <div className="pt-grid" onMouseLeave={() => { if (!preview) setShown(null); }}>
            {rows.map(({ g, n, os }) => (
              <div key={`${g.name}${n}`} className="pt-row">
                <span className="pt-fam" style={{ '--hue': g.hue } as CSSProperties}>{n || groups.length === 1 ? '' : g.name}</span>
                {os.map(o => (
                  <button key={String(o.value)} type="button" className={`opt opt-pt${matches(o) ? '' : ' dim'}${preview === String(o.value) ? ' pv' : ''}`} {...a(o)} aria-label={`${words(o)}, ${faceOf(o).file ?? ''}`}
                    onMouseOver={() => setShown(String(o.value))} onFocus={() => setShown(String(o.value))}
                    onClick={() => { if (touch.current && preview !== String(o.value) && !o.on) { setPreview(String(o.value)); setShown(String(o.value)); return; } choose(o); }}>
                    <El f={faceOf(o)} />
                  </button>
                ))}
              </div>
            ))}
          </div>
        </>
      );
    } else {
      body = (
        <>
          <div className="dd-head"><h4>{say(head)}</h4>{filtered ? filterBox(ui('dd.filter.n', { n: opts.length })) : null}</div>
          <div className="dd-body" id="ddBody">
            <div className={`dd-grid ${curves ? 'sm' : 'list'}`}>
              {opts.map(o => (
                <button key={String(o.value)} type="button" className="opt opt-el" {...a(o)} hidden={!matches(o)} onClick={() => choose(o)}>
                  {curves ? <Spark curve={faceOf(o).curve ?? []} w={120} h={44} pad={4} /> : <El f={faceOf(o)} />}<b>{nbHyphen(words(o))}</b><code>{faceOf(o).file}</code>
                </button>
              ))}
            </div>
          </div>
          {none}
        </>
      );
    }
  } else if (variant === 'model') {
    /* one row per model; its quantisations are chips on the row (G-D-46) */
    const groups: { name: string; os: Opt[] }[] = [];
    for (const o of opts) { const n = faceOf(o).name ?? words(o); let g = groups.find(x => x.name === n); if (!g) groups.push(g = { name: n, os: [] }); g.os.push(o); }
    const mem = (f: Face) => (f.mem !== undefined ? <span className="mem"><i style={{ '--w': `${f.mem * 100}%` } as CSSProperties} /></span> : null);
    body = (
      <>
        <div className="dd-model-title">{say(title ?? p.label)} <span>{say(ui('dd.model.vram'))}</span></div>
        {groups.map(g => {
          if (g.os.length === 1) { const o = g.os[0]!, f = faceOf(o); return <button key={String(o.value)} type="button" className="opt opt-model" {...a(o)} onClick={() => choose(o)}><b>{g.name}</b>{f.quant ? <span className={`qb q-${f.qc}`}>{f.quant}</span> : <span />}<span className="gb">{f.gbText ?? say(ui('dd.model.variant'))}</span>{mem(f)}</button>; }
          const show = g.os.find(o => o.on) ?? g.os[0]!, f = faceOf(show);
          return (
            <div key={g.name} className={`opt-mrow${g.os.some(o => o.on) ? ' on' : ''}`}>
              <b>{g.name}</b>
              <span className="qchips">{g.os.map(o => <button key={String(o.value)} type="button" className="opt opt-q" {...a(o)} title={words(o)} onClick={() => choose(o)}><span className={`qb q-${faceOf(o).qc}`}>{faceOf(o).quant}</span></button>)}</span>
              <span className="gb">{f.gbText ?? ''}</span>{mem(f)}
            </div>
          );
        })}
      </>
    );
  } else if (variant === 'stamp') {
    const stamp = (o: Opt) => { const i = opts.indexOf(o); return (
      <button key={String(o.value)} type="button" className="opt opt-stamp" {...a(o)} hidden={!matches(o)} onClick={() => choose(o)} style={{ '--ink': faceOf(o).ink, '--r': `${[-6, 4, -2, 7, -4, 3, -7, 5][i % 8]}deg` } as CSSProperties}>
        {faceOf(o).native}<small>{words(o)}</small>
      </button>
    ); };
    const rec = recent.map(v => opts.find(o => String(o.value) === v)).filter((o): o is Opt => !!o && !o.on).slice(0, 4);
    body = (
      <>
        <div className="dd-stamp-title">{say(title ?? ui('dd.stamp.title'))}{filterBox(ui('dd.filter.find', { n: opts.length }))}</div>
        {rec.length ? <div className="stamps-rec"><span className="dd-fam">{say(ui('dd.recent'))}</span><div className="stamps">{rec.map(stamp)}</div></div> : null}
        <div className="stamps">{opts.map(stamp)}</div>
        {none}
      </>
    );
  } else {
    body = (
      <>
        <div className="dd-lcd-title">{label}</div>
        {opts.map(o => variant === 'chip'
          ? <button key={String(o.value)} type="button" className="opt opt-chip" {...a(o)} onClick={() => choose(o)}><b>{words(o)}</b>{o.detail ? <small>{say(o.detail)}</small> : null}</button>
          : <button key={String(o.value)} type="button" className="opt opt-lcd" {...a(o)} onClick={() => choose(o)}><span><LcdFace text={o.pick} /></span><small>{o.detail ? say(o.detail) : ''}</small></button>)}
      </>
    );
  }
  return createPortal(body, el);
}
