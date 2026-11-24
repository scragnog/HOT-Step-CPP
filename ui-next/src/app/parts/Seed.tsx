// The Seed tile's instruments (the concept's 25-module-seed.js):
//
//   splitFlap   the seed as ten split-flap cells (a seed is right-aligned with blank cells before it, RANDOM in the
//               middle, so the row never changes width: G-M-11); pressing it swaps in a field in the cells' own
//               footprint (Q-07, Q-23) to type or paste a seed. Any change of the seed shown flips the cells that differ
//               (A-20): each flips like a real split-flap, at most four times through random digits to its own, the
//               cells starting 35ms apart left to right, the last one landing with a 1px bounce. Its first child (the
//               lock) stands beside the cells, the second (the LM seed's switch) under them with what it does; then
//               Roll once and Random each time
//   savedSeeds  the saved seeds: a heading with Save on it and its one-column form, the search and a random saved
//               seed, and the rows (star, name and number, description, tags, delete that asks first)
//
// What a seed is, what each press sets and how each step is named are the core's (act: type, roll, random, save, star,
// delete, load); the cells' characters, the rows and the form's defaults are its model (PartModel.instrument).

import { Children, useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent } from 'react';
import { definePart, Icon, ui, useSay, type PartView } from '../kit/index.ts';
import { motionVar, mtime, reduced } from './motion.ts';

/* a flip is two folds on one hinge: the old character's top leaf falls, then the new one's bottom leaf lands */
const FLIP = 55, FOLD = 28;
function leaf(ch: string, half: string): HTMLElement { const l = document.createElement('i'); l.className = `leaf ${half}`; l.textContent = ch; return l; }
function flip(cell: HTMLElement, ch: string): void {
  const old = cell.textContent ?? ''; cell.textContent = ch; cell.classList.toggle('blank', ch === '');
  if (reduced()) return;
  const a = leaf(old, 'top'), b = leaf(ch, 'bot'); cell.append(a, b);
  a.animate([{ transform: 'perspective(140px) rotateX(0deg)' }, { transform: 'perspective(140px) rotateX(-90deg)' }], { duration: FOLD, easing: motionVar('--e-in'), fill: 'forwards' });
  b.animate([{ transform: 'perspective(140px) rotateX(90deg)' }, { transform: 'perspective(140px) rotateX(0deg)' }], { duration: FOLD, delay: FOLD, easing: motionVar('--e-out'), fill: 'both' })
    .finished.then(() => { a.remove(); b.remove(); }, () => {});
}
/** Draws the cells at once (a display just drawn). */
function setCells(el: HTMLElement, to: string): void {
  el.replaceChildren(...[...to].map(c => { const s = document.createElement('span'); s.className = `flap${c === ' ' ? ' blank' : ''}`; s.setAttribute('aria-hidden', 'true'); s.textContent = c === ' ' ? '' : c; return s; }));
}
/** Rolls the cells from what they showed to what they show now; returns when it ends, in milliseconds. */
function roll(el: HTMLElement, from: string, to: string, timers: ReturnType<typeof setTimeout>[]): number {
  setCells(el, to);
  if (reduced()) return 0;
  const cells = [...el.querySelectorAll<HTMLElement>('.flap')];
  cells.forEach((c, i) => { c.textContent = from[i] === ' ' ? '' : from[i] ?? ''; c.classList.toggle('blank', from[i] === ' '); });
  let last = 0;
  cells.forEach((c, i) => {
    const end = to[i] === ' ' ? '' : to[i]!;
    if (c.textContent === end && from[i] === to[i]) return;
    const n = end === '' ? 1 : 1 + Math.floor(Math.random() * 4), seq = [...Array.from({ length: n - 1 }, () => String(Math.floor(Math.random() * 10))), end];
    seq.forEach((ch, k) => {
      const at = i * 35 + k * FLIP; last = Math.max(last, at);
      timers.push(setTimeout(() => {
        if (!c.isConnected) return; flip(c, ch);
        if (k === seq.length - 1 && at === last) timers.push(setTimeout(() => c.animate([{ transform: 'none' }, { transform: 'translateY(1px)' }, { transform: 'none' }], { duration: 120, easing: motionVar('--e-out') }), FOLD * 2));
      }, at));
    });
  });
  return last + FLIP + 40;
}

export function SplitFlap(v: PartView) {
  const say = useSay(), f = v.instrument?.flap;
  const btn = useRef<HTMLButtonElement>(null), inp = useRef<HTMLInputElement>(null), dice = useRef<HTMLButtonElement>(null);
  const shown = useRef<string | null>(null), timers = useRef<ReturnType<typeof setTimeout>[]>([]);
  const [editing, setEditing] = useState(false), [bad, setBad] = useState(false);
  const text = f ? f.cells.join('') : '';
  /* the cells: drawn at once the first time, rolled on every change after (the page owns their characters) */
  useLayoutEffect(() => {
    const el = btn.current; if (!el || !f) return;
    el.classList.toggle('rand', f.rand);
    const was = shown.current; shown.current = text;
    if (was === null) { setCells(el, text); return; }
    if (was === text) return;
    timers.current.forEach(clearTimeout); timers.current = [];
    el.classList.add('rolling');
    const t = roll(el, was, text, timers.current);
    timers.current.push(setTimeout(() => el.classList.remove('rolling'), t));
  }, [text, f]);
  useEffect(() => () => timers.current.forEach(clearTimeout), []);
  if (!f) return null;
  const [lock, ...link] = Children.toArray(v.children);

  /* the field takes the cells' exact footprint: the width of the ten flaps, their height and left edge (Q-07, Q-23) */
  const edit = () => {
    const b = btn.current!, i = inp.current!, fl = b.querySelectorAll('.flap');
    const w = fl.length ? fl[fl.length - 1]!.getBoundingClientRect().right - fl[0]!.getBoundingClientRect().left : b.offsetWidth - 4;
    i.style.height = `${(fl[0] as HTMLElement | undefined)?.offsetHeight ?? b.offsetHeight - 4}px`; i.style.flex = `0 0 ${Math.round(w)}px`;
    i.value = f.edit; setBad(false); setEditing(true);
    requestAnimationFrame(() => { i.focus(); i.select(); });
  };
  const close = (focus: boolean) => { setEditing(false); setBad(false); if (focus) requestAnimationFrame(() => btn.current?.focus({ preventScroll: true })); };
  const key = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') { e.preventDefault(); if (v.act('type', e.currentTarget.value)) close(true); else { setBad(true); e.currentTarget.select(); } }
    else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close(true); }
  };
  const blur = () => { if (!editing) return; v.act('left', inp.current?.value ?? ''); close(false); };
  const rollOnce = () => { const d = dice.current; if (d) { d.classList.remove('roll'); void d.offsetWidth; d.classList.add('roll'); } v.act('roll', Math.random()); };
  return (
    <>
      <div className="seed-row">
        <button ref={btn} type="button" className="seed-num" data-seednum="" data-seededit="" hidden={editing} title={say(f.title)} aria-label={say(f.aria)} onClick={edit} />
        <input ref={inp} className={`seed-in${bad ? ' bad' : ''}`} data-seedin="" hidden={!editing} type="text" inputMode="numeric" maxLength={20} autoComplete="off" spellCheck={false}
          placeholder={say(ui('seed.empty'))} aria-label={say(ui('seed.type'))} aria-invalid={bad || undefined} onKeyDown={key} onBlur={blur} />
        {lock}
      </div>
      {f.lm !== null ? <div className="seed-link">{link}<span className="lbl" data-lmseed="">{say(f.lm)}</span></div> : null}
      <div className="seed-btns">
        <button ref={dice} type="button" className="sbtn dice" data-dice="" onClick={rollOnce}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="4" /><circle cx="8.5" cy="8.5" r="1.2" fill="currentColor" /><circle cx="15.5" cy="15.5" r="1.2" fill="currentColor" /><circle cx="12" cy="12" r="1.2" fill="currentColor" /></svg>
          {say(ui('seed.roll'))}
        </button>
        <button type="button" className="sbtn" data-seedrand="" aria-pressed={f.rand} title={say(ui('seed.random.title'))} data-ctl={f.lock ? undefined : v.params[0]?.key} onClick={() => v.act('random')}>{say(ui('seed.random'))}</button>
      </div>
    </>
  );
}

/* one drawn star: stroked when not starred, filled when starred (G-D-22) */
const Star = () => <svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3.2l2.7 5.6 6.1.8-4.5 4.2 1.1 6.1L12 17l-5.4 2.9 1.1-6.1-4.5-4.2 6.1-.8z" fill="var(--star-fill, none)" stroke="currentColor" strokeWidth="1.75" strokeLinejoin="round" /></svg>;
const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

export function SavedSeeds(v: PartView) {
  const say = useSay(), m = v.instrument?.stars;
  const [open, setOpen] = useState(false), [q, setQ] = useState(''), [armed, setArmed] = useState<string | null>(null);
  const form = useRef<HTMLDivElement>(null), saveBtn = useRef<HTMLButtonElement>(null), arm = useRef<ReturnType<typeof setTimeout>>(undefined);
  /* the form arrives from just above (G-D-05) and its name is ready to type over */
  useLayoutEffect(() => {
    const el = form.current; if (!open || !el) return;
    if (!reduced()) el.animate([{ opacity: 0, transform: 'translateY(-6px)' }, { opacity: 1, transform: 'none' }], { duration: mtime('--t-base'), easing: motionVar('--e-out') });
    el.querySelector('input')?.select();
  }, [open]);
  useEffect(() => () => clearTimeout(arm.current), []);
  if (!m) return null;
  const rows = q ? m.rows.filter(r => new RegExp(escape(q), 'i').test(r.find)) : m.rows;
  const back = () => requestAnimationFrame(() => saveBtn.current?.focus({ preventScroll: true }));
  const cancel = () => { setOpen(false); back(); };
  const save = () => {
    const f = (k: string) => form.current?.querySelector<HTMLInputElement>(`[data-smf="${k}"]`)?.value ?? '';
    v.act('save', { name: f('n'), desc: f('d'), tags: f('t') }); setOpen(false); back();
  };
  const formKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (!(e.target as HTMLElement).matches('input')) return;
    if (e.key === 'Enter') { e.preventDefault(); e.stopPropagation(); save(); }
    else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); cancel(); }
  };
  /* delete asks first: the button says so for a moment, and a second press deletes (14-undo.js armed) */
  const del = (key: string) => {
    if (armed !== key) { setArmed(key); clearTimeout(arm.current); arm.current = setTimeout(() => setArmed(null), 2600); return; }
    setArmed(null); v.act('delete', key);
  };
  const searchKey = (e: KeyboardEvent<HTMLInputElement>) => { if (e.key === 'Escape' && q) { e.preventDefault(); e.stopPropagation(); setQ(''); } };
  const fm = m.form;
  return (
    <div className="seedmgr">
      <div className="sm-head">
        <span className="kl">{say(ui('stars.title'))}</span>
        <button ref={saveBtn} type="button" className="sbtn sm-savebtn" data-star="" aria-expanded={open} title={say(ui('stars.save.title'))} onClick={() => { if (v.act('open')) setOpen(true); }}><Star />{say(ui('stars.save'))}</button>
      </div>
      {open && fm ? (
        <div ref={form} className="sm-save" role="group" aria-label={say(ui(fm.old ? 'stars.form.update' : 'stars.form.save', { seed: String(v.state.find(s => s.kind === 'seed')?.value ?? '') }))} onKeyDown={formKey}>
          {fm.old ? <span className="sm-was">{say(ui('stars.was'))}</span> : null}
          <label className="sm-f"><span className="kl">{say(ui('stars.name'))}</span><input data-smf="n" type="text" maxLength={40} defaultValue={fm.name} /></label>
          <label className="sm-f"><span className="kl">{say(ui('stars.desc'))}</span><input data-smf="d" type="text" maxLength={120} defaultValue={fm.desc} placeholder={say(ui('stars.optional'))} /></label>
          <label className="sm-f"><span className="kl">{say(ui('stars.tags'))}</span><input data-smf="t" type="text" maxLength={80} defaultValue={fm.tags} placeholder={say(ui('stars.comma'))} /></label>
          <div className="sm-acts">
            <button type="button" className="small-btn sm-cancel" data-smcancel="" onClick={cancel}>{say(ui('stars.cancel'))}</button>
            <button type="button" className="small-btn primary" data-smsave="" onClick={save}>{say(ui(fm.old ? 'stars.update' : 'stars.saveSeed'))}</button>
          </div>
        </div>
      ) : null}
      <div className="sm-bar">
        <input className="sm-q" data-smq="" type="text" placeholder={say(ui('stars.search'))} value={q} aria-label={say(ui('stars.search.label'))} onChange={e => setQ(e.currentTarget.value)} onKeyDown={searchKey} />
        <button type="button" className="sbtn" data-smrand="" title={say(ui('stars.random.title'))} onClick={() => v.act('random', Math.random())}>{say(ui('stars.random'))}</button>
      </div>
      <div className="stars" data-stars="">
        {rows.length ? rows.map(r => {
          const k = String(r.v), long = r.desc.length > 44;
          return (
            <div key={k} className={`sm-row${r.star ? ' starred' : ''}${r.cur ? ' cur' : ''}`}>
              <button type="button" className="sm-star" data-smstar={k} aria-pressed={r.star} aria-label={say(ui(r.star ? 'stars.unstar' : 'stars.star', { name: r.name }))} onClick={() => v.act('star', k)}><Star /></button>
              <button type="button" className="sm-load" data-loadseed={k} aria-label={say(ui(r.desc ? 'stars.load.desc' : 'stars.load', { name: r.name, seed: k, desc: r.desc }))} title={long ? r.desc : undefined} data-tip-side={long ? 'beside' : undefined} onClick={() => v.act('load', k)}>
                <b>{r.name}</b><small>{k}</small>{r.desc ? <span className="sm-dt">{r.desc}</span> : null}
                {r.tags.length ? <span className="sm-d">{r.tags.map(t => <i key={t} className="sm-tag">{t}</i>)}</span> : null}
              </button>
              <button type="button" className={`sm-del${armed === k ? ' armed' : ''}`} data-smdel={k} data-armed={armed === k ? '1' : undefined} aria-label={say(ui('stars.delete', { name: r.name }))} onClick={() => del(k)} onBlur={() => { if (armed === k) setArmed(null); }}>
                {armed === k ? say(ui('stars.delete.ask')) : <Icon name="x" />}
              </button>
            </div>
          );
        }) : <p className="empty-note">{say(ui(q ? 'stars.noMatch' : 'stars.none'))}</p>}
      </div>
    </div>
  );
}

definePart('splitFlap', SplitFlap);
definePart('savedSeeds', SavedSeeds);
