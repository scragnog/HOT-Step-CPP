// The Workflows list (create/10-workflows-panel.html's #presetList, 32-workflows.js renderPresets): the user's
// workflows, then the factory's, each with its revisions under it; the open row, the modified rows with Revert and
// Update (or Save as…), a revision's details, renaming in place, the row menu, and the list as one Tab stop moved
// through with the arrow keys (F-07). Every word, "modified", and what each row holds are the core's (listModel).

import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type KeyboardEvent, type MouseEvent } from 'react';
import { engineName } from '../../definitions/engines.ts';
import { findRevision, listModel, nameKey, normName, revisionDetail, type Msg, type RevRow, type WfRow } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { deleteRev, deleteWf, loadRev, openWorkflow, renameRev, renameWf, updateRevision, updateWorkflow, useWf } from '../store/workflows.ts';
import { ui, useSay } from '../i18n/index.ts';
import { Icon, type IconName } from '../kit/icons.tsx';
import { CONCEPT_ID, ENGINE_COLOUR, logoOf } from '../shell/engines.ts';
import { useEdgeFade } from '../shell/edgeFade.ts';
import { exportRevision } from './io.ts';
import { Pop } from './pop.tsx';
import { closePop } from './popPlace.ts';
import { openSave } from './save.ts';
import { rowSel, useWfView, type RowRef } from './state.ts';
import { useWfModel } from './useWfModel.ts';

const sel = (r: RowRef) => rowSel(r, e => CONCEPT_ID[e]);
const ico = (name: IconName, size = 14) => <Icon name={name} size={size} />;
/** A row action whose button goes with it (Revert, Update) leaves focus on its row (F-03). */
const keepOn = (r: RowRef) => useDesk.getState().setFocus(sel(r));

const rowRef = (el: Element): RowRef | null => {
  const d = (el as HTMLElement).dataset;
  if (el.matches('.pre.wfrow')) return { kind: 'wf', name: d.preset!, user: useWf.getState().saved.workflows.some(w => w.name === d.preset) };
  if (el.matches('.pre.rev')) { const e = Object.entries(CONCEPT_ID).find(([, c]) => c === d.rb)?.[0]; return e ? { kind: 'rev', id: { workflow: d.rev!, engine: e as keyof typeof CONCEPT_ID, name: d.rn! }, user: d.ru === '1' } : null; }
  return null;
};

/* ---------------------------------------------------------------- renaming in place (G-D-30, F-11) */

function Rename({ row, name, commit }: { row: RowRef; name: string; commit: (v: string) => Msg | null }) {
  const say = useSay(), input = useRef<HTMLInputElement>(null), err = useWfView(s => s.renameErr), done = useRef(false);
  const setErr = (e: Msg | null) => useWfView.setState({ renameErr: e });
  /* the field holds the name as its value, not as an attribute (as the concept's renameInline sets it) */
  useEffect(() => { const f = input.current; if (!f) return; f.value = name; f.focus(); f.select(); }, [name]);
  const end = (ok: boolean, blurTo?: Element | null) => {
    if (done.current) return;
    const v = normName(input.current?.value ?? '');
    if (ok && v && v !== name) {
      const e = commit(v);
      if (e) { setErr(e); useDesk.getState().say(e); input.current?.focus(); return; }
    }
    done.current = true;
    useWfView.setState({ renaming: null, renameErr: null });
    /* focus stays on the row (under its new name), unless the blur took it elsewhere */
    if (!blurTo) requestAnimationFrame(() => document.querySelector<HTMLElement>(sel(row.kind === 'wf' ? { ...row, name: useWf.getState().saved.workflows.some(w => w.name === v) && ok ? v : name } : { ...row, id: { ...row.id, name: ok && findRevision(useWf.getState().saved, { ...row.id, name: v }, true) ? v : name } }))?.focus({ preventScroll: true }));
  };
  return (
    <>
      <input ref={input} className="pren" maxLength={28} aria-label={say(ui('wf.newName'))} aria-describedby="prenHint" {...(err ? { 'aria-invalid': true } : {})}
        onInput={() => setErr(null)} onClick={e => e.stopPropagation()}
        onKeyDown={e => { e.stopPropagation(); if (e.key === 'Enter') end(true); if (e.key === 'Escape') end(false); }}
        onBlur={e => end(true, e.relatedTarget)} />
    </>
  );
}
/** A refused rename's reason, under its row (the concept appends it to the row). */
function RenameHint() {
  const say = useSay(), err = useWfView(s => s.renameErr);
  return <p className="save-hint warn pren-hint" id="prenHint" hidden={!err}>{err ? say(err) : ''}</p>;
}

/* ---------------------------------------------------------------- the row menu (G-D-29) */

type MenuAct = 'wfren' | 'delpreset' | 'rren' | 'rexp' | 'rdel';
const MENU_ICON: Record<MenuAct, IconName> = { wfren: 'pencil', delpreset: 'trash', rren: 'pencil', rexp: 'download', rdel: 'trash' };
const MENU_WORD: Record<MenuAct, string> = { wfren: 'wf.rename', delpreset: 'wf.delete', rren: 'wf.rename', rexp: 'wf.export', rdel: 'wf.delete' };

function runAct(row: RowRef, act: MenuAct): void {
  if (act === 'wfren' || act === 'rren') { useWfView.setState({ renaming: row }); return; }
  if (act === 'rexp' && row.kind === 'rev') { exportRevision(row.id, row.user); return; }
  /* deleting: focus goes to the row that takes its place, or to + Save (C-12) */
  const here = document.querySelector(sel(row)), rows = [...document.querySelectorAll<HTMLElement>(row.kind === 'wf' ? '#presetList .pre.wfrow' : '.revs .pre.rev')].filter(x => row.kind === 'wf' || x.closest('.revs') === here?.closest('.revs'));
  const i = here ? rows.indexOf(here as HTMLElement) : -1, next = rows[i + 1] ?? rows[i - 1], to = next ? rowRef(next) : null;
  const up = row.kind === 'rev' ? `.pre.wfrow[data-preset="${CSS.escape(row.id.workflow)}"]` : null;
  if (act === 'delpreset' && row.kind === 'wf') deleteWf(row.name);
  if (act === 'rdel' && row.kind === 'rev') deleteRev(row.id);
  useDesk.getState().setFocus([...(to ? [sel(to)] : []), ...(up ? [up] : []), '#presetList .pre', '#saveBtn']);
}

function RowMenu({ row, acts, armed0 }: { row: RowRef; acts: MenuAct[]; armed0?: MenuAct | undefined }) {
  const say = useSay(), [armed, setArmed] = useState<MenuAct | null>(armed0 ?? null);
  useEffect(() => { if (!armed) return; const t = setTimeout(() => setArmed(null), 2600); return () => clearTimeout(t); }, [armed]);
  const press = (a: MenuAct) => {
    if ((a === 'delpreset' || a === 'rdel') && armed !== a) { setArmed(a); return; }
    closePop(false);
    /* the menu's action is the row's: focus is back on the row's ⋯ first, so it stays there (Export) unless the action
       moves it on (Rename's field) (F-03) */
    document.querySelector<HTMLElement>(`${sel(row)} [data-rowmore]`)?.focus({ preventScroll: true });
    runAct(row, a);
  };
  return (
    <div className="rowmenu" role="menu">
      {acts.map(a => {
        const danger = a === 'delpreset' || a === 'rdel';
        return <button key={a} type="button" role="menuitem" className={`${danger ? 'danger' : ''}${armed === a ? ' armed' : ''}`} data-rowact={a} onClick={() => press(a)}>
          {armed === a ? say(ui('wf.delete.armed')) : <>{ico(MENU_ICON[a])}<span>{say(ui(MENU_WORD[a]))}</span></>}
        </button>;
      })}
    </div>
  );
}

/* ---------------------------------------------------------------- rows */

interface RowProps { tab: (r: RowRef) => number; tier: string }

function More({ row, name, tab }: { row: RowRef; name: string; tab: number }) {
  const say = useSay(), open = useWfView(s => s.pop?.kind === 'menu' && sel(s.pop.row) === sel(row));
  return <button type="button" className="jump ico-btn" data-rowmore="" aria-haspopup="menu" {...(open ? { 'aria-expanded': true } : {})} aria-label={say(ui('wf.more.label', { name }))} title={say(ui('wf.more'))} tabIndex={tab}
    onClick={e => { e.stopPropagation(); if (open) { closePop(); return; } useWfView.setState({ pop: { kind: 'menu', row, anchor: e.currentTarget } }); }}>{ico('more')}</button>;
}

/** A row's own key handling: Enter and Space act as a click; the arrows move through the list (F-07). */
function rowKeys(e: KeyboardEvent<HTMLDivElement>, act: () => void): void {
  const row = e.currentTarget;
  if (e.target !== row || e.altKey || e.ctrlKey || e.metaKey) return;
  if (e.key === 'Enter' || (e.key === ' ' && !e.repeat)) { e.preventDefault(); act(); return; }
  const rows = [...document.querySelectorAll<HTMLElement>('#presetList .pre')].filter(x => x.getClientRects().length), i = rows.indexOf(row);
  const chev = row.querySelector<HTMLButtonElement>('[data-wfopen]'), open = chev?.getAttribute('aria-expanded') === 'true';
  const go = (r: HTMLElement | undefined) => { if (!r) return; const ref = rowRef(r); if (ref) useWfView.setState({ rove: sel(ref) }); r.focus(); r.scrollIntoView({ block: 'nearest' }); };
  const again = () => requestAnimationFrame(() => { const ref = rowRef(row); if (ref) document.querySelector<HTMLElement>(sel(ref))?.focus(); });
  /* F2 renames a row of yours, Delete arms its Delete in its menu, a letter moves to the next row starting with it
     (F-36) */
  const ref = rowRef(row);
  if (ref?.user && e.key === 'F2') { e.preventDefault(); runAct(ref, ref.kind === 'wf' ? 'wfren' : 'rren'); return; }
  if (ref?.user && e.key === 'Delete') { const more = row.querySelector<HTMLElement>('[data-rowmore]'); if (more) { e.preventDefault(); useWfView.setState({ pop: { kind: 'menu', row: ref, anchor: more, armed: ref.kind === 'wf' ? 'delpreset' : 'rdel' } }); } return; }
  if (e.key.length === 1 && e.key !== ' ') {
    const name = (r: HTMLElement) => r.dataset.rn ?? r.dataset.preset ?? '', k = nameKey(e.key);
    const next = [...rows.slice(i + 1), ...rows.slice(0, i)].find(r => nameKey(name(r)).startsWith(k));
    if (next) { e.preventDefault(); go(next); }
    return;
  }
  let done = true;
  if (e.key === 'ArrowDown') go(rows[i + 1]); else if (e.key === 'ArrowUp') go(rows[i - 1]); else if (e.key === 'Home') go(rows[0]); else if (e.key === 'End') go(rows[rows.length - 1]);
  else if (e.key === 'ArrowRight') { if (row.matches('.wfrow') && chev && !chev.disabled) { if (open) go(rows[i + 1]); else { chev.click(); again(); } } }
  else if (e.key === 'ArrowLeft') { if (row.matches('.rev')) go(row.closest('.wf')?.querySelector<HTMLElement>('.wfrow') ?? undefined); else if (open && chev) { chev.click(); again(); } }
  else done = false;
  if (done) e.preventDefault();
}

function Detail({ row }: { row: RevRow }) {
  const say = useSay();
  const d = useWfModel((s, w) => { const r = findRevision(w.saved, row.id, row.user); return r ? revisionDetail(ix, r, s.desk, row.modified) : null; }, JSON.stringify([row.id, row.user, row.modified]));
  if (!d) return null;
  return (
    <div className="pdetail">
      {d.empty ? <span className="empty-note">{say(d.empty)}</span> : d.groups.map((g, i) => (
        <div key={i} className="pd-g"><i>{say(g.name)}</i>
          {g.rows.map((x, j) => x.wide ? <span key={j} className="pd-wide">{say(x.value)}</span>
            : x.now !== undefined ? <span key={j} className="pd-ch">{say(x.label)}<b>{say(x.value)} <em aria-label={say(ui('wf.now'))}>{'→ '}{say(x.now)}</em></b></span>
              : <span key={j}>{say(x.label)}<b>{say(x.value)}</b></span>)}
        </div>))}
    </div>
  );
}

function RevisionRow({ r, tab }: { r: RevRow } & Pick<RowProps, 'tab'>) {
  const say = useSay(), ref: RowRef = { kind: 'rev', id: r.id, user: r.user }, t = tab(ref), renaming = useWfView(s => s.renaming?.kind === 'rev' && sel(s.renaming) === sel(ref));
  const key = `${r.id.workflow}|${r.id.engine}|${r.id.name}|${r.user ? 1 : 0}`, info = useWf(s => s.info === key), name = r.id.name;
  const attrs = { 'data-rev': r.id.workflow, 'data-rn': name, 'data-rb': CONCEPT_ID[r.engine], 'data-ru': r.user ? '1' : '0' };
  const load = () => loadRev(r.id, r.user);
  const btn = (o: { data: string; icon: IconName; label: Msg; title?: Msg; menu?: boolean; mlabel?: string; onClick: () => void; extra?: Record<string, unknown> }) => (
    <button type="button" className={`jump ico-btn${o.menu ? ' in-menu' : ''}`} {...{ [o.data]: '' }} {...attrs} {...(o.mlabel ? { 'data-mlabel': say(ui(o.mlabel)) } : {})} {...o.extra}
      title={o.title ? say(o.title) : undefined} aria-label={say(o.label)} tabIndex={t} onClick={e => { e.stopPropagation(); o.onClick(); }}>{ico(o.icon)}</button>);
  return (
    <>
      <div className={`pre rev${r.on ? ' on' : ''}${r.modified ? ' mod' : ''}`} {...attrs} role="button" tabIndex={t} aria-label={say(r.label)} {...(r.on ? { 'aria-current': true } : {})} title={say(r.tip)}
        onClick={(e: MouseEvent) => { if (!(e.target as Element).closest('button, input')) load(); }} onKeyDown={e => rowKeys(e, load)} onFocus={() => useWfView.setState({ rove: sel(ref) })}>
        <span className="rev-eng" style={{ '--ec': ENGINE_COLOUR[r.engine] } as CSSProperties} title={engineName(r.engine)} dangerouslySetInnerHTML={{ __html: logoOf(r.engine, 13, `rev ${key}`) }} />
        {renaming ? <Rename row={ref} name={name} commit={v => renameRev(r.id, v)} /> : <b>{name}</b>}
        {r.mine ? <i className="pmine" title={say(ui('wf.yoursMark.title'))} aria-hidden="true">{say(ui('wf.yoursMark'))}</i> : null}
        <span className="pslot">
          {r.modified || !r.holds ? null : <span className="ptier" title={r.holdsTip ? say(r.holdsTip) : undefined}>{ico(r.holds === 'lyrics' ? 'note' : 'lines', 13)}<span className="sr-only">{say(ui(r.holds === 'lyrics' ? 'wf.holds.lyrics' : 'wf.holds.box'))}</span></span>}
          <span className="pacts">
            {btn({ data: 'data-rinfo', icon: 'info', label: { id: 'rev.row.info', slots: { name } }, title: { id: r.modified ? 'rev.row.info.title.mod' : 'rev.row.info.title' }, extra: { 'aria-expanded': info }, onClick: () => useWf.setState({ info: info ? null : key }) })}
            {r.user ? btn({ data: 'data-rren', icon: 'pencil', label: ui('wf.rename.label', { name }), menu: true, mlabel: 'wf.rename', onClick: () => runAct(ref, 'rren') }) : null}
            {btn({ data: 'data-rexp', icon: 'download', label: ui('wf.export.label', { name }), title: ui('wf.export'), menu: r.user, mlabel: 'wf.export', onClick: () => exportRevision(r.id, r.user) })}
            {r.user ? <>{btn({ data: 'data-rdel', icon: 'trash', label: ui('wf.delete.label', { name }), menu: true, mlabel: 'wf.delete', onClick: () => runAct(ref, 'rdel') })}<More row={ref} name={name} tab={t} /></> : null}
          </span>
        </span>
        {r.modified ? <span className="pmodrow">
          <button type="button" className="jump show" data-rrevert="" {...attrs} title={say({ id: 'rev.revert.title', slots: { name } })} tabIndex={t} onClick={e => { e.stopPropagation(); keepOn(ref); loadRev(r.id, r.user, true); }}>{ico('undo', 13)}{say(ui('wf.revert'))}</button>
          {r.user
            ? <button type="button" className="jump show" data-rover="" {...attrs} title={say({ id: 'rev.update.title' })} tabIndex={t} onClick={e => { e.stopPropagation(); keepOn(ref); updateRevision(r.id); }}>{ico('saveIn', 13)}{say(ui('wf.update'))}</button>
            : <button type="button" className="jump show" data-saveasnew={`rev|${name}`} title={say({ id: 'rev.saveAs.title' })} tabIndex={t} onClick={e => { e.stopPropagation(); openSave('rev', { id: 'wf.saveAs.name', slots: { name } }); }}>{ico('pagePlus', 13)}{say(ui('wf.saveAs'))}</button>}
        </span> : null}
        {renaming ? <RenameHint /> : null}
      </div>
      {info ? <Detail row={r} /> : null}
    </>
  );
}

function WorkflowRow({ w, tab, tier }: { w: WfRow } & RowProps) {
  const say = useSay(), ref: RowRef = { kind: 'wf', name: w.name, user: w.user }, t = tab(ref), renaming = useWfView(s => s.renaming?.kind === 'wf' && s.renaming.name === w.name);
  const shown = useWf(s => w.revisions.length > 0 && ((w.on && !s.fold) || (!w.on && s.expanded.includes(w.name))));
  const n = w.revisions.length, open = () => openWorkflow(w.name);
  const chev = () => { const s = useWf.getState(); if (w.on) useWf.setState({ fold: !s.fold }); else useWf.setState({ expanded: s.expanded.includes(w.name) ? s.expanded.filter(x => x !== w.name) : [...s.expanded, w.name] }); };
  const btn = (data: string, icon: IconName, label: Msg, mlabel: string, run: () => void) => (
    <button type="button" className="jump ico-btn in-menu" {...{ [data]: w.name }} data-mlabel={say(ui(mlabel))} aria-label={say(label)} tabIndex={t} onClick={e => { e.stopPropagation(); run(); }}>{ico(icon)}</button>);
  return (
    <div className={`wf${shown ? ' open' : ''}`}>
      <div className={`pre wfrow${w.on ? ' on' : ''}${w.modified ? ' mod' : ''}`} data-preset={w.name} role="button" tabIndex={t} aria-label={say(w.label)} {...(w.on ? { 'aria-current': true } : {})} title={say(w.tip)}
        onClick={(e: MouseEvent) => { if (!(e.target as Element).closest('button, input')) open(); }} onKeyDown={e => rowKeys(e, open)} onFocus={() => useWfView.setState({ rove: sel(ref) })}>
        <button type="button" className="pchev" data-wfopen={w.name} aria-expanded={shown} disabled={!n} title={say({ id: 'wf.row.count', slots: { n } })} aria-label={say({ id: 'wf.row.chev', slots: { name: w.name } })} tabIndex={t}
          onClick={e => { e.stopPropagation(); chev(); }}>{ico('chevR', 12)}</button>
        <span className={`pdot ${w.tier}`} title={say(w.tierWord)}><span className="sr-only">{say(w.tierWord)}</span></span>
        {renaming ? <Rename row={ref} name={w.name} commit={v => renameWf(w.name, v)} /> : <b>{w.name}</b>}
        <span className="pslot">
          {w.modified ? null : <span className="ptier">{n ? <i className="rc" aria-label={say({ id: 'wf.row.count', slots: { n } })}>{n}</i> : null}</span>}
          {w.tier !== tier || w.user ? <span className="pacts">
            {w.tier !== tier ? <button type="button" className="jump ico-btn" data-presettier={w.name} title={say({ id: 'wf.row.openIn', slots: { tier: w.tierWord } })} aria-label={say({ id: 'wf.row.openIn.label', slots: { name: w.name, tier: w.tierWord } })} tabIndex={t}
              onClick={e => { e.stopPropagation(); keepOn(ref); openWorkflow(w.name, { tier: true }); }}>{ico('openIn')}</button> : null}
            {w.user ? <>{btn('data-wfren', 'pencil', ui('wf.rename.label', { name: w.name }), 'wf.rename', () => runAct(ref, 'wfren'))}{btn('data-delpreset', 'trash', ui('wf.delete.label', { name: w.name }), 'wf.delete', () => runAct(ref, 'delpreset'))}<More row={ref} name={w.name} tab={t} /></> : null}
          </span> : null}
        </span>
        {w.modified ? <span className="pmodrow">
          <button type="button" className="jump show" data-revert={w.name} title={say({ id: 'wf.revert.wf.title' })} tabIndex={t} onClick={e => { e.stopPropagation(); keepOn(ref); openWorkflow(w.name, { revert: true }); }}>{ico('undo', 13)}{say(ui('wf.revert'))}</button>
          {w.user
            ? <button type="button" className="jump show" data-wfupd={w.name} title={say({ id: 'wf.update.wf.title' })} tabIndex={t} onClick={e => { e.stopPropagation(); keepOn(ref); updateWorkflow(w.name); }}>{ico('saveIn', 13)}{say(ui('wf.update'))}</button>
            : <button type="button" className="jump show" data-saveasnew={`wf|${w.name}`} title={say({ id: 'wf.saveAs.wf.title' })} tabIndex={t} onClick={e => { e.stopPropagation(); openSave('wf', { id: 'wf.saveAs.name', slots: { name: w.name } }); }}>{ico('pagePlus', 13)}{say(ui('wf.saveAs'))}</button>}
        </span> : null}
        {renaming ? <RenameHint /> : null}
      </div>
      {shown ? <div className="revs">{w.revisions.map(r => <RevisionRow key={`${r.user ? 1 : 0}|${r.engine}|${r.id.name}`} r={r} tab={tab} />)}</div> : null}
    </div>
  );
}

/* ---------------------------------------------------------------- the list */

export function WorkflowList() {
  const say = useSay(), model = useWfModel((d, w) => listModel(ix, w.saved, w.open, d.desk, d.mods, d.board)), tier = useDesk(s => s.desk.tier);
  const rove = useWfView(s => s.rove), fold = useWf(s => s.fold), expanded = useWf(s => s.expanded), list = useRef<HTMLElement | null>(null);
  /* the list is #presetList, the shell's (drawn once this is): its edges fade where there is more (edgeFade) */
  useLayoutEffect(() => { list.current = document.getElementById('presetList'); }, []);
  useEdgeFade(list);
  /* the row that takes Tab: the one last focused, else the open workflow, else the first (F-07) */
  const rows: RowRef[] = [...model.yours, ...model.factory].flatMap(w => [{ kind: 'wf' as const, name: w.name, user: w.user },
    ...(w.revisions.length && ((w.on && !fold) || (!w.on && expanded.includes(w.name))) ? w.revisions.map(r => ({ kind: 'rev' as const, id: r.id, user: r.user })) : [])]);
  const sels = rows.map(sel), openRow = [...model.yours, ...model.factory].find(w => w.on);
  const stop = rove && sels.includes(rove) ? rove : openRow ? sel({ kind: 'wf', name: openRow.name, user: openRow.user }) : sels[0] ?? null;
  const tab = (r: RowRef) => (sel(r) === stop ? 0 : -1);
  /* the row that first takes Tab keeps it until another row is focused (32-workflows.js roveRows) */
  useEffect(() => { if (stop && !useWfView.getState().rove) useWfView.setState({ rove: stop }); });
  /* a name cut short leads its row's tooltip (F-05) */
  useLayoutEffect(() => {
    document.querySelectorAll<HTMLElement>('#presetList .pre > b').forEach(b => { const row = b.closest<HTMLElement>('.pre')!; if (b.scrollWidth > b.clientWidth + 1 && !row.title.startsWith(b.textContent + ' · ')) row.title = `${b.textContent} · ${row.title}`; });
  });
  /* a renamed or imported row is brought into view and washed green (G-D-30, F-20) */
  const wash = useWf(s => s.wash);
  useEffect(() => {
    if (!wash) return;
    const row = document.querySelector<HTMLElement>(wash.sel); if (!row) return;
    const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
    row.scrollIntoView({ block: 'nearest', behavior: still ? 'auto' : 'smooth' });
    if (!still) row.animate([{ backgroundColor: '#1d9a5b33' }, { backgroundColor: 'transparent' }], { duration: 500, easing: 'cubic-bezier(.2,.8,.2,1)' });
  }, [wash]);
  const menu = useWfView(s => (s.pop?.kind === 'menu' ? s.pop : null));
  return (
    <>
      <div className="pre-group">{say(ui('wf.yours'))}</div>
      {model.yours.length ? model.yours.map(w => <WorkflowRow key={'u:' + w.name} w={w} tab={tab} tier={tier} />) : <p className="empty-note">{model.empty ? say(model.empty) : null}</p>}
      <div className="pre-group">{say(ui('wf.factory'))}</div>
      {model.factory.map(w => <WorkflowRow key={'f:' + w.name} w={w} tab={tab} tier={tier} />)}
      {menu ? <Pop label={menu.anchor.getAttribute('aria-label') ?? say(ui('wf.more'))}><RowMenu key={sel(menu.row) + (menu.armed ?? '')} row={menu.row} acts={menuActs(menu.row)} armed0={menu.armed} /></Pop> : null}
    </>
  );
}
/** The row menu's actions: the row's own, in its order (rename, export, delete). */
const menuActs = (r: RowRef): MenuAct[] => (r.kind === 'wf' ? ['wfren', 'delpreset'] : ['rren', 'rexp', 'rdel']);
