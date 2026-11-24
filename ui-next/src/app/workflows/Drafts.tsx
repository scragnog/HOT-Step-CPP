// The go bar's Drafts (35-drafts.js): a popover above the go bar with a name field (focus starts there, Enter saves:
// F-18), Save draft (resting while the desk holds what the last draft holds), and the drafts, newest first, each with
// its engine, when, and how it differs; Load (switching the engine for another engine's draft) and Delete, both with
// Undo. What a draft holds and every word are the core's. It opens in the page's one popover (features/pop.ts), as
// the fixed panel above the go bar.

import { useRef, useState, type CSSProperties } from 'react';
import { engineName } from '../../definitions/engines.ts';
import { draftOf, draftSub, draftTitle, draftsAbout, sameDraft } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { deleteDraft, loadDraft, saveDraft } from '../store/workflows.ts';
import { ui, useSay } from '../i18n/index.ts';
import { Icon } from '../kit/icons.tsx';
import { ENGINE_COLOUR, logoOf } from '../shell/engines.ts';
import { useEdgeFade } from '../shell/edgeFade.ts';
import { closePop, openPop } from '../features/pop.ts';
import { usePop } from '../store/overlays.ts';
import { useWfModel } from './useWfModel.ts';

/** Opens the drafts from their button (or closes them, from it again). */
function openDrafts(b: HTMLElement): boolean {
  /* above the whole go bar (F-18) */
  const bar = document.querySelector<HTMLElement>('.gobar');
  return openPop(b, () => <DraftsList />, { label: ui('drafts'), fixed: true, focus: '#draftName', place: { prefer: 'up', ...(bar ? { over: bar } : {}) } });
}

export function DraftsButton() {
  const say = useSay(), open = usePop(s => s.anchor?.id === 'draftsBtn');
  return (
    <button className="tool drafts-btn" id="draftsBtn" type="button" aria-haspopup="dialog" aria-expanded={open} title={say(ui('drafts.title'))}
      onClick={e => { openDrafts(e.currentTarget); }}>{say(ui('drafts'))}</button>
  );
}

function DraftsList() {
  const say = useSay(), [name, setName] = useState(''), field = useRef<HTMLInputElement>(null), list = useRef<HTMLDivElement>(null);
  /* the list's edges fade where there is more (edgeFade) */
  useEdgeFade(list);
  const m = useWfModel((d, w) => {
    const now = Date.now(), e = d.desk.engine;
    return {
      about: draftsAbout(ix, d.desk),
      same: sameDraft(w.drafts.find(x => x.engine === e), draftOf(ix, d.desk, '')),
      rows: w.drafts.map(x => ({ id: x.id, engine: x.engine, other: x.engine !== e, title: draftTitle(x), sub: draftSub(x, e, now) })),
    };
  });
  /* after a change the list keeps focus on the same row's control, or the next one, or the field (C-25) */
  const keep = (sel: string) => requestAnimationFrame(() => (list.current?.querySelector<HTMLElement>(sel) ?? document.querySelector<HTMLElement>('#pop [data-draftload], #draftName'))?.focus({ preventScroll: true }));
  const save = () => { if (m.same && !name.trim()) return; saveDraft(name); setName(''); if (field.current) field.current.value = ''; field.current?.focus({ preventScroll: true }); };
  return (
    <>
      <h4>{say(ui('draft.h'))}</h4>
      <p className="pop-p" title={say(m.about.tip)}>{say(m.about.says)}</p>
      <div className="pop-row">
        <input ref={field} id="draftName" data-enter="" maxLength={40} placeholder={say(ui('draft.name'))} aria-label={say(ui('draft.name.label'))} {...(m.same ? { 'aria-describedby': 'draftSame' } : {})}
          onInput={e => setName(e.currentTarget.value)} onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); e.stopPropagation(); save(); } }} />
        <button type="button" className="small-btn primary" data-draftsave="" disabled={m.same && !name.trim()} onClick={save}>{say(ui('draft.save'))}</button>
      </div>
      {m.same ? <p className="pop-note" id="draftSame">{say({ id: 'draft.same' })}</p> : null}
      <div className="drafts" ref={list}>
        {m.rows.length ? m.rows.map((d, i) => (
          <div key={d.id} className="draft">
            <span className="rev-eng" style={{ '--ec': ENGINE_COLOUR[d.engine] } as CSSProperties} title={engineName(d.engine)} dangerouslySetInnerHTML={{ __html: logoOf(d.engine, 12, `draft ${d.id}`) }} />
            <span className="dr-t"><b title={say(d.title)}>{say(d.title)}</b><small>{say(d.sub)}</small></span>
            <button type="button" className="small-btn" data-draftload={i} {...(d.other ? { title: say({ id: 'draft.load.title', slots: { engine: engineName(d.engine) } }) } : {})} onClick={() => { closePop(false); loadDraft(d.id); }}>{say(ui('draft.load'))}</button>
            <button type="button" className="jump ico-btn" data-draftdel={i} aria-label={say({ id: 'draft.delete.label', slots: { name: d.title } })} title={say(ui('draft.delete'))}
              onClick={() => { deleteDraft(d.id); keep(`[data-draftdel="${Math.min(i, m.rows.length - 2)}"]`); }}><Icon name="x" size={13} /></button>
          </div>
        )) : <p className="empty-note">{say({ id: 'draft.none' })}</p>}
      </div>
    </>
  );
}
