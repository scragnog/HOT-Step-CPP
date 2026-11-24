// The Workflows panel's head buttons (+ Save, Import) and its save row (10-workflows-panel.html, 32-workflows.js
// openSave, paintSave, doSave): a revision of the open workflow or a new workflow, named in the row; a name the core
// refuses is said beside the field and focus stays in it (C-22), a name in use asks to Replace (C-02), Escape closes
// the row (from inside it only: F-27), and a name typed and abandoned does not come back (C-22).

import { useEffect, useRef, useState } from 'react';
import { findWorkflow, saveCheck, saveHint, type Msg } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { layoutChanged, saveRevision, saveWorkflow, useWf } from '../store/workflows.ts';
import { ui, useSay } from '../i18n/index.ts';
import { Icon } from '../kit/icons.tsx';
import { importChosen } from './io.ts';
import { closeSave, openSave, openWf } from './save.ts';
import { useWfView } from './state.ts';
import { useWfModel } from './useWfModel.ts';

const MAX = 28;

export function HeadButtons() {
  const say = useSay(), open = useWfView(s => s.saveOpen), file = useRef<HTMLInputElement>(null);
  return (
    <>
      <button type="button" className="small-btn" id="saveBtn" aria-controls="saveRow" aria-expanded={open} onClick={() => (useWfView.getState().saveOpen ? closeSave(false) : openSave())}>
        {/* the concept's own plus, a heavier stroke than the icon set's */}
        <svg className="ico" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M12 5v14M5 12h14" /></svg>{say(ui('panel.save'))}
      </button>
      <button type="button" className="icon-btn" id="importBtn" title={say(ui('panel.import.title'))} aria-label={say(ui('panel.import'))} onClick={() => file.current?.click()}><Icon name="upload" size={15} /></button>
      <input ref={file} type="file" id="importFile" accept="application/json,.json" hidden onChange={e => { const f = e.currentTarget.files?.[0]; e.currentTarget.value = ''; if (f) importChosen(f); }} />
    </>
  );
}

export function SaveRow() {
  const say = useSay(), v = useWfView(), field = useRef<HTMLInputElement>(null), inc = useRef<HTMLInputElement>(null), [count, setCount] = useState(0);
  const m = useWfModel((d, w) => {
    const wf = findWorkflow(w.saved, w.open.workflow)?.w.name ?? null, rev = v.saveAs === 'rev' && !!wf;
    return { wf, rev, hint: saveHint(ix, rev ? 'rev' : 'wf', wf, d.desk, rev && layoutChanged()) };
  }, v.saveAs);
  /* opening the row puts focus in the field, with what it starts with selected */
  useEffect(() => {
    const f = field.current; if (!f || !v.saveOpen) return;
    if (v.prefill) { f.value = v.prefill.text ? say(JSON.parse(v.prefill.text) as Msg).slice(0, MAX) : ''; setCount(f.value.length); }
    f.focus(); f.select();
  }, [v.saveOpen, v.prefill]); // eslint-disable-line react-hooks/exhaustive-deps -- say is stable for a language
  /* the row closes empty (C-22) */
  useEffect(() => { if (!v.saveOpen && field.current) { field.current.value = ''; setCount(0); } }, [v.saveOpen]);
  const refuse = (says: Msg, confirm: string | null = null) => {
    useWfView.setState({ err: { says, n: (useWfView.getState().err?.n ?? 0) + 1 }, confirm });
    useDesk.getState().say(says); field.current?.focus();
  };
  const save = () => {
    const s = useWfView.getState(), wf = openWf(), as = s.saveAs === 'rev' && wf ? 'rev' : 'wf', typed = field.current?.value ?? '';
    const c = saveCheck(useWf.getState().saved, as, wf, useDesk.getState().desk.engine, typed);
    if ('refuse' in c) { refuse(c.refuse); return; }
    if ('replace' in c && s.confirm !== c.replace) { refuse(c.says, c.replace); return; }
    const name = 'ok' in c ? c.ok : null, replace = 'replace' in c ? c.replace : null, withContent = !!inc.current?.checked;
    closeSave(false);
    useDesk.getState().setFocus('#saveBtn');
    if (as === 'rev') saveRevision(name ?? replace!, withContent, replace);
    else saveWorkflow(name ?? replace!, replace, () => openSave('rev'));
  };
  const typed = () => { setCount(field.current?.value.length ?? 0); if (useWfView.getState().err || useWfView.getState().confirm) useWfView.setState({ err: null, confirm: null }); };
  const hint = v.err ? v.err.says : m.hint.says, warn = v.saveOpen && (!!v.err || m.hint.warn);
  return (
    <div className="save-row" id="saveRow" hidden={!v.saveOpen}>
      <div className="save-as" role="group" aria-label={say(ui('wf.saveAs.group'))}>
        <button type="button" data-saveas="rev" aria-pressed={m.rev} disabled={!m.wf} title={say(!v.saveOpen ? ui('wf.saveAs.rev.title') : m.wf ? { id: 'wf.saveAs.rev.title', slots: { name: m.wf } } : { id: 'wf.saveAs.rev.none' })}
          onClick={() => { useWfView.setState({ saveAs: 'rev', confirm: null, err: null }); field.current?.focus(); }}>{say(ui('wf.saveAs.rev'))}</button>
        <button type="button" data-saveas="wf" aria-pressed={!m.rev} onClick={() => { useWfView.setState({ saveAs: 'wf', confirm: null, err: null }); field.current?.focus(); }}>{say(ui('wf.saveAs.wf'))}</button>
      </div>
      <span className="save-field">
        <input ref={field} id="saveName" placeholder={say(ui(!v.saveOpen ? 'wf.name.it' : m.rev ? 'wf.name.rev' : 'wf.name.wf'))} maxLength={MAX} aria-label={say(ui('wf.name'))} aria-describedby="saveHint saveCount" onInput={typed}
          onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); e.stopPropagation(); save(); } if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); closeSave(true); } }} />
        <span className="save-count mono" id="saveCount" hidden={count < MAX - 5} aria-live="polite">{v.saveOpen ? say(ui('wf.count', { n: count, max: MAX })) : ''}</span>
      </span>
      <button type="button" className={`small-btn primary${v.confirm ? ' warn' : ''}`} id="saveOk" onClick={save}>{say(ui(v.confirm ? 'wf.replace' : 'wf.save'))}</button>
      <label className="save-inc" id="saveInc" hidden={!m.rev}><input ref={inc} type="checkbox" id="incContent" defaultChecked /> {say(ui('wf.include'))}</label>
      <p className={`save-hint${warn ? ' warn' : ''}`} id="saveHint" hidden={!v.saveOpen} aria-live="polite">{v.saveOpen ? say(hint) : ''}</p>
    </div>
  );
}
