// promptBox: the prompt card (create/20-centre.html, 33-prompt.js): the main box (a description, a caption, a prompt or
// tags, by engine), the caption a dataset supplies and how to edit a copy, the dataset's tags, the suggestions the box's
// words call for, the tools row and Song info. What it shows (the box's words, the suggestions, which tools show, the
// supplied caption) is the core's (PartModel.prompt); typing sets the box, one step per visit; every tool is one of
// its node's named actions, run by the prompt feature (features/prompt-tools.ts), which names its step and says what it
// did. The cover art subject is song info while the Cover art unit is on.

import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import { definePart, Icon, ui, useSay, type ParamView, type PartView, type Slot } from '../kit/index.ts';
import { Pick } from './Pick.tsx';
import { subView } from './subView.ts';
import { useGrow } from './textGrow.ts';

const INFO = ['song.artist', 'song.title', 'song.subject'];
/* the box grows to a third of the window (150px at least), then scrolls (19-text-boxes.js growCap) */
const CAP = () => Math.max(150, Math.round(innerHeight * 0.3));
/* the placeholder turns through the examples every 3.5 s while the box is empty and has no focus (S-35: not under
   reduced motion) */
const TURN_MS = 3500;
const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

const LOCK = (
  <svg className="ico" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <rect x="5" y="11" width="14" height="10" rx="2" /><path d="M8 11V7a4 4 0 0 1 8 0v4" />
  </svg>
);

export function PromptBox(v: PartView) {
  const say = useSay(), ta = useRef<HTMLTextAreaElement>(null), info = useRef<HTMLDivElement>(null), m = v.prompt, box = v.params[0];
  const of = (k: string) => v.params.find(p => p.key === k);
  const act = (name: string, arg?: string) => { v.act(name, arg); };
  const words = (name: string) => v.actions.find(a => a.name === name);
  const [ph, setPh] = useState(0), [ping, setPing] = useState(0);
  const any = !!m?.info.any, seen = useRef(any), [open, setOpen] = useState(false);
  const g = useGrow(ta, CAP, m?.shown ?? '');
  /* Song info opens by itself only when info arrives from outside (a draft, a revision), never again after it was closed */
  useEffect(() => { if (any && !seen.current) setOpen(true); seen.current = any; }, [any]);
  useEffect(() => {
    const t = setInterval(() => { if (!reduced() && !ta.current?.value && document.activeElement !== ta.current) setPh(i => i + 1); }, TURN_MS);
    return () => clearInterval(t);
  }, []);
  /* a supplied caption can't be typed over: the way to edit it rings, and the page says why (at most every 6 s) */
  const said = useRef(-Infinity);
  const ring = () => { setPing(n => n + 1); };
  useEffect(() => { if (!ping) return; const t = setTimeout(() => setPing(0), 1100); return () => clearTimeout(t); }, [ping]);
  if (!m || !box) return null;
  const locked = !!m.locked;
  const lockedKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (!locked || e.ctrlKey || e.metaKey || e.altKey || (e.key.length !== 1 && e.key !== 'Backspace' && e.key !== 'Delete')) return;
    ring(); if (performance.now() - said.current < 6000) return; said.current = performance.now(); act('promptLocked');
  };
  const field = (p: ParamView | undefined, cls?: string) => p ? (
    <label key={p.key} className={cls} hidden={cls === 'si-art' ? !m.info.art : undefined}><span>{say(p.label)}</span>
      <input data-bind={p.key} maxLength={p.maxLength} placeholder={p.placeholder ? say(p.placeholder) : undefined} value={String(p.value ?? '')}
        onFocus={() => v.gesture.begin(p.key)} onBlur={() => v.gesture.end(p.key)} onChange={e => { v.gesture.begin(p.key); p.set(e.currentTarget.value); }} />
    </label>
  ) : null;
  const toggleInfo = () => { const to = !open; setOpen(to); if (to) requestAnimationFrame(() => info.current?.querySelector('input')?.focus({ preventScroll: true })); };
  const src = of('cap.src'), ds = of('cap.ds'), wild = of('wild.auto'), trig = of('ad.trigger');
  /* the caption source and its dataset are chips (G-D-8); choosing a dataset is the feature's, so a track it lacks goes
     back to Custom in the same step (E-22) */
  const chip = (p: ParamView, label: Slot, title?: Slot | null, set?: (x: string | number | boolean) => void) =>
    <Pick {...subView(v, { params: [set ? { ...p, set } : p], opts: { variant: 'chip' }, label, title: title ?? undefined })} />;
  const tool = (name: string, id: string, hidden: boolean, extra: Record<string, unknown> = {}) => {
    const w = words(name);
    return <button className="tool" id={id} hidden={hidden} title={w?.tip ? say(w.tip) : undefined} onClick={() => act(name, extra.arg as string | undefined)} {...(extra.data as object)}>{say(w?.label)}</button>;
  };
  return (
    <div className="prompt-card">
      <textarea id="prompt" ref={ta} rows={2} aria-label={say(m.aria)} placeholder={say(m.examples[ph % m.examples.length])} value={m.shown} readOnly={locked}
        className={[locked ? 'locked' : '', g.cls].filter(Boolean).join(' ') || undefined}
        onFocus={() => v.gesture.begin('text')} onBlur={() => v.gesture.end('text')}
        onChange={e => { if (locked) return; v.gesture.begin('text'); box.set(e.currentTarget.value); }}
        onKeyDown={lockedKey} onClick={() => { if (locked) ring(); }} />
      <div className="cap-lock" id="capLock" hidden={!locked}>
        {m.locked ? <>{LOCK}<span>{say(m.locked.line)}</span><button type="button" className={`linkbtn${ping ? ' ping' : ''}`} data-capcustom="" onClick={() => act('editCaptionCopy')}>{say(words('editCaptionCopy')?.label)}</button></> : null}
      </div>
      <div className="dstags" id="dsTags" hidden={!m.tools.tags}>
        {m.tools.tags ? <><div className="dst-head" title={say(m.tagsTip)}><b>{say(words('datasetTag')?.label)}</b></div>
          <div className="dst-chips">{m.tags.map(t => <button key={t.tag} type="button" className={`dst ${t.on ? 'on' : ''}`} data-dstag={t.tag} aria-pressed={t.on} onClick={() => act('datasetTag', t.tag)}>{t.tag}</button>)}</div></> : null}
      </div>
      <div className="suggest" id="suggest">
        {m.suggestions.map(s => s.kind === 'wild' ? (
          <span key={s.kind} className="sg-note">{say(s.lead)}{(s.items ?? []).map((x, i) => <span key={i}>{i ? ', ' : ''}<b>{x}</b></span>)}{'. '}{say(s.tail)}{s.link ? <>{' '}<button type="button" className="linkbtn" data-wildx="prompt" onClick={() => act('expandWildcards', 'prompt')}>{say(s.link)}</button></> : null}</span>
        ) : (
          <button key={s.kind} type="button" data-sg={s.kind} data-v={s.bpm} onClick={() => act(s.kind === 'lyrics' ? 'moveToLyrics' : s.kind === 'bpm' ? 'tempoFromPrompt' : 'plainCaption', s.bpm === undefined ? undefined : String(s.bpm))}>{say(s.lead)}<b>{say(s.bold)}</b></button>
        ))}
      </div>
      <div className="pc-foot">
        <button className="tool" id="ideasBtn" aria-haspopup="dialog" aria-expanded="false" hidden={!m.tools.presets} onClick={() => act('stylePresets')}>{say(words('stylePresets')?.label)}</button>
        {tool('promptWrite', 'aiBtn', !m.tools.write)}
        <button className={`tool${any ? ' on' : ''}`} id="songInfoBtn" aria-expanded={open} aria-controls="songInfo" onClick={toggleInfo}><i className="si-dot" aria-hidden="true" />{say(ui('prompt.songInfo'))}</button>
        <label className="tool" id="latBtn" title={say(words('importLatent')?.tip)} hidden={!m.tools.latent}
          onDragOver={e => { if (e.dataTransfer.types.includes('Files')) e.preventDefault(); }} onDrop={e => { const f = e.dataTransfer.files[0]; if (!f) return; e.preventDefault(); e.stopPropagation(); act('importLatent', f.name); }}>{say(words('importLatent')?.label)}
          <input type="file" className="sr-file" accept=".latent,.hslat" data-file="latent" aria-label={say(words('importLatent')?.label)} onChange={e => { const f = e.currentTarget.files?.[0]; if (f) act('importLatent', f.name); e.currentTarget.value = ''; }} />
        </label>
        {tool('composeCaption', 'structBtn', !m.tools.compose)}
        {tool('expandWildcards', 'wildBtn', !m.tools.expand, { arg: 'prompt', data: { 'data-wildx': 'prompt' } })}
        <span className="pc-sep" aria-hidden="true" />
        <span className="capsrc" id="capSrc" hidden={!m.tools.source}>
          {m.tools.source && src ? chip(src, src.label, m.sourceTip) : null}
          {m.tools.source && m.tools.dataset && ds ? chip(ds, words('chooseDataset')?.label ?? ds.label, null, x => act('chooseDataset', String(x))) : null}
        </span>
        <span id="latChip">{m.latent ? <span className="pchip latchip" title={say(ui('prompt.latent.tip'))}><span className="k">{say(ui('prompt.latent'))}</span>{m.latent}<button type="button" data-latclear="" aria-label={say(words('clearLatent')?.label)} title={say(words('clearLatent')?.label)} onClick={() => act('clearLatent')}><Icon name="x" size={13} /></button></span> : null}</span>
        {wild ? <button type="button" className={`pchip wild-auto${wild.value === true ? ' on' : ''}${wild.changed ? ' changed' : ''}`} id="wildAuto" data-ledtog={wild.key} data-ctl={wild.key} aria-pressed={wild.value === true} hidden={!m.tools.wildAuto} title={say(ui('prompt.wildAuto.tip'))} onClick={() => wild.set(wild.value !== true)}><i aria-hidden="true" />{say(ui('prompt.wildAuto'))}</button> : null}
        {trig ? <label className={`pchip trig${trig.changed ? ' changed' : ''}`} id="trigTape" data-ctl={trig.key} hidden={!m.tools.trigger}><span className="k">{say(trig.label)}</span>
          <input data-bind={trig.key} placeholder={trig.placeholder ? say(trig.placeholder) : undefined} aria-label={say(ui('prompt.trigger.aria'))} title={trig.tip ? say(trig.tip) : undefined} maxLength={trig.maxLength} value={String(trig.value ?? '')}
            onFocus={() => v.gesture.begin(trig.key)} onBlur={() => v.gesture.end(trig.key)} onChange={e => { v.gesture.begin(trig.key); trig.set(e.currentTarget.value); }} /></label> : null}
      </div>
      <div className="songinfo" id="songInfo" ref={info} hidden={!open}>{INFO.map(k => field(of(k)))}{field(of('pp.art.subject'), 'si-art')}</div>
      {g.mark ? <i className={`smark${g.mark.drag ? ' drag' : ''}`} aria-hidden="true" style={g.mark.style} onPointerDown={g.mark.onPointerDown} /> : null}
    </div>
  );
}

definePart('promptBox', PromptBox);
