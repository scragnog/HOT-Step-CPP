// The go bar (create/20-centre.html .gobar, 35-gobar.js): the player when a song plays, the status line, Drafts, Cancel
// and Generate. Generate keeps its width, label and place; the queue is a badge on it and its progress sweeps across
// it under the label; Cancel slides out from under its left edge and Drafts makes room by sliding (B-11, M-12, G-M-06).
// What every piece says is the core's (derive/generate.ts goLine, goModel).

import { Fragment, useLayoutEffect, useRef, type CSSProperties, type MouseEvent } from 'react';
import { goLine, goModel } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useModel } from '../render/useModel.ts';
import { Slot } from '../register/Slot.tsx';
import { ui, useSay } from '../i18n/index.ts';
import { useGen } from '../store/generate.ts';
import { cancelAll, generate, showChanges, takesFit } from './actions.ts';
import { places, slide } from './motion.ts';
import { easing, mtime, reduced } from '../kit/motion.ts';
import { PlayerHost } from './Player.tsx';

/** The status line: the next run judged where Generate is (G-B-08), the tier, what is on the desk, the changes. Where
 *  the bar is narrow its container queries keep the warning and the tier (C-05); all of it is the line's title. */
function Line() {
  const say = useSay(), m = useModel(s => goLine(ix, s.desk), 'go-line'), el = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => { if (el.current) el.current.title = el.current.textContent ?? ''; });
  return (
    <div className="meter" id="meter" ref={el}>
      {m.wontFit ? <><span className="m-warn"><span className="warn">{say(ui('gen.wontFit'))}<span className="m-long">{say(m.wontFit.needs)}</span>.</span>{' '}<button type="button" className="linkbtn" data-takesfit={m.wontFit.fits} onClick={() => takesFit(m.wontFit!.fits)}>{say(m.wontFit.use)}</button></span><span className="m-sep">{' · '}</span></> : null}
      {m.passes ? <><span className="m-warn"><span className="c">{say(m.passes.takes)}<span className="m-long">{say(m.passes.long)}</span></span></span><span className="m-sep">{' · '}</span></> : null}
      <b>{say(m.tier)}</b>
      <span className="m-rest">
        {m.rest.map((r, i) => <Fragment key={i}>{' · '}{r.hidden ? <span className="l">{say(r.says)}</span> : say(r.says)}</Fragment>)}
        {' · '}{m.changes ? <><button type="button" className="linkbtn c" data-showchanges="" title={say(ui('gen.showChanges'))} onClick={showChanges}>{say(m.changesSay)}</button>{say(m.rested)}</> : say(m.rested)}
      </span>
    </div>
  );
}

/** Drafts is the drafts package's (package 7 fills `centre.drafts`). */
const Drafts = () => <Slot name="centre.drafts" />;

/** The queue as Generate shows it: how many songs render now (the head run's) and how many wait. */
const counts = (s: ReturnType<typeof useGen.getState>) => {
  const head = s.runs[0], pend = s.cards.filter(c => c.pending), run = head ? pend.filter(c => c.pending!.run === head.id).length : 0;
  return { run, wait: pend.length - run };
};

function Go() {
  const say = useSay(), run = useGen(s => counts(s).run), wait = useGen(s => counts(s).wait), gf = useGen(s => s.lastF);
  /* the sweep stays where the last run left it once the queue is empty, under a fill that fades out (running) */
  const m = useModel(s => goModel(ix, s.desk, run, wait), `go:${run}:${wait}`);
  /* a second press of one double-click is the same press; separate presses each queue a run (F-22) */
  const press = (e: MouseEvent) => { if (e.detail > 1) return; generate(); };
  return (
    <button className={`go${m.wontFit ? ' wontfit' : ''}`} id="go" title={say(m.title)} aria-keyshortcuts="Control+Enter" style={gf !== null ? { '--gf': gf.toFixed(4) } as CSSProperties : undefined} onClick={press}>
      <i className="go-fill" aria-hidden="true" /><svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M7 4v16l13-8z" /></svg><span id="goLabel">{say(m.label)}</span>
      <b className="go-q" id="goQ" hidden={!m.queued}>{m.queued ? <><span className="sr-only">{say(ui('gen.running'))}</span>{m.queued}</> : null}</b>
    </button>
  );
}

function Cancel() {
  const say = useSay(), run = useGen(s => counts(s).run), wait = useGen(s => counts(s).wait), m = useModel(s => goModel(ix, s.desk, run, wait), `go:${run}:${wait}`);
  return <button className="cancel-go" id="cancelGo" type="button" hidden={!m.queued} aria-label={say(m.cancelLabel)} onClick={cancelAll}>{say(m.cancel)}<span id="qCount">{m.count}</span></button>;
}

export function GoBar() {
  const live = useGen(s => s.cards.some(c => c.pending)), bar = useRef<HTMLElement | null>(null), was = useRef<Map<string, DOMRect> | null>(null), seen = useRef<boolean | null>(null);
  const key = (el: HTMLElement) => el.id || el.className;
  /* while a run is live the page says so (the go fill shows; the document's title carries the count) */
  useLayoutEffect(() => { document.body.classList.toggle('running', live); }, [live]);
  /* Cancel slides out from under Generate's left edge, and what stands before it slides to make room (G-M-06) */
  useLayoutEffect(() => {
    const g = bar.current ?? (bar.current = document.querySelector('.gobar'));
    if (seen.current !== null && seen.current !== live) {
      if (was.current) slide(g, ':scope > *', key, was.current, '--t-base');
      const cg = document.getElementById('cancelGo');
      if (live && cg && !reduced()) cg.animate([{ opacity: 0, translate: '16px 0' }, { opacity: 1, translate: '0 0' }], { duration: mtime('--t-base'), easing: easing('--e-out') });
    }
    seen.current = live; was.current = places(g, ':scope > *', key);
  });
  return (
    <>
      <PlayerHost />
      <Line />
      <Drafts />
      <Cancel />
      <Go />
    </>
  );
}
