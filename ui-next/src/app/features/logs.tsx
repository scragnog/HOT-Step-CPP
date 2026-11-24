// The engine log (the concept's 34-logs.js, overlays/24-engine-log.html, 34-engine-panel.js paintLog), opened from the
// VRAM meter in the top bar. A dock, not a popover (F-15): it sits along the window's foot and the go bar stands on top
// of it, so what it reports on stays in view and in reach. It rises from the bottom edge (the concept's stylesheet)
// with focus on its Close, and stays open until its Close, Escape (once every overlay above it has had its Escape) or
// a second press on the meter, which says whether it is open (aria-expanded); it does not close on a press elsewhere,
// and Ctrl+Enter still generates while it is open (logs-dock.ts). Its edges fade where there is more to read.
//
// What it says is the core's (src/derive/logs.ts engineLogModel), from the desk, what the server says about itself
// (the logs service, services/logs.ts) and what the queue holds (store/notes-logs.ts tellQueue). It is drawn whether or
// not it is open, as the concept paints it, into the skeleton's #logs.

import { useEffect, useLayoutEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { engineLogModel } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { fill } from '../register/slots.ts';
import { service } from '../register/services.ts';
import { useModel } from '../render/useModel.ts';
import { useEdgeFade } from '../shell/edgeFade.ts';
import { ui, useSay } from '../i18n/index.ts';
import { useNotesLogs } from '../store/notes-logs.ts';
import type { LogsService } from '../services/logs.ts';
import '../services/logs.ts';
import { dockLogs, logsDock, toggleLogs, vramMeter } from './logs-dock.ts';

export function Logs() {
  const say = useSay(), open = useNotesLogs(s => s.logs), ref = useRef<HTMLElement | null>(logsDock());
  /* what the server says and what the queue holds are part of what it is asked with */
  const status = useNotesLogs(s => s.status), queue = useNotesLogs(s => s.queue);
  const m = useModel(s => engineLogModel(ix, s.desk, status, queue), `engine-log ${JSON.stringify([status, queue])}`);
  useLayoutEffect(() => {
    const l = logsDock(); if (!l) return;
    l.hidden = !open; l.setAttribute('aria-label', say(ui('log.label')));
    vramMeter()?.setAttribute('aria-expanded', String(open));
    dockLogs();
    if (!open) return;
    const ro = new ResizeObserver(() => dockLogs()); ro.observe(l);
    return () => ro.disconnect();
  }, [open, say]);
  useLayoutEffect(() => { logsDock()?.classList.add('fade-own'); }, []);
  useEdgeFade(ref);
  /* the meter opens and closes it */
  useEffect(() => {
    const press = (e: MouseEvent) => { if ((e.target as Element).closest?.('#engMeter')) toggleLogs(); };
    document.addEventListener('click', press);
    return () => document.removeEventListener('click', press);
  }, []);
  const l = logsDock(); if (!l) return null;
  return createPortal(
    <>
      <div className="logs-head"><b>{say(ui('log.engine'))}</b><span className="mono" id="logEngine">{say(m.head)}</span><button type="button" className="small-btn" id="logsClose" onClick={() => toggleLogs(false)}>{say(ui('log.close'))}</button></div>
      <pre id="logText">{m.lines.map(x => say(x)).join('\n')}</pre>
    </>, l);
}
fill('overlays', Logs);

/** What the server says about itself, told once when the page loads. */
void service<LogsService>('logs').status().then(status => useNotesLogs.setState({ status }), () => { /* the log says it is waiting for the server */ });
