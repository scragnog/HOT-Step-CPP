// The toast (36-toast.js): one message at a time, with its action (Undo, Redo) and a way to dismiss it; it lasts
// longer for a longer message and holds while hovered or while focus is in it. And the live region the page speaks
// through.

import { useEffect, useState, type CSSProperties } from 'react';
import { createPortal } from 'react-dom';
import { useDesk } from '../store/desk.ts';
import { ui, useSay } from '../i18n/index.ts';
import { Icon } from '../parts/icons.tsx';

export function Toast() {
  const say = useSay(), t = useDesk(s => s.toast), [hold, setHold] = useState(false);
  const host = document.getElementById('toast');
  const text = t ? say(t.text) : '', life = Math.min(8000, Math.max(4000, 3500 + text.length * 30));
  useEffect(() => { const h = document.getElementById('toast'); if (h) h.hidden = !t; }, [t]);
  useEffect(() => {
    const host = document.getElementById('toast'); if (!host) return;
    const on = () => setHold(true), off = () => setHold(host.contains(document.activeElement) || host.matches(':hover'));
    host.addEventListener('pointerenter', on); host.addEventListener('pointerleave', off); host.addEventListener('focusin', on); host.addEventListener('focusout', off);
    return () => { host.removeEventListener('pointerenter', on); host.removeEventListener('pointerleave', off); host.removeEventListener('focusin', on); host.removeEventListener('focusout', off); };
  }, []);
  useEffect(() => {
    if (!t || hold) return;
    const id = window.setTimeout(() => useDesk.getState().showToast(null), life);
    return () => clearTimeout(id);
  }, [t, hold, life]);
  if (!host || !t) return null;
  return createPortal(
    <>
      <span className="toast-msg">{text}</span>
      {t.action ? <button className="toast-act" type="button" onClick={() => { const a = t.action!; useDesk.getState().showToast(null); a.run(); }}>{say(t.action.label)}</button> : null}
      <button className="toast-x" type="button" aria-label={say(ui('toast.close'))} title={say(ui('toast.close'))} onClick={() => useDesk.getState().showToast(null)}><Icon name="x" size={14} /></button>
      {t.action ? <i className="toast-life" aria-hidden="true" style={{ '--life': `${life}ms` } as CSSProperties} /> : null}
    </>, host);
}

export function LiveRegion() {
  const say = useSay(), said = useDesk(s => s.said), host = document.getElementById('say');
  if (!host) return null;
  return createPortal(<>{said ? say(said.text) : ''}</>, host);
}
