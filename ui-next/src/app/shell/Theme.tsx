// The theme switch (31-topbar.js): System, Light or Dark, remembered; a radio group, arrow keys between them. System
// follows the operating system; the choice applies before the first paint (index.html reads it).

import { useEffect, useLayoutEffect, useState, type KeyboardEvent } from 'react';
import { flushSync } from 'react-dom';
import { ui, useSay } from '../i18n/index.ts';

export const THEME_KEY = 'hotstep.theme';
const THEMES = ['system', 'light', 'dark'] as const;
type Theme = (typeof THEMES)[number];
const ICON: Record<Theme, React.ReactNode> = {
  system: <><rect x="3" y="4" width="18" height="12" rx="2" /><path d="M8 20h8M12 16v4" /></>,
  light: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>,
  dark: <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" />,
};

const read = (): Theme => { let t: string | null = null; try { t = localStorage.getItem(THEME_KEY); } catch { t = null; } return (THEMES as readonly string[]).includes(t ?? '') ? t as Theme : 'system'; };
function apply(t: Theme) {
  if (t === 'system') delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = t;
  try { if (t === 'system') localStorage.removeItem(THEME_KEY); else localStorage.setItem(THEME_KEY, t); } catch { /* not remembered */ }
}

export function Theme() {
  const say = useSay(), [t, setT] = useState<Theme>(read);
  useLayoutEffect(() => { apply(t); }, [t]);
  /* a theme chosen in another tab is followed here */
  useEffect(() => { const f = (e: StorageEvent) => { if (e.key === THEME_KEY || e.key === null) setT(read()); }; addEventListener('storage', f); return () => removeEventListener('storage', f); }, []);
  const choose = (n: Theme) => {
    if (n === t) return;
    const h = document.documentElement; h.classList.add('theme-switching');
    const end = () => requestAnimationFrame(() => requestAnimationFrame(() => h.classList.remove('theme-switching')));
    const go = () => setT(n);
    type VT = { finished: Promise<void>; ready: Promise<void>; updateCallbackDone: Promise<void> };
    const start = (document as Document & { startViewTransition?: (f: () => void) => VT }).startViewTransition;
    /* a switch made while another runs skips the first (F-20): that is expected, never an error */
    if (start && !matchMedia('(prefers-reduced-motion: reduce)').matches) { const vt = start.call(document, () => flushSync(go)); vt.finished.then(end, end); vt.ready.catch(() => {}); vt.updateCallbackDone.catch(() => {}); }
    else { go(); end(); }
  };
  const key = (e: KeyboardEvent<HTMLDivElement>) => {
    const i = THEMES.indexOf(t), j = ({ ArrowRight: i + 1, ArrowDown: i + 1, ArrowLeft: i - 1, ArrowUp: i - 1, Home: 0, End: 2 } as Record<string, number>)[e.key];
    if (j === undefined) return; e.preventDefault();
    const n = THEMES[(j + 3) % 3]!; choose(n); document.querySelector<HTMLElement>(`#themeSeg [data-themeset="${n}"]`)?.focus();
  };
  return (
    <div className="theme-seg" id="themeSeg" role="radiogroup" aria-label={say(ui('theme'))} onKeyDown={key}>
      {THEMES.map(n => (
        <button key={n} type="button" role="radio" data-themeset={n} aria-checked={n === t} tabIndex={n === t ? 0 : -1}
          aria-label={say(ui(n === 'system' ? 'theme.system.label' : `theme.${n}`))} title={say(n === 'system' ? ui('theme.system.title') : ui('theme.title', { name: ui(`theme.${n}`) }))} onClick={() => choose(n)}>
          <svg className="ico" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{ICON[n]}</svg>
        </button>
      ))}
    </div>
  );
}
