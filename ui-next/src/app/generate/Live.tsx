// The go bar's fill: the go bar itself, and what the page does while songs render: the document's title carries their
// count, and leaving asks first, since they would be lost (F-37).

import { useEffect } from 'react';
import { docTitle } from '../../derive.ts';
import { sayNow } from '../i18n/index.ts';
import { useGen } from '../store/generate.ts';
import { GoBar } from './GoBar.tsx';

const BASE = document.title;
function Live() {
  const q = useGen(s => s.cards.filter(c => c.pending).length);
  useEffect(() => { document.title = sayNow(docTitle(BASE, q)); }, [q]);
  /* songs still rendering are lost when the page closes: leaving asks first (F-37) */
  useEffect(() => {
    if (!q) return;
    const ask = (e: BeforeUnloadEvent) => { e.preventDefault(); };
    addEventListener('beforeunload', ask);
    return () => removeEventListener('beforeunload', ask);
  }, [q]);
  return null;
}
export function GoBarFill() { return <><GoBar /><Live /></>; }

