// lyricSheet: the ruled lyric paper (25-module-lyrics.js lyricsInner): at Lab the section tabs over it and its foot with
// the line count and the wildcard tools, the language stamp on it, the structure strip under it (each section a
// coloured segment that puts the caret at its first line), and the tile's foot it holds: Instrumental and the vocal
// gender (its children), and below Lab the line count. With Instrumental on the text is kept read-only and everything
// that edits or measures it stands back (L-22, Q-27). What it shows is the core's (PartModel.sheet); typing sets the
// text, one step per visit; a tab and Expand are its node's named actions, run by the prompt feature.

import { useLayoutEffect, useRef, type CSSProperties } from 'react';
import { definePart, ui, useSay, type PartView } from '../kit/index.ts';
import { Pick } from './Pick.tsx';
import { Pill } from './Toggles.tsx';
import { subView } from './subView.ts';
import { useGrow } from './textGrow.ts';

/* the sheet grows to 62% of the window (420px at least), then scrolls (19-text-boxes.js growCap) */
const CAP = () => Math.max(420, Math.round(innerHeight * 0.62));
const LINE = 26;

export function LyricSheet(v: PartView) {
  const say = useSay(), ta = useRef<HTMLTextAreaElement>(null), strip = useRef<HTMLDivElement>(null), s = v.sheet, p = v.params[0];
  const lang = v.params.find(x => x.options && x !== p), wild = v.params.find(x => typeof x.value === 'boolean');
  const act = (name: string, arg?: string) => { const i = v.actions.findIndex(a => a.name === name); if (i >= 0) v.run(i, arg); };
  const words = (name: string) => v.actions.find(a => a.name === name);
  const text = String(p?.value ?? '');
  const g = useGrow(ta, CAP, text);
  /* a strip label that doesn't fit its segment shortens (Pre-Chorus, Pre, P); the full name stays in its tip (L-20) */
  const names = (s?.strip ?? []).map(x => x.name).join('\n');
  useLayoutEffect(() => {
    const el = strip.current; if (!el) return;
    const fitAll = () => el.querySelectorAll<HTMLButtonElement>('[data-secjump]').forEach((b, j) => {
      for (const t of s?.strip[j]?.labels ?? []) { b.textContent = t; if (b.scrollWidth <= b.clientWidth + 1) break; }
    });
    fitAll();
    const ro = new ResizeObserver(fitAll); ro.observe(el);
    return () => ro.disconnect();
  }, [names, s]);
  if (!s || !p) return null;
  const L = s.lab, off = s.instr;
  /* the caret goes to the start of the section, its line near the top of the sheet */
  const jump = (at: number) => {
    const t = ta.current; if (!t) return;
    t.focus({ preventScroll: true }); t.setSelectionRange(at, at);
    t.scrollTop = Math.max(0, text.slice(0, at).split('\n').length - 2) * LINE;
    t.scrollIntoView({ block: 'nearest' });
  };
  const tabs = s.tabs ? (
    <div className="itabs" role="group" aria-label={say(s.tabsLabel)}>
      {s.tabs.map(t => <button key={t.tag} type="button" data-tag={t.tag} style={{ '--c': t.colour } as CSSProperties} title={say(t.tip)} aria-disabled={off || undefined}
        onMouseDown={e => e.preventDefault()} onClick={() => act('insertSectionTag', t.tag)}>{t.tag.slice(1, -1)}</button>)}
    </div>
  ) : null;
  const stamp = s.stamp && lang ? <Pick {...subView(v, { params: [lang], opts: { variant: 'stamp' }, label: lang.label, title: lang.tip, control: { faces: s.faces }, dim: off ? s.readOnlyTip : null })} /> : null;
  const stat = <span className="stat" data-lstat="">{say(s.stat)}</span>;
  const expand = words('expandWildcards');
  return (
    <>
      <div className={off ? `paper-wrap${L ? ' lab' : ''} instr` : `paper-wrap ${L ? 'lab' : ''}`}>
        {tabs}
        <div className="paper">
          {stamp}
          <textarea ref={ta} data-bind={p.key} aria-label={say(p.label)} placeholder={say(s.placeholder)} readOnly={off} title={s.readOnlyTip ? say(s.readOnlyTip) : ''} value={text} className={g.cls || undefined}
            onFocus={() => v.gesture.begin('text')} onBlur={() => v.gesture.end('text')}
            onChange={e => { if (off) return; v.gesture.begin('text'); p.set(e.currentTarget.value); }} />
          {L && s.wild ? (
            <div className="paper-foot">{stat}<span className="wild">
              <button type="button" className="wild-x" data-wildx="lyrics" title={say(s.wild.tip)} aria-disabled={off || undefined} disabled={off} onClick={() => act('expandWildcards', 'lyrics')}>{say(expand?.label ?? s.wild.expand)}</button>
              {s.wild.auto && wild ? <Pill {...subView(v, { params: [wild], label: ui('lyrics.wildAuto'), title: wild.tip })} /> : null}
            </span></div>
          ) : null}
          {g.mark ? <i className={`smark${g.mark.drag ? ' drag' : ''}`} aria-hidden="true" style={g.mark.style} onPointerDown={g.mark.onPointerDown} /> : null}
        </div>
      </div>
      {' '}
      <div className="strip" data-strip="" ref={strip}>
        {s.stripSays ? <span className="empty">{say(s.stripSays)}</span> : s.strip.map((x, j) => (
          <button key={j} type="button" data-secjump={j} style={{ '--n': Math.max(1, x.n), background: x.colour } as CSSProperties} title={say(x.tip)} aria-label={say(x.aria)} data-full={x.name} onClick={() => jump(x.start)}>{x.name}</button>
        ))}
      </div>
      {' '}
      <div className="tile-foot">{v.children}{L ? null : stat}</div>
    </>
  );
}

definePart('lyricSheet', LyricSheet);
