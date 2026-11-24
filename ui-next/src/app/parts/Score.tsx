// abcFoot: the lead sheet's foot (the concept's 25-module-score.js abcFoot). Insert example while the sheet is empty,
// the preview (of the score on the sheet, or of one the planner writes), and, with Chain of Thought off and nothing on
// the sheet, the way back to a score; then one line saying how the sheet reads. Which of its node's actions apply and
// every word are the core's (instrument.abc); each button runs its node's action (v.run), the preview the named action
// scorePreview (features/score-preview.tsx). The way back hands focus to the preview it brings (E-12).

import { useLayoutEffect, useRef } from 'react';
import { definePart, useSay, type PartView } from '../kit/index.ts';

export function AbcFoot(v: PartView) {
  const say = useSay(), m = v.instrument?.abc, box = useRef<HTMLDivElement>(null), back = useRef(false);
  useLayoutEffect(() => { if (back.current) { const b = box.current?.querySelector<HTMLElement>('[data-scoreprev]'); if (b) { back.current = false; b.focus({ preventScroll: true }); } } });
  if (!m) return null;
  return (
    <div ref={box} className="tile-foot" data-abcfoot="">
      {m.show.map(s => {
        const a = v.actions[s.at]; if (!a) return null;
        if (s.kind === 'example') return <button key={s.at} type="button" className="linkbtn" data-example={s.key} onClick={() => v.run(s.at)}>{say(a.label)}</button>;
        if (s.kind === 'back') return <button key={s.at} type="button" className="small-btn" data-bcot="" onClick={() => { back.current = true; v.run(s.at); }}>{say(a.label)}</button>;
        return <button key={s.at} type="button" className="small-btn" data-scoreprev="" onClick={() => v.run(s.at)}>{say(m.preview)}</button>;
      })}
      {m.note ? <span className={`lbl${m.note.warn ? ' warn' : ''}`}>{say(m.note.text)}</span> : null}
    </div>
  );
}

definePart('abcFoot', AbcFoot);
