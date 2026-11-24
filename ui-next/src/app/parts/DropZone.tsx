// dropZone: a file drop zone (the concept's label.drop, 13-files.js, 25-module-timbre.js, 25-module-cont.js,
// 27-pedals.js): a record sleeve on a tile or a slot on a pedal, that takes an audio file dropped on it or chosen with
// its file field, shows the file's name, and clears it with ×. Its node's state is the file. Whether a file is taken,
// and the step it is, are the core's (StateView.load: a file that is no audio is refused and said); a file another
// definition lends it (its `from` words) is shown and never taken or cleared here.

import { useLayoutEffect, useRef, useState, type DragEvent } from 'react';
import { definePart, Icon, ui, useSay, type PartView } from '../kit/index.ts';

const hasFiles = (e: DragEvent) => [...(e.dataTransfer?.types ?? [])].includes('Files');

/** A loaded file's waveform on the sleeve (the concept's drawMini): a bar every 3px in the reference's own blue, as its
 *  label and circle (G-D-23); the bars' heights are the core's (StateModel.wave). Drawn again when the sleeve resizes. */
function MiniWave({ wave }: { wave: number[] }) {
  const cv = useRef<HTMLCanvasElement>(null);
  useLayoutEffect(() => {
    const c = cv.current; if (!c) return;
    const draw = () => {
      const w = c.clientWidth, h = 36, dpr = devicePixelRatio || 1; if (!w) return;
      c.width = w * dpr; c.height = h * dpr; const g = c.getContext('2d'); if (!g) return; g.scale(dpr, dpr); g.fillStyle = '#4c6fff';
      for (let i = 0; i < w / 3 && i < wave.length; i++) { const a = wave[i]! * (h / 2 - 2); g.fillRect(i * 3, h / 2 - a, 2, a * 2); }
    };
    draw();
    const ro = new ResizeObserver(draw); ro.observe(c); return () => ro.disconnect();
  }, [wave]);
  return <canvas ref={cv} className="minwave" data-minwave="" />;
}

export function DropZone(v: PartView) {
  const say = useSay(), st = v.state[0], [over, setOver] = useState(false);
  if (!st) return null;
  const name = typeof st.value === 'string' ? st.value : '', lent = st.lent, slot = v.opts.look === 'slot';
  const over$ = {
    onDragOver: (e: DragEvent<HTMLLabelElement>) => { if (!hasFiles(e) || lent) return; e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; setOver(true); },
    onDragLeave: () => setOver(false),
    onDrop: (e: DragEvent<HTMLLabelElement>) => { if (!hasFiles(e) || lent) return; e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f) st.load(f.name, f.type); },
  };
  const input = lent ? null : (
    <input type="file" accept="audio/*" data-file={st.owner} aria-label={say(ui('drop.choose', { what: st.noun }))}
      onChange={e => { const f = e.currentTarget.files?.[0]; e.currentTarget.value = ''; if (f) st.load(f.name, f.type); }} />
  );
  const clear = name && !lent ? (
    <button type="button" className="drop-x" data-fileclear={st.owner} aria-label={say(ui('drop.clear', { what: st.noun }))} title={say(ui('drop.clear.title'))}
      onClick={e => { e.preventDefault(); st.apply('clear', ''); }}><Icon name="x" size={14} /></button>
  ) : null;
  const cls = ['drop', slot ? 'pslot' : 'sleeve', name ? 'loaded' : '', lent && name ? 'lent' : '', over ? 'over' : ''].filter(Boolean).join(' ');
  if (slot) {
    return (
      <label className={cls} {...over$}>
        <span className="ps-eng">{v.label !== undefined ? say(v.label) : null}</span><span className="ps-name">{name || (v.words.empty ? say(v.words.empty) : null)}</span>{input}{clear}
      </label>
    );
  }
  return (
    <label className={cls} {...over$}>
      <i className="vinyl" aria-hidden="true" />
      <span className="sl-txt">
        {name ? <><b title={name}>{name}</b>{lent && v.words.from ? <small className="sl-from">{say(v.words.from)}</small> : st.detail ? <small>{say(st.detail)}</small> : null}{st.wave ? <MiniWave wave={st.wave} /> : null}</>
          : <><b>{v.words.empty ? say(v.words.empty) : null}</b>{v.words.hint ? <small>{say(v.words.hint)}</small> : null}</>}
      </span>
      {input}{clear}
    </label>
  );
}

definePart('dropZone', DropZone);
