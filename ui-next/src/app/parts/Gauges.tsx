// The read-outs that draw a figure: each shows what the core worked out from the values it reads
// (src/derive/readouts.ts) and edits nothing.
//
//   beatPulse    the lamps beside the BPM, pulsing on the beat (25-module-tempo.js pulseHTML)
//   dice         one die per allowed attempt, under the re-planning switch (25-module-replan.js)
//   flowWindows  MiniMax-Music3's flow windows, and the count a 3:00 song needs (25-module-flow.js)
//   speedGauge   how much faster the trades make a run, on a small LCD (25-module-perf.js)
//   adhereCurve  plan strength across the run, with the Plan CFG band and the CADS ramp (25-module-adhere.js)
//   planText     the plan the planner starts from, as terminal text (25-module-planner.js planText)

import { Fragment, type CSSProperties } from 'react';
import { definePart, ui, useSay, type PartView } from '../kit/index.ts';

export function BeatPulse(v: PartView) {
  const p = v.readout?.pulse;
  if (!p) return null;
  return (
    <span className="lcd-pulse" data-pulse="" aria-hidden="true">
      {Array.from({ length: p.beats }, (_, i) => <i key={i} className={p.idle ? 'idle' : undefined} style={{ '--i': i, '--n': p.beats, '--beat': p.beat } as CSSProperties} />)}
    </span>
  );
}

export function Dice(v: PartView) {
  const say = useSay(), d = v.readout?.dice;
  if (!d) return null;
  return (
    <div className="dicebox" data-dice-box="" title={say(d.tip)}>
      {Array.from({ length: d.n }, (_, i) => <i key={i} className={['die', d.off ? 'off' : '', i < d.used ? 'used' : ''].filter(Boolean).join(' ')}>{i + 1}</i>)}
    </div>
  );
}

export function FlowWindows(v: PartView) {
  const say = useSay(), f = v.readout?.flow;
  if (!f) return null;
  return (
    <>
      <div className="flowviz" data-flowviz="" aria-label={say(ui('flow.label'))}>
        {f.windows.map((w, i) => <i key={i} style={{ '--x': w.x.toFixed(4), '--w': w.w.toFixed(4), '--lane': w.lane } as CSSProperties} />)}
        <span className="fv-axis">{f.axis.map((a, i) => <b key={i}>{a}</b>)}</span>
      </div>
      <div className="cap" data-flowcap=""><span><b>{f.count}</b> {say(f.compare)}</span><span>{say(f.seam)}</span></div>
    </>
  );
}

export function SpeedGauge(v: PartView) {
  const say = useSay(), s = v.readout?.speed;
  if (!s) return null;
  return <div className="gauge"><span data-speed="">{s.text}</span><small>{say(ui('gauge.faster'))}</small></div>;
}

export function AdhereCurve(v: PartView) {
  const say = useSay(), c = v.readout?.curve;
  if (!c) return null;
  const T = 8, B = 88, L = 8, R = 292;
  return (
    <div className="scope">
      <div data-adhere="">
        <svg viewBox="0 0 300 110" preserveAspectRatio="none" aria-label={say(ui('curve.label'))}>
          {c.grid.map((x, i) => <line key={i} x1={x} y1={T} x2={x} y2={B} stroke="#262131" vectorEffect="non-scaling-stroke" />)}
          <line x1={L} y1={c.unity} x2={R} y2={c.unity} stroke="#ffffff22" strokeDasharray="3 4" vectorEffect="non-scaling-stroke" />
          {c.cads ? <>
            <defs><linearGradient id="cadsG" x1="0" x2="1"><stop offset="0" stopColor="#9b85ff" stopOpacity={c.cads.opacity} /><stop offset={c.cads.stop} stopColor="#9b85ff" stopOpacity={c.cads.opacity} /><stop offset="1" stopColor="#9b85ff" stopOpacity="0" /></linearGradient></defs>
            <rect x={L} y={B + 12} width={c.cads.width} height="6" fill="url(#cadsG)" />
          </> : null}
          {c.band ? <rect x={c.band.x} y={B + 4} width={c.band.width} height="6" fill="#f5b544" opacity={c.band.opacity} /> : null}
          <polyline points={c.points} fill="none" stroke="#a6f5bd" strokeWidth="2" vectorEffect="non-scaling-stroke" />
        </svg>
        <span className="ad-axis"><b>{say(ui('curve.noise'))}</b><span className="ad-leg">{c.legend.map((l, i) => <b key={i} className={l.cls}>{say(l.words)}</b>)}</span><b>{say(ui('curve.clean'))}</b></span>
      </div>
    </div>
  );
}

/* the terminal's text keeps its own line breaks (white-space: pre-wrap) */
const NL = '\n';

export function PlanText(v: PartView) {
  const say = useSay(), p = v.readout?.plan;
  if (!p) return null;
  return (
    <div data-term="">
      <span className="p">$</span> {say(p.command)}{p.after ? <> <span className="k">{say(p.after)}</span></> : null}{NL}
      {p.note ? <><span className="k">{say(p.note)}</span>{NL}</> : null}
      {p.lines.map((l, i) => (
        <Fragment key={i}>
          <span className="k">{say(l.key).padEnd(p.pad)}</span>{l.tone === 'k' ? <span className="v"><span className="k">{say(l.value)}</span></span> : <span className={l.tone}>{say(l.value)}</span>}{NL}
        </Fragment>
      ))}
      <span className="cursor" />
    </div>
  );
}

definePart('beatPulse', BeatPulse);
definePart('dice', Dice);
definePart('flowWindows', FlowWindows);
definePart('speedGauge', SpeedGauge);
definePart('adhereCurve', AdhereCurve);
definePart('planText', PlanText);
