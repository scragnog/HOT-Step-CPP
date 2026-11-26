// Injected before the page's own scripts (addInitScript). Collects, for the action being measured:
//  - Event Timing entries (the browser's own interaction-to-next-paint figures, 8 ms resolution, >= 16 ms only),
//  - a finer "input to next frame" figure: from the event's timeStamp to a task queued from the next animation frame
//    (when the frame's script, style and layout are done), for every input event,
//  - long tasks and long animation frames,
//  - frames (only while __perf.frames is switched on, so idle stays idle).
(() => {
  const P = window.__perf = { events: [], marks: [], lt: [], loaf: [], frames: null, on: true };
  const push = (a, x) => { if (P.on) a.push(x); };
  try { new PerformanceObserver(l => { for (const e of l.getEntries()) push(P.events, { name: e.name, dur: e.duration, id: e.interactionId || 0, start: e.startTime, proc: e.processingEnd - e.processingStart, wait: e.processingStart - e.startTime }); }).observe({ type: 'event', buffered: true, durationThreshold: 16 }); } catch (e) {}
  try { new PerformanceObserver(l => { for (const e of l.getEntries()) push(P.lt, { start: e.startTime, dur: e.duration }); }).observe({ type: 'longtask', buffered: true }); } catch (e) {}
  try { new PerformanceObserver(l => { for (const e of l.getEntries()) push(P.loaf, { start: e.startTime, dur: e.duration, block: e.blockingDuration, render: e.renderStart ? e.startTime + e.duration - e.renderStart : 0, style: e.styleAndLayoutStart ? e.startTime + e.duration - e.styleAndLayoutStart : 0, scripts: (e.scripts || []).slice(0, 4).map(s => ({ inv: s.invoker, dur: Math.round(s.duration), fn: s.sourceFunctionName, at: s.sourceCharPosition })) }); }).observe({ type: 'long-animation-frame', buffered: true }); } catch (e) {}
  const T = ['pointerdown', 'pointerup', 'pointermove', 'keydown', 'keyup', 'input', 'click', 'dragover', 'drop', 'dragstart', 'dragend', 'wheel'];
  for (const t of T) addEventListener(t, e => {
    if (!P.on) return;
    const t0 = e.timeStamp, h0 = performance.now();
    requestAnimationFrame(() => setTimeout(() => push(P.marks, { type: e.type, lat: performance.now() - t0, handled: h0 - t0 }), 0));
  }, true);
  P.startFrames = () => { P.frames = []; const f = ts => { if (!P.frames) return; P.frames.push(ts); requestAnimationFrame(f); }; requestAnimationFrame(f); };
  P.stopFrames = () => { const f = P.frames; P.frames = null; return f || []; };
  P.reset = () => { P.events = []; P.marks = []; P.lt = []; P.loaf = []; };
  P.settle = (ms = 250) => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(r, ms))));
})();
