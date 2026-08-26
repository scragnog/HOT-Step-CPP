// heapWatch.ts — a heap sampler, for finding out why the tab runs out of memory.
//
// Two different crashes get reported as "Out of Memory". One follows clicking
// between tracks, and that one is explained: the player used to decode whole
// files into AudioBuffers. The other happens with the app sitting idle, which
// that explanation does not cover, so rather than guess we measure.
//
// Every 30 seconds this logs the JS heap plus a few DOM counts that catch the
// usual leaks (media elements and canvases that are created but never removed).
// The numbers to watch are the two deltas: `30s` should hover around zero once
// the app is idle, and `total` should not climb without bound.
//
// performance.memory is Chrome-only and reports the JS heap alone. Decoded
// audio, canvas backing stores and image bitmaps live outside it, so a crash
// with a flat line here means the leak is in one of those, which is itself a
// useful answer.

interface ChromeMemory {
  usedJSHeapSize: number;
  totalJSHeapSize: number;
  jsHeapSizeLimit: number;
}

const SAMPLE_MS = 30_000;

let timer: ReturnType<typeof setInterval> | null = null;
let firstUsed = 0;
let lastUsed = 0;
let peakUsed = 0;
let samples = 0;

function memory(): ChromeMemory | null {
  const m = (performance as Performance & { memory?: ChromeMemory }).memory;
  return m && typeof m.usedJSHeapSize === 'number' ? m : null;
}

const mb = (bytes: number) => (bytes / 1048576).toFixed(1);
const delta = (bytes: number) => `${bytes >= 0 ? '+' : ''}${mb(bytes)}`;

/** One sample, logged. Returns the reading so `window.__heap()` is useful at
 *  the console as well as in the log. */
export function sampleHeap(label = 'tick'): Record<string, unknown> | null {
  const m = memory();
  if (!m) {
    console.log('[Heap] performance.memory unavailable (not Chrome?) — no reading');
    return null;
  }

  samples++;
  if (!firstUsed) firstUsed = m.usedJSHeapSize;
  if (m.usedJSHeapSize > peakUsed) peakUsed = m.usedJSHeapSize;

  const media = document.querySelectorAll('audio, video').length;
  const canvases = document.querySelectorAll('canvas').length;
  const reading = {
    label,
    sample: samples,
    usedMB: +mb(m.usedJSHeapSize),
    totalMB: +mb(m.totalJSHeapSize),
    limitMB: +mb(m.jsHeapSizeLimit),
    peakMB: +mb(peakUsed),
    since30sMB: +delta(m.usedJSHeapSize - (lastUsed || m.usedJSHeapSize)),
    sinceStartMB: +delta(m.usedJSHeapSize - firstUsed),
    mediaElements: media,
    canvases,
  };

  console.log(
    `[Heap] ${mb(m.usedJSHeapSize)}/${mb(m.jsHeapSizeLimit)} MB used`
    + ` | 30s ${delta(m.usedJSHeapSize - (lastUsed || m.usedJSHeapSize))}`
    + ` | total ${delta(m.usedJSHeapSize - firstUsed)}`
    + ` | peak ${mb(peakUsed)}`
    + ` | media ${media} canvas ${canvases}`
    + (label !== 'tick' ? ` | ${label}` : '')
  );

  lastUsed = m.usedJSHeapSize;
  return reading;
}

/** Start sampling. Safe to call twice — the second call is ignored. */
export function startHeapWatch(): void {
  if (timer) return;
  if (!memory()) {
    console.log('[Heap] sampler not started: performance.memory unavailable');
    return;
  }
  sampleHeap('start');
  timer = setInterval(() => sampleHeap('tick'), SAMPLE_MS);
  (window as Window & { __heap?: typeof sampleHeap }).__heap = sampleHeap;
  console.log(`[Heap] sampling every ${SAMPLE_MS / 1000}s — call __heap() for a reading on demand`);
}

export function stopHeapWatch(): void {
  if (timer) { clearInterval(timer); timer = null; }
}
