/**
 * pitchShift.ts — the file decks' route to the speakers, with a pitch shifter
 * in it.
 *
 *     deck <audio> ──┐
 *     deck <audio> ──┼─> input ─> [shifter] ─> tap ─> destination
 *     deck <audio> ──┘                          └──> audioMotion (analyser)
 *
 * This module owns the AudioContext and the MediaElementSourceNodes, and hands
 * `tap` to the spectrum analyser as its source. It used to be the other way
 * round — audioMotion captured the elements and we spliced in behind it — but
 * that made the shifter's existence depend on the analyser being constructed,
 * which is a thing that can fail (double capture of an element after an HMR
 * remount, for one) and take the shifter down with it silently.
 *
 * An element can only be captured once for the life of the page, so ownership
 * has to sit in exactly one place. This is that place.
 *
 * All three decks are attached, not just the audible one. They run in lockstep
 * with volume deciding who is heard, and a muted element contributes true
 * silence (volume is applied ahead of the source node), so summing them costs
 * nothing and means a variant switch cannot land on an unattached deck.
 *
 * The chain is built synchronously; the worklet loads async and splices itself
 * between input and tap when it arrives. Until then — or forever, if it fails
 * — the chain is a plain pass-through and isPitchShiftReady() reports false.
 */

import workletUrl from './pitchShiftWorklet.js?url';

let ctx: AudioContext | null = null;
let input: GainNode | null = null;
let tap: GainNode | null = null;
let node: AudioWorkletNode | null = null;
let installing = false;
let failure: string | null = null;
let ratio = 1;

const attached = new WeakSet<HTMLMediaElement>();
const readyListeners = new Set<() => void>();

export function isPitchShiftReady(): boolean {
  return node !== null;
}

/** Why the shifter is not available, or null if it is (or is still loading). */
export function pitchShiftError(): string | null {
  return failure;
}

/** Fires once the shifter is in the chain. Returns an unsubscribe. */
export function onPitchShiftReady(fn: () => void): () => void {
  if (node) { fn(); return () => {}; }
  readyListeners.add(fn);
  return () => { readyListeners.delete(fn); };
}

/**
 * Pitch multiplier for the audible signal — 1 is bypass, 0.91875 drops
 * ~1.47 semitones. Tempo is untouched either way. Safe to call before the
 * worklet finishes loading; the value is applied on arrival.
 */
export function setPitchRatio(r: number): void {
  const changed = r !== ratio;
  ratio = r;
  node?.port.postMessage({ type: 'ratio', value: r });
  // Dump the state a few seconds after a change, once the shifter has settled.
  // `starves` is the number to watch: it should stop climbing after priming,
  // and a figure that keeps rising means audio is being produced slower than
  // realtime — the only way this node could drag the tempo.
  if (changed && r !== 1) {
    setTimeout(() => { void pitchMeasure(); }, 1500);
  }
}

function ensureGraph(): boolean {
  if (ctx) return true;
  try {
    ctx = new AudioContext();
    input = ctx.createGain();
    tap = ctx.createGain();
    input.connect(tap);
    tap.connect(ctx.destination);
    // Autoplay policy: the context starts suspended and only a gesture lifts
    // it. audioMotion used to do this for us, back when it owned the context.
    const unlock = () => {
      if (ctx && ctx.state === 'suspended') {
        void ctx.resume().then(() => window.removeEventListener('click', unlock));
      }
    };
    window.addEventListener('click', unlock);
    void loadWorklet();
    return true;
  } catch (err) {
    failure = `AudioContext unavailable: ${String(err)}`;
    console.error('[pitchShift]', failure);
    return false;
  }
}

async function loadWorklet(): Promise<void> {
  if (node || installing || !ctx || !input || !tap) return;
  installing = true;
  try {
    if (!ctx.audioWorklet) throw new Error('AudioWorklet missing (page is not a secure context?)');
    await ctx.audioWorklet.addModule(workletUrl);
    const n = new AudioWorkletNode(ctx, 'hot-pitch-shift', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [2],
      channelCount: 2,
      channelCountMode: 'explicit',
      channelInterpretation: 'speakers',
    });
    n.port.postMessage({ type: 'ratio', value: ratio });
    input.disconnect(tap);
    input.connect(n);
    n.connect(tap);
    node = n;
    failure = null;
    console.log('[pitchShift] shifter online');
    for (const fn of readyListeners) fn();
    readyListeners.clear();
  } catch (err) {
    failure = String(err);
    console.error('[pitchShift] shifter unavailable — the 44.1 kHz toggle will do nothing:', err);
  } finally {
    installing = false;
  }
}

/**
 * Route one deck's media element through the chain. Idempotent per element —
 * a second capture of the same element throws, and there is no way to undo the
 * first, so the WeakSet is load-bearing.
 */
export function pitchAttachElement(el: HTMLMediaElement | null): void {
  if (!el || attached.has(el)) return;
  if (!ensureGraph() || !ctx || !input) return;
  try {
    ctx.createMediaElementSource(el).connect(input);
    attached.add(el);
    if (ctx.state === 'suspended') void ctx.resume();
  } catch (err) {
    // Already captured by something else. Leave it playing direct to the
    // speakers rather than losing its audio entirely.
    attached.add(el);
    console.error('[pitchShift] could not capture deck element:', err);
  }
}

/** Lift the autoplay suspension. Called whenever the decks start, because a
 *  suspended context means silence, and the context is now ours to unlock. */
export function pitchResume(): void {
  if (ctx && ctx.state === 'suspended') void ctx.resume();
}

// ── Diagnostics ─────────────────────────────────────────────────────────────
//
// The play bar has four things that can make sound and three AudioContexts
// between them, so "the pitch button did the wrong thing" has more than one
// possible author. This dumps every one of them at once, on demand:
// window.__hotPitch() in the console.

type DeckProbe = () => Record<string, unknown>;
let deckProbe: DeckProbe | null = null;

/** Let the playback store contribute what it knows about the decks. */
export function registerDeckProbe(fn: DeckProbe): void {
  deckProbe = fn;
}

/** Ask the worklet for its counters. Null if there is no node to ask. */
async function workletStats(): Promise<Record<string, number> | null> {
  if (!node) return null;
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(null), 500);
    const once = (e: MessageEvent) => {
      if (e.data?.type !== 'stats') return;
      clearTimeout(timer);
      node?.port.removeEventListener('message', once);
      resolve(e.data);
    };
    node!.port.addEventListener('message', once);
    node!.port.start();
    node!.port.postMessage({ type: 'stats' });
  });
}

/**
 * Watch the audio path for a few seconds and report RATES rather than state.
 * State has said "all correct" twice while the tempo moved anyway, so this
 * measures the three clocks that could disagree:
 *
 *   - the media element's own currentTime, against the wall clock
 *   - the worklet's output frames, against the context rate
 *   - the worklet's input consumption, against the same
 *
 * Whichever one reads ~0.919 instead of ~1.0 is the culprit, and they are
 * mutually exclusive, so this narrows it to one line.
 */
export async function pitchMeasure(seconds = 4): Promise<void> {
  const el = deckProbe ? (deckProbe().audibleElement as HTMLMediaElement | null) : null;
  const a = await workletStats();
  const t0 = performance.now();
  const m0 = el?.currentTime ?? NaN;
  const c0 = ctx?.currentTime ?? NaN;

  await new Promise((r) => setTimeout(r, seconds * 1000));

  const b = await workletStats();
  const wall = (performance.now() - t0) / 1000;
  const m1 = el?.currentTime ?? NaN;
  const c1 = ctx?.currentTime ?? NaN;
  const sr = ctx?.sampleRate ?? 48000;

  const mediaRate = (m1 - m0) / wall;
  const ctxRate = (c1 - c0) / wall;
  const outRate = a && b ? (b.framesOut - a.framesOut) / wall / sr : NaN;
  const inRate = a && b ? (b.inWrite - a.inWrite) / wall / sr : NaN;
  const starveRate = a && b ? (b.starves - a.starves) / wall : NaN;

  const f = (x: number) => (Number.isFinite(x) ? x.toFixed(4) : 'n/a');
  console.log('[pitchShift] ── measured over ' + wall.toFixed(2) + ' s ──');
  console.log('  media element advances     ' + f(mediaRate) + ' s/s   (1.0 = deck at normal speed)');
  console.log('  audio context advances     ' + f(ctxRate) + ' s/s');
  console.log('  worklet output             ' + f(outRate) + ' x realtime');
  console.log('  worklet input consumed     ' + f(inRate) + ' x realtime');
  console.log('  starved frames             ' + f(starveRate) + ' /s   (0 = healthy, >0 = producing too slowly)');
  console.log('  ratio ' + ratio + '  active ' + String(b?.active) + '  ctx ' + sr + ' Hz');

  if (Number.isFinite(mediaRate) && mediaRate < 0.97 && mediaRate > 0.85) {
    console.warn('  VERDICT: the media element itself is running slow — the deck, not the shifter.');
  } else if (Number.isFinite(starveRate) && starveRate > sr * 0.02) {
    console.warn('  VERDICT: the shifter is starving — it cannot produce at realtime.');
  } else if (Number.isFinite(outRate) && outRate < 0.97) {
    console.warn('  VERDICT: the shifter is emitting fewer frames than realtime.');
  } else {
    console.warn('  VERDICT: every clock reads normal speed. Nothing here is changing tempo.');
  }
}

if (typeof window !== 'undefined') {
  (window as unknown as Record<string, unknown>).__hotMeasure = pitchMeasure;
}

export async function pitchDiagnostics(): Promise<Record<string, unknown>> {
  const worklet = await new Promise<unknown>((resolve) => {
    if (!node) { resolve('no worklet node'); return; }
    const timer = setTimeout(() => resolve('worklet did not answer'), 500);
    const once = (e: MessageEvent) => {
      if (e.data?.type !== 'stats') return;
      clearTimeout(timer);
      node?.port.removeEventListener('message', once);
      resolve(e.data);
    };
    node.port.addEventListener('message', once);
    node.port.start();
    node.port.postMessage({ type: 'stats' });
  });

  return {
    graph: {
      contextRate: ctx?.sampleRate ?? null,
      contextState: ctx?.state ?? 'no context',
      shifterInChain: node !== null,
      requestedRatio: ratio,
      installError: failure,
    },
    worklet,
    decks: deckProbe ? deckProbe() : 'no deck probe registered',
  };
}

if (typeof window !== 'undefined') {
  (window as unknown as Record<string, unknown>).__hotPitch = async () => {
    const d = await pitchDiagnostics();
    console.log('[pitchShift] diagnostics', d);
    return d;
  };
}

/** The node the spectrum analyser should read, or null if there is no graph
 *  yet. Reading here rather than the raw element keeps capture in one place —
 *  and means the analyser sees post-shift audio. */
export function pitchTapNode(): AudioNode | null {
  return tap;
}
