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
  ratio = r;
  node?.port.postMessage({ type: 'ratio', value: r });
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

/** The node the spectrum analyser should read, or null if there is no graph
 *  yet. Reading here rather than the raw element keeps capture in one place —
 *  and means the analyser sees post-shift audio. */
export function pitchTapNode(): AudioNode | null {
  return tap;
}
