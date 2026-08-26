/**
 * pitchShift.ts — owns the one pitch-shifting node in the play bar's graph.
 *
 * The file decks' audio already runs through a Web Audio graph: audioMotion
 * takes the media element with createMediaElementSource() and pushes the result
 * at the speakers. So rather than build a second graph and fight it for the
 * element, we splice into the one that exists:
 *
 *     media element -> audioMotion -> [pitch shifter] -> destination
 *
 * The node stays in the chain for the life of the page and bypasses itself at
 * ratio 1, so nothing about ordinary playback changes — no reconnect races, no
 * chance of ending up with a graph that has no path to the speakers.
 *
 * Installing needs an AudioWorklet module, which loads asynchronously and can
 * fail (old browser, blocked asset). Callers ask isPitchShiftReady() and keep
 * a fallback for the answer being no; onPitchShiftReady() fires once when it
 * flips, so a store can re-apply itself at that point.
 */

import workletUrl from './pitchShiftWorklet.js?url';

/** Minimal slice of the audioMotion instance we need. */
interface AnalyzerLike {
  audioCtx: AudioContext;
  connectOutput(node?: AudioNode): void;
  disconnectOutput(node?: AudioNode): void;
}

let node: AudioWorkletNode | null = null;
let installing = false;
let ratio = 1;
const readyListeners = new Set<() => void>();

export function isPitchShiftReady(): boolean {
  return node !== null;
}

/** Fires once the shifter is in the graph. Returns an unsubscribe. */
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

/**
 * Splice the shifter in after `analyzer`. Idempotent — later calls are no-ops,
 * which matters because the analyzer is created once but its effect can run
 * again on a deck change.
 */
export async function installPitchShifter(analyzer: AnalyzerLike): Promise<void> {
  if (node || installing) return;
  installing = true;
  try {
    const ctx = analyzer.audioCtx;
    await ctx.audioWorklet.addModule(workletUrl);
    const n = new AudioWorkletNode(ctx, 'hot-pitch-shift', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [2],
      channelCount: 2,
      channelCountMode: 'explicit',
      channelInterpretation: 'speakers',
    });
    // Add before removing, so the analyzer is never momentarily output-less —
    // audioMotion tears down its internal connections when its last output
    // goes away.
    analyzer.connectOutput(n);
    analyzer.disconnectOutput(ctx.destination);
    n.connect(ctx.destination);
    n.port.postMessage({ type: 'ratio', value: ratio });
    node = n;
    for (const fn of readyListeners) fn();
    readyListeners.clear();
  } catch (err) {
    console.error('[pitchShift] worklet unavailable, falling back to rate-based detune:', err);
  } finally {
    installing = false;
  }
}
