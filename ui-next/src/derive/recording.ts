// Continue from recording (YuE2's recordingSpan, the concept's 25-module-cont.js): the part of a recording, up to 30
// seconds, that a new song continues from, and what a hand or a key does to it. The span is its node's state, in
// seconds; the recording's length is what the page was told of the file (desk.sessions.recording, services/recording.ts).
// The span in force is the stored one while it lies inside the recording, else the first 20 seconds, so a recording
// replaced by a shorter one never holds a part it does not have.
//
//   spanModel(ix, node, desk)    InstrumentModel.span: the selection, its words and the ruler under it
//   spanMove(a, z, dur, how)     where a key, a drag or a double-click takes it (B-28, G-B-01)

import type { Node } from '../definitions/types.ts';
import { type Desk, type Index, mmss, stateOf } from './core.ts';
import { EN, type Slot, msg } from './messages.ts';
import { rulerStep } from './controls.ts';
import { defineInstrument } from './instruments.ts';

Object.assign(EN, {
  'span.text': '{a} to {b}',
  'span.cap': '{a} – {b} · {n} s of {max} s max',
  'span.aria': 'Part to continue from: drag, or use the arrow keys to move it and Shift with the arrows to resize it, up to {max} seconds',
  'span.tip': 'Drag to choose up to {max} seconds',
  'span.play': 'Play the chosen part',
  'span.stop': 'Stop',
  'file.long': '{t} long',
});

/** The longest part YuE2 continues from, the part a new recording starts with, and the shortest a hand can make. */
export const SPAN_MAX = 30, SPAN_FIRST = 20, SPAN_MIN = 1;
/** A recording's length while the page has not been told it (the concept's). */
const UNTOLD = 180;
const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));

export interface SpanModel {
  file: string; dur: number; a: number; z: number;
  /** The span's ends as percents of the recording, for the drawing. */
  pa: string; pz: string;
  text: Slot; cap: Slot; aria: Slot; tip: Slot; play: Slot; stop: Slot;
  /** The ruler under the waveform: round steps, chosen as the length rulers choose them, and the file's end (B-28). */
  ticks: { at: number; text: string; end: boolean }[];
}

/** The recording, its length and the span in force. */
export function spanAt(ix: Index, node: Node, desk: Desk): { file: string; dur: number; a: number; z: number; key: string | undefined } | null {
  const fileKey = node.reads?.[0], key = node.state?.find(k => ix.states.get(k)?.s.kind === 'span');
  const file = fileKey ? stateOf(ix, fileKey, desk) : undefined;
  if (typeof file !== 'string' || !file) return null;
  const told = desk.sessions?.recording as { [file: string]: number } | undefined, dur = typeof told?.[file] === 'number' && told[file]! > 0 ? told[file]! : UNTOLD;
  const s = key ? stateOf(ix, key, desk) as { start?: number; end?: number } | undefined : undefined;
  const ok = s && typeof s.start === 'number' && typeof s.end === 'number' && s.end <= dur && s.end - s.start <= SPAN_MAX;
  return { file, dur, a: ok ? s.start! : 0, z: ok ? s.end! : Math.min(SPAN_FIRST, dur), key };
}

export function spanModel(ix: Index, node: Node, desk: Desk): SpanModel | undefined {
  const r = spanAt(ix, node, desk);
  if (!r) return undefined;
  const { file, dur, a, z } = r, st = rulerStep(dur), at: number[] = [];
  for (let t = 0; t < dur - st * 0.4; t += st) at.push(t);
  at.push(dur);
  return {
    file, dur, a, z, pa: (a / dur * 100).toFixed(3), pz: (z / dur * 100).toFixed(3),
    text: msg('span.text', { a: mmss(a), b: mmss(z) }), cap: msg('span.cap', { a: mmss(a), b: mmss(z), n: z - a, max: SPAN_MAX }),
    aria: msg('span.aria', { max: SPAN_MAX }), tip: msg('span.tip', { max: SPAN_MAX }), play: msg('span.play'), stop: msg('span.stop'),
    ticks: at.map((t, i) => ({ at: t / dur * 100, text: mmss(t), end: i === at.length - 1 })),
  };
}

/** What a hand on the waveform is: near an edge (within 6px) that edge, inside the span moving it, outside drawing a new
 *  one. `x` and the ends are pixels along the track. */
export const spanHit = (x: number, ax: number, zx: number): 'a' | 'z' | 'move' | 'new' => (Math.abs(x - ax) <= 6 ? 'a' : Math.abs(x - zx) <= 6 ? 'z' : x > ax && x < zx ? 'move' : 'new');

/** Where a key, a drag or a double-click takes the span, in whole seconds. Keys (B-28): arrows move it a second, Shift
 *  with the arrows moves its end, Page Up and Page Down move it by its own length, Home and End take it to the file's
 *  start and end (with Shift: its end to the shortest or the longest it may be). A drag (G-B-01) moves the edge it
 *  pressed, carries the span, or draws a new one from where it began; `t` and `t0` are fractions of the file. A
 *  double-click puts back the first 20 seconds. Null for a key it does not take. */
export function spanMove(a: number, z: number, dur: number, how: { key?: string; shift?: boolean; mode?: 'a' | 'z' | 'move' | 'new'; t?: number; t0?: number; a0?: number; z0?: number; reset?: boolean }): [number, number] | null {
  if (how.reset) return [0, Math.min(SPAN_FIRST, dur)];
  const w = z - a;
  if (how.key) {
    const k = how.key, dir = k === 'ArrowRight' || k === 'ArrowUp' ? 1 : k === 'ArrowLeft' || k === 'ArrowDown' ? -1 : 0;
    if (dir) { if (how.shift) return [a, clamp(z + dir, a + SPAN_MIN, Math.min(dur, a + SPAN_MAX))]; const na = clamp(a + dir, 0, dur - w); return [na, na + w]; }
    if (k === 'PageUp' || k === 'PageDown') { const na = clamp(a + (k === 'PageUp' ? w : -w), 0, dur - w); return [na, na + w]; }
    if (k === 'Home') return how.shift ? [a, a + SPAN_MIN] : [0, w];
    if (k === 'End') return how.shift ? [a, Math.min(dur, a + SPAN_MAX)] : [dur - w, dur];
    return null;
  }
  const sec = (f: number | undefined) => Math.round(clamp(f ?? 0, 0, 1) * dur), t = sec(how.t), a0 = how.a0 ?? a, z0 = how.z0 ?? z;
  if (how.mode === 'a') return [clamp(t, Math.max(0, z0 - SPAN_MAX), z0 - SPAN_MIN), z0];
  if (how.mode === 'z') return [a0, clamp(t, a0 + SPAN_MIN, Math.min(dur, a0 + SPAN_MAX))];
  if (how.mode === 'move') { const len = z0 - a0, na = clamp(a0 + t - sec(how.t0), 0, dur - len); return [na, na + len]; }
  const s0 = sec(how.t0); let na = Math.min(s0, t), nz = Math.max(s0, t);
  if (nz - na > SPAN_MAX) { if (t > s0) nz = na + SPAN_MAX; else na = nz - SPAN_MAX; }
  if (nz - na < SPAN_MIN) nz = na + SPAN_MIN;
  return [na, nz];
}

defineInstrument('recordingSpan', (ix, node, desk) => { const s = spanModel(ix, node, desk); return s ? { span: s } : {}; }, {
  /* a key, a drag or a double-click on the waveform (`arg`: spanMove's `how`) */
  move: x => {
    const r = spanAt(x.ix, x.node, x.desk), st = r?.key ? x.v.state[r.key] : undefined;
    if (!r || !st) return false;
    const to = spanMove(r.a, r.z, r.dur, (x.arg ?? {}) as Parameters<typeof spanMove>[3]);
    if (!to) return false;
    if (to[0] === r.a && to[1] === r.z) return;
    st.apply('move', { start: to[0], end: to[1] });
  },
});
