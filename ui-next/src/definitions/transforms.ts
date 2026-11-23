// The named transforms a wire may use with `via`, for what a wire's fields cannot say. Each one is a pair: `send`
// turns the desk's value (or, for a packing transform, the values of every parameter it packs) into the real field,
// and `read` turns a real field back into desk values, for loading a saved profile or a song's settings. Adding a
// transform is adding an entry here; the validator refuses a `via` that names none, and checks that every
// parameter a packing transform lists names it.

import type { Json, Value } from './types.ts';

export interface Transform {
  doc: string;
  /** A packing transform: the parameters (or state) whose values make the one field. */
  packs?: readonly string[];
  /** The real value, or undefined to send nothing. */
  send(value: Value | Json | { [key: string]: Value | Json | undefined }): Json | undefined;
  /** The desk value (or, packing, the values) a real value stands for. */
  read(real: Json): Value | Json | { [key: string]: Value | Json } | undefined;
}

const num = (v: unknown, d: number) => (typeof v === 'number' && Number.isFinite(v) ? v : d);
const f2 = (v: unknown, d: number) => num(v, d).toFixed(2);

export const TRANSFORMS = {
  int: {
    doc: 'A whole number typed as text, sent as a number: the retake seed, StableStep\'s fixed seed.',
    send: v => (typeof v === 'string' && /^-?\d+$/.test(v.trim()) ? Number(v.trim()) : undefined),
    read: real => (typeof real === 'number' ? String(real) : undefined),
  },
  text: {
    doc: 'A number sent as its decimal text: YuE2\'s composer CFG is a text knob.',
    send: v => (typeof v === 'number' ? String(v) : undefined),
    read: real => (typeof real === 'string' && real.trim() !== '' && Number.isFinite(Number(real)) ? Number(real) : undefined),
  },
  scheduler: {
    doc: 'The schedule and its shape packed into the scheduler string the engine reads: "beta:<alpha>:<beta>", "power:<exponent>", "composite:<A>+<B>:<crossover>:<split>", or the schedule\'s id.',
    packs: ['samp.sched', 'sch.alpha', 'sch.beta', 'sch.exp', 'sch.a', 'sch.b', 'sch.x', 'sch.split'],
    send: v => {
      const o = v as { [key: string]: Value | undefined };
      const s = o['samp.sched'];
      if (typeof s !== 'string') return undefined;
      if (s === 'beta') return `beta:${f2(o['sch.alpha'], 0.5)}:${f2(o['sch.beta'], 0.7)}`;
      if (s === 'power') return `power:${f2(o['sch.exp'], 2)}`;
      if (s === 'composite') return `composite:${o['sch.a'] ?? 'bong_tangent'}+${o['sch.b'] ?? 'linear'}:${f2(o['sch.x'], 0.5)}:${f2(o['sch.split'], 0.5)}`;
      return s;
    },
    read: real => {
      if (typeof real !== 'string') return undefined;
      let m = /^beta:([\d.]+):([\d.]+)$/.exec(real);
      if (m) return { 'samp.sched': 'beta', 'sch.alpha': +m[1]!, 'sch.beta': +m[2]! };
      m = /^power:([\d.]+)$/.exec(real);
      if (m) return { 'samp.sched': 'power', 'sch.exp': +m[1]! };
      m = /^composite:([^+]+)\+([^:]+):([\d.]+):([\d.]+)$/.exec(real);
      if (m) return { 'samp.sched': 'composite', 'sch.a': m[1]!, 'sch.b': m[2]!, 'sch.x': +m[3]!, 'sch.split': +m[4]! };
      return { 'samp.sched': real };
    },
  },
  firstPath: {
    doc: 'The first loaded adapter\'s file, or "" with none: the single adapter of Simple mode.',
    send: v => (Array.isArray(v) && v.length && v[0] && typeof v[0] === 'object' && 'file' in v[0] ? (v[0] as { file: Json }).file : ''),
    read: real => (typeof real === 'string' && real ? [{ file: real }] : undefined),
  },
  stackPaths: {
    doc: 'The loaded adapters as the stack\'s entries, each with its file as `path`; their strengths are the slots\' own parameters.',
    send: v => (Array.isArray(v) && v.length ? v.map(x => ({ path: (x as { file: Json }).file, scale: 1 })) : undefined),
    read: real => (Array.isArray(real) ? real.map(x => ({ file: (x as { path: Json }).path })) : undefined),
  },
  continuation: {
    doc: 'The recording and the part of it to continue from, as YuE2\'s yue2Continuation: { sourceAudioUrl, start, end }.',
    packs: ['contFile', 'contSel'],
    send: v => {
      const o = v as { [key: string]: Json | undefined }, span = o.contSel as { start?: number; end?: number } | null | undefined;
      return typeof o.contFile === 'string' && o.contFile
        ? { sourceAudioUrl: o.contFile, start: span?.start ?? 0, end: span?.end ?? 30 } : undefined;
    },
    read: real => {
      const r = real as { sourceAudioUrl?: string; start?: number; end?: number } | null;
      return r && r.sourceAudioUrl ? { contFile: r.sourceAudioUrl, contSel: { start: r.start ?? 0, end: r.end ?? 30 } } : undefined;
    },
  },
} as const satisfies Record<string, Transform>;

export type TransformName = keyof typeof TRANSFORMS;
