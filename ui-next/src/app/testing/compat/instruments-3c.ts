// What package 3c's scenarios need of the concept's globals beyond the layer's core (testing/compat.ts): the build as
// the concept keeps it (S.build: { planned, seed, cur, plan, takes, kept, locked, rendered }, the record and the
// session's takes together) and redrawBuild(); the recording's length and the part chosen (S.contDur, S.contSel as
// [start, end] in seconds); the score preview (openScore, closeScore, SC, scoreLayout). Each read and set as a scene: a
// scene's build is given to the simulated build session and its record to the desk, never as a step.

import type { Json, Node } from '../../../definitions/types.ts';
import { buildAt, scoreLayout } from '../../../derive.ts';
import { defOf, ix } from '../../model/index.ts';
import { useDesk } from '../../store/desk.ts';
import { buildSession } from '../../services/mock/build.ts';
import type { Take } from '../../services/build.ts';
import { recordings } from '../../services/mock/recording.ts';
import { close, open, scoreShown } from '../../score/state.ts';
import { asScene, expose, onMirror, onScene, type Scene } from '../compat.ts';

const J = (x: unknown) => JSON.stringify(x ?? null);
const BOARD = () => defOf('build')?.face?.find(n => n.part === 'buildBoard') as Node | undefined;
const SPAN = 'contSel', FILE = 'contFile', RECORD = 'build';
let scenes = 0;

/** The build in the concept's shape. */
function conceptBuild(): unknown {
  const node = BOARD(), s = useDesk.getState(); if (!node) return null;
  const b = buildAt(ix, node, s.desk);
  if (!b.rec || !b.planned) return b.told.planning ? { planning: true } : null;
  return { planned: true, seed: b.rec.seed, cur: b.cur?.id ?? null, plan: b.rec.plan, takes: b.takes, kept: b.kept, locked: Object.keys(b.kept).length > 0, rendered: b.rec.rendered, ...(b.told.planning ? { planning: true } : {}) };
}

export function install(): void {
  const last: Record<string, string> = {};
  const fresh = (S: Scene, k: string) => !(k in last) || J(S[k]) === last[k];
  onMirror((S: Scene, s) => {
    const f = s.desk.state[FILE], told = s.desk.sessions?.recording as { [f: string]: number } | undefined;
    const dur = typeof f === 'string' && told?.[f] ? told[f]! : 180, sp = s.desk.state[SPAN] as { start?: number; end?: number } | undefined;
    const ok = sp && typeof sp.start === 'number' && typeof sp.end === 'number' && sp.end <= dur;
    const mirror: Record<string, unknown> = { build: conceptBuild(), contDur: dur, contSel: ok ? [sp.start, sp.end] : [0, Math.min(20, dur)] };
    for (const [k, v] of Object.entries(mirror)) if (fresh(S, k)) { S[k] = v; last[k] = J(v); }
  });
  onScene(S => {
    let s = useDesk.getState();
    const f = typeof S.contFile === 'string' && S.contFile ? S.contFile : s.desk.state[FILE];
    if (!fresh(S, 'contDur') && typeof f === 'string' && f && typeof S.contDur === 'number') recordings.tell(f, S.contDur);
    s = useDesk.getState();
    const state = { ...s.desk.state }; let n = 0;
    if (!fresh(S, 'contSel') && Array.isArray(S.contSel)) { const [a, z] = S.contSel as number[]; state[SPAN] = { start: a!, end: z! }; n++; }
    if (!fresh(S, 'build')) {
      const B = S.build as { planned?: boolean; seed?: number; cur?: string | null; plan?: string[]; takes?: { [k: string]: Take[] }; kept?: { [k: string]: string }; rendered?: unknown } | null;
      if (B?.planned) {
        const id = `scene${++scenes}`, st = buildSession.state();
        buildSession.replace({ ...st, planning: false, sessions: { ...st.sessions, [id]: { seed: B.seed ?? 1, takes: B.takes ?? {}, ...(B.cur ? { cur: B.cur } : {}) } } });
        state[RECORD] = { session: id, seed: B.seed ?? 1, plan: B.plan ?? [], kept: B.kept ?? {}, dropped: [], rendered: (B.rendered ?? null) as Json, songs: [] };
      } else delete state[RECORD];
      n++;
    }
    /* the scene is now what is mirrored: the next change of the store mirrors the desk again */
    for (const k of ['build', 'contDur', 'contSel']) last[k] = J(S[k]);
    if (n) useDesk.setState({ desk: { ...useDesk.getState().desk, state } });
  });
  /* the build redrawn after a scenario moved its section (the concept's redrawBuild) */
  const redrawBuild = () => asScene(() => {
    const S = (window as unknown as { S: Scene }).S, B = S.build as { cur?: string } | null, rec = useDesk.getState().desk.state[RECORD] as { session?: string } | undefined;
    delete last.build;
    if (B?.cur && rec?.session) buildSession.look(rec.session, B.cur);
  });
  expose({
    redrawBuild,
    openScore: (seed?: number) => { void open(seed); },
    closeScore: close,
    scoreLayout,
    get SC() { return scoreShown(); },
  });
}
