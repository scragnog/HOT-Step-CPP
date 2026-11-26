// What the instruments' scenarios (package 3a) need of the concept's globals beyond the layer's core
// (testing/compat.ts): the seed and the saved seeds as the concept keeps them (S.seed, null for random; S.stars), the
// recent references (S.refs), and the song a repaint works on with its region as fractions of the song (S.source,
// S.region), each read and set as a scene. The concept's page starts with two saved seeds; the test page is given the
// same two, as a scene, so its scenarios find them (the product starts with none).

import { cleanSeeds, cleanState } from '../../../derive.ts';
import { ix } from '../../model/index.ts';
import { useDesk } from '../../store/desk.ts';
import { references } from '../../services/mock/references.ts';
import { asScene, onMirror, onScene, type Scene } from '../compat.ts';
import type { Json } from '../../../definitions/types.ts';

const DEMO_STARS = [{ v: 1834920213, n: 'kick-heavy take', d: 'Punchy low end, the chorus lands hard', tags: ['drums', 'keeper'], star: true }, { v: 77104, n: 'airy chorus', d: '', tags: ['vocals'] }];
const J = (x: unknown) => JSON.stringify(x ?? null);

export function install(): void {
  /* what was last mirrored: a value the scenario has set since is its scene, kept for render() to apply */
  const last: Record<string, string> = {};
  const fresh = (S: Scene, k: string) => !(k in last) || J(S[k]) === last[k];
  onMirror((S: Scene, s) => {
    const st = s.desk.state, src = st.source as { dur?: number } | undefined, r = st.region as { start?: number; end?: number } | undefined;
    const mirror: Record<string, unknown> = {
      seed: st.seed ?? null,
      stars: cleanSeeds(st.stars),
      refs: [...references.list()],
      source: src ?? null,
      region: src?.dur && r && typeof r.start === 'number' && typeof r.end === 'number' ? [r.start / src.dur, r.end / src.dur] : [0.35, 0.55],
    };
    for (const [k, v] of Object.entries(mirror)) if (fresh(S, k)) { S[k] = v; last[k] = J(v); }
  });
  onScene(S => {
    const s = useDesk.getState(), state = { ...s.desk.state }; let n = 0;
    const put = (k: string, v: Json | undefined) => { if (J(v) === J(state[k])) return; if (v === undefined || v === null) delete state[k]; else state[k] = v; n++; };
    if (!fresh(S, 'seed')) { const d = ix.states.get('seed')?.s; put('seed', d && S.seed !== null ? cleanState(d, S.seed) : null); }
    if (!fresh(S, 'stars')) put('stars', cleanSeeds(S.stars as Json) as unknown as Json);
    if (!fresh(S, 'refs') && Array.isArray(S.refs)) references.replace(S.refs as string[]);
    const src = (fresh(S, 'source') ? state.source : S.source) as { dur?: number } | null | undefined;
    if (!fresh(S, 'source')) put('source', src && typeof src === 'object' && typeof src.dur === 'number' ? src as Json : null);
    if ((!fresh(S, 'region') || state.region === undefined) && src?.dur && Array.isArray(S.region)) { const [a, b] = S.region as number[]; put('region', { start: a! * src.dur, end: b! * src.dur }); }
    for (const k of ['seed', 'stars', 'refs', 'source', 'region']) delete last[k];
    if (n) useDesk.setState({ desk: { ...s.desk, state } });
  });
  /* the concept's two saved seeds, on a page that has none yet */
  if (useDesk.getState().desk.state.stars === undefined) asScene(() => { const s = useDesk.getState(); useDesk.setState({ desk: { ...s.desk, state: { ...s.desk.state, stars: DEMO_STARS as unknown as Json } } }); });
}
