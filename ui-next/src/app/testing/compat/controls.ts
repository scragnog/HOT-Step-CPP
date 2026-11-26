// What the shared controls' scenarios need of the concept's globals beyond the layer's core (testing/compat.ts): `get`
// says a choice the server lists (a solver, a schedule, a language) by its words, as the concept's get does, where the
// core layer knows only the options a definition lists itself ("dpm3m" is "DPM++ 3M"); and a scene that sets such a
// choice by its words (S.vals['samp.solver'] = 'Heun') sets the value the server lists under them.

import { optionsOf, valueOf } from '../../../derive.ts';
import { ix } from '../../model/index.ts';
import { useDesk } from '../../store/desk.ts';
import { sayNow } from '../../i18n/index.ts';
import { expose, onScene, toConcept } from '../compat.ts';

export function install(): void {
  onScene(S => {
    const s = useDesk.getState(), vals = { ...s.desk.vals }; let n = 0;
    for (const [k, v] of Object.entries(S.vals ?? {})) {
      if (ix.params.get(k)?.p.type !== 'choice' || typeof v !== 'string') continue;
      const o = optionsOf(ix, k, s.desk).find(x => x.value !== v && sayNow(x.label) === v);
      if (o) { vals[k] = o.value; n++; }
    }
    if (n) useDesk.setState({ desk: { ...s.desk, vals } });
  });
  expose({
    get(key: string): unknown {
      const desk = useDesk.getState().desk, v = valueOf(ix, key, desk);
      if (v === undefined) return undefined;
      if (ix.params.get(key)?.p.type === 'choice') { const o = optionsOf(ix, key, desk).find(x => x.value === v); if (o) return sayNow(o.label); }
      return toConcept(key, v, desk.engine);
    },
  });
}
