// What the engine panel's scenarios need of the concept's globals beyond the layer's core (testing/compat.ts): `get`
// says a model choice as the concept does, by the file in force. The concept's default is the first file of a bucket;
// the definitions' is "" (the engine's default), which stands for the file the server has selected (the mock selects
// the first), so a model choice left at "" is said as that file.

import { modelDefault, valueOf } from '../../../derive.ts';
import { ix } from '../../model/index.ts';
import { useDesk } from '../../store/desk.ts';
import { expose } from '../compat.ts';

export function install(): void {
  const before = (window as unknown as { get: (key: string) => unknown }).get;
  expose({
    get(key: string): unknown {
      const desk = useDesk.getState().desk;
      if (ix.params.has(key) && valueOf(ix, key, desk) === '') { const f = modelDefault(ix, key, desk); if (f !== undefined) return f; }
      return before(key);
    },
  });
}
