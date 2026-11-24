// What the fold's scenarios need of the concept's globals beyond the layer's core (testing/compat.ts): the view state
// the concept keeps with its desk save, as S.qAll (Quick's fold standing open) and S.songInfoOpen (Song info open),
// read and set as a scene; and the concept's defineModule, defineFace, FACES and catalogued(), with which its suite
// registers two stand-in modules to make a desk the fold takes three of. The rebuild's stand-ins are definitions of the
// test build (testing/standins.ts); defineModule names one of them in the concept's catalogue (MODS), and refuses any
// other.

import { keptStates, type Kept } from '../../kit/session.ts';
import { STAND_IN_IDS } from '../standins.ts';
import { S, expose, onMirror, onScene, type Scene } from '../compat.ts';

/** The concept's names for the page's kept view state (kit/session.ts). */
const NAMES: Record<string, string> = { qAll: 'desk.fold', songInfoOpen: 'prompt.songInfo' };
const kept = (k: string): Kept<unknown> | undefined => keptStates().get(NAMES[k]!);

export function install(): void {
  /* what was last mirrored: a value the scenario has set since is its scene, kept for render() to apply */
  const last: Record<string, unknown> = {};
  const fresh = (s: Scene, k: string) => !(k in last) || s[k] === last[k];
  const mirror = (s: Scene) => { for (const k of Object.keys(NAMES)) { const x = kept(k); if (x && fresh(s, k)) { s[k] = x.get(); last[k] = s[k]; } } };
  onMirror(s => mirror(s));
  onScene(s => {
    for (const k of Object.keys(NAMES)) { const x = kept(k); if (x && !fresh(s, k) && typeof s[k] === 'boolean') x.set(s[k]); delete last[k]; }
    mirror(s);
  });
  /* a person's click on the fold or on Song info changes no desk: follow the kept state itself (load.ts has loaded
     every feature and part, which register theirs, before this layer installs) */
  for (const k of Object.keys(NAMES)) kept(k)?.subscribe(() => { delete last[k]; mirror(S); });

  const MODS = (window as unknown as { MODS: Record<string, { name: string }> }).MODS;
  expose({
    defineModule(id: string, o: { name: string }) {
      if (!(STAND_IN_IDS as readonly string[]).includes(id)) throw new Error(`the test page has no stand-in module "${id}" (testing/standins.ts)`);
      MODS[id] = { name: o.name };
    },
    defineFace() { /* the stand-ins' faces are their definitions' */ },
    FACES: {},
    catalogued() { /* the catalogue is the definitions' */ },
  });
}
