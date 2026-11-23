// What the containers' and read-outs' scenarios need of the concept's globals beyond the layer's core
// (testing/compat.ts): the files the drop zones hold, as the concept keeps them (S.timbreFile, S.contFile, S.masterRef,
// '' when empty), read and set as a scene; and isAuto, as the concept's.

import { cleanState, isAuto } from '../../../derive.ts';
import { ix } from '../../model/index.ts';
import { useDesk } from '../../store/desk.ts';
import { expose, onMirror, onScene, type Scene } from '../compat.ts';

/** The concept's names for the files its drop zones hold, and the state that holds each here. */
const FILES: Record<string, string> = { timbreFile: 'timbreFile', contFile: 'contFile', masterRef: 'masterRef' };
/** The adapters loaded (S.slots), so a scene can put the fader bank on the desk (the cartridges are package 3's). */
const SLOTS = 'slots';

export function install(): void {
  /* what was last mirrored: a value the scenario has set since is its scene, kept for render() to apply (the store
     changes, and mirrors, before a package's part of the scene is applied) */
  const last: Record<string, string> = {};
  onMirror((S: Scene, s) => {
    for (const [k, st] of Object.entries(FILES)) { if (k in last && S[k] !== last[k]) continue; S[k] = last[k] = typeof s.desk.state[st] === 'string' ? s.desk.state[st] as string : ''; }
    const sl = JSON.stringify(s.desk.state[SLOTS] ?? []); if (!('slots' in last) || JSON.stringify(S.slots) === last.slots) { S.slots = JSON.parse(sl); last.slots = sl; }
  });
  onScene(S => {
    const s = useDesk.getState(), state = { ...s.desk.state }; let n = 0;
    for (const [k, st] of Object.entries(FILES)) if (typeof S[k] === 'string' && S[k] !== (state[st] ?? '')) { if (S[k]) state[st] = S[k] as string; else delete state[st]; n++; }
    const def = ix.states.get(SLOTS)?.s, slots = Array.isArray(S.slots) && def ? cleanState(def, S.slots) : undefined;
    if (slots !== undefined && JSON.stringify(slots) !== JSON.stringify(state[SLOTS] ?? [])) { state[SLOTS] = slots; n++; }
    for (const [k, st] of Object.entries(FILES)) last[k] = typeof state[st] === 'string' ? state[st] as string : '';
    last.slots = JSON.stringify(state[SLOTS] ?? []);
    if (n) useDesk.setState({ desk: { ...s.desk, state } });
  });
  expose({ isAuto: (key: string) => ix.params.has(key) && isAuto(ix, key, useDesk.getState().desk) });
}
