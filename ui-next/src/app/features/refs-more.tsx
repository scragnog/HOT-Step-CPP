// "N more…" under the Timbre reference's recent tracks (the concept's [data-refsmore], 25-module-timbre.js): the
// named action moreRefs opens every recent reference in the page's popover, under the button that ran it; one press
// loads it as the reference (its file state's `load` step, through the node's own binding), closes the list and puts
// focus on the sleeve's file field. Before, the button ran nothing: the tile's latest two were the only way back.

import type { Definition, Node } from '../../definitions/types.ts';
import { bindPart, nodeKeys } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { ui, useSay } from '../i18n/index.ts';
import { defineAction } from '../register/actions.ts';
import { useModel } from '../render/useModel.ts';
import { closePop, openPop } from './pop.ts';

/** The node that draws the recent references, its definition, and the file state a choice loads. */
function findRefs(): { def: Definition; node: Node; key: string; file: string } | null {
  for (const d of ix.defs) {
    const n = ('face' in d ? d.face ?? [] : []).find(x => x.part === 'recentRefs');
    const file = n?.state?.find(k => ix.states.get(k)?.s.kind === 'file');
    if (n && file) return { def: d, node: n, key: nodeKeys(d).get(n) ?? '', file };
  }
  return null;
}
const REFS = findRefs();

/* the button a hand pressed last: the list opens under it (a click does not focus a button everywhere) */
let pressed: HTMLElement | null = null;
document.addEventListener('click', e => { pressed = (e.target as HTMLElement | null)?.closest?.('button') ?? null; }, true);
defineAction('moreRefs', () => {
  const f = document.activeElement instanceof HTMLElement && document.activeElement !== document.body ? document.activeElement : null;
  const a = pressed?.isConnected ? pressed : f;
  if (a) openPop(a, () => <MoreRefs />, { label: ui('refs.all') });
});

function load(file: string): void {
  if (!REFS) return;
  const s = useDesk.getState(), desk = () => useDesk.getState().desk;
  closePop(false);
  try {
    bindPart(ix, REFS.def, REFS.node, desk, c => s.commit(c, `${REFS.def.id}:${REFS.key}`), m => s.showToast({ text: m })).state[REFS.file]?.apply('load', file, undefined, { file });
  } catch { /* the tile is not on the desk any more */ }
  s.setFocus(`[data-m="${REFS.def.id}"] .drop.sleeve input`);
}

/** Every recent reference, the one in use pressed. */
export function MoreRefs() {
  const say = useSay();
  const m = useModel(s => ({ refs: (s.desk.lists?.['refs.recent'] ?? []).map(x => String(x.value)), on: REFS ? String(s.desk.state[REFS.file] ?? '') : '' }), 'refs-more');
  return (
    <>
      <h4>{say(ui('refs.all'))}</h4>
      <div className="refs refs-all">
        {m.refs.map(r => <button key={r} type="button" className={`pref${m.on === r ? ' on' : ''}`} data-pref={r} title={say(ui('refs.use', { file: r }))} onClick={() => load(r)}>{r}</button>)}
      </div>
    </>
  );
}
