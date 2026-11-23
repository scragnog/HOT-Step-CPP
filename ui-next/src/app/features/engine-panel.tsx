// The engine panel's models (the concept's 34-engine-panel.js fillEngine), drawn under its engine cards
// (shell/TopBar.tsx) through the engine.panel slot: the heading of the engine's models, what the tier shows of them,
// and the Models definition's face, drawn by the generic renderer (its modelPicker rows; Quick shows the sound model
// only, as the definition says). And what the page is told about the models: every engine's model catalogue and the
// VRAM figures, from the models service (services/models.ts), once when the page loads.

import { ENGINES } from '../../definitions/engines.ts';
import { enginePanelModel, listKey, selectionKey } from '../../derive.ts';
import { defOf, ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { useSay } from '../i18n/index.ts';
import { fill } from '../register/slots.ts';
import { service } from '../register/services.ts';
import { Face } from '../render/Face.tsx';
import { useModel } from '../render/useModel.ts';
import type { ModelsService } from '../services/models.ts';
import '../services/models.ts';

const MODELS = defOf('models')!;

export function EngineModels() {
  const say = useSay(), m = useModel(s => enginePanelModel(ix, s.desk.tier, s.desk.engine), 'engine-panel');
  return (
    <div className="ep-models">
      <div className="ep-mh"><b>{say(m.head)}</b><span>{say(m.sub)}{m.testing ? <>{say(m.sep)}<em className="ep-test">{say(m.testing)}</em></> : null}</span></div>
      <Face def={MODELS} surface="tile" />
    </div>
  );
}
fill('engine.panel', EngineModels);

/** Every engine's catalogue (its buckets as lists, its selection by `<engine>.<bucket>`) and the figures, told to the
 *  page as each arrives. A catalogue that cannot be read leaves that engine's model rows on "Engine default". */
async function tell(): Promise<void> {
  const models = service<ModelsService>('models');
  await Promise.all([
    models.vram().then(vram => useDesk.getState().told({ vram })),
    ...ENGINES.map(e => models.catalogue(e.id).then(c => useDesk.getState().told({
      lists: Object.fromEntries(Object.entries(c.buckets).map(([b, files]) => [listKey({ from: 'models', bucket: b }, e.id), files])),
      selection: Object.fromEntries(Object.entries(c.selection).map(([b, v]) => [selectionKey(e.id, b), v])),
    }))),
  ]).catch(() => { /* the panel shows what it was told; the real adapter says why it failed */ });
}
void tell();
