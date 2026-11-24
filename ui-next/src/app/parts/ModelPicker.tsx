// modelPicker: the engine panel's model rows (the concept's pick(k, 'model') for each model, 34-engine-panel.js
// fillEngine). One model picker for each model choice the node binds that applies on the engine: its role (Sound,
// Planner), the file's name, its quantisation, its size and a bar of the card it takes; its list of the server's files,
// one row per model with its quantisations as chips (Pick's and Dropdown's model variant). The rows, their words and
// every option's face are the core's (derive/vram.ts modelRows); a choice is the parameter's own setter, one step
// named by the model's role ("Sound model XL Turbo · Q8_0 → Turbo · Q8_0").

import { definePart, type PartView } from '../kit/index.ts';
import { Pick } from './Pick.tsx';

export function ModelPicker(v: PartView) {
  return (
    <>
      {(v.models ?? []).map(r => {
        const p = v.params.find(x => x.key === r.key);
        return p ? <Pick key={r.key} {...v} params={[p]} label={r.role} opts={{ variant: 'model' }} control={{ faces: r.faces }} words={{ title: r.title, aria: r.aria, note: r.note }} /> : null;
      })}
    </>
  );
}

definePart('modelPicker', ModelPicker);
