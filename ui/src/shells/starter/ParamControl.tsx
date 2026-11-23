// ParamControl.tsx — draws one generation parameter from its descriptor. This is the pattern
// for every control: useParamMeta(id) says what it is, useParam(id) reads and writes it.

import React from 'react';
import Select from '../../components/shared/kit/Select';
import { KitToggle } from '../../components/shared/kit/Toggle';
import { useParam, useParamMeta, type ParamId } from '../../skin-api';

export const ParamControl: React.FC<{ id: ParamId; options?: readonly string[] }> = ({ id, options }) => {
  const meta = useParamMeta(id);
  const { value, set, reset, changed } = useParam<unknown>(id);
  if (!meta || !meta.editable) return null;

  let input: React.ReactNode;
  if (meta.type === 'toggle') {
    input = <KitToggle checked={!!value} onChange={set} aria-label={meta.label} />;
  } else if (meta.type === 'select' || options) {
    // A select may arrive without options (models, solvers): the caller fills them from
    // useBackendModels() and the like. With none at all, fall back to a text field.
    const list = meta.options?.map(o => o.value) ?? options;
    input = list
      ? <Select<string> value={String(value ?? '')} onChange={set} placeholder="(default)" aria-label={meta.label}
          options={list.map(v => ({ value: v, label: meta.options?.find(o => o.value === v)?.label ?? v }))} />
      : <input className="st-input" value={String(value ?? '')} onChange={e => set(e.target.value)} />;
  } else if (meta.type === 'slider') {
    // Some sliders have no min and max yet: draw a number field instead of guessing a range.
    const ranged = meta.min !== undefined && meta.max !== undefined;
    input = <input className={ranged ? undefined : 'st-input'} type={ranged ? 'range' : 'number'} min={meta.min} max={meta.max} step={meta.step ?? 'any'}
      value={Number(value ?? 0)} onChange={e => set(Number(e.target.value))} />;
  } else if (meta.type === 'text') {
    input = <input className="st-input" value={String(value ?? '')} onChange={e => set(e.target.value)} />;
  } else {
    return null; // arrays and objects need their own editors
  }

  return (
    <div className="st-field" title={meta.hint}>
      <span className="st-row">
        <span className={changed ? 'st-changed' : undefined}>{meta.label}{changed ? ' •' : ''}</span>
        <span className="st-muted">{typeof value === 'object' ? '' : String(value ?? '')}</span>
        {changed && <button className="st-btn" onClick={e => { e.preventDefault(); reset(); }}>Reset</button>}
      </span>
      {input}
    </div>
  );
};
