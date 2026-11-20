// KitGallery.tsx — every kit element, every size and variant, in rest / hover /
// pressed / disabled / focus, in the default tint and again inside a teal
// data-tint box. Dev-only: open http://localhost:3000/kit (no sidebar link).

import React, { useState } from 'react';
import { Play, Plus, Trash2 } from 'lucide-react';
import { Button, IconButton, type ButtonSize, type ButtonVariant } from './Button';
import { Card, Panel } from './Panel';
import { StatusLed } from './StatusLed';
import { Tabs } from './Tabs';
import { Slider } from '../Slider';
import { StyledSelect } from '../StyledSelect';
import { Toggle } from '../Toggle';

const SIZES: ButtonSize[] = ['sm', 'md', 'lg'];
const VARIANTS: ButtonVariant[] = ['primary', 'secondary', 'ghost', 'danger'];
const STATES = [
  { name: 'rest', props: {} },
  { name: 'hover', props: { 'data-force': 'hover' } },
  { name: 'pressed', props: { 'data-force': 'press' } },
  { name: 'focus', props: { 'data-force': 'focus' } },
  { name: 'disabled', props: { disabled: true } },
] as const;

const Section: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <div className="mb-5">
    <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-ink-muted">{title}</h3>
    {children}
  </div>
);

const KitSet: React.FC = () => {
  const [tab, setTab] = useState('create');
  const [on, setOn] = useState(true);
  const [val, setVal] = useState(40);
  const [sel, setSel] = useState('b');
  return (
    <div className="space-y-1">
      <Section title="Button: variant x state, per size">
        {SIZES.map(size => (
          <div key={size} className="mb-3">
            <div className="mb-1 text-xs text-ink-muted">size {size}</div>
            <div className="grid grid-cols-[auto_repeat(5,auto)] items-center gap-x-3 gap-y-2 justify-start">
              {VARIANTS.map(variant => (
                <React.Fragment key={variant}>
                  <span className="text-xs text-ink-muted">{variant}</span>
                  {STATES.map(st => (
                    <Button key={st.name} size={size} variant={variant} {...st.props}>
                      <Play size={size === 'lg' ? 18 : 14} /> {size === 'lg' ? 'Generate' : st.name}
                    </Button>
                  ))}
                </React.Fragment>
              ))}
            </div>
          </div>
        ))}
      </Section>

      <Section title="IconButton">
        <div className="flex items-center gap-3">
          {SIZES.map(size => VARIANTS.map(variant => (
            <IconButton key={size + variant} size={size} variant={variant} aria-label={`${variant} ${size}`}>
              {variant === 'danger' ? <Trash2 size={14} /> : <Plus size={14} />}
            </IconButton>
          )))}
          <IconButton disabled aria-label="disabled"><Plus size={14} /></IconButton>
        </div>
      </Section>

      <Section title="Tabs">
        <Tabs
          aria-label="Gallery tabs"
          value={tab}
          onChange={setTab}
          items={[{ value: 'create', label: 'Create' }, { value: 'library', label: 'Library' }, { value: 'training', label: 'Training studio' }, { value: 'off', label: 'Disabled', disabled: true }]}
        />
      </Section>

      <Section title="Panel and Card">
        <div className="grid grid-cols-2 gap-3">
          <Panel title="Panel" actions={<IconButton size="sm" variant="ghost" aria-label="Add"><Plus size={14} /></IconButton>}>
            <Card>Static card</Card>
          </Panel>
          <Panel>
            <Card onClick={() => undefined}>Interactive card: hover, press, focus</Card>
          </Panel>
        </div>
      </Section>

      <Section title="StatusLed">
        <div className="flex flex-wrap gap-5">
          <StatusLed state="idle" label="Idle" />
          <StatusLed state="ok" label="Ready" />
          <StatusLed state="warn" label="Low VRAM" />
          <StatusLed state="error" label="Failed" />
          <StatusLed state="generating" label="Generating" />
        </div>
      </Section>

      <Section title="Slider, Toggle, StyledSelect (live; hover and drag them)">
        <div className="grid grid-cols-3 items-start gap-6">
          <Slider label="Guidance" value={val} onChange={setVal} min={0} max={100} step={1} defaultValue={50} />
          <div className="space-y-3">
            <Toggle checked={on} onChange={setOn} label="Enabled" info="Kit toggle in the current tint." />
            <Toggle checked={!on} onChange={v => setOn(!v)} size="sm" label="Small" />
            <Toggle checked disabled onChange={() => undefined} label="Disabled on" />
            <Toggle checked={false} disabled onChange={() => undefined} label="Disabled off" />
          </div>
          <div className="space-y-3">
            <StyledSelect value={sel} onChange={setSel} options={[{ value: 'a', label: 'Alpha' }, { value: 'b', label: 'Bravo', hint: 'With a hint' }, { value: 'c', label: 'Charlie' }]} />
            <StyledSelect size="sm" value={sel} onChange={setSel} options={[{ value: 'a', label: 'Alpha' }, { value: 'b', label: 'Bravo' }]} />
            <StyledSelect disabled value={sel} onChange={setSel} options={[{ value: 'b', label: 'Disabled' }]} />
          </div>
        </div>
      </Section>
    </div>
  );
};

export const KitGallery: React.FC = () => (
  <div className="flex-1 overflow-y-auto bg-surface-chassis p-6 text-ink">
    <h1 className="mb-1 text-xl font-semibold">UI kit gallery</h1>
    <p className="mb-5 text-sm text-ink-muted">Dev only. Hover, pressed and focus rows are forced; sliders, toggles and selects are live.</p>
    <h2 className="mb-3 text-sm font-semibold text-tint">Amber (default tint)</h2>
    <KitSet />
    <h2 className="mb-3 mt-8 text-sm font-semibold text-tint" data-tint="teal">Teal (data-tint="teal")</h2>
    <div data-tint="teal" className="rounded-panel border border-tint/30 p-4">
      <KitSet />
    </div>
  </div>
);
