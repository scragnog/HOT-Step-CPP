// KitGallery.tsx — the kit's interactive test page. Every control works; the
// tint switcher re-tints the whole page live through data-tint. Dev-only: open
// http://localhost:3000/kit (no sidebar link). Nothing here touches the app.

import React, { useEffect, useRef, useState } from 'react';
import { Play, Plus, Trash2 } from 'lucide-react';
import { Button, IconButton, type ButtonSize, type ButtonVariant } from './Button';
import { Card, Panel } from './Panel';
import { StatusLed } from './StatusLed';
import { Tabs } from './Tabs';
import { KitSlider } from './Slider';
import { KitToggle } from './Toggle';
import { Select } from './Select';

const SIZES: ButtonSize[] = ['sm', 'md', 'lg'];
const VARIANTS: ButtonVariant[] = ['primary', 'secondary', 'ghost', 'danger'];
const STATES = [
  { name: 'rest', props: {} },
  { name: 'hover', props: { 'data-force': 'hover' } },
  { name: 'pressed', props: { 'data-force': 'press' } },
  { name: 'focus', props: { 'data-force': 'focus' } },
  { name: 'disabled', props: { disabled: true } },
] as const;
const TINTS = [
  { value: 'amber', label: 'Amber' },
  { value: 'teal', label: 'Teal' },
  { value: 'violet', label: 'Violet' },
  { value: 'rose', label: 'Rose' },
];

const Section: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <div className="mb-6">
    <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-ink-muted">{title}</h3>
    {children}
  </div>
);

export const KitGallery: React.FC = () => {
  const [tint, setTint] = useState('amber');
  const [reduce, setReduce] = useState(false);
  const [tab, setTab] = useState('create');
  const [on, setOn] = useState(true);
  const [small, setSmall] = useState(false);
  const [val, setVal] = useState(40);
  const [sel, setSel] = useState('b');
  const [many, setMany] = useState('opt3');
  const [generating, setGenerating] = useState(false);
  const [clicks, setClicks] = useState(0);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);

  const generate = () => {
    setGenerating(true);
    timer.current = setTimeout(() => setGenerating(false), 4000);
  };

  return (
    // Default amber is the :root value, so only a non-default tint sets the attribute.
    <div data-tint={tint === 'amber' ? undefined : tint} className={`flex-1 overflow-y-auto bg-surface-chassis p-6 text-ink${reduce ? ' kit-reduce' : ''}`}>
      <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold">UI kit test page</h1>
          <p className="text-sm text-ink-muted">Dev only. Everything here is live. Forced hover / pressed / focus rows show the states without a pointer.</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-xs text-ink-muted">Tint
            <Tabs aria-label="Tint" value={tint} onChange={setTint} items={TINTS} />
          </div>
          <KitToggle size="sm" checked={reduce} onChange={setReduce} label="Reduced-motion preview" />
        </div>
      </div>

      <Section title="Try it: Generate (LED runs for 4 seconds)">
        <div className="flex items-center gap-4">
          <Button size="lg" variant="primary" disabled={generating} onClick={generate}>
            <Play size={18} /> {generating ? 'Generating…' : 'Generate'}
          </Button>
          <StatusLed state={generating ? 'generating' : 'ok'} label={generating ? 'Generating' : 'Ready'} />
          <Button variant="secondary" onClick={() => setClicks(c => c + 1)}>Clicked {clicks}</Button>
          <Button variant="danger" onClick={() => setClicks(0)}>Reset</Button>
        </div>
      </Section>

      <Section title="Button: variant x state, per size">
        {SIZES.map(size => (
          <div key={size} className="mb-3">
            <div className="mb-1 text-xs text-ink-muted">size {size}</div>
            <div className="grid grid-cols-[5rem_repeat(5,auto)] items-center justify-start gap-x-3 gap-y-2">
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

      <Section title="Tabs (sliding indicator)">
        <Tabs
          aria-label="Gallery tabs"
          value={tab}
          onChange={setTab}
          items={[{ value: 'create', label: 'Create' }, { value: 'library', label: 'Library' }, { value: 'training', label: 'Training studio' }, { value: 'off', label: 'Disabled', disabled: true }]}
        />
      </Section>

      <Section title="Panel (collapsible) and Card">
        <div className="grid grid-cols-2 gap-3">
          <Panel collapsible title="Collapsible panel" actions={<IconButton size="sm" variant="ghost" aria-label="Add"><Plus size={14} /></IconButton>}>
            <Card>Static card</Card>
          </Panel>
          <Panel title="Panel with interactive cards">
            <div className="space-y-2">
              <Card onClick={() => setClicks(c => c + 1)}>Interactive card: hover, press, click (count {clicks})</Card>
              <Card onClick={() => setClicks(c => c + 1)}>Another one</Card>
            </div>
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

      <Section title="Slider, Toggle, Select">
        <div className="grid grid-cols-3 items-start gap-6">
          <div className="space-y-4">
            <KitSlider label="Guidance" value={val} onChange={setVal} min={0} max={100} step={1} defaultValue={50} info="Drag me, or use the arrow keys." />
            <KitSlider label="With number box" value={val} onChange={setVal} min={0} max={100} step={1} showInput />
            <KitSlider label="Disabled" value={30} onChange={() => undefined} min={0} max={100} step={1} disabled />
          </div>
          <div className="space-y-3">
            <KitToggle checked={on} onChange={setOn} label="Enabled" info="Flips; the track takes the tint when on." />
            <KitToggle checked={small} onChange={setSmall} size="sm" label="Small" />
            <KitToggle checked disabled onChange={() => undefined} label="Disabled on" />
            <KitToggle checked={false} disabled onChange={() => undefined} label="Disabled off" />
          </div>
          <div className="space-y-3">
            <Select value={sel} onChange={setSel} options={[{ value: 'a', label: 'Alpha' }, { value: 'b', label: 'Bravo', hint: 'With a hint' }, { value: 'c', label: 'Charlie' }]} />
            <Select size="sm" value={sel} onChange={setSel} options={[{ value: 'a', label: 'Alpha' }, { value: 'b', label: 'Bravo' }]} />
            <Select value={many} onChange={setMany} options={Array.from({ length: 12 }, (_, i) => ({ value: `opt${i}`, label: `Option ${i}` }))} />
            <Select disabled value={sel} onChange={setSel} options={[{ value: 'b', label: 'Disabled' }]} />
          </div>
        </div>
      </Section>
    </div>
  );
};
