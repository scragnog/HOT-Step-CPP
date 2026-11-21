// MockLayouts.tsx — clickable 1920x1080 mocks of the two UI Refresh layout
// options: A "Console" (icon rail, right Inspector, bottom dock) and B "Guided"
// (labelled grouped nav, Source / Shape / Render columns, Advanced drawer,
// Activity column). Kit parts and fake data only; nothing here reads the app.
// Nav, tabs, folds, drawer, dock and song menu really open and close; most
// fields are inert. Shared settings live in one state object so A and B show
// the same values. Test page only (see KitGallery).

import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import {
  AudioLines, ChevronDown, Disc3, Layers, Library, ListMusic, Mic2, MoreHorizontal, Music, Pause, Play,
  Radio, Scissors, Search, Settings, SlidersHorizontal, Upload, Wand2, X, GitCompare, Hammer, Piano,
} from 'lucide-react';
import { Button, IconButton } from './Button';
import { Panel } from './Panel';
import { EmptyState, Module, Rail, SectionPanel, SongItem } from './Panels';
import { Select } from './Select';
import { KitSlider } from './Slider';
import { StatusLed } from './StatusLed';
import { Tabs } from './Tabs';
import { KitToggle } from './Toggle';
import { useTheme } from '../../../hooks/useTheme';

// ---------- fake data ----------

const COVERS = [
  'linear-gradient(135deg, #5b8def, #1b2a5c)', 'linear-gradient(135deg, #e0668f, #5c1b3a)',
  'linear-gradient(135deg, #6fd3a0, #1b5c3f)', 'linear-gradient(135deg, #c98bf0, #3f1b5c)',
  'linear-gradient(135deg, #f0b36b, #5c3a1b)',
];
interface Song { id: number; title: string; meta: string; duration: string }
const SONGS: Song[] = [
  { id: 1, title: 'Studio sketch 01', meta: 'Lo-fi hip hop, mellow electric piano', duration: '2:51' },
  { id: 2, title: 'Evening texture', meta: 'Ambient, warm pads, slow', duration: '3:56' },
  { id: 3, title: 'Untitled take', meta: 'Folk rock, acoustic guitar', duration: '3:03' },
  { id: 4, title: 'Ambient fragment', meta: 'Drone, field recording', duration: '0:28' },
  { id: 5, title: 'Field recording', meta: 'Spoken word, room tone', duration: '5:00' },
  { id: 6, title: 'Late bus', meta: 'Synth-pop, driving bass', duration: '3:41' },
];
// Invented names, 360 in the real picker; enough here to need a search box.
const ARTISTS = ['Alder Voss', 'Bryony Hale', 'Calder Finch', 'Dara Quill', 'Elio Marsh', 'Fenna Roake', 'Gideon Pryce', 'Hollis Wren', 'Isla Thorne', 'Jory Blaine', 'Kestrel Moon', 'Lenka Ashby'];
const GENRES = ['Lo-fi', 'Ambient', 'Folk', 'Synth-pop', 'Jazz', 'Post-rock', 'House', 'Soul'];

const NAV_GROUPS = [
  { label: 'Make', items: [{ id: 'create', label: 'Create', icon: Wand2 }, { id: 'lyrics', label: 'Lyric Studio', icon: Mic2 }, { id: 'builder', label: 'Song Builder', icon: Hammer }, { id: 'storm', label: 'STORM', icon: Radio }] },
  { label: 'Edit', items: [{ id: 'cover', label: 'Cover', icon: Disc3 }, { id: 'repaint', label: 'Repaint', icon: Scissors }, { id: 'stems', label: 'Stems', icon: AudioLines }, { id: 'layers', label: 'Layers', icon: Layers }, { id: 'midi', label: 'MIDI', icon: Piano }] },
  { label: 'Review', items: [{ id: 'library', label: 'Library', icon: Library }, { id: 'compare', label: 'Compare', icon: GitCompare }] },
  { label: 'Train', items: [{ id: 'training', label: 'Training', icon: Music }] },
];
const ALL_NAV = [...NAV_GROUPS.flatMap(g => g.items), { id: 'settings', label: 'Settings', icon: Settings }];
const SEND_TO = [
  { id: 'cover', label: 'Cover' }, { id: 'repaint', label: 'Repaint' }, { id: 'stems', label: 'Stems' }, { id: 'layers', label: 'Layers' }, { id: 'midi', label: 'MIDI' },
];
const dur = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;

// ---------- shared state ----------

interface Settings { model: string; adapter: string; adapterScale: number; seed: string; steps: number; cfg: number; sampler: string; lm: boolean; lmModel: string; denoise: boolean; normalise: boolean; format: string }
interface Mock {
  page: string; go: (p: string) => void;
  settings: Settings; set: (p: Partial<Settings>) => void;
  songs: Song[]; coverSong: Song | null; sendToCover: (s: Song) => void;
  compare: Song[]; addCompare: (s: Song) => void;
  generating: boolean; generate: () => void;
  playing: Song | null; play: (s: Song | null) => void;
  menu: { song: Song; x: number; y: number } | null; openMenu: (s: Song, el: HTMLElement) => void; closeMenu: () => void;
  menuSend: (to: string, s: Song) => void;
}
const Ctx = createContext<Mock>(null as unknown as Mock);
const useMock = () => useContext(Ctx);

const field = 'kit-well kit-mono w-full resize-none rounded-ctl p-3 text-sm text-ink outline-none placeholder:text-ink-muted/60 focus-visible:border-tint/60';
const input = 'kit-well kit-mono w-full rounded-ctl px-3 py-2 text-sm text-ink outline-none placeholder:text-ink-muted/60 focus-visible:border-tint/60';

const GlobalBadge: React.FC = () => (
  <span title="Shared setting: a change applies on every page" className="kit-mono rounded-full border border-tint/40 px-1.5 py-px text-[10px] normal-case tracking-normal text-tint-ink">global</span>
);

// ---------- shared building blocks ----------

/** One Source picker for every studio: Upload / Library / Recent. */
const SourcePicker: React.FC<{ value: Song | null; onPick: (s: Song) => void }> = ({ value, onPick }) => {
  const { songs } = useMock();
  const [tab, setTab] = useState('library');
  const [dropped, setDropped] = useState(false);
  return (
    <div className="space-y-3">
      <Tabs aria-label="Source" value={tab} onChange={setTab} items={[{ value: 'upload', label: 'Upload' }, { value: 'library', label: 'Library' }, { value: 'recent', label: 'Recent' }]} />
      {tab === 'upload' ? (
        <button type="button" onClick={() => setDropped(true)} className="kit-well kit-focus flex h-40 w-full flex-col items-center justify-center gap-2 rounded-ctl border border-dashed border-ink-muted/40 text-sm text-ink-muted">
          <Upload size={20} />{dropped ? 'my-demo.wav (mock file)' : 'Drop audio here or click to browse'}
        </button>
      ) : (
        <div className="max-h-56 space-y-1 overflow-y-auto">
          {(tab === 'recent' ? songs.slice(0, 3) : songs).map((s, i) => (
            <SongItem key={s.id} title={s.title} meta={s.meta} duration={s.duration} cover={COVERS[i % 5]} selected={value?.id === s.id} onClick={() => onPick(s)} />
          ))}
        </div>
      )}
    </div>
  );
};

const ArtistPicker: React.FC = () => {
  const [artist, setArtist] = useState('Alder Voss');
  return <Select aria-label="Artist" searchable searchPlaceholder="Search artists…" value={artist} onChange={setArtist} options={ARTISTS.map(a => ({ value: a, label: a }))} />;
};

/** The five folds both options share. Values live in the shared state, so closing a fold keeps them. */
const Folds: React.FC = () => {
  const { settings: s, set } = useMock();
  return (
    <div className="space-y-2">
      <Panel collapsible title="Model and adapter" actions={<GlobalBadge />}>
        <div className="space-y-3">
          <Select aria-label="Model" value={s.model} onChange={v => set({ model: v })} options={[{ value: 'xl', label: 'Base XL (turbo)' }, { value: 'sft', label: 'SFT 4B' }, { value: 'base', label: 'Base' }]} />
          <Select aria-label="Adapter" value={s.adapter} onChange={v => set({ adapter: v })} options={[{ value: 'none', label: 'No adapter' }, { value: 'a', label: 'Warm tape LoRA' }, { value: 'b', label: 'Brushed drums LoRA' }]} />
          <KitSlider label="Adapter scale" value={s.adapterScale} onChange={v => set({ adapterScale: v })} min={0} max={2} step={0.05} defaultValue={1} disabled={s.adapter === 'none'} />
          {s.adapter === 'none' && <p className="text-xs text-ink-muted">Pick an adapter to set its scale.</p>}
        </div>
      </Panel>
      <Panel collapsible title="Sampling" actions={<GlobalBadge />}>
        <div className="space-y-3">
          <label className="block text-xs font-medium uppercase tracking-wider text-ink-muted">Seed
            <input aria-label="Seed" className={`${input} mt-1.5`} value={s.seed} placeholder="Random" onChange={e => set({ seed: e.target.value })} />
          </label>
          <KitSlider label="Steps" showInput value={s.steps} onChange={v => set({ steps: v })} min={1} max={100} step={1} defaultValue={8} />
          <KitSlider label="Guidance" value={s.cfg} onChange={v => set({ cfg: v })} min={1} max={15} step={0.5} defaultValue={7} />
          <Select aria-label="Sampler" value={s.sampler} onChange={v => set({ sampler: v })} options={[{ value: 'euler', label: 'Euler' }, { value: 'heun', label: 'Heun' }, { value: 'dpm', label: 'DPM++' }]} />
        </div>
      </Panel>
      <Panel collapsible defaultOpen={false} title="LM" actions={<GlobalBadge />}>
        <div className="space-y-3">
          <KitToggle size="sm" checked={s.lm} onChange={v => set({ lm: v })} label="Use language model" info="One control: the switch and the model choice are the same setting." />
          <Select aria-label="LM model" disabled={!s.lm} value={s.lmModel} onChange={v => set({ lmModel: v })} options={[{ value: '1.7', label: 'LM 1.7B' }, { value: '4', label: 'LM 4B' }]} />
          {!s.lm && <p className="text-xs text-ink-muted">Switch the language model on to choose one.</p>}
        </div>
      </Panel>
      <Panel collapsible defaultOpen={false} title="Post-processing" actions={<GlobalBadge />}>
        <div className="space-y-3">
          <KitToggle size="sm" checked={s.denoise} onChange={v => set({ denoise: v })} label="Denoise" />
          <KitToggle size="sm" checked={s.normalise} onChange={v => set({ normalise: v })} label="Normalise loudness" />
        </div>
      </Panel>
      <Panel collapsible defaultOpen={false} title="Output" actions={<GlobalBadge />}>
        <Select aria-label="Format" value={s.format} onChange={v => set({ format: v })} options={[{ value: 'wav', label: 'WAV 24-bit' }, { value: 'flac', label: 'FLAC' }, { value: 'mp3', label: 'MP3 320' }]} />
      </Panel>
    </div>
  );
};

const GenerateButton: React.FC<{ label?: string }> = ({ label = 'Generate' }) => {
  const { generate, generating } = useMock();
  return <Button size="lg" variant="primary" className="w-full" disabled={generating} onClick={generate}><Play size={18} /> {generating ? 'Generating…' : label}</Button>;
};

const SongMenu: React.FC = () => {
  const { menu, closeMenu, menuSend, addCompare } = useMock();
  useEffect(() => {
    if (!menu) return;
    const off = (e: MouseEvent) => { if (!(e.target as HTMLElement).closest('[data-song-menu]')) closeMenu(); };
    document.addEventListener('mousedown', off);
    return () => document.removeEventListener('mousedown', off);
  }, [menu, closeMenu]);
  if (!menu) return null;
  return (
    <div role="menu" data-song-menu style={{ position: 'fixed', left: menu.x, top: menu.y }} className="kit-section z-[110] w-52 p-1.5 text-sm shadow-xl">
      <div className="px-2.5 pb-1 pt-1.5 text-[11px] font-semibold uppercase tracking-wider text-ink-muted">Send to</div>
      {SEND_TO.map(t => <button key={t.id} role="menuitem" type="button" onClick={() => menuSend(t.id, menu.song)} className="kit-item kit-focus w-full text-left">{t.label}</button>)}
      <Rail className="my-1.5" />
      <button role="menuitem" type="button" onClick={() => { addCompare(menu.song); closeMenu(); }} className="kit-item kit-focus w-full text-left">Add to compare</button>
    </div>
  );
};

// ---------- pages ----------

const LibraryPage: React.FC = () => {
  const { songs, openMenu, play, playing, compare } = useMock();
  return (
    <SectionPanel title="Library" actions={<span className="text-xs text-ink-muted">{compare.length} in compare</span>} bodyClassName="overflow-y-auto p-3" className="h-full">
      <div className="mx-auto max-w-4xl space-y-1">
        {songs.map((s, i) => (
          <div key={s.id} className="flex items-center gap-2">
            <SongItem className="flex-1" title={s.title} meta={s.meta} duration={s.duration} cover={COVERS[i % 5]} playing={playing?.id === s.id} onClick={() => play(playing?.id === s.id ? null : s)} />
            <IconButton aria-label={`Song menu: ${s.title}`} variant="ghost" onClick={e => openMenu(s, e.currentTarget)}><MoreHorizontal size={16} /></IconButton>
          </div>
        ))}
      </div>
    </SectionPanel>
  );
};

const ComparePage: React.FC = () => {
  const { compare } = useMock();
  return (
    <SectionPanel title="Compare" className="h-full" bodyClassName="grid grid-cols-2 gap-4">
      {['A', 'B'].map((slot, i) => (
        <Module key={slot} label={`Track ${slot}`}>
          {compare[i] ? <SongItem title={compare[i].title} meta={compare[i].meta} duration={compare[i].duration} cover={COVERS[i]} selected /> : <EmptyState title="Empty slot">Use “Add to compare” in a song menu.</EmptyState>}
        </Module>
      ))}
    </SectionPanel>
  );
};

const Placeholder: React.FC = () => {
  const { page } = useMock();
  return <SectionPanel title={ALL_NAV.find(n => n.id === page)?.label ?? page} className="h-full"><EmptyState icon={<ListMusic size={20} />} title="Not part of this mock">Create, Cover, Library and Compare are clickable; the rest of the nav only shows where each page lives.</EmptyState></SectionPanel>;
};

type CreateMode = 'quick' | 'custom';
const ModeTabs: React.FC<{ mode: CreateMode; set: (m: CreateMode) => void }> = ({ mode, set }) => (
  <Tabs<CreateMode> aria-label="Create mode" value={mode} onChange={set} items={[{ value: 'quick', label: 'Quick' }, { value: 'custom', label: 'Custom' }]} />
);

const CreateSource: React.FC<{ mode: CreateMode }> = ({ mode }) => {
  const [style, setStyle] = useState('Dreamy indie folk, warm acoustic guitar, soft female vocals');
  const [lyrics, setLyrics] = useState('[Verse 1]\nWalking through the morning light\nEvery shadow fading bright\n\n[Chorus]\nWe\'re alive, we\'re alive tonight');
  const [instr, setInstr] = useState(false);
  const [picked, setPicked] = useState<string[]>(['Lo-fi', 'Ambient']);
  if (mode === 'quick') return (
    <Module label="Pick a few genres">
      <div className="flex flex-wrap gap-2">
        {GENRES.map(g => (
          <Button key={g} size="sm" variant={picked.includes(g) ? 'primary' : 'secondary'} aria-pressed={picked.includes(g)} onClick={() => setPicked(p => p.includes(g) ? p.filter(x => x !== g) : [...p, g])}>{g}</Button>
        ))}
      </div>
      <p className="mt-3 text-xs text-ink-muted">The app writes the style and lyrics for you. Switch to Custom to edit them.</p>
    </Module>
  );
  return (
    <div className="space-y-3">
      <Module label="Style description"><textarea aria-label="Style description" rows={3} value={style} onChange={e => setStyle(e.target.value)} className={field} /></Module>
      <Module label="Lyrics">
        <div className="mb-2"><KitToggle size="sm" checked={instr} onChange={setInstr} label="Instrumental (no vocals)" /></div>
        <textarea aria-label="Lyrics" rows={9} value={lyrics} disabled={instr} onChange={e => setLyrics(e.target.value)} className={`${field} disabled:opacity-40`} />
      </Module>
    </div>
  );
};

const CreateShape: React.FC = () => {
  const [bpm, setBpm] = useState(96);
  const [d, setD] = useState(180);
  const [key, setKey] = useState('auto');
  return (
    <Module label="Music parameters">
      <div className="space-y-4">
        <KitSlider label="BPM" value={bpm} onChange={setBpm} min={40} max={220} step={1} defaultValue={96} />
        <KitSlider label="Duration" value={d} onChange={setD} min={30} max={300} step={5} suffix="s" defaultValue={180} />
        <Select aria-label="Key" value={key} onChange={setKey} options={[{ value: 'auto', label: 'Key: Auto' }, { value: 'c', label: 'C major' }, { value: 'am', label: 'A minor' }]} />
      </div>
    </Module>
  );
};

/** B's short "Basics", beside Generate. Seed and steps are NOT here: they live in Advanced. */
const Basics: React.FC = () => {
  const [q, setQ] = useState('std');
  const [n, setN] = useState(2);
  return (
    <Module label="Basics">
      <div className="space-y-4">
        <Select aria-label="Quality" value={q} onChange={setQ} options={[{ value: 'draft', label: 'Quality: Draft' }, { value: 'std', label: 'Quality: Standard' }, { value: 'best', label: 'Quality: Best' }]} />
        <KitSlider label="Takes" value={n} onChange={setN} min={1} max={4} step={1} defaultValue={2} />
      </div>
    </Module>
  );
};

const CoverSource: React.FC = () => {
  const { coverSong, sendToCover } = useMock();
  return (
    <Module label="Source">
      {coverSong && <div className="mb-3 rounded-ctl border border-tint/40 p-1"><SongItem title={coverSong.title} meta="Loaded as the cover source" duration={coverSong.duration} cover={COVERS[0]} selected /></div>}
      <SourcePicker value={coverSong} onPick={sendToCover} />
    </Module>
  );
};
const CoverShape: React.FC = () => {
  const [strength, setStrength] = useState(60);
  const [lyr, setLyr] = useState(true);
  return (
    <Module label="Artist and strength">
      <div className="space-y-4">
        <ArtistPicker />
        <KitSlider label="Cover strength" value={strength} onChange={setStrength} min={0} max={100} step={1} defaultValue={60} suffix="%" />
        <KitToggle size="sm" checked={lyr} onChange={setLyr} label="Keep original lyrics" />
      </div>
    </Module>
  );
};

// ---------- player / queue / recent ----------

const Player: React.FC = () => {
  const { playing, play, songs } = useMock();
  return (
    <div className="flex items-center gap-3">
      <IconButton aria-label={playing ? 'Pause' : 'Play'} variant="secondary" onClick={() => play(playing ? null : songs[0])}>{playing ? <Pause size={16} /> : <Play size={16} />}</IconButton>
      <div className="min-w-0 flex-1">
        <div className="truncate text-sm">{playing ? playing.title : 'Nothing playing'}</div>
        <div className="kit-well mt-1 h-1.5 rounded-full"><div className="h-full rounded-full bg-tint" style={{ width: playing ? '38%' : '0%' }} /></div>
      </div>
      <span className="kit-mono text-xs text-ink-muted">{playing ? `1:05 / ${playing.duration}` : '0:00'}</span>
    </div>
  );
};
const QueueList: React.FC = () => {
  const { generating } = useMock();
  return generating
    ? <div className="space-y-1.5"><StatusLed state="generating" label="Generating: Take (step 4 of 8)" /><div className="kit-well h-1.5 rounded-full"><div className="h-full w-1/2 rounded-full bg-tint" /></div></div>
    : <p className="text-xs text-ink-muted">Queue is empty.</p>;
};
const RecentList: React.FC = () => {
  const { songs, play, playing } = useMock();
  return <div className="space-y-1">{songs.slice(0, 5).map((s, i) => <SongItem key={s.id} title={s.title} meta={s.meta} duration={s.duration} cover={COVERS[i % 5]} playing={playing?.id === s.id} onClick={() => play(s)} />)}</div>;
};

// ---------- shells ----------

const TopBar: React.FC<{ search?: boolean }> = () => (
  <header className="kit-section-head !rounded-none">
    <span className="kit-heading text-sm font-semibold">HOT-Step</span>
    <StatusLed state="ok" label="Engine ready" />
    <div className="ml-auto flex items-center gap-3">
      <div className="relative w-72"><Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-muted" /><input aria-label="Search" placeholder="Search songs, artists…" className={`${input} !pl-9`} /></div>
      <div className="w-44"><Select size="sm" aria-label="Profile" value="studio" onChange={() => undefined} options={[{ value: 'studio', label: 'Profile: Studio' }, { value: 'live', label: 'Profile: Live' }]} /></div>
    </div>
  </header>
);

const PageBody: React.FC<{ create: React.ReactNode; cover: React.ReactNode }> = ({ create, cover }) => {
  const { page } = useMock();
  return page === 'create' ? <>{create}</> : page === 'cover' ? <>{cover}</>
    : page === 'library' ? <LibraryPage /> : page === 'compare' ? <ComparePage /> : <Placeholder />;
};

const ConsoleLayout: React.FC = () => {
  const { page, go, generating } = useMock();
  const [mode, setMode] = useState<CreateMode>('custom');
  const [dock, setDock] = useState(true);
  const [dockTab, setDockTab] = useState('queue');
  const inspector = (
    <SectionPanel title="Inspector" actions={<span className="text-xs text-ink-muted">Shared by every page</span>} className="w-[400px] shrink-0" bodyClassName="overflow-y-auto" footer={<div className="w-full"><GenerateButton label={page === 'cover' ? 'Generate cover' : 'Generate'} /></div>}>
      <Folds />
    </SectionPanel>
  );
  const create = (
    <>
      <SectionPanel title="Create" led={{ state: generating ? 'generating' : 'ok', label: generating ? 'Generating' : 'Ready' }} actions={<ModeTabs mode={mode} set={setMode} />} className="min-w-0 flex-1" bodyClassName="grid grid-cols-2 gap-4 overflow-y-auto">
        <CreateSource mode={mode} /><CreateShape />
      </SectionPanel>
      {inspector}
    </>
  );
  const cover = (
    <>
      <SectionPanel title="Cover Studio" led={{ state: generating ? 'generating' : 'ok', label: generating ? 'Generating' : 'Ready' }} className="min-w-0 flex-1" bodyClassName="grid grid-cols-2 gap-4 overflow-y-auto"><CoverSource /><CoverShape /></SectionPanel>
      {inspector}
    </>
  );
  const wide = page === 'create' || page === 'cover';
  return (
    <div className="flex h-full flex-col bg-surface-chassis">
      <TopBar />
      <div className="flex min-h-0 flex-1 gap-3 p-3">
        <nav aria-label="Navigation" className="kit-section flex w-14 shrink-0 flex-col items-center gap-1 p-1.5">
          {ALL_NAV.map(n => (
            <button key={n.id} type="button" title={n.label} aria-label={n.label} aria-current={page === n.id || undefined} onClick={() => go(n.id)} className="kit-item kit-focus !justify-center !p-2.5">
              <n.icon size={18} className={page === n.id ? 'text-tint-ink' : 'text-ink-muted'} />
            </button>
          ))}
        </nav>
        {wide ? <PageBody create={create} cover={cover} /> : <div className="min-w-0 flex-1"><PageBody create={create} cover={cover} /></div>}
      </div>
      <section aria-label="Dock" className="kit-section mx-3 mb-3 shrink-0 !rounded-panel">
        <div className="flex items-center gap-4 px-4 py-2.5">
          <div className="flex-1"><Player /></div>
          <IconButton aria-label={dock ? 'Collapse dock' : 'Expand dock'} aria-expanded={dock} variant="ghost" onClick={() => setDock(d => !d)}><ChevronDown size={16} className={dock ? '' : 'rotate-180'} /></IconButton>
        </div>
        {dock && (
          <div className="h-52 space-y-3 overflow-y-auto border-t border-black/30 p-3">
            <Tabs aria-label="Dock" value={dockTab} onChange={setDockTab} items={[{ value: 'queue', label: 'Queue' }, { value: 'recent', label: 'Recent' }]} />
            {dockTab === 'queue' ? <QueueList /> : <RecentList />}
          </div>
        )}
      </section>
    </div>
  );
};

const GuidedLayout: React.FC = () => {
  const { page, go, generating } = useMock();
  const [mode, setMode] = useState<CreateMode>('custom');
  const [drawer, setDrawer] = useState(false);
  const [actTab, setActTab] = useState('queue');
  const renderCol = (label: string) => (
    <div className="space-y-3">
      <Basics />
      <Button variant="secondary" className="w-full" aria-expanded={drawer} onClick={() => setDrawer(d => !d)}><SlidersHorizontal size={16} /> Advanced</Button>
      <GenerateButton label={label} />
    </div>
  );
  const colHead = (n: number, t: string) => <div className="kit-module-label !px-1 !pb-1.5">{n}. {t}</div>;
  const create = (
    <SectionPanel title="Create" led={{ state: generating ? 'generating' : 'ok', label: generating ? 'Generating' : 'Ready' }} actions={<ModeTabs mode={mode} set={setMode} />} className="h-full" bodyClassName="grid grid-cols-[1.3fr_1fr_0.9fr] gap-5 overflow-y-auto">
      <div>{colHead(1, 'Source')}<CreateSource mode={mode} /></div>
      <div>{colHead(2, 'Shape')}<CreateShape /></div>
      <div>{colHead(3, 'Render')}{renderCol('Generate')}</div>
    </SectionPanel>
  );
  const cover = (
    <SectionPanel title="Cover Studio" led={{ state: generating ? 'generating' : 'ok', label: generating ? 'Generating' : 'Ready' }} className="h-full" bodyClassName="grid grid-cols-[1.3fr_1fr_0.9fr] gap-5 overflow-y-auto">
      <div>{colHead(1, 'Source')}<CoverSource /></div>
      <div>{colHead(2, 'Shape')}<CoverShape /></div>
      <div>{colHead(3, 'Render')}{renderCol('Generate cover')}</div>
    </SectionPanel>
  );
  return (
    <div className="flex h-full flex-col bg-surface-chassis">
      <TopBar />
      <div className="flex min-h-0 flex-1 gap-3 p-3">
        <nav aria-label="Navigation" className="kit-section flex w-52 shrink-0 flex-col gap-3 overflow-y-auto p-2">
          {NAV_GROUPS.map(g => (
            <div key={g.label}>
              <div className="px-2.5 pb-1 text-[11px] font-semibold uppercase tracking-wider text-ink-muted">{g.label}</div>
              {g.items.map(n => (
                <button key={n.id} type="button" aria-current={page === n.id || undefined} onClick={() => go(n.id)} className="kit-item kit-focus w-full">
                  <n.icon size={16} className={page === n.id ? 'text-tint-ink' : 'text-ink-muted'} /><span className="text-sm">{n.label}</span>
                </button>
              ))}
            </div>
          ))}
          <button type="button" aria-current={page === 'settings' || undefined} onClick={() => go('settings')} className="kit-item kit-focus mt-auto w-full"><Settings size={16} className="text-ink-muted" /><span className="text-sm">Settings</span></button>
        </nav>
        <div className="min-w-0 flex-1"><PageBody create={create} cover={cover} /></div>
        {/* In the flow, not over it: the page squeezes so Generate stays in view. */}
        <div className={`shrink-0 overflow-hidden transition-[width,visibility] duration-200 ${drawer ? 'w-[420px]' : 'invisible w-0'}`}>
          <aside aria-label="Advanced" aria-hidden={!drawer} className="kit-section h-full w-[420px]">
            <header className="kit-section-head"><h2 className="kit-section-title">Advanced</h2><span className="text-xs text-ink-muted">Shared by every page</span><IconButton className="ml-auto" aria-label="Close advanced" variant="ghost" onClick={() => setDrawer(false)}><X size={16} /></IconButton></header>
            <div className="kit-section-body overflow-y-auto"><Folds /></div>
          </aside>
        </div>
        <SectionPanel title="Activity" className="w-[340px] shrink-0" bodyClassName="space-y-3 overflow-y-auto" footer={<div className="w-full"><Player /></div>}>
          <Tabs aria-label="Activity" value={actTab} onChange={setActTab} items={[{ value: 'queue', label: 'Queue' }, { value: 'recent', label: 'Recent' }]} />
          {actTab === 'queue' ? <QueueList /> : <RecentList />}
        </SectionPanel>
      </div>
    </div>
  );
};

// ---------- host ----------

export const MockLayouts: React.FC = () => {
  const [layout, setLayout] = useState<'A' | 'B'>('A');
  const [page, go] = useState('create');
  const [settings, setSettings] = useState<Settings>({ model: 'xl', adapter: 'none', adapterScale: 1, seed: '', steps: 8, cfg: 7, sampler: 'euler', lm: true, lmModel: '1.7', denoise: false, normalise: true, format: 'wav' });
  const [songs, setSongs] = useState(SONGS);
  const [coverSong, setCoverSong] = useState<Song | null>(null);
  const [compare, setCompare] = useState<Song[]>([]);
  const [generating, setGenerating] = useState(false);
  const [playing, play] = useState<Song | null>(null);
  const [menu, setMenu] = useState<Mock['menu']>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);

  const mock: Mock = {
    page, go, settings, set: p => setSettings(s => ({ ...s, ...p })), songs, coverSong, compare, generating, playing, play, menu,
    sendToCover: s => setCoverSong(s),
    addCompare: s => setCompare(c => (c.some(x => x.id === s.id) ? c : [...c, s].slice(-2))),
    generate: () => {
      setGenerating(true);
      timer.current = setTimeout(() => { setGenerating(false); setSongs(s => [{ id: Date.now(), title: `Take ${s.length + 1}`, meta: `${settings.steps} steps, seed ${settings.seed || 'random'}`, duration: dur(180) }, ...s]); }, 3000);
    },
    openMenu: (s, el) => { const r = el.getBoundingClientRect(); setMenu({ song: s, x: Math.min(r.right - 208, window.innerWidth - 220), y: Math.min(r.bottom + 4, window.innerHeight - 260) }); },
    closeMenu: () => setMenu(null),
    menuSend: (to, s) => { setMenu(null); if (to === 'cover') setCoverSong(s); go(to); },
  };

  const [full, setFull] = useState(false);
  const { theme, toggleTheme } = useTheme();
  useEffect(() => {
    if (!full) return;
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') setFull(false); };
    document.addEventListener('keydown', esc);
    return () => document.removeEventListener('keydown', esc);
  }, [full]);
  const layoutTabs = <Tabs aria-label="Layout option" value={layout} onChange={setLayout} items={[{ value: 'A', label: 'A: Console' }, { value: 'B', label: 'B: Guided' }]} />;

  return (
    <Ctx.Provider value={mock}>
      <div className="mb-3 flex items-center gap-3 text-xs text-ink-muted">
        Layout {layoutTabs}
        <Button size="sm" variant="primary" onClick={() => setFull(true)}>Open at 1920x1080</Button>
        <span>Fake data only. The full-screen view is a fixed 1920x1080 canvas; Esc closes it.</span>
      </div>
      {!full && <div className="kit-well flex h-24 items-center justify-center rounded-panel text-sm text-ink-muted">Open the mock full-screen to try it.</div>}
      {full && (
        <div id="layout-mock" className="fixed left-0 top-0 z-[100] h-[1080px] w-[1920px] overflow-hidden bg-surface-chassis text-ink">
          {layout === 'A' ? <ConsoleLayout /> : <GuidedLayout />}
          <div className="fixed left-1/2 top-1.5 z-[101] flex -translate-x-1/2 items-center gap-3 rounded-panel bg-surface-chassis/90 px-2 py-1 text-xs text-ink-muted">
            {layoutTabs}
            <Tabs aria-label="Theme" value={theme} onChange={v => v !== theme && toggleTheme()} items={[{ value: 'dark', label: 'Dark' }, { value: 'light', label: 'Light' }]} />
            <Button size="sm" variant="secondary" onClick={() => setFull(false)}><X size={14} /> Exit</Button>
          </div>
        </div>
      )}
      <SongMenu />
    </Ctx.Provider>
  );
};
