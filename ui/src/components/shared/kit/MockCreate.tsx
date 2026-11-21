// MockCreate.tsx — a static, illustrative Create page built only from kit
// parts, to judge the kit at page scale. Every control is live; nothing is
// wired to the app. Test page only (see KitGallery).

import React, { useEffect, useRef, useState } from 'react';
import { AudioLines, Disc3, Library, ListMusic, Mic2, Music, Play, Sparkles, Wand2 } from 'lucide-react';
import { Button } from './Button';
import { EmptyState, Module, Rail, SectionPanel, SongItem } from './Panels';
import { Select } from './Select';
import { KitSlider } from './Slider';
import { Tabs } from './Tabs';
import { KitToggle } from './Toggle';

const NAV = [
  { id: 'create', label: 'Create', icon: Wand2 },
  { id: 'library', label: 'Library', icon: Library },
  { id: 'lyrics', label: 'Lyric Studio', icon: Mic2 },
  { id: 'cover', label: 'Cover Studio', icon: Disc3 },
  { id: 'stems', label: 'Stem Separator', icon: AudioLines },
  { id: 'training', label: 'Training', icon: Music },
];

const COVERS = [
  'linear-gradient(135deg, #5b8def, #1b2a5c)',
  'linear-gradient(135deg, #e0668f, #5c1b3a)',
  'linear-gradient(135deg, #6fd3a0, #1b5c3f)',
  'linear-gradient(135deg, #c98bf0, #3f1b5c)',
  'linear-gradient(135deg, #f0b36b, #5c3a1b)',
];
const SONGS = [
  { title: 'Studio sketch 01', meta: 'Lo-fi hip hop, mellow electric piano', duration: '2:51' },
  { title: 'Evening texture', meta: 'Ambient, warm pads, slow', duration: '3:56' },
  { title: 'Untitled take', meta: 'Folk rock, acoustic guitar', duration: '3:03' },
  { title: 'Ambient fragment', meta: 'Drone, field recording', duration: '0:28' },
  { title: 'Field recording', meta: 'Spoken word, room tone', duration: '5:00' },
];

const field = 'kit-well kit-mono w-full resize-none rounded-ctl p-3 text-sm text-ink outline-none placeholder:text-ink-muted/60 focus-visible:border-tint/60';

export const MockCreate: React.FC = () => {
  const [nav, setNav] = useState('create');
  const [mode, setMode] = useState('oneshot');
  const [style, setStyle] = useState('Dreamy indie folk, warm acoustic guitar, soft female vocals');
  const [lyrics, setLyrics] = useState('[Verse 1]\nWalking through the morning light\nEvery shadow fading bright\n\n[Chorus]\nWe\'re alive, we\'re alive tonight');
  const [instrumental, setInstrumental] = useState(false);
  const [bpm, setBpm] = useState(96);
  const [dur, setDur] = useState(180);
  const [key, setKey] = useState('auto');
  const [sig, setSig] = useState('4');
  const [generating, setGenerating] = useState(false);
  const [songs, setSongs] = useState(SONGS);
  const [selected, setSelected] = useState(0);
  const [playing, setPlaying] = useState<number | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);

  const generate = () => {
    setGenerating(true);
    timer.current = setTimeout(() => {
      setGenerating(false);
      setSongs(s => [{ title: `Take ${s.length + 1}`, meta: style.slice(0, 40), duration: `${Math.floor(dur / 60)}:${String(dur % 60).padStart(2, '0')}` }, ...s]);
      setSelected(0);
      setPlaying(null);
    }, 3000);
  };

  return (
    <div className="grid h-[920px] grid-cols-[180px_minmax(0,1fr)_320px] gap-3 rounded-panel bg-surface-chassis p-3">
      {/* Left nav strip */}
      <nav aria-label="Mock navigation" className="kit-section flex flex-col gap-1 p-2">
        {NAV.map(n => (
          <button key={n.id} type="button" aria-current={nav === n.id || undefined} onClick={() => setNav(n.id)} className="kit-item kit-focus">
            <n.icon size={16} className={nav === n.id ? 'text-tint' : 'text-ink-muted'} />
            <span className="text-sm">{n.label}</span>
          </button>
        ))}
      </nav>

      {/* Centre: Create */}
      <SectionPanel
        title="Custom-Gen"
        led={{ state: generating ? 'generating' : 'ok', label: generating ? 'Generating' : 'Ready' }}
        actions={<Button size="sm" variant="secondary"><Sparkles size={14} /> Generate with AI</Button>}
        bodyClassName="space-y-3 overflow-y-auto"
        footer={<><span>Draft saved</span><span className="kit-mono">{bpm} BPM · {Math.floor(dur / 60)}:{String(dur % 60).padStart(2, '0')}</span></>}
      >
        <Tabs aria-label="Mode" value={mode} onChange={setMode} items={[{ value: 'oneshot', label: 'One-shot' }, { value: 'sections', label: 'Build by section' }]} />
        <Module label="Style description">
          <textarea aria-label="Style description" rows={2} value={style} onChange={e => setStyle(e.target.value)} className={field} />
        </Module>
        <Module label="Lyrics">
          <div className="mb-2"><KitToggle size="sm" checked={instrumental} onChange={setInstrumental} label="Instrumental (no vocals)" /></div>
          <textarea aria-label="Lyrics" rows={6} value={lyrics} disabled={instrumental} onChange={e => setLyrics(e.target.value)} className={`${field} disabled:opacity-40`} />
        </Module>
        <Module label="Music parameters">
          <div className="flex items-stretch gap-4">
            <div className="min-w-0 flex-1"><KitSlider label="BPM" value={bpm} onChange={setBpm} min={40} max={220} step={1} defaultValue={96} /></div>
            <Rail vertical />
            <div className="min-w-0 flex-1"><KitSlider label="Duration" value={dur} onChange={setDur} min={30} max={300} step={5} suffix="s" defaultValue={180} /></div>
          </div>
          <Rail className="my-3" />
          <div className="grid grid-cols-2 gap-4">
            <Select value={key} onChange={setKey} options={[{ value: 'auto', label: 'Key: Auto' }, { value: 'c', label: 'C major' }, { value: 'am', label: 'A minor' }, { value: 'g', label: 'G major' }]} />
            <Select value={sig} onChange={setSig} options={[{ value: '4', label: 'Time: 4/4' }, { value: '3', label: 'Time: 3/4' }, { value: '6', label: 'Time: 6/8' }]} />
          </div>
        </Module>
        <Button size="lg" variant="primary" className="w-full" disabled={generating} onClick={generate}>
          <Play size={18} /> {generating ? 'Generating…' : 'Generate'}
        </Button>
      </SectionPanel>

      {/* Right: song list */}
      <SectionPanel
        title="Playlist"
        actions={<Button size="sm" variant="ghost" onClick={() => { setSongs([]); setPlaying(null); }}>Clear</Button>}
        footer={<><span>{songs.length} songs</span>{playing !== null && <span>Now playing</span>}</>}
        bodyClassName="overflow-y-auto p-2"
      >
        {songs.length === 0 ? (
          <EmptyState icon={<ListMusic size={20} />} title="Nothing here yet" action={<Button size="sm" variant="secondary" onClick={() => setSongs(SONGS)}>Restore samples</Button>}>
            Generate a song and it lands in this list.
          </EmptyState>
        ) : (
          <div className="space-y-1">
            {songs.map((s, i) => (
              <SongItem
                key={s.title}
                title={s.title}
                meta={s.meta}
                duration={s.duration}
                cover={COVERS[i % COVERS.length]}
                selected={selected === i}
                playing={playing === i}
                onClick={() => { setSelected(i); setPlaying(p => (p === i ? null : i)); }}
              />
            ))}
          </div>
        )}
      </SectionPanel>
    </div>
  );
};
