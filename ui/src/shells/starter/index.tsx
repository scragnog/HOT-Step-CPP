// index.tsx — the starter shell. It uses every skin-api 0 export once, so it doubles as the
// reference and as the canary: `npx tsc -b` fails if the API drifts under it.
//
// Rules for a shell (docs/dev/skins.md#shell-skins):
//   - The default export is the root component. It takes no props. ShellHost has already
//     mounted the app runtime, the providers and the offline/restart/update overlays: never
//     call them yourself.
//   - Import only from this folder, '../../skin-api', 'react', 'lucide-react' and the kit
//     (components/shared/kit). `npm run lint` and `npm run check:skins` enforce it.
//   - No fetch, no localStorage: the API has hooks for data and useSkinStorage for preferences.

import React, { useState } from 'react';
import { Moon } from 'lucide-react';
import {
  ClassicPage, SKIN_API_VERSION, backend, cancel, generate, openClassic, openInStudio, params,
  useActiveBackend, useBackendModels, useBackends, useCapabilities, useEngineLog, useParamHistory,
  useQueue, useRecentSongs, useSkinStorage, useSongs, useTheme, useThemeSkin, useVram,
} from '../../skin-api';
import Select from '../../components/shared/kit/Select';
import { ParamControl } from './ParamControl';
import './starter.css';

const TABS = ['create', 'songs', 'engine', 'training'] as const;
type Tab = typeof TABS[number];
const TUNING = ['inferenceSteps', 'guidanceScale', 'shift'];

const Create: React.FC = () => {
  const [caption, setCaption] = useSkinStorage('caption', '');
  const [lyrics, setLyrics] = useState('');
  const [seconds, setSeconds] = useState(20);
  const { items, active } = useQueue();
  const models = useBackendModels();
  const caps = useCapabilities();
  const history = useParamHistory();
  const ext = caps?.extensionIds[0]; // a knob this backend declares, if any

  return (
    <>
      <section className="st-panel">
        <span className="st-title">Prompt</span>
        <textarea className="st-input" rows={2} placeholder="Style, mood, instruments" value={caption} onChange={e => setCaption(e.target.value)} />
        <textarea className="st-input" rows={4} placeholder="Lyrics (empty for an instrumental)" value={lyrics} onChange={e => setLyrics(e.target.value)} />
        <div className="st-row">
          {/* generate() returns at once; progress is in the queue below. Call it from a click. */}
          <button className="st-btn st-btn-on" disabled={!caption.trim()} onClick={() => generate({ caption, lyrics, instrumental: !lyrics.trim(), duration: seconds })}>Generate</button>
          <input className="st-input" style={{ width: '5rem' }} type="number" min={5} aria-label="Seconds" value={seconds} onChange={e => setSeconds(Number(e.target.value))} />
          <span className="st-muted">{active} in the queue</span>
        </div>
      </section>

      <section className="st-panel">
        <div className="st-row">
          <span className="st-title st-bar-grow">Controls</span>
          <button className="st-btn" disabled={!history.canUndo} onClick={history.undo}>Undo</button>
          <button className="st-btn" disabled={!history.canRedo} onClick={history.redo}>Redo</button>
          {/* A transaction is one undo step, however many controls it touches. */}
          <button className="st-btn" onClick={() => params.transaction('Reset tuning', () => TUNING.forEach(id => params.reset(id)))}>Reset tuning</button>
        </div>
        {/* Model selects ship without options: fill them from the catalogue. */}
        <ParamControl id="ditModel" options={models?.buckets.dit} />
        {TUNING.map(id => <ParamControl key={id} id={id} />)}
        {ext && <ParamControl id={ext} />}
        <span className="st-muted">Last: {history.cursor > 0 ? history.steps[history.cursor - 1].label : 'nothing yet'}</span>
      </section>

      <section className="st-panel">
        <span className="st-title">Queue</span>
        {items.length === 0 && <span className="st-muted">Nothing queued.</span>}
        {items.map(i => (
          <div key={i.id} className="st-row">
            <span className="st-bar-grow">{i.title} <span className="st-muted">{i.state}{i.stage ? ` · ${i.stage}` : ''}</span></span>
            <span className="st-meter"><span className="st-meter-fill" style={{ width: `${i.progress}%` }} /></span>
            {i.error && <span className="st-error">{i.error}</span>}
            {(i.state === 'waiting' || i.state === 'running') && <button className="st-btn" onClick={() => cancel(i.id)}>Cancel</button>}
          </div>
        ))}
      </section>
    </>
  );
};

const Songs: React.FC = () => {
  const { songs } = useSongs();
  const recent = useRecentSongs('create');
  return (
    <>
      <section className="st-panel">
        <span className="st-title">Library ({songs.length})</span>
        {songs.slice(0, 8).map(s => (
          <div key={s.id} className="st-row">
            <span className="st-bar-grow">{s.title} <span className="st-muted">{Math.round(s.duration)} s</span></span>
            {/* Cover, stems, MIDI and repaint are not in api 0: hand off to classic. */}
            <button className="st-btn" onClick={() => openInStudio('cover', s.id)}>Cover studio</button>
          </div>
        ))}
      </section>
      <section className="st-panel">
        <span className="st-title">Latest from Create</span>
        {recent.songs.slice(0, 3).map(s => <audio key={s.id} controls src={s.audioUrl} />)}
      </section>
    </>
  );
};

const Engine: React.FC = () => {
  const { backends } = useBackends();
  const vram = useVram();
  const { lines } = useEngineLog();
  const [showLog, setShowLog] = useSkinStorage('show-log', true);
  return (
    <>
      <section className="st-panel">
        <span className="st-title">Backend</span>
        <div className="st-row">
          {backends.map(b => <button key={b.id} className={`st-btn${b.active ? ' st-btn-on' : ''}`} onClick={() => void backend.switch(b.id)}>{b.name}</button>)}
        </div>
        {vram && <span className="st-muted">VRAM {vram.usedMb} / {vram.totalMb} MB</span>}
      </section>
      <section className="st-panel">
        <div className="st-row">
          <span className="st-title st-bar-grow">Engine log</span>
          <button className="st-btn" onClick={() => setShowLog(!showLog)}>{showLog ? 'Hide' : 'Show'}</button>
        </div>
        {showLog && <div className="st-log">{lines.slice(-60).map(l => l.text).join('\n')}</div>}
      </section>
    </>
  );
};

const StarterApp: React.FC = () => {
  const [tab, setTab] = useSkinStorage<Tab>('tab', 'create');
  const theme = useTheme();
  const active = useActiveBackend();
  const { skin, setSkin, available } = useThemeSkin(); // optional: only if you wear the kit's colour skins

  return (
    <div className="st-page">
      <div className="st-bar">
        <span className="st-title">Starter shell</span>
        <span className="st-muted">api {SKIN_API_VERSION} · {active?.name ?? '…'}</span>
        <span className="st-bar-grow" />
        {TABS.map(t => <button key={t} className={`st-btn${tab === t ? ' st-btn-on' : ''}`} onClick={() => setTab(t)}>{t}</button>)}
        <button className="st-btn" title="Cycles system, light, dark" onClick={theme.cycle}><Moon size={14} /> {theme.mode}</button>
        {available.length > 0 && (
          <Select<string> size="sm" className="w-36" value={skin ?? ''} onChange={v => setSkin(v || null)} aria-label="Colour skin"
            options={[{ value: '', label: 'No skin' }, ...available.map(a => ({ value: a.id, label: a.name }))]} />
        )}
        <button className="st-btn" onClick={openClassic}>Classic</button>
      </div>
      {tab === 'create' && <Create />}
      {tab === 'songs' && <Songs />}
      {tab === 'engine' && <Engine />}
      {/* A classic page, lazy-loaded inside this frame, for what the shell has not built. */}
      {tab === 'training' && <ClassicPage view="training-studio" className="st-panel" />}
    </div>
  );
};

export default StarterApp;
