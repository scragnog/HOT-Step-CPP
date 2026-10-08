# ui-feature-dev — reference

Deep-dive companion to [SKILL.md](SKILL.md). All paths/line refs verified against the repo on 2026-07-02.

## 1. Worked example — Stem Studio end-to-end

Files: `ui/src/components/stem-studio/{StemStudio,SourceSelector,TrackSelector,RecentExtractions}.tsx`, `ui/src/services/stemStudioApi.ts`, `server/src/routes/stemStudio.ts` (mounted `app.use('/api/stem-studio', stemStudioRoutes)` at `server/src/index.ts:89`).

### 1.1 Registration (the five edits)

- Import: `App.tsx:38` — `import { StemStudio } from './components/stem-studio/StemStudio';`
- URL → view: `App.tsx:88` — `if (path.startsWith('/stem-studio')) return 'stem-studio';`
- View → URL: `App.tsx:109` — `if (view === 'stem-studio') return '/stem-studio';`
- Render: `App.tsx:681-687` — `<DiscoPulseWrapper hue={DISCO.assistant} className="flex-1 overflow-hidden"><StemStudio /></DiscoPulseWrapper>`
- Nav: `Sidebar.tsx:111-117` — `NavItem` with `icon={<Scissors size={20} />}`, `label={t('sidebar.stemStudio')}`, `active={activeView === 'stem-studio'}`, `onClick={() => onViewChange('stem-studio')}`, `isExpanded={isOpen}`.

### 1.2 Orchestrator state (StemStudio.tsx:40-104)

All local `useState` — no dedicated store. Persistence is piecemeal:

- `hs-stem-sourceUrl` / `hs-stem-sourceFile`: lazy `useState(() => localStorage.getItem(...))` init + setter wrappers that write-through (lines 43-46).
- `hs-stem-extractModel`, `hs-stem-sepLevel`: persisted in `useEffect`s (lines 97-104).
- Cross-studio state touched: shared queue count via `useAudioGenQueueSelector(s => s.items.filter(i => i.status === 'pending' || i.status === 'loading-adapter' || i.status === 'generating').length)` (lines 77-79) and shared sidebar width via `usePersistedState('hs-activitySidebarWidth', 320)` (line 76).

### 1.3 Server data on mount (lines 81-94)

```tsx
useEffect(() => {
  modelApi.list()
    .then(data => {
      const base = getBaseModels(data.models.dit || []);   // name-prefix filter, lines 32-38
      setBaseModels(base);
      if (base.length > 0) {
        const stored = localStorage.getItem('hs-stem-extractModel');
        setExtractModel(stored && base.includes(stored) ? stored : base[0]);
      }
    })
    .catch(err => console.error('[StemStudio] Failed to load models:', err))
    .finally(() => setModelsLoading(false));
}, []);
```

Error handling convention: log to console, show an inline warning banner when the resulting list is empty (lines 307-312) — no toast spam, no crash.

### 1.4 Job submit + poll (`handleExtract`, lines 150-203)

1. Guard clauses (`!sourceAudioUrl`, empty tracks, no model) → early return.
2. `setIsExtracting(true)` → `submitExtraction({ sourceAudioUrl, sourceFileName, tracks, style?, lyrics?, ditSettings })` returns `jobId` (POST `/api/stem-studio/extract`, `stemStudioApi.ts:107-119` — server responds `{ id }`).
3. `waitForExtraction(jobId, setExtractProgress)` (`stemStudioApi.ts:190-211`) polls `GET /api/stem-studio/:id/progress` every 1000 ms until `status ∈ {done, failed, cancelled}`; on `done` fetches `GET /:id/result` → `{ id, stems: [{ trackName, audioUrl, durationSec, index }] }`.
4. `loadResultIntoMixer` (lines 205-215) maps stems into shared `StemMixer` props.
5. **Failures are folded back into the same `ExtractProgress` shape** with `status: 'failed'` + `error` (lines 190-199) so one render path handles progress AND errors.
6. `finally { setIsExtracting(false); }`.

Progress shape (`stemStudioApi.ts:22-31`):

```ts
interface ExtractProgress {
  status: 'pending' | 'extracting' | 'separating' | 'saving' | 'done' | 'failed' | 'cancelled';
  progress: number;          // 0-100
  currentTrack: string;
  completedStems: string[];
  totalTracks: number;
  warning?: string; error?: string; sepMessage?: string;
}
```

### 1.5 Layout skeleton (lines 264-415)

```
div.flex.flex-col.w-full.h-full.overflow-hidden          ← root
  div.flex-1.flex.overflow-hidden                         ← main row
    div (style={{width: 300}}, border-r, flex-shrink-0)   ← left: source + optional fields
    div.flex-1 (overflow-y-auto, border-r)                ← center: selectors, progress, mixer
    div.w-1.5.cursor-col-resize (onMouseDown=resize)      ← drag handle
    div (style={{width: sidebarWidth}}, border-l)         ← right: Section×2 (Recent + Queue)
```

Right sidebar reuses `Section` from `shared/ActivitySidebar.tsx` and `InlineAudioQueue` from `lyric-studio/` (lines 390-413) — the standard "Recent X + Queue" pattern. Resize handler clamps `Math.min(700, Math.max(240, ...))` (line 249) and sets/clears `document.body.style.cursor/userSelect`.

### 1.6 Downloads (lines 227-241)

Synthesized `<a download>` clicks against direct GET URLs (`getStemUrl`, `getDownloadAllUrl`) — no blob juggling.

## 2. Store templates

### 2.1 Hand-rolled external store (copy `abCompareStore.ts`)

```ts
import { useSyncExternalStore, useRef, useCallback } from 'react';

export interface MyState { value: number }
let _state: MyState = { value: 0 };

const _listeners = new Set<() => void>();
function notify() { _listeners.forEach(cb => cb()); }
function setState(updates: Partial<MyState>) { _state = { ..._state, ...updates }; notify(); }
function subscribe(cb: () => void) { _listeners.add(cb); return () => _listeners.delete(cb); }
function getSnapshot() { return _state; }

// Plain action functions — callable from non-React code
export function setValue(v: number) { setState({ value: v }); }

// Hooks
export function useMyStore(): MyState { return useSyncExternalStore(subscribe, getSnapshot); }
export function useMyStoreSelector<T>(selector: (s: MyState) => T): T {
  const selectorRef = useRef(selector); selectorRef.current = selector;
  const selectedRef = useRef<T>(selector(_state));
  const getSelected = useCallback(() => {
    const next = selectorRef.current(_state);
    if (Object.is(selectedRef.current, next)) return selectedRef.current;
    selectedRef.current = next; return next;
  }, []);
  return useSyncExternalStore(subscribe, getSelected);
}
```

This is exactly `abCompareStore.ts:23-101`. The selector hook's `Object.is` guard prevents render loops from fresh-object selectors.

### 2.2 Adding a field to globalParamsStore (zustand, per-key localStorage)

Three touch points in `ui/src/stores/globalParamsStore.ts`:

1. State init: `myKnob: readKey("hs-myKnob", <default>),`
2. Setter: `setMyKnob: (v: number) => { set({ myKnob: v }); writeKey('hs-myKnob', v); },`
3. **`getGlobalParams()`** (starts line 324): add the field to the returned `Partial<GenerationParams>`, gated `undefined` unless its feature toggle is on (match the existing conditional-gating style). Also add the field to `GenerationParams` in `ui/src/types.ts`.
4. **Server side**: map the field in `server/src/services/generation/translateParams.ts` (entry point `translateParams()` at :11, consumed by `server/src/routes/generate.ts:191`) so it lands in `aceReq`. **Sideband rule**: ServerFields-only params are not part of the C++ `AceRequest` struct and never survive the `/lm` round trip — `generate.ts` rebuilds each LM result from the current `aceReq`, taking only LM-generated fields from the echo (comment at generate.ts:311-319). Add your param to `aceReq` before the LM phase; never extend the LM-echo pick-list into a copy-across whitelist.

Consume with a selector: `const myKnob = useGlobalParamsStore(s => s.myKnob);`

### 2.3 usePersistedState / writePersistedState

`usePersistedState(key, default)` (`hooks/usePersistedState.ts:9-39`) = useState + localStorage + storage-event listener. To update it from OUTSIDE React (store code, another component tree), you MUST use `writePersistedState(key, value)` (lines 43-49) — it dispatches a synthetic `StorageEvent` because native ones only fire cross-tab.

## 3. Per-studio API client template (from stemStudioApi.ts)

```ts
// myStudioApi.ts — API client for My Studio
const API_BASE = '/api/my-studio';

export interface MyJobProgress { status: 'pending' | 'running' | 'done' | 'failed'; progress: number; error?: string }

export async function submitJob(params: MyParams): Promise<string> {
  const res = await fetch(`${API_BASE}/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
    throw new Error(err.error || `Submit failed: ${res.status}`);
  }
  return (await res.json()).id;
}

export async function waitForJob(jobId: string, onProgress?: (p: MyJobProgress) => void, pollMs = 1000): Promise<MyResult> {
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const p: MyJobProgress = await (await fetch(`${API_BASE}/${jobId}/progress`)).json();
        onProgress?.(p);
        if (p.status === 'done') resolve(await (await fetch(`${API_BASE}/${jobId}/result`)).json());
        else if (p.status === 'failed') reject(new Error(p.error || 'failed'));
        else setTimeout(poll, pollMs);
      } catch (err) { reject(err); }
    };
    poll();
  });
}
```

Conventions: no shared client, no token (newer endpoints skip auth), errors as `Error` with the server's `error` field, submit-then-poll (no websockets).

For the main generation pipeline instead, use `generateApi` (`api.ts:148+`): `submit(params, token)` → `{ jobId, status }`; `status(jobId)` → `GenerationJob` (`types.ts:266-281`, statuses `pending|lm_running|synth_running|saving|succeeded|failed|cancelled`); also `cancel`, `cancelAll`, `queueStatus`. Enqueue via `audioGenQueueStore` rather than reimplementing the loop.

## 4. DISCO hue table (App.tsx:279-289)

| Key | Hue | Colour | Typical use |
|---|---|---|---|
| `sidebar` | 120 | green | left nav |
| `createPanel` | 330 | hot pink | create/insta-gen/repaint/song-builder panels |
| `songGrid` | 195 | deep sky blue | library grids, lyric studio |
| `activity` | 270 | purple | settings, cover studio, activity sidebar |
| `playlist` | 20 | orange | playlist sidebar |
| `assistant` | 175 | cyan | assistant, stem studio, stem builder |
| `terminal` | 135 | green | terminal panel |
| `player` | 45 | gold | player bar |
| `rightSidebar` | 300 | magenta | right sidebars |

`DISCO` is a local const inside the `App` component — new render branches reference it directly; pick the hue matching your studio's family.

## 5. localStorage / persistence namespaces

| Prefix / key | Owner |
|---|---|
| `hs-*` | global params (per-field, globalParamsStore), UI layout (`hs-activitySidebarWidth` — shared across studios), studio-local (`hs-stem-*`), `hs-lastLyricStudioUrl` |
| `ace-settings` | AppSettings blob (SettingsPanel; read back inside `getGlobalParams()` at globalParamsStore.ts:326) |
| `lireek-audio-gen-queue` | LEGACY queue localStorage key — migration shim only (audioGenQueueStore.ts:73,164-179) |
| IndexedDB `lireek-queue-store` | current queue persistence (store `queue`, key `state`) |
| `i18nextLng` | language detector cache (i18n/index.ts:48) |

## 6. Shared components inventory (`ui/src/components/shared/`)

`ABCompareModal`, `Accordion`, `ActivitySidebar` (exports **`Section`** — props `title: string`, `icon: ReactNode`, `count?: number`, `countColor?: string`, `defaultOpen?: boolean`, `children`; verified ActivitySidebar.tsx:21-31), `ConfirmDialog`, `CoverArtSubjectSection`, `DiscoPulseWrapper`, `EditableSlider`, `FileBrowserModal`, `HiHatParticles`, `HoverFullText`, `LatentImport`, `Slider`, `SnareFlashOverlay`, `StemMixer`, `Toast` (`ToastType = 'success' | 'error' | 'info' | 'warning'`), `UnifiedRecentSongs`, `VramIndicator`.

Hooks (`ui/src/hooks/`): `usePersistedState`, `useEventSource` (SSE), `usePolling`, `useTheme`, `usePluginRegistry`, `useStreamGeneration`, `useDisguiseMode`, `useAssistantActions`.

## 7. Status ledger (what's current vs legacy)

- **Current**: hand-rolled external stores (majority), zustand `globalParamsStore` with `hs-*` per-key persistence, IndexedDB queue persistence, pushState URL routing, per-studio API client files, `getGlobalParams()` as sole request assembler.
- **Legacy — do not extend**: `GlobalParamsContext.tsx` (compat wrapper, header says so), `lireek-audio-gen-queue` localStorage persistence (migration shim).
- **Known discrepancies**: docs say "Zustand" but 6/8 stores are hand-rolled; the i18n rule coexists with hard-coded English placeholders (StemStudio.tsx:288,295,310); `ui/README.md` is the untouched Vite template.
- **Unverified**: behaviour of `ui/add_strings.mjs` / `ui/generate_translations.mjs`; full `NavItem` prop surface beyond `icon/label/active/onClick/isExpanded`.
