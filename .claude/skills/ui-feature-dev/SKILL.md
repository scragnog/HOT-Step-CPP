---
name: ui-feature-dev
description: Guides adding or modifying UI features (React "studios") in HOT-Step CPP — studio-folder convention, store patterns, API wiring, routing registration, Tailwind/i18n rules, and the Vite HMR dev loop. Use when creating a new studio/panel/view, adding UI controls or sliders, wiring UI to a server endpoint, or changing anything under ui/src/. NOT for solver/scheduler/guidance knobs — those are declared by Lua plugins in engine/plugins/ (see the lua-plugin-authoring skill), with zero ui/src/ changes.
---

# UI Feature Development (React Studios)

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

The UI tier is a single-page React 19 app in `ui/`, with Vite, Tailwind,
TanStack Query, Zustand, react-i18next, axios and lucide-react icons. Each studio has
a component folder under `ui/src/components/`. Routing uses `history.pushState`
in `App.tsx`. TanStack Query owns server snapshots; services remain the HTTP
transports. Read [the server-state contract](../../../docs/dev/ui-server-state.md)
and the owning `ui/src/queries/<area>/README.md` before adding a reader.

Terms used below:
- **DiT**: the diffusion transformer, the C++ engine's core music-generation model. The UI only ever references it as a model name string.
- **Studio**: one feature area (Lyric Studio, Stem Studio, Cover Studio...), i.e. one folder under `ui/src/components/` plus a branch in `App.tsx`.
- **Backend**: the active generation engine family (`ace`, `minimax-m3`, `yue2`, `sa3`), held in `ui/src/stores/backendStore.ts`; its capability manifest (`useCapabilities()`) decides which controls and studios show.
- **hs-\***: the localStorage key namespace used for persisted UI state.

## When to use this skill

- Adding a new studio/view/panel to the UI.
- Adding a control (slider, toggle, dropdown) to an existing studio or the global parameter bar.
  - **Exception:** a knob for a solver/scheduler/guidance mode is NOT a UI task. Lua plugins declare their own UI params (`docs/dev/plugins-authoring.md`, lua-plugin-authoring skill). Wiring one into `getGlobalParams()` is the wrong implementation entirely. The same goes for a non-ACE backend's knobs: the backend declares them in its capability manifest (`capabilities().extensions`) and `global-bar/BackendExtensionControls.tsx` renders them.
- Wiring the UI to a new or existing server API endpoint.
- Debugging UI state, routing, persistence, or "my new param does nothing" issues.

## Golden rules (hard constraints)

0. **Controls come from the shared primitives; read [docs/dev/ui-design.md](../../../docs/dev/ui-design.md) before adding any.** A dropdown is `StyledSelect`, an on/off setting (selection-list rows included) is `Toggle`, a parameter label is `ParamLabel` with `info`, all from `ui/src/components/shared/`. Never a native `<select>` or `<input type="checkbox">`; `node tools/docs/check-docs.mjs` fails on a new one. WHY: the native widgets look nothing like the app (grey OS box, white text) and a knob with no explanation is a support question.
1. **NEVER use a browser agent for visual verification; ask the human user for screenshots/feedback.** The browser agent is too slow/unreliable in this environment. It is acceptable ONLY for non-visual checks (hitting API endpoints). Workflow: make the change → confirm type-check is clean → ask the user "please check X on the /stem-studio page".
2. **Don't `npm run build` during dev.** Vite HMR means UI edits need no build and no restart. Type-check with `npx tsc -b` from `ui\` (or `npx tsc --noEmit -p tsconfig.app.json`; `tsconfig.app.json` and `tsconfig.node.json` both set `noEmit: true`, so nothing is emitted). A bare `npx tsc --noEmit` in `ui\` checks nothing, because `ui/tsconfig.json` has no files of its own. Only build right before user prod testing. WHY: builds are slow and pointless mid-dev; the dev server already serves source.
3. **Node 20 to 24 LTS, 24 recommended.** `ui/package.json` enforces `"node": ">=20.0.0 <25.0.0"`.
4. **Registering a new studio requires SIX edits**: four in `App.tsx` (the import + `viewFromUrl` + `urlForView` + the `renderContent` branch), one `NavItem` in `Sidebar.tsx`, and one entry in `PALETTE_VIEWS` in `ui/src/components/shared/overlays/CommandPalette.tsx` (see procedure below). Skipping `viewFromUrl()`/`urlForView()` gives a studio that renders but breaks on refresh/back-button; skipping the palette entry leaves it out of Ctrl+K.
5. **Any new generation parameter MUST be added to `getGlobalParams()`** in `ui/src/stores/globalParamsStore.ts` (starts line 741). That function is the SINGLE assembly point for the request body the UI sends. A slider not wired there silently does nothing. Submissions resolved on the server (`generation-intent/1`, used by the durable audio queue and headless callers) are built by `resolveGenerationParams()` in `server/src/services/generation/defaults.ts`, which mirrors `getGlobalParams()` by hand; add the field there too and list it in `server/src/contracts/generationControls.ts` ([frontend-create-controls.md](../../../docs/dev/frontend-create-controls.md) has the table).
6. **Every string through `t()`** (react-i18next), written as `t('key', 'English default')`, with the key in `ui/src/i18n/locales/en.json`; other locales fall back to English. Note: en.json nests everything under a top-level `"translation"` key, and the keys inside are flat dotted strings (`"stem.queue": "Queue"`), not nested objects.
7. **Git discipline (repo-wide):** all work on `master`, never `git add -A`, never `git add -f` on gitignored paths; stage explicit paths only. Push only with explicit user approval, commit locally often. Never push a `v*` tag casually: it triggers a full multi-platform CI release build.
8. **Cross-tier:** if your UI feature needs engine (C++) changes, rebuild via `dev-rebuild.bat` at repo root, NEVER `engine/build.cmd` directly, under any circumstances (Node auto-respawns ace-server; only dev-rebuild.bat performs the clean shutdown). Never `cmake --clean-first` (20+ min CUDA recompile).
9. **Sideband rule for new generation params:** server-side params outside the C++ `AceRequest` struct never survive the `/lm` round trip. Synth requests are rebuilt by spreading the original `aceReq` and taking ONLY LM-generated fields from the echo (`server/src/services/backends/ace/generate.ts:192-213`). Add new sideband params to `aceReq` before the LM phase; NEVER extend the LM-field pick-list into a copy-across whitelist (bug class fixed in 8ea519b/168dcb5).
10. **When verifying generation params, never delete generated test audio**; the user verifies results by ear. Hand over song IDs/URLs and ask the user to listen, just as visual changes get a screenshot request.

## Dev loop (PowerShell, from repo root)

```powershell
pwsh -NoProfile -File tools\detach.ps1 dev.bat   # dev.bat, detached; the app then lives at http://localhost:3000
```

This starts BOTH tiers (verified `dev.bat`): the Node server via `server\restart-loop.cmd` on **:3001** (tsx watch, auto-restart) and `npx vite --port 3000 --host` in `ui\`. Develop against **http://localhost:3000**: Vite proxies `/api`, `/audio`, and `/references` to :3001 (`ui/vite.config.ts:46-69`). `LAUNCH.bat` is the end-user prod launcher (Node :3001 serving prebuilt `ui/dist/`, `server/src/index.ts:182-206`); agents never run it.

Checks while developing:

```powershell
cd D:\Ace-Step-Latest\hot-step-cpp\ui
npx tsc -b          # type-check UI (project references; catches what a plain tsc --noEmit on one file misses)
node --import ../server/node_modules/tsx/dist/loader.mjs --test --test-force-exit --test-timeout=60000 "src/**/*.test.ts"
npx eslint .        # lint; also bans queryClient.fetchQuery (use queryClient.query())
```

If you touched `server/src/`: `cd D:\Ace-Step-Latest\hot-step-cpp\server; npx tsc --noEmit`.

Only before handing to the user for prod testing: `npm run build` in `ui\` (runs `tsc -b && vite build`).

## Procedure: adding a new studio

Say the studio is "My Studio", slug `my-studio`.

1. **Create the folder** `ui/src/components/my-studio/` with an orchestrator `MyStudio.tsx` (PascalCase files, kebab-case folder). Export both named and default (`export const MyStudio: React.FC = ...` + `export default MyStudio`; see `StemStudio.tsx:44,536`). Layout convention (documented in-source at `StemStudio.tsx:12-13`): flex columns with border dividers, resizable right sidebar.
2. **Register the route** in `ui/src/App.tsx`: add the `import { MyStudio } ...` at the top (cf. App.tsx:54), then three places:
   - `viewFromUrl()` (App.tsx:103-119): `if (path.startsWith('/my-studio')) return 'my-studio';`
   - `urlForView()` (App.tsx:121-150): `if (view === 'my-studio') return '/my-studio';`
   - `renderContent()` (App.tsx:910 onward): add a branch, wrapped in `DiscoPulseWrapper` (a beat-reactive visual wrapper) with a hue from the `DISCO` constants (App.tsx:351-363, e.g. `DISCO.assistant` = 175 cyan):
     ```tsx
     if (activeView === 'my-studio') {
       return (
         <DiscoPulseWrapper hue={DISCO.assistant} className="flex-1 overflow-hidden">
           <MyStudio />
         </DiscoPulseWrapper>
       );
     }
     ```
   Navigation flows through `navigateTo()` (App.tsx:486-499) → `setActiveView` + `history.pushState`; back/forward is a `popstate` listener (App.tsx:503-507). Only Lyric Studio and Training Studio persist a deep URL (`hs-lastLyricStudioUrl`, `hs-lastTrainingStudioUrl`); copy that only if your studio has sub-routes. The studio area is wrapped in an `ErrorBoundary` keyed on the active view, so a render throw in your studio does not blank the app.
3. **Add a nav item** in `ui/src/components/sidebar/Sidebar.tsx`: copy an existing `NavItem` block inside `SidebarNav` (e.g. Stem Studio at Sidebar.tsx:144-150): `icon` (lucide-react), `label={t('sidebar.myStudio')}`, `active={activeView === 'my-studio'}`, `onClick={() => onNav('my-studio')}`, `isExpanded={isExpanded}`. If the studio only works on some backends, add a predicate to `STUDIO_NEEDS` (Sidebar.tsx:20-27) and render the item as `{available('my-studio') && <NavItem ... />}`; the sidebar then moves the user to Custom-Gen when they switch to an engine that lacks it.
4. **Add it to the command palette**: `{ view: 'my-studio', labelKey: 'sidebar.myStudio' }` in `PALETTE_VIEWS` (`ui/src/components/shared/overlays/CommandPalette.tsx:15-30`).
5. **If it needs server endpoints**: create `server/src/routes/myStudio.ts`, mount it in `server/src/index.ts` (`app.use('/api/my-studio', ...)`; see the mount block at index.ts:104-151), put shared response types in `server/src/contracts/myStudio.ts`, and create a per-studio client `ui/src/services/myStudioApi.ts` over `apiClient` (see API layer below). Keep all endpoints under `/api/...` or the Vite proxy won't forward them in dev.
6. **Add i18n keys** to `ui/src/i18n/locales/en.json` (under the `"translation"` root) for every user-visible string.
7. **Document it**: a user page `docs/user/studios/my-studio.md`, a row in `STUDIO_PAGES` in `tools/docs/check-docs.mjs`, and a line in `FEATURES.md`; then `node tools/docs/check-docs.mjs`.
8. Type-check, then **ask the user to visually verify**, never a browser agent.

## Server state and client state

Query owns fetched lists, details, status and progress. One HMR-stable
`queryClient` in `queries/client.ts` is provided above AuthProvider and App in
`main.tsx`. Use `useQuery` for view snapshots, or `queryClient.query()`/a Query observer
for imperative execution (`fetchQuery` is deprecated here and the lint rule rejects it).
Never add a mount-effect fetch, module cache or
network interval beside that owner. Keep form buffers separate: seed once for
the captured selection and preserve unsaved edits across background refetches.

Build keys with `serverKey`, `resourceKey` and `sharedKeys` from `queries/keys.ts`.
Capture backend/worker with the request. Worker `null` means this PC; omit worker
scope on local run/review/mirror resources. Authenticated requests use an opaque
session epoch, never a credential in the key. Read `sessionEpoch` from `useAuth`
so a changed session rebuilds the key. Pass Query's `signal` through every read
helper. Throw HTTP errors; do not cache failures as empty results.

Cross-area invalidation uses foundation's canonical keys, never another area's
hooks. Preserve each key's wire shape and use observer `select` for projections.
Defaults are 30 s freshness, 5 min retention, no focus refetch, reconnect refetch
and at most two retries for network/5xx failures. Mutations never auto-retry.
Use `pollWhile` for active jobs with terminal stop and capped error backoff;
document any resource override. Abort cancels the read, not the server job.

Capture mutation targets before awaiting. After success, update or invalidate
all related list/detail keys. For deletion, cancel reads before the command and
again after success, then remove cached rows and invalidate. Keep streams and
their log/text/audio reducers; map events and reconnects into scoped cache
updates or invalidation with `bridgeEvents`, coalesce bursts and tear down
subscriptions. Replay must never submit a command or deliver completion twice.

The following store patterns are for client state: forms, selections, playback,
generation parameters and execution. Match the nearest existing store. Scoped
compatibility projections may expose Query data but cannot own a second cache
or independent fetch loop. The integration audit and open migration requests
live in [architecture.md](../../../docs/dev/architecture.md#ui-server-state-integration-audit).

**Pattern A: hand-rolled external store (majority; canonical example `ui/src/stores/abCompareStore.ts`, 101 lines).** Module-level `let _state`, a `Set` of listeners, `setState()` that spreads + notifies, exported *plain action functions* (importable from non-React code), plus a `useXxx()` hook (`useSyncExternalStore(subscribe, getSnapshot)`) and a ref-cached `useXxxSelector(selector)` hook. Selector hooks are the preferred subscription style: `useAudioGenQueueSelector(s => s.items.filter(...).length)` (StemStudio.tsx:101-103). Used by: `playbackStore`, `audioGenQueueStore`, `discoStore`, `streamingStore`, `mm3StreamStore`, `postProcessStore`, `compareStore`, `abCompareStore`, `components/lyric-studio/playlistStore` (studio-local stores may live inside the studio folder).

**Pattern B: zustand + per-field localStorage (`ui/src/stores/globalParamsStore.ts`, ~1040 lines).** `create<any>()((set, get) => ({...}))` (line 313); the `any` is deliberate. Every field initialises via `readKey('hs-<name>', GLOBAL_PARAM_DEFAULTS.<name>)` (defaults table at line 44, so reset-to-default controls can read it) and every setter does `set({...}); writeKey('hs-<name>', v)`: per-key `hs-*` persistence, NOT zustand/persist middleware. Fields listed in `BACKEND_SCOPED_FIELDS` (line 219) are remembered per backend under `hs-<name>@<backendId>` (ACE keeps the bare key); `scopedKey()` maps them. `getGlobalParams()` (line 741) assembles the knobs into a `Partial<GenerationParams>` request body with conditional gating; **new generation params go here or nowhere**. `ui/src/context/GlobalParamsContext.tsx` is a legacy compat shim; its own header (lines 1-8) says new code must import the store directly. Don't extend the context. Other zustand stores (`backendStore`, `trainingStore`, `vstChainStore`, `coverStemsStore`) follow the plain `create<State>()` form.

**Pattern C: plain `useState` + localStorage inside the studio.** StemStudio persists studio-local state via manual `localStorage` with `hs-stem-*` keys (StemStudio.tsx:64-67, 105-122) or via `usePersistedState(key, default)` (`ui/src/hooks/usePersistedState.ts:9`). **Cross-component/same-tab sync requires `writePersistedState()`** (usePersistedState.ts:43-49), which manually dispatches a `StorageEvent`; native StorageEvents only fire cross-tab.

The generation queue persists client execution state in IndexedDB. Its executor
retains serialization, manual rows, legacy owner migration and completion
effects; `queries/shell-queue/queue.ts` owns HTTP snapshots. Keep the existing
projection loop as the scheduler. Do not add a parallel poller or resubmit on
remount. Server-owned rows reattach to server work; browser-owned recovery
follows the existing migration and resume rules.

## API layer: how the UI reaches the server

Two tiers, both in `ui/src/services/`:

- `api.ts` exports domain clients such as `songApi`, `generateApi` and
  `settingsApi` over the shared `httpClient.ts` (`apiClient`), which runs JSON
  traffic through the axios instance in `axiosClient.ts`. Reads take trailing
  `ReadOptions` with `signal`; `ApiError` preserves HTTP status and message.
  The axios response interceptor is the one global catch: it normalizes
  failures to `ApiError` and shows the error toast (pass `{ silent: true }` to
  opt a request out). Multipart uploads use XHR for progress; SSE stays on
  `fetch` (`services/sharedEventSource.ts`).
- Per-studio clients retain endpoint types and command functions. Their reads
  feed area query options. Existing extraction/separation waiters use Query
  observers; follow that pattern instead of adding a submit-then-poll timer.
- `trainingApi.ts` reads accept captured `{ worker, signal }` for worker routes
  and `{ signal }` for local-only routes. Pass explicit `worker: null` for a
  local target; omitted worker can mean the live selection in legacy callers.

**Auth**: single-user local app. `AuthContext` auto-logs-in on mount via `authApi.autoLogin()` (`GET /api/auth/auto`), hands the token to the transport (`setAuthToken`, `axiosClient.ts`), and logs in again when a request answers 401. The axios request interceptor adds the bearer to every call, so **no API function takes a token**. `useAuth()` still exposes `token` (the queue store's runner takes it) and `sessionEpoch`. Server routes that call `getUserId` need the bearer; stem-studio, vst, settings and song detail do not check it. Any Query key whose read needs the session sets `{ auth: true }` so it carries `sessionEpoch` (see above); the song-detail key carries one too, but only so sign-out drops it.

**snake_case gotcha**: the server returns SQLite snake_case rows. `normalizeSong()` (api.ts:51-91) maps to camelCase while keeping BOTH spellings (`audioUrl` and `audio_url`) and parses `generation_params` if it arrives as a JSON string. Any new endpoint returning songs must route through it or the player/library break.

**Shared wire types**: for newer routes the response types live in `server/src/contracts/*.ts`, and the UI imports them by relative path (`import type { HealthResponse } from '../../../../server/src/contracts/health'`, `components/shared/status/noModelsPoll.ts:5`). Reuse them instead of re-declaring a shape.

**Generation job protocol** (verified `generateApi` at `api.ts:181-222`, `GenerationJob` at `types.ts:380`, Query reads in `queries/shell-queue/queue.ts`):

```
POST /api/generate            → { jobId, status }
GET  /api/generate/status/:id → GenerationJob { status: 'pending|lm_running|synth_running|saving|succeeded|failed|cancelled',
                                                progress?, stage?, error?,
                                                result?: { audioUrls[], songIds[], masteredAudioUrl?, duration?, ... } }
```

The server can also report `running` (YuE2 jobs and post-processing re-runs). Active status reads normally use a 1.5 s cadence through Query. Progress may be
0-1 or 0-100; preserve the queue's normalization. SSE also carries logs, model
downloads, training metrics, MIDI notes and workflows. Shared connections and
stream reducers stay; reconnect and terminal events reconcile cached snapshots.

## Styling and i18n conventions

- Tailwind **v3** (not v4). `ui/tailwind.config.js`: `darkMode: 'class'`, custom `suno` dark palette (`#09090b` bg / `#121214` panel / `#18181b` card / `#27272a` hover-border) and `brand` pink `#ec4899`; font Inter; a `portrait-wide` variant for portrait screens at desktop width. `ui/src/index.css` adds CSS-variable design tokens on top of the `@tailwind` directives.
- **Dual-theme classes are mandatory**: always write light+dark pairs: `border-zinc-200 dark:border-white/5`, `text-zinc-700 dark:text-zinc-300` (see StemStudio.tsx:306-343).
- Accents: purple (`focus:border-purple-500/40`, `#a78bfa` labels) inside studios; brand pink for global CTAs. Pass the matching `accent` to `StyledSelect` and `Toggle` (table in docs/dev/ui-design.md).
- Mixing Tailwind with a bottom-of-file `const styles: Record<string, React.CSSProperties>` object for one-off widgets is accepted convention (StemStudio.tsx:472-534); not tech debt to "fix".
- Resizable sidebars: hand-rolled mousemove handler, width via `usePersistedState('hs-activitySidebarWidth', 320)`. This key is **shared across studios** so sidebar width is consistent app-wide (StemStudio.tsx:100, 262-281).
- Narrow screens: studios switch to one pane at a time below 1024 px and on `portrait-wide` (StemStudio.tsx:287-303 is the pattern); phone song actions use `BottomSheet` (docs/dev/ui-design.md).
- Icons: lucide-react everywhere.
- i18n: `const { t } = useTranslation()`, namespaced keys per studio (`t('stem.recentExtractions')`), keys in `locales/en.json` under the `"translation"` root; `ja/ko/ru/zh` fall back to en. Two helper scripts sit in `ui/`: `add_strings.mjs <file.json>` merges a JSON object of flat keys into en.json's `translation`, and `generate_translations.mjs` machine-translates en.json into the ru/zh/ja/ko files through Google Translate's public endpoint (network calls; it rewrites the locale files). Some existing strings are hard-coded English (StemStudio.tsx:300, 330, 341); use `t()` anyway for new strings.

## Key files

| Path | Role |
|---|---|
| `ui/src/App.tsx` (~1690 ln) | Root: hand-rolled routing (`viewFromUrl`/`urlForView`/`navigateTo`), `renderContent()`, `DISCO` hues, player wiring |
| `ui/src/main.tsx` | Entry: StrictMode > ErrorBoundary > QueryClientProvider > AuthProvider > (ToastContainer, Suspense > App) |
| `ui/src/components/<studio>/` | One folder per studio; orchestrator component named after the studio |
| `ui/src/components/shared/` | `StyledSelect`, `Toggle`, `ParamLabel`, `Slider`, `EditableSlider`, `Section` (in `ActivitySidebar.tsx`), `Toast`, `ConfirmDialog`, `StemMixer`, `DiscoPulseWrapper`, `CommandPalette`, `BottomSheet`, `ErrorBoundary`, `BackendCapabilityGate`, ... |
| `ui/src/components/sidebar/Sidebar.tsx` | Left nav: one `NavItem` per studio, `STUDIO_NEEDS` engine gating |
| `ui/src/stores/globalParamsStore.ts` | zustand; all generation knobs; `getGlobalParams()` = sole request assembler (line 741) |
| `ui/src/stores/backendStore.ts` | Active backend, capability and model projections |
| `ui/src/stores/abCompareStore.ts` | Smallest canonical hand-rolled store; copy this as a template |
| `ui/src/stores/audioGenQueueStore.ts` | Generation executor, persistence, projection scheduling and completion guards |
| `ui/src/queries/` | Canonical keys/client/policy, area options, observers and event mappings |
| `ui/src/services/httpClient.ts`, `axiosClient.ts` | Shared transport: bearer injection, `ApiError`, global error toast |
| `ui/src/services/api.ts` | Domain clients + `normalizeSong()` (line 51) |
| `ui/src/services/stemStudioApi.ts` | Per-studio transport and Query-observed progress waiter |
| `ui/src/hooks/usePersistedState.ts` | Persisted useState + `writePersistedState()` same-tab sync |
| `ui/src/hooks/useCapabilities.ts` | The active backend's capability manifest |
| `ui/src/context/AuthContext.tsx` | Auto-login, `useAuth()` exposes user, token and sessionEpoch |
| `ui/src/context/GlobalParamsContext.tsx` | LEGACY compat shim; do not extend |
| `ui/src/i18n/locales/en.json` | Source-of-truth strings (root key `"translation"`) |
| `ui/vite.config.ts` | Dev proxy: `/api`, `/audio`, `/references` → :3001 |
| `ui/tailwind.config.js` | `suno` palette, `brand` pink, `darkMode: 'class'` |
| `server/src/index.ts` | Route mounts (lines 104-151), serves `ui/dist` in prod |
| `server/src/routes/stemStudio.ts` | Model server route paired with stemStudioApi.ts |

## Failure signatures

| Symptom | Cause → fix |
|---|---|
| `npm install`/dev fails with dep/engine errors | Check Node 20 to 24 range (`ui/package.json` engines field) and lockfile consistency. |
| New studio renders but URL resets to `/` on refresh, or back-button breaks | Added `renderContent()` branch but missed `viewFromUrl()` and/or `urlForView()`; all four App.tsx edits + Sidebar + palette are required. |
| New studio missing from the sidebar on one engine | Its `STUDIO_NEEDS` predicate is false for the active backend's capabilities; check `useCapabilities()`. |
| API 404 in dev but works in prod (or vice versa) | Route prefix not proxied: only `/api`, `/audio`, `/references` are in `vite.config.ts`. Keep endpoints under `/api/...`. |
| Songs missing audio/cover in one studio only | Endpoint bypassed `normalizeSong()`; raw snake_case leaked (api.ts:51-91). |
| Effects/API calls fire twice in dev | React StrictMode double-invoke (main.tsx:18). Dev-only; make mount effects idempotent, don't "fix" with flags. |
| A failed request shows an unexpected toast, or none | The axios response interceptor decides (`axiosClient.ts`); pass `{ silent: true }` for a call that handles its own errors. |
| New slider has no effect on generations | Value not wired into `getGlobalParams()` (globalParamsStore.ts:741), the only place the UI request body is assembled. If it reaches `getGlobalParams()` but never reaches `/synth`: the server chain is `routes/generate.ts` → backend `generate()` → `translateParams()` (`server/src/services/generation/translateParams.ts:47`) → `aceReq`; sideband params must join `aceReq` BEFORE the LM phase (Golden rule 9, never whitelist across the LM echo). If it works from Create but not from the audio queue or an MCP call, `resolveGenerationParams()` in `server/src/services/generation/defaults.ts` lacks it (Golden rule 5). |
| Setting change in one studio doesn't reflect in another | Wrote `localStorage` directly instead of `writePersistedState()`; same-tab listeners need the manually-dispatched StorageEvent. |
| A sampler pick changes when switching backend | The field is in `BACKEND_SCOPED_FIELDS`, so each backend keeps its own value (`hs-<name>@<backendId>`). Intended. |
| Progress bar jumps 0→100 or shows 5000% | Server progress is 0-1 on some paths, 0-100 on others; normalise like audioGenQueueStore.ts:1196. |
| Queue items stuck after reload | Check server/browser ownership, migration fences and reattachment before changing execution. Never blindly resubmit an existing server job. |
| UI shows a raw key like `stem.queue` | Key missing from `i18n/locales/en.json`. |
| `npx tsc --noEmit` passes but `npm run build` fails | A bare `tsc --noEmit` in `ui\` checks nothing; `build` runs `tsc -b` across ALL project references. Reproduce with `npx tsc -b` from `ui\`. |
| Generation fails and the UI just shows "failed" | Debug server-side: newest `<data root>/logs/YYYY-MM-DD_HH-MM-SS/` folder → `generations/gen_*.log`, then `ace_engine.log` / `node_console.log`. |

## Institutional knowledge

- **VALIDATED (lead engineer, verbatim rule): Never use a browser agent for visual verification; ask the human user for screenshots/feedback. Browser agent is acceptable only for non-visual API checks.**
- VALIDATED: `getGlobalParams()` is the sole assembly point for UI generation request bodies; params bypass it → silently dropped. Confirmed by code and by commit 8ea519b ("adapter param plumbing bugs").
- Client stores use Zustand or hand-rolled `useSyncExternalStore` subscriptions. Query owns server reads; match existing client-state patterns without copying fetched snapshots into an unscoped store.
- VALIDATED: queue persistence moved localStorage → IndexedDB because 600+-item queues exceeded the 5 MB localStorage cap (comment at audioGenQueueStore.ts:243-246).
- VALIDATED: `GlobalParamsContext.tsx` is a migration shim by its own header; extend the store, not the context.
- NOTE: `ui/README.md` is the untouched Vite template; ignore it as documentation. `ui/PWA.md` documents the installable app and phone sheets.

## Deeper reading

- [reference.md](reference.md) (same folder): full Stem Studio end-to-end dissection, store code templates, DISCO hue table, localStorage key namespaces.
- `docs/dev/ui-design.md` (control rules), `docs/dev/ui-server-state.md` and `docs/dev/tanstack-query.md` (Query contract), `docs/dev/frontend-studios.md` and the other `docs/dev/frontend-*.md` pages.
- `AGENTS.md` (repo root; `CLAUDE.md` only imports it): cross-tier build/git rules.
- `FEATURES.md`: feature catalogue.
- `docs/dev/plugins-authoring.md`: if your UI feature exposes a new solver/scheduler/guidance knob, that's a Lua plugin, not C++ (plugins declare their own UI params).
- `engine/docs/ARCHITECTURE.md`: engine request JSON, generation modes.
- `docs/plans/`: internal design docs. **Gitignored, local-only; may be absent on a fresh clone.**
