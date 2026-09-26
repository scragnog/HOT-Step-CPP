---
name: ui-feature-dev
description: Guides adding or modifying UI features (React "studios") in HOT-Step CPP — studio-folder convention, store patterns, API wiring, routing registration, Tailwind/i18n rules, and the Vite HMR dev loop. Use when creating a new studio/panel/view, adding UI controls or sliders, wiring UI to a server endpoint, or changing anything under ui/src/. NOT for solver/scheduler/guidance knobs — those are declared by Lua plugins in engine/plugins/ (see the lua-plugin-authoring skill), with zero ui/src/ changes.
---

# UI Feature Development (React Studios)

The UI tier of HOT-Step CPP is a single-page React 19 app in `ui/` (Vite 8, Tailwind 3.4, zustand 5, react-i18next, lucide-react icons). Every user-facing feature is a **"studio"** — a folder of components under `ui/src/components/` mounted as a top-level view. There is **no react-router**: routing is hand-rolled `history.pushState` in `App.tsx`. All server calls are plain `fetch()` against relative `/api/...` URLs (no axios, no react-query).

Terms used below:
- **DiT** — the diffusion transformer, the C++ engine's core music-generation model. The UI only ever references it as a model name string.
- **Studio** — one feature area (Lyric Studio, Stem Studio, Cover Studio...), i.e. one folder under `ui/src/components/` plus a branch in `App.tsx`.
- **hs-\*** — the localStorage key namespace used for persisted UI state.

## When to use this skill

- Adding a new studio/view/panel to the UI.
- Adding a control (slider, toggle, dropdown) to an existing studio or the global parameter bar.
  - **Exception:** a knob for a solver/scheduler/guidance mode is NOT a UI task — Lua plugins declare their own UI params (`docs/dev/plugins-authoring.md`, lua-plugin-authoring skill). Wiring one into `getGlobalParams()` is the wrong implementation entirely.
- Wiring the UI to a new or existing server API endpoint.
- Debugging UI state, routing, persistence, or "my new param does nothing" issues.

## Golden rules (hard constraints)

0. **Controls come from the shared primitives — read [docs/dev/ui-design.md](../../../docs/dev/ui-design.md) before adding any.** A dropdown is `StyledSelect`, an on/off setting is `Toggle`, a parameter label is `ParamLabel` with `info`. Never a native `<select>` or `<input type="checkbox">`; `node tools/docs/check-docs.mjs` fails on a new one. WHY: the native widgets look nothing like the app (grey OS box, white text) and a knob with no explanation is a support question.
1. **NEVER use a browser agent for visual verification — ask the human user for screenshots/feedback.** The browser agent is too slow/unreliable in this environment. It is acceptable ONLY for non-visual checks (hitting API endpoints). Workflow: make the change → confirm type-check is clean → ask the user "please check X on the /stem-studio page".
2. **Don't `npm run build` during dev.** Vite HMR means UI edits need no build and no restart. Type-check with `npx tsc -b` from `ui\` (all tsconfigs have `noEmit: true` — nothing is emitted). Only build right before user prod testing. WHY: builds are slow and pointless mid-dev; the dev server already serves source.
3. **Node 18–22 only.** `ui/package.json` enforces `"node": ">=18.0.0 <24.0.0"`. Node 24+ breaks dependencies.
4. **Registering a new studio requires FIVE edits** (four in `App.tsx`: the import + `viewFromUrl` + `urlForView` + the `renderContent` branch, plus one `NavItem` in `Sidebar.tsx` — see procedure below). Skipping `viewFromUrl()`/`urlForView()` gives a studio that renders but breaks on refresh/back-button.
5. **Any new generation parameter MUST be added to `getGlobalParams()`** in `ui/src/stores/globalParamsStore.ts` (starts line 324) — that function is the SINGLE assembly point for the request body. A slider not wired there silently does nothing.
6. **Every string through `t()`** (react-i18next) with a key in `ui/src/i18n/locales/en.json`; other locales fall back to English. Note: en.json nests everything under a top-level `"translation"` key.
7. **Git discipline (repo-wide):** all work on `master`, never `git add -A`, never `git add -f` on gitignored paths — stage explicit paths only. Push only with explicit user approval, commit locally often. Never push a `v*` tag casually — it triggers a full multi-platform CI release build.
8. **Cross-tier:** if your UI feature needs engine (C++) changes, rebuild via `dev-rebuild.bat` at repo root, NEVER `engine/build.cmd` directly, under any circumstances (Node auto-respawns ace-server; only dev-rebuild.bat performs the clean shutdown). Never `cmake --clean-first` (20+ min CUDA recompile).
9. **Sideband rule for new generation params:** server-side params outside the C++ `AceRequest` struct never survive the `/lm` round trip. Synth requests are rebuilt by spreading the original `aceReq` and taking ONLY LM-generated fields from the echo (`server/src/routes/generate.ts:312-328`). Add new sideband params to `aceReq` before the LM phase; NEVER extend the LM-field pick-list into a copy-across whitelist (bug class fixed in 8ea519b/168dcb5).
10. **When verifying generation params, never delete generated test audio** — the user verifies results by ear. Hand over song IDs/URLs and ask the user to listen, just as visual changes get a screenshot request.

## Dev loop (PowerShell, from repo root)

```powershell
.\dev.bat
```

This starts BOTH tiers (verified `dev.bat`): the Node server via `server\restart-loop.cmd` on **:3001** (tsx watch, auto-restart) and `npx vite --port 3000 --host` in `ui\`. Develop against **http://localhost:3000** — Vite proxies `/api`, `/audio`, and `/references` to :3001 (`ui/vite.config.ts:10-24`). `LAUNCH.bat` = prod mode, serving prebuilt `ui/dist/` from :3001 (`server/src/index.ts:126`).

Checks while developing:

```powershell
cd D:\Ace-Step-Latest\hot-step-cpp\ui
npx tsc -b          # type-check UI (project references — catches what a plain tsc --noEmit on one file misses)
npx eslint .        # lint
```

If you touched `server/src/`: `cd D:\Ace-Step-Latest\hot-step-cpp\server; npx tsc --noEmit`.

Only before handing to the user for prod testing: `npm run build` in `ui\` (runs `tsc -b && vite build`).

## Procedure: adding a new studio

Say the studio is "My Studio", slug `my-studio`.

1. **Create the folder** `ui/src/components/my-studio/` with an orchestrator `MyStudio.tsx` (PascalCase files, kebab-case folder). Export both named and default (`export const MyStudio: React.FC = ...` + `export default MyStudio` — see `StemStudio.tsx:40,503`). Layout convention (documented in-source at `StemStudio.tsx:9-10`): flex columns with border dividers, resizable right sidebar.
2. **Register the route** in `ui/src/App.tsx` — add the `import { MyStudio } ...` at the top (cf. App.tsx:38), then three places:
   - `viewFromUrl()` (App.tsx:84-95): `if (path.startsWith('/my-studio')) return 'my-studio';`
   - `urlForView()` (App.tsx:98-116): `if (view === 'my-studio') return '/my-studio';`
   - `renderContent()` (App.tsx:652 onward): add a branch, wrapped in `DiscoPulseWrapper` (a beat-reactive visual wrapper) with a hue from the `DISCO` constants (App.tsx:279-289, e.g. `DISCO.assistant` = 175 cyan):
     ```tsx
     if (activeView === 'my-studio') {
       return (
         <DiscoPulseWrapper hue={DISCO.assistant} className="flex-1 overflow-hidden">
           <MyStudio />
         </DiscoPulseWrapper>
       );
     }
     ```
   Navigation flows through `navigateTo()` (App.tsx:391-401) → `setActiveView` + `history.pushState`; back/forward is a `popstate` listener (App.tsx:404-408). Only Lyric Studio persists a deep URL (`hs-lastLyricStudioUrl`) — copy that only if your studio has sub-routes.
3. **Add a nav item** in `ui/src/components/sidebar/Sidebar.tsx` — copy an existing `NavItem` block (e.g. Stem Studio at Sidebar.tsx:111-117): `icon` (lucide-react), `label={t('sidebar.myStudio')}`, `active={activeView === 'my-studio'}`, `onClick={() => onViewChange('my-studio')}`, `isExpanded={isOpen}`.
4. **If it needs server endpoints**: create `server/src/routes/myStudio.ts`, mount it in `server/src/index.ts` (`app.use('/api/my-studio', ...)` — see the mount block at index.ts:72-95), and create a per-studio client `ui/src/services/myStudioApi.ts` (see API layer below). Keep all endpoints under `/api/...` or the Vite proxy won't forward them in dev.
5. **Add i18n keys** to `ui/src/i18n/locales/en.json` (under the `"translation"` root) for every user-visible string.
6. Type-check, then **ask the user to visually verify** — never a browser agent.

## State management — three coexisting patterns

Top-level docs say "Zustand" but only 2 of 8 files in `ui/src/stores/` use zustand's `create()` (`globalParamsStore.ts:9`, `vstChainStore.ts:6` — verified by grep). The majority pattern is a hand-rolled `useSyncExternalStore` module singleton. **Match the pattern of the nearest existing store; don't introduce a fourth.**

**Pattern A — hand-rolled external store (majority; canonical example `ui/src/stores/abCompareStore.ts`, 101 lines).** Module-level `let _state`, a `Set` of listeners, `setState()` that spreads + notifies, exported *plain action functions* (importable from non-React code), plus a `useXxx()` hook (`useSyncExternalStore(subscribe, getSnapshot)`) and a ref-cached `useXxxSelector(selector)` hook. Selector hooks are the preferred subscription style: `useAudioGenQueueSelector(s => s.items.filter(...).length)` (StemStudio.tsx:77-79). Used by: `playbackStore`, `audioGenQueueStore`, `discoStore`, `streamingStore`, `abCompareStore`, `components/lyric-studio/playlistStore` (studio-local stores may live inside the studio folder).

**Pattern B — zustand + per-field localStorage (`ui/src/stores/globalParamsStore.ts`, 460 lines).** `create<any>()((set, get) => ({...}))` — the `any` is deliberate (line 39-40). Every field initialises via `readKey('hs-<name>', default)` and every setter does `set({...}); writeKey('hs-<name>', v)` — per-key `hs-*` persistence, NOT zustand/persist middleware. `getGlobalParams()` (line 324-459, end of file) assembles ~100 knobs into a `Partial<GenerationParams>` request body with conditional gating — **new generation params go here or nowhere**. `ui/src/context/GlobalParamsContext.tsx` is a legacy compat shim; its own header (lines 1-8) says new code must import the store directly. Don't extend the context.

**Pattern C — plain `useState` + localStorage inside the studio.** StemStudio persists studio-local state via manual `localStorage` with `hs-stem-*` keys (StemStudio.tsx:43-46,97-104) or via `usePersistedState(key, default)` (`ui/src/hooks/usePersistedState.ts:9`). **Cross-component/same-tab sync requires `writePersistedState()`** (usePersistedState.ts:43-49), which manually dispatches a `StorageEvent` — native StorageEvents only fire cross-tab.

The generation queue store `audioGenQueueStore.ts` (992 lines) persists to **IndexedDB** (`lireek-queue-store` DB) with debounced 2 s writes because large queues blew localStorage's 5 MB cap (lines 65-73,126-133); on reload it resets in-flight items to `pending` + "Reconnecting…" (`_sanitizeItems`, lines 142-156).

## API layer — how the UI reaches the server

Two tiers, both in `ui/src/services/`:

- **Central client `api.ts`** (406 lines): private `get/post/patch/del<T>` fetch helpers, `BASE = '/api'`, optional `Bearer` token, uniform errors (non-OK → `throw new Error(err.error || 'API error: <status>')`, api.ts:10-64). Exports per-domain objects (`songApi`, `generateApi`, `modelApi`, `adapterApi`, `settingsApi`, ...).
- **Per-studio client files** (model: `stemStudioApi.ts`, 220 lines): standalone exported async functions, own `API_BASE = '/api/stem-studio'`, own types/constants, and a submit-then-poll helper `waitForExtraction(jobId, onProgress, pollMs=1000)` (stemStudioApi.ts:190-211). New studios with their own routes follow this shape.

**Auth**: single-user local app. `AuthContext` auto-logs-in on mount via `authApi.autoLogin()` (`GET /api/auth/auto`); components do `const { token } = useAuth()` and pass `token` to api methods. Newer endpoints (stem-studio, vst, settings) skip the token entirely — both styles coexist.

**snake_case gotcha**: the server returns SQLite snake_case rows. `normalizeSong()` (api.ts:76-114) maps to camelCase while keeping BOTH spellings (`audioUrl` and `audio_url`) and parses `generation_params` if it arrives as a JSON string. Any new endpoint returning songs must route through it or the player/library break.

**Generation job protocol** (verified `api.ts:148-152`, `types.ts:266-281`, polling loop `audioGenQueueStore.ts:534-600`):

```
POST /api/generate            → { jobId, status }
GET  /api/generate/status/:id → GenerationJob { status: 'pending|lm_running|synth_running|saving|succeeded|failed|cancelled',
                                                progress?, stage?, error?,
                                                result?: { audioUrls[], songIds[], masteredAudioUrl?, duration?, ... } }
```

Poll every 1.5 s. Progress may arrive as 0–1 OR 0–100 — normalise both, as `audioGenQueueStore.ts:548-550` does. No websockets. Job status is polled (`audioGenQueueStore`); SSE exists in three places: logs (`hooks/useEventSource.ts` → `TerminalPanel`), model-download progress (`components/model-manager/useDownloadStream.ts`, its own `EventSource`), and streaming audio previews during generation (`hooks/useStreamGeneration.ts` → `GET /api/generate/stream/:jobId`).

## Styling and i18n conventions

- Tailwind **v3** (not v4). `ui/tailwind.config.js`: `darkMode: 'class'`, custom `suno` dark palette (`#09090b` bg / `#121214` panel / `#18181b` card / `#27272a` hover-border) and `brand` pink `#ec4899`; font Inter. `ui/src/index.css` adds CSS-variable design tokens on top of the `@tailwind` directives.
- **Dual-theme classes are mandatory** — always write light+dark pairs: `border-zinc-200 dark:border-white/5`, `text-zinc-700 dark:text-zinc-300` (see StemStudio.tsx:271,289).
- Accents: purple (`focus:border-purple-500/40`, `#a78bfa` labels) inside studios; brand pink for global CTAs.
- Mixing Tailwind with a bottom-of-file `const styles: Record<string, React.CSSProperties>` object for one-off widgets is accepted convention (StemStudio.tsx:419-501) — not tech debt to "fix".
- Resizable sidebars: hand-rolled mousemove handler, width via `usePersistedState('hs-activitySidebarWidth', 320)` — this key is **shared across studios** so sidebar width is consistent app-wide (StemStudio.tsx:76,244-262).
- Icons: lucide-react everywhere.
- i18n: `const { t } = useTranslation()`, namespaced keys per studio (`t('stem.recentExtractions')`), keys in `locales/en.json` under the `"translation"` root; `ja/ko/ru/zh` fall back to en. Helper scripts `ui/add_strings.mjs` and `ui/generate_translations.mjs` exist (purpose inferred from names — unverified, read before running). Some existing placeholders are hard-coded English (StemStudio.tsx:288,295); use `t()` anyway for new strings.

## Key files

| Path | Role |
|---|---|
| `ui/src/App.tsx` (1437 ln) | Root: hand-rolled routing (`viewFromUrl`/`urlForView`/`navigateTo`), `renderContent()`, `DISCO` hues, player wiring |
| `ui/src/main.tsx` | Entry: StrictMode > AuthProvider > Suspense > App |
| `ui/src/components/<studio>/` | One folder per studio; orchestrator component named after the studio |
| `ui/src/components/shared/` | `Section` (in `ActivitySidebar.tsx` — props `title, icon, count?, countColor?, defaultOpen?`), `Slider`, `EditableSlider`, `Toast`, `ConfirmDialog`, `StemMixer`, `DiscoPulseWrapper`, ... |
| `ui/src/components/sidebar/Sidebar.tsx` | Left nav — one `NavItem` per studio |
| `ui/src/stores/globalParamsStore.ts` | zustand; all generation knobs; `getGlobalParams()` = sole request assembler (line 324) |
| `ui/src/stores/abCompareStore.ts` | Smallest canonical hand-rolled store — copy this as a template |
| `ui/src/stores/audioGenQueueStore.ts` | Generation queue; IndexedDB persistence; submit/poll loop (line 534) |
| `ui/src/services/api.ts` | Central fetch client + `normalizeSong()` (line 76) |
| `ui/src/services/stemStudioApi.ts` | Model per-studio API client (submit-then-poll) |
| `ui/src/hooks/usePersistedState.ts` | Persisted useState + `writePersistedState()` same-tab sync |
| `ui/src/context/AuthContext.tsx` | Auto-login, `useAuth()` → `{ user, token }` |
| `ui/src/context/GlobalParamsContext.tsx` | LEGACY compat shim — do not extend |
| `ui/src/i18n/locales/en.json` | Source-of-truth strings (root key `"translation"`) |
| `ui/vite.config.ts` | Dev proxy: `/api`, `/audio`, `/references` → :3001 |
| `ui/tailwind.config.js` | `suno` palette, `brand` pink, `darkMode: 'class'` |
| `server/src/index.ts` | Route mounts (lines 72-95), serves `ui/dist` in prod |
| `server/src/routes/stemStudio.ts` | Model server route paired with stemStudioApi.ts |

## Failure signatures

| Symptom | Cause → fix |
|---|---|
| `npm install`/dev fails with dep/engine errors | Node 24+. Use Node 18–22 (`ui/package.json` engines field). |
| New studio renders but URL resets to `/` on refresh, or back-button breaks | Added `renderContent()` branch but missed `viewFromUrl()` and/or `urlForView()` — all three + Sidebar are required. |
| API 404 in dev but works in prod (or vice versa) | Route prefix not proxied — only `/api`, `/audio`, `/references` are in `vite.config.ts`. Keep endpoints under `/api/...`. |
| Songs missing audio/cover in one studio only | Endpoint bypassed `normalizeSong()` — raw snake_case leaked (api.ts:76-114). |
| Effects/API calls fire twice in dev | React StrictMode double-invoke (main.tsx:9). Dev-only; make mount effects idempotent, don't "fix" with flags. |
| New slider has no effect on generations | Value not wired into `getGlobalParams()` (globalParamsStore.ts:324) — the only place the request body is assembled. If it reaches `getGlobalParams()` but never reaches `/synth`: the server chain is `routes/generate.ts` → `translateParams()` (`server/src/services/generation/translateParams.ts:11`) → `aceReq`; sideband params must join `aceReq` BEFORE the LM phase (Golden rule 9 — never whitelist across the LM echo). |
| Setting change in one studio doesn't reflect in another | Wrote `localStorage` directly instead of `writePersistedState()` — same-tab listeners need the manually-dispatched StorageEvent. |
| Progress bar jumps 0→100 or shows 5000% | Server progress is 0–1 on some paths, 0–100 on others; normalise like audioGenQueueStore.ts:548-550. |
| Queue items stuck "generating" after reload | Expected behaviour: `_sanitizeItems` resets to `pending`+"Reconnecting…" and `resumeQueue()` re-submits (audioGenQueueStore.ts:142-156,610-621). |
| UI shows a raw key like `stem.queue` | Key missing from `i18n/locales/en.json`. |
| `npx tsc --noEmit` on a file passes but `npm run build` fails | `build` runs `tsc -b` across ALL project references — reproduce with `npx tsc -b` from `ui\`. |
| Generation fails and the UI just shows "failed" | Debug server-side: newest `logs/YYYY-MM-DD_HH-MM-SS/` folder → `generations/gen_*.log`, then `ace_engine.log` / `node_console.log`. |

## Institutional knowledge

- **VALIDATED (lead engineer, verbatim rule): Never use a browser agent for visual verification — ask the human user for screenshots/feedback. Browser agent is acceptable only for non-visual API checks.**
- VALIDATED: `getGlobalParams()` is the sole assembly point for generation request bodies; params bypass it → silently dropped. Confirmed by code and by commit 8ea519b ("adapter param plumbing bugs").
- VALIDATED: the "Zustand" label in top-level docs is aspirational — 5 of the 8 files in `ui/src/stores/` are hand-rolled `useSyncExternalStore` stores (a 6th, `playlistStore`, lives in `components/lyric-studio/`); 2 use zustand; `playbackConverters.ts` is pure helpers, not a store. Match your neighbours.
- VALIDATED: queue persistence moved localStorage → IndexedDB because 600+-item queues exceeded the 5 MB localStorage cap (comment at audioGenQueueStore.ts:66-68).
- VALIDATED: `GlobalParamsContext.tsx` is a migration shim by its own header — extend the store, not the context.
- UNVERIFIED: exact behaviour of `ui/add_strings.mjs` / `ui/generate_translations.mjs` (files exist; purpose inferred from names — read before running).
- NOTE: `ui/README.md` is the untouched Vite template — ignore it as documentation.

## Deeper reading

- [reference.md](reference.md) (same folder) — full Stem Studio end-to-end dissection, store code templates, DISCO hue table, localStorage key namespaces.
- `CLAUDE.md` (repo root) — cross-tier build/git rules.
- `FEATURES.md` — feature catalogue.
- `docs/dev/plugins-authoring.md` — if your UI feature exposes a new solver/scheduler/guidance knob, that's a Lua plugin, not C++ (plugins declare their own UI params).
- `engine/docs/ARCHITECTURE.md` — engine request JSON, generation modes.
- `docs/plans/` — internal design docs. **Gitignored, local-only — may be absent on a fresh clone.**
