# Skins

A skin changes how HOT-Step looks without changing what it does. There are two
kinds: **themes** (variables only, below) and [shell skins](#shell-skins) (a whole
React UI, for developers who replace the layout). Studio themes apply to the studio shell and the `/kit`
page. The classic UI has no semantic colour layer, so a theme reaches it only if its manifest targets `classic` (rules to follow).

Status: the theme API is `api: 0` (unstable). Renaming or removing a token is a
major API bump; adding one is not.

## Themes

A theme is a folder `ui/src/themes/<id>/` with two files. The registry finds it
with `import.meta.glob`, so adding a theme never edits a central list.

```
ui/src/themes/<id>/manifest.json
ui/src/themes/<id>/theme.css
```

`manifest.json`:

| Field | Rule |
|---|---|
| `id` | Lowercase letters, digits, dashes. Equals the folder name. |
| `name` | Shown in the picker. |
| `author` | Who to credit. |
| `version` | Semver, `1.0.0`. |
| `license` | Optional. The licence of an imported skin (e.g. `MIT`). |
| `api` | `0`. |
| `kind` | `"theme"`. |
| `targets` | `["studio"]`, `["classic"]` or both. Studio themes set kit tokens only (the rules below). Classic skins follow [Classic overlay skins](#classic-overlay-skins). |
| `preview` | 2 to 4 hex colours (`#rrggbb`) for the picker swatch. |

The selection is stored as `hs-skin` (`hs-theme` is light or dark). The active
theme is `html[data-skin="<id>"]`, set by `ui/public/theme-boot.js` before first paint and dropped by `main.tsx` when the skin does not target the running UI. With no skin set you get the default kit.

### Rules for `theme.css`

- Two selectors only: `html[data-skin="<id>"]` (dark) and
  `html[data-skin="<id>"]:not(.dark)` (light). Nothing else.
- Only the tokens below. No other property, no `@import`, no `url()`, no nested
  rules.
- Set every colour token in **both** blocks. Both selectors have the same
  specificity as the kit's own defaults, so a token left out of the light block
  would show its dark value in light mode. Fonts need only the first block.
- Values follow a fixed grammar per token: colour channels are `R G B`; gradients
  may use only `linear-gradient()`, `rgb()`, `rgba()` and `var()` of a documented
  token; shadows only `rgb()`, `rgba()` and `var()`; fonts are plain family names.
  No image function of any kind, no escapes, no strings outside fonts. A gradient
  is `linear-gradient(<angle or to side>, <opaque colour> [positions], ...)` with
  `#hex` or `rgb()` colours; names and transparency are rejected because the
  checker cannot compute their contrast. `!important` is not allowed.
- Text must keep 4.5:1 contrast on flat surfaces and 4.6:1 on gradient fills (the extra 0.1 is the sampling margin), in both modes: ink and
  muted on the flat surfaces and on every colour stop, and the blends between stops, of the raised, panel and well
  gradients, and on-tint on the accent gradients.

Run `npm run check:skins` from `ui/`. It parses each theme with PostCSS, checks
the manifest, the rules above and the contrast pairs (ink and muted on the four
surfaces, on-tint on the tint fills, tint-ink, ok, warn and error on the panel),
then confirms the bad fixtures in `ui/scripts/skin-fixtures/` are rejected.

Colours are RGB channels written `R G B` (no commas), so the kit can add alpha:
`rgb(var(--tint) / 0.5)`.

### Token list

Tint (the accent). The `--grad-tint*` gradients are built from these; override
them only to replace the gradient.

| Token | Role |
|---|---|
| `--tint` | Accent fill and rings. |
| `--tint-hover` | Lighter accent: hover borders, top of the fill gradient. |
| `--tint-glow` | Glow colour on hover. |
| `--tint-deep` | Darker accent: bottom of the fill gradient. |
| `--tint-ink` | Accent as text or an icon. In light mode it must be darker than `--tint`. |

Surfaces, ink and status:

| Token | Role |
|---|---|
| `--k-well` | Recessed areas (inputs, tracks). |
| `--k-chassis` | Page background. |
| `--k-panel` | Panels and sections. |
| `--k-raised` | Raised controls and cards. |
| `--k-ink` | Main text. |
| `--k-muted` | Secondary text. |
| `--k-on-tint` | Text on accent fills. |
| `--k-ok`, `--k-warn`, `--k-error` | Status text and LEDs. |
| `--k-error-fill` | Danger button fill and ring. |
| `--k-error-hover`, `--k-error-glow` | Danger hover border and glow. |
| `--k-shade` | Shadow and engraving channel (default black). |
| `--k-hi` | Highlight channel: top edges and glints (default white). |

Depth (a complete CSS value each):

| Token | Role |
|---|---|
| `--grad-raised` | Raised surface gradient. |
| `--grad-raised-press` | Same, pressed. |
| `--grad-panel` | Panel gradient. |
| `--grad-well` | Well gradient (darker at the top). |
| `--grad-tint`, `--grad-tint-hover`, `--grad-tint-press` | Accent fill gradients. |
| `--edge-side` | 1px border colour on surfaces. |
| `--edge-neutral` | Top-edge highlight on neutral surfaces (an inset shadow). |
| `--edge-tint` | Top-edge highlight on accent fills. |
| `--drop` | Drop shadow under raised surfaces. |
| `--press-shadow` | Inset shadow when pressed. |
| `--well-shadow` | Inset shadow of wells. |

Fonts:

| Token | Role |
|---|---|
| `--font-body` | Body and UI text. |
| `--font-heading` | Headings and panel titles. |
| `--font-mono` | Numeric readouts. |

A font must already be on the page: a theme cannot load one (no `@import`, no
`url()`). The app runs offline, so a theme may only name system fonts or fonts
the app ships.

### Adding a theme

1. Copy `ui/src/themes/ice/` to `ui/src/themes/<id>/` and edit both files.
2. Open `/kit`, pick it in the Skin row, and look at dark and light.
3. `npm run check:skins` from `ui/`.

## Classic overlay skins

A theme whose manifest targets `classic` is an overlay: plain CSS that restyles
classic's class names (`ui/src/themes/skeuo/` is the first). The classic UI has no
semantic colour layer, so an overlay is **tied to classic's markup**. That works
because classic is meant to stay put; if you change a classic class combination or
a hook (`sk-fader*` on `EditableSlider`), check the overlays.

To recolour the brand accent, set `--pink-50` to `--pink-950` and `--rose-50` to
`--rose-950` (RGB channels, `R G B`). Every `pink-*` and `rose-*` utility reads
them (`tailwind.config.js`); the defaults in `index.css` equal Tailwind's stock
values, so with no skin classic is unchanged.

The picker is **Settings > Skin** in classic. It sets `html[data-skin]` at once and
saves `hs-skin`; the default is no skin.

`check-skins.mjs` enforces, for overlays:

- Every selector is scoped to `html[data-skin="<id>"]`, `html.dark[data-skin="<id>"]`
  or `:root[data-skin="<id>"]`, so a skin is inert until selected.
- No `@import`, `@font-face` or other at-rules except `@media` and
  `@supports` (no `@keyframes`: names are global). No `url()` except `data:` (the embedded images need `img-src data:`
  in the CSP). Fonts: name system fonts only; a skin cannot ship a font file.
- No `position: fixed` and no `content` property, so a skin cannot overlay the page
  or inject fake UI. `!important` is allowed.

There is no token or contrast check: a person reviews an overlay in both modes.
Keep an `IMPORT.md` beside an imported skin recording the source and every edit,
so the next drop from its author is a diff.

## Shell skins

A shell skin is a whole React UI: its own layout, its own pages, its own look. It is for
a developer who wants to replace what the app looks like and how it is arranged, not just
recolour it. It runs inside the app, in the same browser tab, on the same data. This section
assumes you have never seen this repo.

Status: **skin-api version 0, unstable.** Names can still change while the first shells are
ported (see [Versioning](#versioning)). Shells ship inside the app build; nothing is loaded at
run time from outside.

### What you get, and what you must not do

The app has a *host* (`ui/src/app/ShellHost.tsx`). It starts everything a UI needs: the render
queue, the library, the connection to the engine, the offline, restart and update screens, light
and dark mode. Your shell is only the part you can see. So:

- **Never** start the runtime, call the server, read `localStorage`, or import app code.
  You get data and actions from `skin-api` (below) and nothing else.
- Your root component takes no props and is the default export of `index.tsx`.
- The host puts your shell inside `<div data-shell="<id>">`, which sets font, margin and
  `color-scheme`. Style under that element.
- You may import: your own folder, `skin-api`, `react`, `lucide-react`, and the kit
  (`components/shared/kit/`, buttons, panels, sliders; optional).

### Adding a shell

1. Copy `ui/src/shells/starter/` to `ui/src/shells/<id>/`. The id is lowercase letters,
   digits and dashes, 2 to 32 characters, and equals the folder name.
2. Edit `manifest.json`:

| Field | Rule |
|---|---|
| `id` | Equals the folder name. |
| `name` | Shown in Settings > Interface. |
| `author` | Who to credit. |
| `version` | Semver, `1.0.0`. |
| `license` | Optional. The licence you give the app (`MIT`). |
| `api` | `0`: the skin-api major you wrote against. |
| `kind` | `"shell"`. |

3. Edit `index.tsx` and its CSS. The registry finds the folder by itself
   (`import.meta.glob`); there is no list to edit.
4. Run it: from `ui/`, `npm run dev`, then open `http://localhost:3000/?ui=<id>`. The choice is
   remembered; **Settings > Interface** (in classic) or `openClassic()` goes back. With no engine
   running the app shows its offline screen: start the app as the README says.
5. Check it: `npx tsc -b`, `npm run lint` and `npm run check:skins`, all from `ui/`.

The starter shell (`ui/src/shells/starter/`) is the working example. It uses every skin-api
export once and draws a Create page, a library, an engine page and a classic Training Studio
page. Read `index.tsx` and `ParamControl.tsx` first.

### The API (version 0)

Import from `'../../skin-api'`. All names are explicit; types are the API's own.

**Theme and look**

| Export | What it does |
|---|---|
| `useTheme()` | `{ mode, resolved, setMode, cycle }`. `mode` is `system`, `light` or `dark`; `resolved` is what shows. The host owns it (`hs-theme`, the `.dark` class on `<html>`, `theme-boot.js`): style from `.dark`, never set it yourself. |
| `useThemeSkin()` | `{ skin, setSkin, available }`: the kit's colour skins ([Themes](#themes)). Optional; a shell with its own look ignores it. |

**Parameters.** An id is a generation control (`inferenceSteps`, `guidanceScale`, `ditModel`...)
or `backendParams.<key>` for a knob the active backend declares.

| Export | What it does |
|---|---|
| `useParam(id)` | `{ value, set, reset, changed }`. `changed` is true when the value differs from its default. |
| `useParamMeta(id)` | `ParamMeta`: `type` (`slider`, `select`, `toggle`, `text`, `array`, `object`), `label`, `hint`, `default`, `min`, `max`, `step`, `options`, `editable`. Undefined while loading. |
| `params.set(id, value)`, `params.reset(id)` | Write a value, or put the default back. Each is one undo step; a drag on one control collapses into one step. |
| `params.transaction(label, fn)` | Everything set inside `fn` is one undo step. |
| `useParamHistory()` | `{ undo, redo, canUndo, canRedo, steps, cursor }`. Host-owned, in memory, 100 steps. Steps from `cursor` onward are undone. |

Rules for parameters:

- Writes go through the same store setters classic uses, so classic sees them and the saved
  settings stay right. Only the host's undo history records them; do not keep your own.
- `editable: false` means a content field (caption, lyrics, bpm, key...). Those travel in
  `generate()`, not `params.set`.
- A `slider` may have no `min` and `max` yet (`adapterSectionIsolation`, `stableStepSeed`):
  draw a number field. A `select` may have no `options` (models, solvers): fill them from
  `useBackendModels()`. `ParamControl.tsx` shows both fallbacks.

**Songs and renders**

| Export | What it does |
|---|---|
| `useSongs()` | `{ songs, loading }`: the library, newest first. |
| `useRecentSongs(source?)` | The latest 50 renders (`all`, `create`, `lyric-studio`, `cover-studio`). |
| `useQueue()` | `{ items, active }`. Each item has `state` (`waiting`, `running`, `done`, `failed`), `stage`, `progress` (0 to 100), `error`, `songId`. |
| `generate(request)` | Starts a render and returns at once. Call it from a click. Follow it with `useQueue()`. `request.caption` is required. |
| `cancel(queueItemId)` | Stops an item, or drops it if it never started. |

**Engine**

| Export | What it does |
|---|---|
| `useBackends()`, `useActiveBackend()`, `backend.switch(id)` | The generation backends, the active one, and switching. Per-backend settings follow the switch. |
| `useCapabilities(id?)` | What the backend supports: longest render, bpm and key support, batch size, and its extra controls as `backendParams.<key>` ids. |
| `useBackendModels(id?)` | Model files by bucket (`dit`, `lm`, `vae`...) and adapters. |
| `useVram()` | `{ usedMb, totalMb, freeMb }`, refreshed while mounted. |
| `useEngineLog()` | The last 500 log lines; the stream is open only while mounted. |

**Navigation and storage**

| Export | What it does |
|---|---|
| `openInStudio(kind, songId)` | `library`, `cover`, `stem`, `midi` or `repaint`: reloads into the classic UI at that page. v0 opens the page; the user picks the song. |
| `openClassic()` | Reload into classic. |
| `<ClassicPage view="training-studio" />` | A classic page, lazy-loaded inside your frame, for what you have not built. v0 has one view: `training-studio`. Library and Create need classic's own handlers, so use `openInStudio`. |
| `useSkinStorage(key, initial)` | `[value, set]`: JSON in `hs-skin-<id>-<key>`, 64 KB per value, for layout and preferences. Never songs, drafts, presets or seeds: those live on the server. |

Not in version 0 (ask for it by opening an issue; each addition is reviewed as core code): seeds,
drafts, presets, adapters and plugin lists, VST and mastering controls, stems, MIDI, repaint,
live-take streaming, the song menu actions.

### Shell CSS rules

Your CSS must not touch anything outside your shell: a `button { }` rule would restyle a classic
page shown inside your frame, and a `body { }` rule would restyle the host's overlays.

- Every selector carries a class. No `:root`, `html`, `body`, `*`, bare element or bare
  attribute selector (`button`, `[hidden]`).
- Custom properties (tokens) are set on `[data-shell="<id>"]` only. Dark mode is
  `.dark [data-shell="<id>"]`: the host puts `.dark` on `<html>`.
- At-rules: `@media`, `@supports`, `@font-face` and `@keyframes <id>-<name>` (keyframe names are
  global, so prefix yours).
- No `@import`. `url()` is a relative file inside your folder, or `data:`. Absolute and
  `https:` URLs are rejected, and the page's security policy would block them anyway.
- Fonts: put the file in your folder and use a relative `@font-face` url. Check its licence
  (OFL fonts are fine).

`npm run check:skins` enforces these with a CSS parser, and also checks every import path in
your folder against the list above. Bad examples live in `ui/scripts/skin-fixtures-shells/`.
`npm run lint` flags the common wrong imports as you type.

### Versioning

`manifest.json` carries `api: <major>`. Rules:

- Shells build with the app, so `npx tsc -b` is the compatibility test, and the starter shell
  uses every export as the canary.
- `skin-api` has explicit named exports and its own types, no `export *`. Adding a name is
  not a break. Renaming or removing one, or changing what it means, is a major bump.
- A refactor inside the app that would break a merged shell is fixed inside `skin-api/`,
  never by editing the shell. If that cannot be done, it is a major bump, and the shells in the
  repo are migrated in the same commit.
- Version 0 is unstable: expect breakage until the app's own interface runs on the same API
  (version 1). You will be told in the change.

### Reviewing a shell

A shell is code that can call everything `skin-api` offers, so it is reviewed like any other
code before it merges. A pull request may touch only its own `ui/src/shells/<id>/` folder
(and later, an entry in the credits). A change to `package.json`, a lockfile, a config file or
`.github/` is a core review, not a skin review.
