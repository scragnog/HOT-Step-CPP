# UI design rules

The track menu's **Measure BPM and key** dialog compares saved metadata with
CPU estimates before its explicit **Replace saved values** action. Analysis
alone never saves; failed or unavailable estimates cannot clear fields.

Read this before adding or changing any control under `ui/src/`. It exists because the same three
mistakes kept coming back: a native `<select>` that renders as a grey OS widget with white text, a
native checkbox where the app uses a toggle everywhere else, and a knob with no explanation. Each
rule names the component to use and shows the call. `node tools/docs/check-docs.mjs` fails on a
new native `<select>` or checkbox, so the rules are not optional.

## The three rules

### 1. A dropdown is `StyledSelect`

Never `<select>`. The house dropdown is the one the global bar's model pickers use: a rounded
trigger with a chevron, a floating panel with a check on the selected row, a filter box from
eight options up. `StyledSelect` is that look for any value list. Its panel is portalled, so it
works inside `<details>` drawers and scrolling panes, and it flips above the trigger when there is
no room below.

```tsx
import { StyledSelect } from '../shared/StyledSelect';

<StyledSelect
  accent="amber"                       // the studio's colour, see below
  value={provider}
  onChange={setProvider}
  options={[
    { value: 'gemini', label: 'Gemini', hint: 'Cloud, hears the audio' },
    { value: 'moss', label: 'MOSS', hint: 'Local, hears the audio' },
  ]}
  placeholder={t('x.pickProvider', 'Pick a provider')}
  className="w-full"
/>
```

`hint` is the second line of a panel row; it never shows on the trigger. `disabled` on an option
greys it out, and `disabled` on the select disables the trigger. `size="sm"` for dense rows.
`searchable` forces the filter box on or off; `searchPlaceholder` and `emptyLabel` set its
placeholder and its no-match line. `triggerLabel` shows a fixed label (such as "Presets") on the
trigger instead of the selected option, and the trigger then sizes to that label; `panelWidth`
sets a minimum panel width in pixels, right-aligned to the trigger when wider.

### 2. An on/off setting is `Toggle`

Never `<input type="checkbox">`. This includes rows in a selection list ("tick the datasets to
train"): those are toggles too. `Toggle` is a `<button role="switch">`, so it needs no id/htmlFor
pairing and it stops the click from reaching an accordion header around it.

```tsx
import { Toggle } from '../shared/Toggle';

// With its label and explanation: one line, nothing else to write.
<Toggle
  accent="amber"
  checked={lyricTiming}
  onChange={setLyricTiming}
  label={t('x.lyricTiming', 'Lyric timing supervision')}
  info={t('x.lyricTimingInfo', 'Uses vocal stems and forced alignment before training so the planner learns where each word lands. Off: skips stems, alignment and the cursor objective.')}
/>

// Bare switch, when the row already has its own label (a table cell, a list row).
<Toggle size="sm" accent="sky" checked={on} onChange={setOn} aria-label="Enable" />
```

`size="md"` (default) is the settings and studio-form size; `size="sm"` is the global bar's dense
row size. `disabled` dims it and ignores clicks. With `label`, `meta` adds the default/range line
to the hover card, as it does on `ParamLabel`.

The older names `Toggle` in `settings/SettingsPrimitives.tsx` and `ToggleSwitch` in
`global-bar/BarSection.tsx` are thin wrappers over this component. Do not add to them; import
from `shared/Toggle`.

### 3. Every parameter explains itself: `ParamLabel` with `info`

A label is a `ParamLabel`, and it carries `info`: what the setting does and what changes when you
raise, lower, or switch it. The default and range go in `meta`. The visible page keeps at most one
short line under a field; anything longer moves into `info`, which is what lets the page stay
uncluttered. The Training Studio's Advanced drawers and every panel under the global bar are the
model to copy.

```tsx
import { ParamLabel } from '../shared/ParamLabel';

<label className="flex flex-col gap-1">
  <ParamLabel
    label={t('x.reconTarget', 'Recon target')}
    meta={t('x.reconTargetMeta', 'default 0.25 · blank = knee')}
    info={t('x.reconTargetInfo', 'Stop the decoder once its reconstruction error reaches this value. Lower is better, typically 0.29–0.35 and falling. Some albums plateau above it; the plateau rule and the step budget then stop the run instead.')}
  />
  <input className={input} type="number" ... />
</label>
```

Write `info` for the person who does not know the code: say what it does, then what happens if
they move it. "Timestep bias" explains nothing; "Which part of the denoising the trainer sees most.
Higher values spend more steps on the noisy end, which shapes structure; lower values on the clean
end, which shapes detail" does. Keep the tone of [writing-style.md](writing-style.md).

A control that is already its own button is explained by passing it as `children`: the card opens
from the control, `label` is its heading, and no native `title` is used. `HintButton` is the
short version for buttons, so a toolbar explains itself the way a field does:

```tsx
<HintButton label={t('x.split', 'Split line')} info={t('x.splitInfo', '...')} className={btnSmall} onClick={split}>
  <Scissors size={12} /> {t('x.split', 'Split line')}
</HintButton>
```

```tsx
<ParamLabel label={t('x.split', 'Split line')} info={t('x.splitInfo', 'Start a new line at the selected word. ...')}>
  <button className={btnSmall} onClick={split}><Scissors size={12} /> {t('x.split', 'Split line')}</button>
</ParamLabel>
```

`className` styles the label text (the studios' field-label style by default) and
`rootClassName` the wrapper, so margins and `block` go there. `underline` (on by default) draws
the dotted hint under an explainable label; section headings turn it off.

Inside a panel that closes when the pointer leaves it (the global bar's dropdowns), mark the panel
`data-hovercard-boundary` so the card lands beside the panel instead of covering the knobs.

The card opens after a 500 ms hover, or at once on keyboard focus or a click on the `?`. Only one
card shows at a time, and while one is up the next hover opens without the delay. Do not add a
delay of your own around a `ParamLabel` or `HintButton`.

### Reset to default

`ParamLabel` takes an `onReset` callback: pass it and a small RotateCcw icon appears after the help
icon, which restores the field's default on click. `Slider`, `EditableSlider` and `Toggle` all take
a `defaultValue` prop and wire it to `ParamLabel` for you. Pass `onReset` directly to `ParamLabel`
only for a bare label next to a `StyledSelect` or a plain `<input>`, where there is no shared
primitive to do the wiring.

```tsx
<Slider label="Guidance Scale" value={gp.guidanceScale} onChange={gp.setGuidanceScale}
  defaultValue={GLOBAL_PARAM_DEFAULTS.guidanceScale} min={0} max={20} step={0.1} />
```

Show the icon only while the value differs from its default. `Slider`/`Toggle`/`EditableSlider` do
this automatically from `defaultValue`. They also take their own `onReset`, called instead of
`onChange(defaultValue)` when the icon is clicked, for the server-default overlay forms below. A
bare `ParamLabel` needs the guard written by hand:

```tsx
<ParamLabel label={t('gen.solver')} info={solverMeta?.description}
  onReset={gp.inferMethod !== GLOBAL_PARAM_DEFAULTS.inferMethod
    ? () => gp.setInferMethod(GLOBAL_PARAM_DEFAULTS.inferMethod) : undefined} />
```

Reset through the field's own setter, never a generic `set({ field: default })`. Some setters carry
side effects (mirroring a model choice to the server, deriving a dependent field) that a bypass would
skip. An accordion section's existing group "Reset" button (resets every field in the panel) and a
per-field `onReset` on one of those same fields (resets just that one) are not the same affordance,
so add the per-field icon without touching the group button. Don't add `onReset` to the header label
itself when it covers several fields with no single value of its own (the group button already speaks
for it).

A schema-driven control (a backend-declared extension in `BackendExtensionControls.tsx`, a Lua
plugin param in `PluginControls.tsx`) has no entry in `GLOBAL_PARAM_DEFAULTS`; its default is
whatever the schema itself declares (`p.default`). Compare the current value against `p.default`
the same way, omit the reset icon (the control itself stays usable) when the schema declares no default, and reset by writing
`p.default` back through the same setter every other change in that control goes through
(`setBackendParam`, `setPluginParam`), never a bespoke path.

A form that overlays edits on the server's own recipe (`Mm3TrainCard.tsx`, `Yue2TrainCard.tsx`,
`Yue2ArTrainCard.tsx` — `const base = {...}; const form = { ...base, ...edits }`, `edits` a
`useState<Partial<FormState>>({})`) has a moving target for "default": the server can ship a new
recipe between sessions, so `base` is re-derived from `status.defaults` on every render rather than
captured once. Reset here means deleting the key from `edits`, never writing `base[k]` back — a
written value freezes that field at today's default and stops following the server, which is the
one thing a field the user never touched should keep doing. Pair the icon with a sibling
`reset(k)` next to the existing `set(k, v)`, gated on `edits[k] !== undefined && edits[k] !== base[k]`
(the same comparison `value !== defaultValue` reduces to once `form` is `{ ...base, ...edits }`).
On a `Slider`, `EditableSlider` or `Toggle`, pass `defaultValue={base.steps}` and
`onReset={() => reset('steps')}`: the primitive does the comparison and calls `reset` instead of
writing the default.
A dependent-field coupling on `onChange` (DiT method's PiSSA forcing HOT-PiZZA off, MM3 train's
same pair) needs the identical coupling on reset — check the delta between the two would-be
states and apply whatever the `onChange` handler would, not a delete-and-hope.

An accordion whose section is one on/off `toggle` (`PostProcessingDropdown.tsx`'s shared `Accordion`)
takes the same `onReset` its `ParamLabel` header already renders — add it to the accordion's own
props and pass it straight through, one level of plumbing, not a new component. Guard it on the
toggle's own field, never on anything the section contains: an accordion's `onReset` and its
content's own per-field resets (or an inner group "Reset" button) stay independent, each touching
only what it says it does.

## Collapsible sections

`Fold` (`ui/src/components/shared/Fold.tsx`) is a titled section that opens and closes from its
header row and remembers that state across page loads in localStorage under `storageKey`.
`defaultOpen` sets the state on a first visit. Create and Cover Studio use it for their optional
blocks (drafts, the negative prompt, dataset caption options). Give each fold its own key.

```tsx
<Fold title={t('studioDrafts.title', 'Drafts')} storageKey="hs-create-fold-drafts">
  ...
</Fold>
```

## Phone bottom sheets

Use `BottomSheet` for phone song actions; desktop actions keep their dropdown.
The responsive parameter bar already provides phone sections, so `BarSection`
keeps its desktop dropdown without a separate phone sheet.

Sheet backdrop/content use z-index 59/60, above settings at 55 and below child
dialogs at 70. A visible higher portal suspends the sheet's Tab and Escape
handlers and backdrop dismissal. Closing the sheet returns focus to its opener.
See [PWA and phone sheets](../../ui/PWA.md) for the full contract and production
fixture, including child-dialog keyboard and focus checks.

## Error boundaries

A render-time throw no longer blanks the tab. `ErrorBoundary`
(`ui/src/components/shared/ErrorBoundary.tsx`) wraps the whole app in `main.tsx`, and the studio
area in `App.tsx` with `resetKey={activeView}`. A broken studio shows the error with Try again and
Reload page, the sidebar and player keep working, and switching studio clears it. Wrap a new
independent region in its own boundary only if one failing part should not take its neighbours
with it.

## The UI kit and the tint (new, not yet in the app)

The kit is one tintable design system: tokens in `ui/src/index.css` and `ui/tailwind.config.js`,
components in `ui/src/components/shared/kit/`. It sits beside the existing controls and changes
nothing in the app yet: the rules above still govern every current page, and the kit's slider,
toggle and select are separate components (`KitSlider`, `KitToggle`, `Select`), not edits of
`Slider`, `Toggle` and `StyledSelect`. A later slice moves pages over. Change a kit element once
and every kit use changes, so never copy its styling to a call site.

To try every element, size, variant and state with a live tint switcher, open
`http://localhost:3000/kit` (dev only, no sidebar link).

| Element | Import | Notes |
|---|---|---|
| `Button`, `IconButton` | `shared/kit/Button` | `size` sm / md / lg; `variant` primary / secondary / ghost / danger. `IconButton` needs `aria-label`. |
| `Tabs` | `shared/kit/Tabs` | `items`, `value`, `onChange`; the active indicator slides; arrow keys move. |
| `Panel`, `Card` | `shared/kit/Panel` | `Panel` takes `title`, `actions`, and `collapsible`; `Card` with `onClick` becomes a button. |
| `StatusLed` | `shared/kit/StatusLed` | `state` idle / ok / warn / error / generating, and a required `label`. Warn shows an icon too. |
| `SectionPanel`, `Well`, `Module`, `Rail`, `SongItem`, `EmptyState` | `shared/kit/Panels` | The big containers. `SectionPanel`: header strip (`title`, optional `led`, `actions`), body, optional `footer`. `Well`: recessed area for lyrics, waveform, log. `Module`: controls under an engraved `label`; `Rail` (`vertical`) is the divider. `SongItem`: rest, hover, `selected`, `playing`. `EmptyState`: `icon`, `title`, text, `action`. |
| `KitSlider`, `KitToggle`, `Select` | `shared/kit/...` | Same props as the originals, minus `accent`. Slider fill needs the `kit-range` class, which `KitSlider` sets. |

```tsx
import { Button } from '../shared/kit/Button';

<Button variant="primary" size="lg">Generate</Button>
```

### How the tint works

The accent is three RGB-channel variables on `:root`: `--tint`, `--tint-hover` and `--tint-glow`.
Tailwind exposes them as `tint`, `tint-hover` and `tint-glow` with alpha support, so `bg-tint`,
`bg-tint/20` and `ring-tint` all work. Set `data-tint="teal"` on any element and everything inside
re-tints; no other wiring. To add a tint, add a block next to `[data-tint="teal"]` in `index.css`
with the three variables. Never hard-code the accent hex in a kit component. A portalled panel
(the `Select` list) sits outside the scope, so it copies the trigger's `data-tint`.

Surfaces (`surface-well`, `-chassis`, `-panel`, `-raised`), text (`ink`, `ink-muted`, `ink-onTint`),
status (`status-ok`, `-warn`, `-error`), radii (`rounded-ctl` 6px, `rounded-panel` 10px), shadows
(`shadow-bevel`, `shadow-press`, `shadow-glow`) and motion (`duration-glow` 140ms,
`duration-press` 70ms) are tokens too. They are additions: the `suno` and `brand` colours and
`--color-primary` are unchanged. Status colour never stands alone: give it a label, and warn an icon.

### Depth recipe

One recipe in the kit block of `index.css`; every element picks it up, so change a number there, never at a call site.

- Tint fills use `--grad-tint` (light top, `--tint-deep` bottom); the hover and press variants are `--grad-tint-hover` and `--grad-tint-press`. Because a `var()` inside a custom property is substituted where it is declared, these gradients are declared on `:root` and on every `[data-tint]` scope. A new tint needs its `--tint-deep` next to the other three channels.
- `.kit-raised` (neutral raised: secondary button, select trigger, card), `.kit-fill` (tint fill: primary, toggle on) and `.kit-well` (recessed) set the gradient, the side border and the edge. Neutral surfaces get a 14% white top edge, tint fills 40%; wells get a darker top inner shadow.
- `.kit-lift` adds the hover (brighter border, 1px ring, outer glow) and the press (darker gradient, inset shadow, 1px drop) over the 140ms token. `data-force="hover"` and `"press"` show those states without a pointer.
- The numbers live in the `--grad-*`, `--edge-*`, `--drop`, `--press-shadow` and `--well-shadow` tokens on `:root`.

### Font roles

Three tokens on `:root`: `--font-body`, `--font-heading` (panel titles, headings, engraved labels) and `--font-mono` (numeric readouts; `.kit-mono` adds tabular figures). Kit surfaces use them through `.kit-root`, `.kit-heading` and `.kit-mono`; the default is Inter for all three. The test page has a font switcher for six pairings and loads those fonts from Google Fonts on that page only. The final pick should be self-hosted, since the app runs locally and may be offline; it then becomes a one-line change to the three tokens.

### Adding a size or variant

Everything hangs off one base class. For a `Button`, add `.kit-btn-<name>` in the kit block of
`index.css` (a size sets height, padding and text size; a variant sets colours and, if the glow
should differ, `--glow`), then add the name to `ButtonSize` or `ButtonVariant` and to its class map
(`SIZE_CLASS`, `VARIANT_CLASS`) in `kit/Button.tsx`. Keep the full class name in the map: Tailwind
drops component classes it cannot find whole in the source, so `kit-btn-${size}` would lose its
styles. Add it to the gallery grid too. Do not fork the base: radius, bevel, border, focus ring,
hover glow and press live in the depth recipe (`.kit-lift`), not in `.kit-btn`. Motion respects `prefers-reduced-motion`; keep that
when adding any (the gallery's reduced-motion preview uses the `kit-reduce` class).

### Light mode

The kit follows the app's theme, the `dark` class on `<html>` (`useTheme`). Dark values are the `:root`
defaults; light values for the kit tokens only sit in one `html:not(.dark)` block in `index.css`
(surfaces `#e9e7e3` / `#f3f2ef` / `#fbfaf8` / `#dedbd6`, ink `#1b1b1d`, muted `#5e5b57`, a white top edge,
soft dark shadows, wells darker at the top). Existing app tokens and rules do not change in either theme.
The surface, ink and status colours are RGB-channel variables (`--k-*`) that Tailwind reads, so
`bg-surface-panel` and `text-ink` follow the theme with no `dark:` variants. Fills keep the tint
gradient with `ink-onTint` text. For tint-coloured text or icons use `text-tint-ink` (`--tint-ink`),
never `text-tint`: amber on a light panel is about 2:1. Light `--tint-ink` values, contrast on
`#f3f2ef` (panel): amber `#8A5200` 5.7, teal `#0B6B62` 5.7, violet `#5B3FC4` 6.4, rose `#B3173F` 6.0;
on the darkest well `#dedbd6` all four stay above 4.6. In dark, `--tint-ink` equals the tint. The hover
glow reads weaker on light, so light hover is a brighter tint border plus a tinted soft shadow; pressed
still sinks 1px. Any new kit rule with a hard-coded dark shadow or white highlight needs a light twin
in that block. The test page has a Dark / Light switch.

## Accent colour

Match the controls around you. Pass the same `accent` to `StyledSelect` and `Toggle`.

| Where | Accent |
|---|---|
| Training Studio | `amber` |
| Create, Lyric Studio | `pink` |
| Global bar panels | the section's own colour (`amber`, `cyan`, `emerald`, `pink`, `purple`, `sky`, `teal`, `violet`); read it off the neighbouring `ToggleSwitch` |
| Anywhere else | the studio's existing highlight colour; `pink` when it has none |

## Strings

Every label, hint and `info` goes through `t('key', 'English default')` with the key added to
`ui/src/i18n/locales/en.json` ([ui-feature-dev](../../.claude/skills/ui-feature-dev/SKILL.md)).

## The check

`node tools/docs/check-docs.mjs` counts native `<select` and `type="checkbox"` per `.tsx` file
under `ui/src`, with comments stripped first, and compares them with
`tools/docs/ui-primitives-baseline.json`, the list of files that still had them when the rule
was written. A file with more than its baseline allows, or a file not in the baseline with any,
fails the check. It runs in CI (`docs.yml`) and inside `check-release-prereqs.mjs`.

The baseline has been empty (`{}`) since every file was converted in `b6eff9e3`, so any native
`<select>` or checkbox in a `.tsx` file now fails the check. Replace it with `StyledSelect` or
`Toggle`; never run `--update-ui-baseline` to make a new one pass. The baseline only ever goes
down.

## Shells

The app has two roots, picked at boot in `ui/src/main.tsx` through `app/shell.ts`: **classic**
(`App.tsx`, the layout described above) and **studio** (`shells/studio/StudioApp.tsx`, a work in
progress). `?ui=classic|studio` wins and is saved to `hs-shell`; without it the saved value is used;
without that, classic. Settings, General, Interface writes the same key. A switch is a full page
reload, never a live swap, and only the chosen shell is downloaded (`import()` under `Suspense`).

Both shells call `useAppRuntime` (`ui/src/app/`) once. It owns everything that is not layout: queue
resume and projection, live MM3 takes and the play-bar handover, song-cache patches from window
events, the library refetch after a reconnect, offline and restart detection, and the `ace-settings`
flags a Create render carries. A new runtime concern goes there, not into a shell.

Rules:

- **Fence.** `src/shells/studio/**` may import only `components/shared/kit/` from `components/`
  (`no-restricted-imports` in `ui/eslint.config.js`). Everything else comes from the runtime, stores,
  queries and services.
- **CSS.** `index.css` is frozen apart from kit classes. Studio CSS lives in a file imported by the
  studio entry only.
- **Persisted keys.** Studio-only keys use an `hs-studio-` prefix. The shape of an existing key never
  changes, because classic reads it too. `hs-shell` holds a JSON string, `"classic"` or `"studio"`.

## Verifying a change

Type-check (`npx tsc -b` from `ui/`), then ask the user to look at the screen; never use a browser
agent for visual checks. Say which page and
which control.
