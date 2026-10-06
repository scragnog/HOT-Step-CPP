# UI design rules

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
greys it out. `size="sm"` for dense rows. `searchable` forces the filter box on or off.

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
row size. `disabled` dims it and ignores clicks.

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

Inside a panel that closes when the pointer leaves it (the global bar's dropdowns), mark the panel
`data-hovercard-boundary` so the card lands beside the panel instead of covering the knobs.

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
this automatically from `defaultValue`; a bare `ParamLabel` needs the guard written by hand:

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
A dependent-field coupling on `onChange` (DiT method's PiSSA forcing HOT-PiZZA off, MM3 train's
same pair) needs the identical coupling on reset — check the delta between the two would-be
states and apply whatever the `onChange` handler would, not a delete-and-hope.

An accordion whose section is one on/off `toggle` (`PostProcessingDropdown.tsx`'s shared `Accordion`)
takes the same `onReset` its `ParamLabel` header already renders — add it to the accordion's own
props and pass it straight through, one level of plumbing, not a new component. Guard it on the
toggle's own field, never on anything the section contains: an accordion's `onReset` and its
content's own per-field resets (or an inner group "Reset" button) stay independent, each touching
only what it says it does.

## Error boundaries

A render-time throw no longer blanks the tab. `ErrorBoundary`
(`ui/src/components/shared/ErrorBoundary.tsx`) wraps the whole app in `main.tsx`, and the studio
area in `App.tsx` with `resetKey={activeView}`. A broken studio shows the error with Try again and
Reload, the sidebar and player keep working, and switching studio clears it. Wrap a new
independent region in its own boundary only if one failing part should not take its neighbours
with it.

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

`node tools/docs/check-docs.mjs` counts native `<select` and `type="checkbox"` per file under
`ui/src` and compares them with `tools/docs/ui-primitives-baseline.json`, the list of files that
still had them when the rule was written. A file with more than its baseline allows, or a file not
in the baseline with any, fails the check. It runs in CI (`docs.yml`) and inside
`check-release-prereqs.mjs`.

When you convert a file, run `node tools/docs/check-docs.mjs --update-ui-baseline` so the
baseline shrinks with it; the check warns while an entry is stale. The baseline only ever goes
down.

## Verifying a change

Type-check (`npx tsc -b` from `ui/`), then ask the user to look at the screen; never use a browser
agent for visual checks ([AGENTS.md](../../AGENTS.md#ui--browser-verification)). Say which page and
which control.
