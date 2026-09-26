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
