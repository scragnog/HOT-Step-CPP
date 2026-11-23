# Re-importing the skeuo skin

Source: `skeuo-skin.zip` from the skin's developer (Helikaon23s), received 2026-11-23, first drop.
Licence: MIT, consent confirmed by the developer on 2026-11-23.

`theme.css` is the zip's `ui/src/skins/skeuo.css` with only the edits below, so the next drop is a diff.
Recipe: diff the new `skeuo.css` against this `theme.css`, apply the new drop, then redo the edits in this list.

## Edits to `theme.css`

1. Removed line 1, `@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono...')`. The app runs offline and the checker rejects `@import`. JetBrains Mono is not bundled (a skin cannot load a font file: no `url()` except `data:`), so `--sk-mono` falls back to the system mono stack. Installed copies of JetBrains Mono still win, as it stays first in the list.
2. Removed the blank line after it. Nothing else changed.

## Not taken from the zip

`index.html`, `main.tsx`, `App.tsx`, `Sidebar.tsx`, `useTheme.ts` and `tailwind.config.js` are edits to an older copy of classic and were not copied. The zip's `data-skin="skeuo"` default-on and "Flat look" toggle are replaced by the Skin picker in Settings (`hs-skin`).

What classic gained instead:

- `ui/tailwind.config.js` and `ui/src/index.css`: `pink-*` and `rose-*` read `--pink-NNN` / `--rose-NNN` channels, defaulting to Tailwind's stock values.
- `ui/src/components/shared/EditableSlider.tsx`: the `sk-fader`, `sk-fader-fill` and `sk-fader-thumb` class hooks the CSS styles. They have no styling of their own.
