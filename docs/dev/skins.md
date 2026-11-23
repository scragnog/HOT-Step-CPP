# Skins

A skin changes how HOT-Step looks without changing what it does. There are two
kinds: **themes** (variables only, below) and shell skins (a separate section,
added with the shell contract). Studio themes apply to the studio shell and the `/kit`
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
| `api` | `0`. |
| `kind` | `"theme"`. |
| `targets` | `["studio"]`, `["classic"]` or both. Studio themes set kit tokens only (the rules below). Classic rules are not defined yet; the checker skips them and says so. |
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
