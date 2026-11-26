# Contribute a HOT-Step skin

## Choose a skin kind

**Theme.** Pick this if you want new colours, depth or typography for the Studio UI and its component kit. A theme sets documented CSS variables. Ice is a blue theme you can try in the kit. [Ice source](ui/src/themes/ice/)

**Classic overlay.** Pick this if you want to restyle the current HOT-Step interface and keep its layout and controls. An overlay applies scoped CSS to Classic's existing classes. Skeuo by Helikaon23s is an example. [Skeuo source](ui/src/themes/skeuo/)

**Shell.** Pick this if you want your own screens, navigation or control layout. A shell is a React UI that uses HOT-Step's shared runtime and data through the skin API. HOT-Step Tiers by iRedsneth is an example of a design for a shell. [Working starter shell](ui/src/shells/starter/)

## Run a preview without a GPU

Fork the repository on GitHub and clone your fork. Use Node.js 24 LTS and run these commands from the clone root:

```sh
npm --prefix server ci --ignore-scripts
npm --prefix ui ci
cd ui
npm run dev -- --config scripts/skin-preview.config.mjs
```

The preview uses the React app from your checkout and serves sample API data on
`127.0.0.1:4174`. It needs no engine, model files or library. You can edit CSS
and components while Vite is running. The library and queue start empty, and
rendering, training and other server writes are unavailable. Use
`Ctrl+C` to stop the preview.

| To preview | Open |
| --- | --- |
| A theme | Go to `http://127.0.0.1:4174/kit?ui=classic` and select your theme in the Skin row. |
| A Classic overlay | Go to `http://127.0.0.1:4174/settings?ui=classic` and choose the overlay under General > Skin. |
| A shell | Go to `http://127.0.0.1:4174/?ui=starter`. Use your own ID in place of `starter` after you copy it. |

## Start your skin

Choose a unique ID made of lowercase letters, digits and dashes. Copy one of
these folders beside the original. Use your ID as the new folder name.

| Kind | Copy | Edit |
| --- | --- | --- |
| Theme | `ui/src/themes/ice/` | Change the ID in `manifest.json` and both skin selectors in `theme.css`. |
| Classic overlay | `ui/src/themes/phosphor/` | Change the ID and the palette variables ([colour variables](docs/dev/skins.md#classic-overlay-skins)), then add scoped rules for anything a palette cannot do. `ui/src/themes/skeuo/` restyles class by class. Write an `IMPORT.md` if you are adapting an outside design. |
| Shell | `ui/src/shells/starter/` | Change the ID in the manifest and `data-shell` selectors in the CSS. Edit `index.tsx`, which default-exports the React component. |

Fill in `id`, `name` for the picker, `author` for credits, `version` in
semver format, `api: 0` and `kind`. Set `license` to `MIT` for your submission.
Themes and Classic overlays both use `kind: "theme"`. Use `targets: ["studio"]`
for a theme or `["classic"]` for an overlay, and supply two to four hex colours
in `preview` for the picker swatch. Shells use `kind: "shell"` with no
`targets` or `preview`. The app discovers your folder automatically.

- **Theme.** Use the documented variables under your skin's dark and light
  selectors. Pick fonts the app already ships or system fonts. Themes cannot
  load assets.
- **Classic overlay.** Scope every selector to your skin ID and use system
  fonts. Images may be embedded as `data:` URLs. Keep the app's controls usable
  in light and dark mode.
- **Shell.** Build with React and the allowed imports. Render text through
  React and keep `innerHTML` and `dangerouslySetInnerHTML` out of the skin.
  Use `useTheme()` for system, light and dark modes. Scope your CSS to your shell
  and avoid global resets. Bundle fonts and assets in the skin folder and use
  relative URLs. Include the licence for each third-party asset, including
  `OFL.txt` for an OFL font. External URLs and hosted fonts are prohibited.
- **Shell data.** The server owns songs, drafts, presets and seeds. If an API
  export you need is missing, open an issue for it. Use `useSkinStorage()` for
  layout and preferences and keep raw `localStorage` out of the skin.
- **Shell networking.** Use `skin-api` for data and actions. Never use `fetch`,
  `XMLHttpRequest`, `WebSocket`, `EventSource` or `navigator.sendBeacon`, with
  or without `window.`, `globalThis.` or `self.`. `check:skins` flags direct
  calls. Human review covers obfuscated calls.

Skin API `0` is unstable, so update your skin against the current API before
submitting. See the full CSS rules, token list, allowed imports and available
API exports in the [skin reference](docs/dev/skins.md).

## Check and submit your skin

Run these from `ui/`:

```sh
npm run check:skins
npx tsc -b
```

Open a pull request from your fork. Add `?template=skin.md` to the GitHub
compare URL (or `&template=skin.md` if it already has query parameters) to
load the [skin PR template](.github/PULL_REQUEST_TEMPLATE/skin.md).
Keep every changed file inside your one skin folder. The scope check also
rejects changes to `README.md` or docs outside that folder. If you need changes
to the API, dependencies, configuration or another skin, ask for a separate
core review. The skin workflow checks the PR's file scope, the skin rules
and TypeScript.

Include screenshots in both light and dark mode and say which screens or
controls you tried. Reviewers check readability, CSS scope, asset licences and,
for shells, React and skin API use. Confirm the MIT licence line in the template.

For an update, keep the same ID and folder, bump the manifest version and open
another skin PR. Describe the changes and include updated screenshots. If the
skin has an `IMPORT.md`, record the new source drop and any changes needed to
fit HOT-Step there.

## Submitting a design instead of a skin

A design-only contribution is welcome. Open an issue with one self-contained
HTML prototype, including any vanilla JavaScript it needs, or screenshots
with notes. Describe which screens you designed, what the controls should do
and how the design should look in light and dark mode. Include any asset
licences and the same MIT agreement used for a skin PR:

> I agree my contribution is licensed under the repository's MIT licence.

We can port the design to a React shell and credit you in its manifest.
