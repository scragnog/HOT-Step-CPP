# Data folder and upgrades

Since v1.4.1, release builds keep your library, downloaded models, adapters, settings and tools in one data folder outside the app. By default this is `HOT-Step-Data` beside the extracted app folder. Your user profile records its location, so a new version uses the same data without downloading the models again.

## What lives there

| Folder | Contents |
|---|---|
| `library/` | Song database, audio, references, stems and caches, VST state, lyrics, training datasets and runs |
| `library/muscriptor/` | MuScriptor tokens and preserved optional-install files |
| `library/models/muscriptor/` | MuScriptor weights, unless overridden |
| `models/`, `adapters/` | Downloaded model weights and adapters |
| `settings/.env` | Saved settings and service credentials |
| `logs/` | Session, generation and training logs |
| `tools/whisper/`, `tools/runtime/` | Downloaded Whisper executable and runtime libraries |
| `plugins/` | Your Lua plugins, overriding shipped plugins with the same name |
| `tmp/`, `cache/` | Reserved working folders |
| `hot-step-data.json` | Marks the folder as a HOT-Step data folder |

Engine binaries, the bundled Node runtime, UI, shipped plugins, noise samples and `server/data/` catalogues stay in the app folder. MuScriptor uses the shipped `ace-midi` executable with weights from the data folder. Preserved Python environments may need rebuilding if an optional integration uses them after relocation.

## Upgrade to a new version

In **Settings > Updates**, Windows portable installs started by a compatible updater launcher offer **Download update**. Downloading verifies and stages the release without changing the running app. You can cancel it. When ready, finish running jobs, click **Restart and install** and confirm. The screen waits for the launcher to restart HOT-Step, shows the outcome and offers **Reload** for the new UI. If the new version fails and rolls back, the old version runs with its restored data. If recovery is required, follow the HOT-Step launcher window. After five minutes without a result, the screen points to that window and keeps checking.

Older releases, Linux and macOS require the manual upgrade below. The Updates panel explains why and links to the release. Source checkouts must close HOT-Step and run `update.bat` in the install folder. Portable installs follow releases only.

1. Stop HOT-Step and back up your data folder, including `settings/.env`.
2. Extract the complete new archive into a separate folder.
3. Start `HOT-Step.bat` on Windows or `./HOT-Step.sh` on Linux and macOS.
4. Check **Settings > Storage & Data > Data folder**. The saved location should be unchanged.

Releases before v1.4.1 kept their data inside the app folder. With no saved location, a new version extracted over such an install carries that install's data into the sibling `HOT-Step-Data` folder the first time `HOT-Step.bat` or `HOT-Step.sh` starts it. If that move stops, the next start resumes it. Conflicting populated destinations stop startup instead of merging libraries.

A fresh extraction elsewhere cannot find an arbitrary older app folder. Run **Import previous install** in Settings first, before you create any songs or settings in the new library, and provide the older app folder. Import needs an empty current library and never merges libraries. It moves the older install's library, models, adapters, logs and Whisper into the data folder, and copies its own Lua plugins and downloaded runtime libraries. The older app folder stays in place without that data, and its `.env` is renamed `.env.migrated`. Its settings become `settings/.env`; a `settings/.env` the data folder already had is kept as `settings/.env.before-import-<id>`.

## Choose or move a folder

Open the **Data folder** panel on the **Storage & Data** tab of Settings, choose an action and enter an absolute path:

- **Move current data** transfers managed data into an empty destination.
- **Use existing folder** selects an existing data folder, one that holds `hot-step-data.json` or `library/hotstep.db`. It copies and deletes nothing.
- **Create empty data folder** starts an empty library in an empty destination while retaining your current folder.
- **Import previous install** moves an older install's data into the current empty library, as described above. Enter the older app folder, not its data subfolder.

The data folder cannot be inside the app folder. On Windows installs with an updater launcher, this means the whole extracted folder containing `HOT-Step.bat`, including locations outside its `app/` subfolder. Settings also refuses a folder that another install recorded in your profile already uses, such as the release app or a development checkout.

Click **Preview change** to see the folders, size and space required. Finish queued and active work and downloads, then click **Confirm and restart**. Transfers run before the library opens; the launch console shows progress. Same-volume moves rename folders. Across volumes, HOT-Step copies and checks file sizes before removing verified source folders. External path overrides remain outside the transfer.

If restart fails, the change remains scheduled. Use **Restart to apply**, or **Cancel scheduled change** before transfer begins. A process `HOT_STEP_DATA_ROOT` override must be cleared before Settings can change the saved location.

## Development checkouts

A development checkout started with `dev.bat` or `dev.sh` uses its own folders until you change them in this panel. Nothing moves at startup. **Move current data** carries the checkout's library, models, adapters, logs, Whisper and `.env` into the destination. It refuses to run when one of those folders holds files git tracks, or leads outside the checkout, so tracked files never move. The checkout's choice is saved in its own `data-location.dev-<hash>.json` in the profile folder below, separate from the release app's `data-location.json`. Settings will not let a checkout and a release share a data folder. Deleting that file points the checkout back at its own folders; data that already moved stays where it is.

## Recover a missing drive or interrupted transfer

The profile pointer is `data-location.json` in:

| Platform | Profile folder |
|---|---|
| Windows | `%LOCALAPPDATA%\HOT-Step` |
| Linux | `$XDG_CONFIG_HOME/hot-step`, or `~/.config/hot-step` |
| macOS | `~/Library/Application Support/HOT-Step` |

Restart state and the transfer journal stay in this profile even while data moves. If the saved data folder is missing or inaccessible, HOT-Step starts with the default `HOT-Step-Data` beside the app. The launch console warns with the saved and fallback paths. This may show an empty library. The saved location stays unchanged, and no data is moved or deleted; reconnect the drive and restart to use it again. A development checkout falls back to its original checkout folders. An explicit `HOT_STEP_DATA_ROOT` override still takes precedence; an invalid override stops startup.

A malformed pointer, a failed transfer, a transfer that cannot start while something still listens on the engine port, or an unsuitable fallback folder shows an explanation page headed "HOT-Step cannot open its data folder" instead of the app, and keeps the saved state. Fix the reported problem and restart. An interrupted transfer resumes. Follow the error page's instructions to abandon an unfinished same-volume move. Keep the pointer and data intact until recovery is complete.

Browser preferences stay in browser localStorage for that address. They do not move with the data folder. Back up the profile pointer separately if you need to restore the saved location on another machine.

## Related

- [Settings](studios/settings.md)
- [Model files](models.md)
- [Configuration for contributors](../dev/config.md)
