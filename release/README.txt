HOT-Step CPP
============

QUICK START
Extract the complete archive. Windows: double-click HOT-Step.bat.
Linux or macOS: run ./HOT-Step.sh.
Open http://localhost:3001 if the browser does not open automatically.
Download models from the Model Manager.

DATA AND UPGRADES
User data defaults to HOT-Step-Data beside the app folder. It includes library,
audio, training, models, adapters, logs, tools and settings/.env. Engine binaries,
UI, bundled runtime, shipped plugins and server/data catalogues stay in the app.

Your profile saves the data location in data-location.json:
  Windows: %LOCALAPPDATA%/HOT-Step
  Linux: $XDG_CONFIG_HOME/hot-step or ~/.config/hot-step
  macOS: ~/Library/Application Support/HOT-Step

Stop the app, back up the data folder, and extract the next version separately.
The new launcher reuses the saved location and downloaded weights.
With no pointer, a new version started in an older install's folder (extracted
over it) moves that install's data to the sibling folder on its first start.
A fresh extraction elsewhere cannot find the old app folder: before creating
anything in it, use Import previous install in Settings (it needs an empty library).

Settings > Storage & Data provides Move current data, Use existing folder,
Create empty data folder and Import previous install. Preview the paths and
space required, finish active work, then confirm and restart. External folder
overrides stay external. Import needs an empty library and moves the older
install's library, models, adapters, logs and Whisper into the data folder; the
older app folder stays in place without them. Repoint copies and deletes nothing.

If a drive is missing or a transfer stops, keep the pointer and data intact.
Read the explanation page, fix the reported problem and restart to resume.

IN-APP UPDATES (Windows packages with an app folder and current.json)
Settings > Updates downloads and installs new releases. HOT-Step.bat installs
the new version beside the old one in app, checks that it starts, and keeps it.
If it does not start, the old version comes back with your data as it was.
Always start HOT-Step.bat; never start anything inside app directly. If the
window says recovery is required, nothing was started or deleted: the reason
is in updater/last-result.json.

CONFIGURATION
Settings writes <data folder>/settings/.env from the shipped .env.example.
Leave managed paths unset. A process HOT_STEP_DATA_ROOT override selects another
existing folder for that launch.

REQUIREMENTS
Use the platform and GPU variant matching your system. Node.js is bundled.
Models require additional disk space and internet for the first download.
Linux ROCm builds require compatible system ROCm libraries.
If macOS blocks the unsigned app: xattr -cr /path/to/extracted-app/

THIRD-PARTY SOFTWARE
Licence texts and notices are in server/data/licenses/ (inside the app folder).
- SoX Resampler Library (libsoxr 0.1.3), LGPL-2.1: statically linked into the
  ace-midi engine (MIDI Studio, dettato). HOT-Step is MIT, which lets you modify
  it and reverse-engineer it for debugging your changes. Complete source of the
  libsoxr used (engine/vendor/soxr, unmodified apart from soxr-config.h) and of
  the rest of the app, with the build files to relink ace-midi, is in
  https://github.com/scragnog/HOT-Step-CPP at the tag named in this release's
  notes. Licence: server/data/licenses/libsoxr-COPYING.LGPL.
- dettato by KobiMusic (Apache-2.0), the engine port and its model weights:
  server/data/licenses/dettato-LICENSE and dettato-NOTICE.

Documentation and support: https://github.com/scragnog/HOT-Step-CPP
