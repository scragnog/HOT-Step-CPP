# Recovery after an update

This page is for Windows portable installs that update from **Settings > Updates**. If an
update goes wrong in a way HOT-Step cannot settle on its own, the HOT-Step window (the console
that `HOT-Step.bat` opens) stops with `Recovery required` and starts nothing. The browser tab
cannot help at that point: no server is running. This page explains what each message means
and what to do.

Paths below are relative to the install folder, the folder that holds `HOT-Step.bat`. "Data
folder" means your HOT-Step data folder, `HOT-Step-Data` beside the install unless you moved
it; see [Data folder and upgrades](data-folder.md).

## What is kept

Nothing is deleted when recovery is required. These files are what the launcher uses to
recover, so leave them in place:

| Path | What it is |
|---|---|
| `updater/last-result.json` | The outcome and the message the window showed |
| `updater/transaction.json` | The unfinished update. The launcher reads it at every start |
| `updater/maintenance-lease.json` and `.hot-step-maintenance.json` in the data folder | Keep any server from opening your databases until the update is settled |
| `updater/snapshots/<id>/` | Your databases and settings as they were just before the update |
| `updater/snapshots/<id>/candidate/` | A copy of what the new version left, made when a restore starts copying. Absent if the snapshot failed its check first |
| `app/<old version>/` and `app/<new version>/` | Both versions of the app |

Do not delete or edit any of them. Deleting `transaction.json` or a lease file does not
fix anything: the launcher would then start a version on data it was not checked against.

## First step, for every message

1. Close the HOT-Step window.
2. Close anything that may hold HOT-Step files open: backup or sync tools (OneDrive, Dropbox
   and the like) and any antivirus scan in progress. If your antivirus quarantined a HOT-Step
   file, restore it and allow the install folder; see
   [Antivirus removes files](troubleshooting.md#antivirus-removes-files).
3. Check that the drive holding the install folder and the drive holding the data folder are
   both connected and have free space.
4. Start `HOT-Step.bat` again.

Each start retries the recovery from the beginning, and a retry never changes the snapshot.
If the cause was a locked file, a full disk or a missing drive, this is usually enough. If the
window stops again with the same message, read on.

## Before anything else: copy

Before you try more than the first step, copy these to another drive or folder:

- the whole `updater` folder from the install folder;
- from the data folder, `settings/.env`, and every file in `library/` whose name starts with
  `hotstep.db` or `lireek.db` (the databases and the `-wal` and `-shm` files beside them).

If you set `DATA_DIR` in `settings/.env`, the databases are in that folder instead. The
snapshot and its restore cover only these databases and settings files; audio, models and
adapters are not part of them.

## The update was interrupted

The message starts with `An update was interrupted`, `The new version could not be activated`
or `The new version failed its health check`, and goes on with `Restoring the data snapshot
failed`.

The update had not finished. HOT-Step switched back to the old version and tried to put your
databases and settings back from the snapshot, and that restore failed. Until it succeeds,
nothing starts, because the old version must not run on data the new version may have changed.

What the rest of the message says decides the next step:

- **A file is locked, access is denied, or the disk is full** (`EPERM`, `EBUSY`, `EACCES`,
  `ENOSPC`): fix that and repeat the first step. The restore starts again from the snapshot.
- **`snapshot file ... is damaged`**, **`does not match the transaction record`** or **`is not a
  format 1 snapshot manifest`**: the snapshot itself cannot be trusted, so a retry gives the same
  result. The restore stopped before it touched your data. There is no safe step to do on your
  own here. Make the copy above, then ask for help (below).
- **Anything else**: repeat the first step once. If the same message comes back, make the copy
  above and ask for help.

## The new version was installed but does not start

The message starts with `The new version was installed but does not start`.

The new version passed its checks and was kept, so your data is already in its format. HOT-Step
does not go back to the old version from here: the old version cannot safely open the newer data.
At each start the launcher tries the new version again.

1. Do the first step above.
2. If it stops again, open the newest session folder in `logs/` in the data folder and read the
   end of `node_console.log`. [Troubleshooting](troubleshooting.md) covers the common causes,
   such as [Port already in use](troubleshooting.md#port-already-in-use).
3. If that does not explain it, make the copy above and ask for help.

Do not point `current.json` back at the old version, and do not copy the snapshot over your
data by hand.

## Other messages

A `Recovery required` message that names `current.json`, `transaction.json` or a lease file
(`is missing`, `is not valid JSON`, `is not a valid version pointer`, `is not a valid update
record`) means one of the launcher's own files is damaged or was edited. Do not recreate it by
guesswork. Make the copy above and ask for help.

`Launcher error` means the launcher failed for a reason that is not one of the above. Do the
first step. If it repeats, ask for help.

## Asking for help

Open an issue on [GitHub](https://github.com/scragnog/HOT-Step-CPP/issues/new) or ask on the
[Discord](https://discord.gg/ezVtmg9GKX), with:

- the full text of the HOT-Step window, from `Recovery required` to the end;
- `updater/last-result.json` and `updater/transaction.json`;
- `node_console.log` from the newest session folder in `logs/` in the data folder, for the
  "does not start" case.

These files contain paths from your machine, which can include your user name. Edit them out
if you prefer. Keep everything listed under [What is kept](#what-is-kept) in place until the
problem is solved.

## Related

- [Data folder and upgrades](data-folder.md)
- [Troubleshooting](troubleshooting.md)
