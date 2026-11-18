# Settings and Lyric Studio backup

Export JSON saves application settings and Lyric Studio data in one file. Keep this
file to restore your setup later or move Lyric Studio to another installation.

The file contains the General and Performance application settings, Lyric Studio
prompt preferences, artists, lyric collections, profiles, generated lyrics, album presets,
and the database links to generated audio. It uses version 1 of the backup format,
with `version`, `exportedAt`, `settings`, and `lyricStudio` fields.

## Export

In Settings, choose **Export JSON** in the backup panel. Your browser downloads a
dated JSON file. This is a copy of the selected settings and data, rather than a
copy of the whole database.

If a saved application setting is missing or has an invalid value, for example from an
older install, the export writes that setting's default instead of failing. Import stays
strict and rejects a file with an invalid setting.

## Import

1. Export your current data first if you want to keep it.
2. Choose **Choose JSON to import** and select a backup file smaller than 500 MB.
3. Review the current and incoming record counts. Import replaces all application
   settings and all Lyric Studio records, including records added after the preview.
4. Choose **Replace settings and Lyric Studio data**, then confirm **Replace data**.

Selecting a file and reviewing its counts do not change anything. Invalid files
and unsupported versions are rejected with a message. Database replacement runs
as one transaction, so a database error rolls back the import. Application settings
apply in the browser after the server confirms the import.

## This is not a backup of your data folder

The JSON file covers settings and Lyric Studio records only. To keep everything else,
copy the whole data folder while HOT-Step is stopped. API keys are in `settings/.env` and
the MuScriptor Hugging Face token is in `library/muscriptor/hf_token`; both are inside that
folder and are not in the JSON file. Treat a copy of the folder as secret.

To move your workspace to another machine, use **Workspace archive** in Settings > Storage & Data
instead of copying by hand: it writes one zip of the library and, if you choose, training data,
adapters, models and secrets. Models and secrets are off by default.

Upgrading does not need a backup restore: extract the new version and start it. It finds
the same data folder through your profile. See [Data folder and upgrades](data-folder.md).

## What stays outside the backup

Audio and image files, model weights, adapter weights, reference tracks, training
data, library songs, playlists, and accounts are not copied. Environment settings
and API keys in `.env`, theme, language, and other browser preferences are also
outside this format. Their existing values are kept on import.

Only custom Lyric Studio prompts, stored as `prompt_*`, and the Inspire system
prompt, stored as `instagen_system_prompt`, are included from the server's settings
table. The active backend, model selections, adapter folders and slots, and training
defaults are kept on the destination machine. Files that include other server
setting keys are rejected during preview and import.

Audio links and preset paths retain their original values. When moving to another
machine, copy their files separately and update paths as needed. Training collections
are kept as they are, so check any links to replaced Lyric Studio collections.
