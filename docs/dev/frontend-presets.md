# Preference presets for frontend clients

The preference API is installation scoped and does not require an auth token. Its named preset families are `vst-chain`, `ai-continue-style`, `ai-continue-lyric` and `yue2-joint`. `ai-continue-template` and `storm-tuning` are singleton settings families. See the [shared schemas](../../server/src/contracts/preferences.ts) and [routes](../../server/src/routes/preferences.ts) for the exact bodies.

Named presets use `GET /api/preferences/presets/:family` returning `{ documents: [...] }`, `POST /api/preferences/presets/:family` with `{ body }` returning `{ document }` (201), `PUT /api/preferences/presets/:family/:id` with `{ expectedRevision, body }` returning `{ document }`, and `DELETE /api/preferences/presets/:family/:id?expectedRevision=N` returning `{ removed: true }`. Each document carries its id, raw body and revision. A stale revision returns 409 with `currentRevision`; unknown families return 404. Invalid requests return 400 with `{ error, issues: [{ path, message }] }`. The server preserves YuE2 preset settings as stored, including its body version.

`POST /api/preferences/presets/:family/import` takes `{ items }` and returns `{ results }`. Each item needs its source `storageKey`, `sourceHash` and raw `body`; a name collision returns an item outcome of `name-conflict`. Retry with that item's explicit `resolution` of `replace` or `keep-both`. Import does not apply a preset. Singleton settings use `GET` and `PUT /api/preferences/settings/:family`, plus `POST /api/preferences/settings/:family/import`; `GET` returns `{ document: null }` before creation. Updating an existing singleton needs `expectedRevision`.

## Applying a YuE2 joint preset

Send the raw preset and the current form to `POST /api/preferences/presets/yue2-joint/resolve`:

```json
{
  "preset": { "name": "Earlier recipe", "version": 1, "settings": { "rank": 64 } },
  "currentForm": { "dataset": "current-dataset", "checkpoint": "current-checkpoint", "output": "current-output", "adapterType": "lokr" },
  "lyricTiming": true
}
```

The response is `{ "result": { "effectiveForm": { ... }, "lyricTiming": true } }`. Apply both returned values together. Resolution starts with `currentForm`, then sets the legacy LoRA adapter, LoRA stop defaults and `cautious: false`, then applies stored settings last. Stored overrides therefore win. Fields absent from settings, including per-run paths, retain their current form values. Only a version 2 preset with a boolean `settings.lyricTiming` changes lyric timing; older, missing-version and absent timing values retain the submitted value. The endpoint writes nothing and returns 400 with field issues for an invalid request. A transport or server error leaves the form unchanged in the bundled UI.

The named preset routes continue to return raw bodies. Keep those bodies for editing and export, and use this resolve result only to populate the effective form. The rule applies only to YuE2 joint presets; other families have no compatibility resolver.
