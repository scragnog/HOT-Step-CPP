# Library, playlist, drafts and Song Builder for frontend clients

This page covers the library's songs, the playlist, saved studio drafts and Song Builder projects. The shared types are in [`contracts/songs.ts`](../../server/src/contracts/songs.ts), [`contracts/studioDrafts.ts`](../../server/src/contracts/studioDrafts.ts), [`contracts/studioDraftFields.ts`](../../server/src/contracts/studioDraftFields.ts) and [`contracts/songBuilder.ts`](../../server/src/contracts/songBuilder.ts). Every path is listed in the [route index](api.md); importing audio files and exporting songs are in [frontend-media.md](frontend-media.md).

Routes marked "token" need `Authorization: Bearer <token>` from `GET /api/auth/auto` and answer `401 { error: 'Unauthorized' }` without it.

## Songs

A song on the wire is its database row ([`Song`](../../server/src/contracts/songs.ts), fields in [`SONG_FIELDS`](../../server/src/contracts/songs.ts)) with two columns decoded: `tags` is an array and `is_public` a boolean. Everything else is sent as stored:

- `generation_params` and `metadata_overrides` stay JSON strings;
- `created_at` is SQLite UTC text (`YYYY-MM-DD HH:MM:SS`);
- empty media fields are `''`.

Display rules (camelCase, fallback titles, artist names) are each client's own.

Media identity: a song is its `id`. Its audio is the relative URL in `audio_url`, with variants in `mastered_audio_url` and `noadapter_audio_url`, and `latent_url` and the stem and disco URLs alongside. Prefix the server origin to fetch any of them.

| Call | Returns |
|---|---|
| `GET /api/songs[?source=]` (token) | `{ songs }`, the user's songs, newest first (`created_at` descending), not paginated. `source` matches `generation_params.source`, for example `create`, `lyric-studio` or `cover-studio`. |
| `GET /api/songs/ids` (token) | `{ ids }`, to prune local references to deleted songs. |
| `GET /api/songs/recent[?source=&limit=50]` (token) | `{ songs }` in the normalised [`RecentSong`](../../server/src/contracts/songs.ts) shape: newest first, at most `limit`. `source=all` or none means every source. Adds `source`, `artist_name`, `artist_image`, `album` and `generation_id` from Lyric Studio where known. |
| `GET /api/songs/:id` | `{ song }`; 404 when not found. This route does not check the token. |
| `POST /api/songs` (token) | `{ song }`. Creates a row from the given fields (`id` optional). |
| `PATCH /api/songs/:id` (token) | `{ song }`. Writes only [`SONG_EDITABLE_FIELDS`](../../server/src/contracts/songs.ts), plus `tags` (an array) and `metadata_overrides` (an object; `null` or `''` clears it). Any other field is ignored. 404 for a song that is not the user's. There is no revision check: the last write wins. |
| `DELETE /api/songs/:id` (token) | `{ success: true }`; also deletes the song's files. 404 when not found. |
| `DELETE /api/songs` (token) | `{ success: true, deletedCount }`: deletes every one of the user's songs and their audio files. |
| `POST /api/songs/bulk-delete` (token) | Takes `{ ids }` and returns `{ success: true, deletedCount }`. An empty `ids` answers 400. |
| `POST /api/songs/import` (token) | Multipart `audio` (up to 50 files) and an optional `description`. Returns `{ songs, errors, accepted }`: 200 when at least one file imported, 400 with the same `errors` when none did. |

## Playlist

The playlist is one revisioned document per user, under `/api/studio-drafts`, with a token on every route. Items keep any extra fields a client stores ([`playlistItemSchema`](../../server/src/contracts/studioDrafts.ts) passes them through). An item `id` may point at a song that no longer exists, or at a queue-only render: the playlist never checks.

- `GET /api/studio-drafts/playlist` returns `{ document }`. It is `null` before the first change, which means revision 0.
- `POST /api/studio-drafts/playlist/commands` takes `{ expectedRevision, command }` and returns `{ document }` at the next revision. The command is one of ([`playlistCommandSchema`](../../server/src/contracts/studioDrafts.ts)):

| `operation` | Fields | Effect |
|---|---|---|
| `add` | `item` | Appends it. An id already present changes nothing, but the revision still moves. |
| `remove` | `id` | Removes it. An unknown id is ignored. |
| `clear` | none | Empties the playlist. |
| `reorder` | `ids` | The new order. It must name every current item exactly once, or the command answers `400 { error: 'Reorder must name every current item exactly once' }`. |
| `update` | `id`, `patch` | Merges `patch` into the item, keeping its `id`. An unknown id is ignored. |

A stale `expectedRevision` answers `409 { error, currentRevision }` and changes nothing. To retry, the client reads the playlist again, reapplies its edit to the current items, and sends it with the new revision. Nothing is retried automatically. A malformed command answers 400.

## Studio drafts

A draft saves a studio's form so another client, or the same one later, can pick it up. Drafts cover the studios `create`, `cover`, `repaint`, `storm`, `stem-studio` and `stem-builder`. Draft documents are at schema version 1.

A draft body is `{ studio, fields, backendId?, sourceAssetId?, sourceSongId?, sourceRevision? }`, plus any extra top-level fields a client keeps. These are stored as given.

**Fields.** `fields` keeps the browser storage names the bundled UI uses, so every client reads the same draft. [`STUDIO_DRAFT_FIELDS`](../../server/src/contracts/studioDraftFields.ts) lists every key by studio, with the value type the server accepts: `string`, `boolean`, finite `number`, `number|null`, `string|null` or `object|null`. Create also accepts the caption-source keys matching `CREATE_CAPTION_SOURCE_KEY`.

A save answers 400 when:

- a key belongs to another studio or to none;
- a value has the wrong type;
- `sourceAssetId` or `sourceSongId` disagrees with the matching field.

The server never interprets a value. Selection pointers, playback state and device handles are never draft fields.

| Call | Returns |
|---|---|
| `GET /api/studio-drafts/drafts[?studio=]` | `{ documents }` |
| `POST /api/studio-drafts/drafts` with `{ body }` | `201 { document }` at revision 1. Sending `expectedRevision` here answers 400. |
| `GET /api/studio-drafts/drafts/:id` | `{ document, sourceError }`. `sourceError` names a source asset that no longer exists; the draft still loads, and the user reuploads the source. |
| `PUT /api/studio-drafts/drafts/:id` with `{ body, expectedRevision }` | `{ document }` at the next revision. A missing `expectedRevision` or a different `studio` answers 400; a stale one answers `409 { error, currentRevision }`. |
| `DELETE /api/studio-drafts/drafts/:id?expectedRevision=N` | `{ removed: true }`. A missing revision answers 400; a stale one answers 409. |

**Source staleness.** A draft's source is its `sourceAssetId`, `sourceSongId` and `sourceRevision`, plus the source fields in [`DRAFT_SOURCE_FIELDS`](../../server/src/contracts/studioDraftFields.ts). When a save changes any of them, the server drops the results computed from the old source, listed in `DRAFT_SOURCE_RESULT_FIELDS`:

- Cover's analysis, metadata, artist caption and lyrics source;
- Create's caption sources.

The user's own text and settings are kept.

**Importing browser storage.** `POST /api/studio-drafts/import` takes `{ storageKey, raw, sourceHash: 'sha256:<hex of raw>', schemaVersion: 1, expectedRevision, resolution? }` and moves one browser storage value into a draft or the playlist. It answers `{ receipt, document, created }`.

- Repeating the same `storageKey` and hash returns the first import.
- Different content under an imported key answers 409 until `resolution` is `keep-both` or `replace`.
- A new draft import needs `expectedRevision: 0`.
- The playlist key `lireek-playQueue` needs `replace` when a playlist already exists.

**Handoffs between studios.** `GET /api/studio-drafts/handoffs/:id` reads a Cover draft or a training audition sent to Create, by the id the sending studio returned. The original revisioned document stays the workflow's input. Any other document id answers 404.

## Song Builder

A project is one song built from an ordered chain of sections ([`contracts/songBuilder.ts`](../../server/src/contracts/songBuilder.ts)). Each section renders `variant_count` candidate songs. Choosing one makes it the source that the next section extends or leads into. All routes are under `/api/builder` and need a token.

**Projects.**

- `POST /projects` and `PATCH /projects/:id` take [`BuilderProjectFields`](../../server/src/contracts/songBuilder.ts) in camelCase and answer the [project view](../../server/src/contracts/songBuilder.ts) `{ project, sections }`. Project fields are stored snake_case.
- A project edit takes no revision: the last write wins per field. It still moves `project.revision`, so section commands based on the old settings are refused.
- `GET /projects` returns `{ projects }`, newest-updated first, each with `section_count`.
- `GET /projects/:id` returns the view.
- `DELETE /projects/:id` returns `{ ok: true }`. Candidate songs stay in the library.

**Sections.** The view lists sections by `position`, then creation time. Each carries its candidate ids, its resolved `candidates` (songs deleted since are dropped from the list) and its `chosen` song.

`status` is:

- `generating` while its job runs;
- then `ready` (some candidates) or `failed` (none);
- `chosen` once a candidate is picked.

A section whose job ended elsewhere (cancelled, or interrupted by a server restart) is settled on the next read. Nothing reruns by itself.

**Commands.** Every section command carries `expectedRevision`, the project revision the client last saw. A stale one answers `409 { error, currentRevision }`; reload the project and decide again. Each accepted command moves the revision by one and answers the whole project view.

| Command | Body | Notes |
|---|---|---|
| `POST /projects/:id/sections/generate` | [`generateSectionSchema`](../../server/src/contracts/songBuilder.ts) | Adds `{ jobId, sectionId }` to the view. Details below. |
| `POST /sections/:id/choose` | `{ songId, expectedRevision }` | `songId` must be one of the section's candidates (409 otherwise). Stops its remaining renders, and fills the project's BPM and key from the chosen song when they are unset. |
| `POST /sections/:id/stop` | `{ expectedRevision }` | Keeps the candidates that landed and stops the rest. |
| `PATCH /sections/:id` | `{ label?, lyrics?, expectedRevision }` | Non-string values answer 400. |
| `DELETE /sections/:id?expectedRevision=N` | none | Also stops its renders. Any later command on it answers 404. |

**Generating a section.** `direction` is `first`, `append` or `prepend`:

- `length` is `{ bars }`, which needs the project BPM, or `{ seconds }`.
- `overlap` is the seconds of existing audio the new section overwrites at the seam.
- `clipPoint` is where an append extends from, or where a prepend connects. `null` means the end, or the start.

Node works out the geometry from the stored project and its chosen sections, so the client sends no positions.

`idempotencyKey` makes a submit safe to repeat:

- The same key and body return the original job and section, even after the revision has moved on.
- The same key with a different body answers 409.

`expectedBackend` must still be the active engine, or the command answers 409 before anything is created. Follow the job with the workflow job routes, or by reading the project; candidates appear as they land.

Errors from Song Builder are `{ error, currentRevision?, issues? }`. A malformed generate body answers 400 with `issues`.
