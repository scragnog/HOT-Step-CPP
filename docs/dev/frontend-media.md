# Media for frontend clients: import, export and live streams

This page covers moving audio in and out of the server: uploading and importing files, resolving and downloading exports, validating an imported profile, and reading the two live audio streams. The shared types are in [`contracts/exportImport.ts`](../../server/src/contracts/exportImport.ts), [`contracts/streaming.ts`](../../server/src/contracts/streaming.ts) and [`contracts/streamSessions.ts`](../../server/src/contracts/streamSessions.ts). Every path is listed in the [route index](api.md).

All URLs the server returns are relative (`/api/download/…`, `/audio/…`). A client on another origin prefixes the server origin itself. Routes marked "token" need `Authorization: Bearer <token>` from `GET /api/auth/auto` and answer `401 { error: 'Unauthorized' }` without it.

## Importing audio

A client never sends a file path. The file goes up as a multipart upload, which returns an asset id, and the import takes asset ids.

1. **Upload** (token): `POST /api/export-import/assets`, `multipart/form-data` with one file in the field `audio`.
   - The extension must be one of [`IMPORT_AUDIO_EXTENSIONS`](../../server/src/contracts/exportImport.ts) (`.wav`, `.mp3`, `.flac`, `.m4a`, `.mp4`, `.aac`, `.ogg`, `.opus`, `.webm`, `.aiff`, `.aif`), and the file at most 500 MiB.
   - Success: `200 { assetId }`.
   - A missing file or another extension: `400 { error: 'Unsupported or missing audio file' }`.
   - The server stores the file under a new name in its references folder; the client's file name is kept only as the asset's display name.
2. **Import** (token): `POST /api/export-import/imports` with `{ items: [{ assetId, description? }] }`, 1 to 50 items.
   - The response is `200 { items }`, one entry per request item in order ([`ImportResultItem`](../../server/src/contracts/exportImport.ts)).
   - Each entry has either `song`, the new library row, or `error`. One bad item does not stop the others, so check every entry.
   - An asset the caller does not own, or whose file is gone, fails its item only.
   - Unknown fields, a non-uuid `assetId` or more than 50 items answer `400 { error: 'Invalid import request', issues }`.
   - The import converts the audio to the library's WAV shape (stereo 16-bit). The uploaded file is left in the references folder: there is no delete route, and the server does not clean it up.

## Exporting audio

There is no export job to poll. A client resolves what to download, then downloads each returned URL.

1. **Resolve** (token): `POST /api/export-import/exports/resolve`. The body is [`exportRequest`](../../server/src/contracts/exportImport.ts):

   ```json
   {
     "items": [{ "songId": "…" }, { "songId": "…", "variant": "mastered" }],
     "format": "flac",
     "downloadVersion": "both",
     "includeLatent": false,
     "bitrate": 192,
     "artist": "…",
     "prepend": "01"
   }
   ```

   - `items`: 1 to 100.
   - `format`: `wav`, `flac` (the default), `mp3` or `opus`.
   - `bitrate`: kbps, 32 to 512, used for MP3 and Opus.
   - `downloadVersion`: `original`, `mastered` (the default) or `both`. It applies to items without an explicit `variant`. `mastered` falls back to the original when a song has no master; `both` gives two entries when it has one.
   - `includeLatent`: adds the latent file as a separate entry.
   - `audioUrl` or `srcUrl` on an item lets a render that exists only in the queue (not yet a library row) be found by its audio file.
2. The response is `200 { items }` ([`ResolvedExport`](../../server/src/contracts/exportImport.ts)). An input item can produce several entries, all carrying its `index`.
   - A resolved entry has `variant`, `filename` and a relative `url`.
   - A failed entry has `error` and no `url`: the song is not found, the variant is unavailable, or its file is missing.
   - A malformed body answers `400 { error: 'Invalid export request', issues }`.
3. **Download**: `GET` each `url`. It points at `/api/download/:id`, which needs no token, and the response is binary:
   - `Content-Type`: `audio/wav`, `audio/flac`, `audio/mpeg`, or `audio/ogg` for Opus. A latent is `application/octet-stream`.
   - `Content-Disposition: attachment; filename="…"`. The name follows the same rule the resolve response's `filename` shows.
   - `Content-Length`.

   Errors come back as JSON instead: 400 for a bad format, 404 for a missing song or file. A non-WAV format, or a WAV with tags to embed, is transcoded into a temporary file. The server deletes that file once it has been read to the end. A download abandoned midway can leave it in the server's `download_temp` folder.

## Importing a profile file

`POST /api/export-import/profiles/validate` (token) with `{ filename, profile }`, where `profile` is the parsed JSON object from the user's file.

- The server accepts either a saved profile wrapper (`{ name, saved_at, data }` whose `data._format` is `hot-step-preset`) or bare preset JSON.
- It names the profile after the file, without `.json` and with characters a filename cannot hold replaced by `_`.
- It answers `200 { name, data }` ([`ProfileImportResult`](../../server/src/contracts/exportImport.ts)). Save `data` under `name` with the profile routes.
- An empty profile, a non-object or unknown top-level fields answer `400 { error: 'Invalid profile', details }`.
- Validation writes nothing.

## Live audio streams

Two routes stream audio while it renders. Each response body is complete RIFF/WAV files concatenated end to end, with no other framing.

| Stream | Request | One WAV is | Ends when |
|---|---|---|---|
| STORM | `POST /api/generate/storm/stream` with [`StormStreamStart`](../../server/src/contracts/streaming.ts) | one generated slot | the client stops it, disconnects, or a slot fails or times out |
| MM3 | `GET /api/generate/mm3/stream/:jobId?take=N` | one rendered window | the take's render finishes (EOF) |

The response is `Content-Type: audio/wav`, and the stream's Node session is named in the `X-Stream-Session` header ([`STREAM_SESSION_HEADER`](../../server/src/contracts/streaming.ts)). The header is absent when Node could not open a session; the audio is unaffected.

**STORM.**

- Start: the body is any Create generation parameters plus `streamId` (default `'default'`, one per DJ deck), `seed`, `coResident` and `pluginParams`. Before any audio, the route answers `503` while the engine is not ready and `409` while generation jobs are active.
- Live changes: `POST /api/generate/storm/control` with [`StormControl`](../../server/src/contracts/streaming.ts) and the same `streamId` applies them from the next slot. It returns `{ ok: true, streamId }`.
  - `next_*`, `seed`, `prompt` and `lyrics` apply to one slot.
  - `stick_prompt` and `stick_lyrics` persist until set to `null`.
  - `stream_pause: true` holds generation, for when the player's buffer is full.
  - Fields of the wrong type are ignored.
- `GET /api/generate/storm/control?streamId=` reads the live state back.
- Stop: `POST /api/generate/storm/stop` with `{ streamId }` cancels the slot in flight and ends the response. Closing the connection does the same.
- Errors after audio has started have no error frame: the response just ends.

**MM3.**

- Submit a generation that streams. Poll `GET /api/generate/status/:id` until `mm3_streaming` is true ([`Mm3StreamStatus`](../../server/src/contracts/streaming.ts)), then open one stream per take, `take` from 0 to `mm3_takes - 1`.
- `mm3_duration` gives the full length before any audio arrives.
- Errors:
  - 404: the job is unknown.
  - 409: the job has not reached the engine yet (poll and reopen), is not streaming, or the engine refused. The engine refuses when the take has finished or already has a reader; it allows one reader per take.
  - 502: the engine stream failed.
- Closing the connection drops the engine connection too. The render itself carries on as a normal job.

### Reading the frames

Read the body as bytes and split it on the RIFF size field: each WAV is `uint32le` at byte 4, plus 8 bytes. [`WavFrameReader`](../../server/src/contracts/streaming.ts) is the reference implementation, and Node's own copy uses the same rule:

```ts
import { WavFrameReader } from '…/contracts/streaming';

const reader = new WavFrameReader();
const body = (await fetch(url, init)).body!.getReader();
for (;;) {
  const { done, value } = await body.read();
  if (done) break;
  for (const wav of reader.push(value)) play(wav);   // zero, one or several per read
}
const { truncatedBytes } = reader.end();             // a WAV cut off by stop or disconnect
```

- A read may hold part of a WAV, several WAVs, or a WAV split anywhere, including inside its 44-byte header. A WAV comes out only once all its bytes are in.
- At the end of the stream (EOF, stop or disconnect), bytes of an incomplete WAV are not playable; discard them.
- Bytes before a `RIFF` marker, and a `RIFF` whose size is impossible (under 44 or over 500 MB), are skipped so the reader resynchronises.
- The reader holds at most one incomplete WAV.

[`wavFraming.test.ts`](../../server/src/contracts/wavFraming.test.ts) runs both readers against every split point, header-boundary splits, several WAVs per read, truncation and resynchronisation.

### What Node does and what the player does

Node forwards the engine's WAVs byte for byte and keeps its own copy in the stream session. Crossfades between STORM slots, scheduling, gapless splicing of MM3 windows, volume and device capture (recording what the speakers play) all happen in the player and never reach Node. So the two kinds of recording differ:

- **Canonical recording**: the stream session records the engine's samples between an explicit start and stop. For MM3 that is the song as rendered; for STORM, the slots are joined end to end without the player's crossfades. Control it with `POST /api/stream-sessions/:id/recording` (`start`, `stop`, `discard`), then download it with `GET /api/stream-sessions/:id/recording/export?format=wav|flac|mp3|opus`.
- **Device capture**: what a client records from its own audio output, including its mix and crossfades. It stays in the client.

Session lifetime, limits, BPM/key analysis and error codes are in [Stream sessions](api-contracts.md#stream-sessions). Briefly:

- A session ends with its stream and is deleted 30 minutes later.
- At most two recording exports run at once; another gets `429`.
- A reconnect opens a new stream and a new session. Recordings do not carry across.
