// YuE2 cover transcription boundary. The existing training queue owns the job.
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { config } from '../config.js';
import { getDb } from '../db/database.js';
import * as queue from './training/labelingQueue.js';
import type { TrainingJob } from './training/labelingQueue.js';
import { runYue2AceTrain } from './training/yue2TrainRunner.js';
import { jobsDir } from './training/paths.js';
import { readDuration } from './training/audioMeta.js';
import { missingYue2SheetModels, readYue2CoverAbc, resolveYue2SheetModel, resolveYue2SheetRefinedModel } from './training/yue2Sheet.js';
import { runYue2CoverSheetJob } from './training/yue2ArTrainRunner.js';
import { resolveCoverDatasetSource, type CoverDatasetSource } from './coverDatasetSource.js';
import { getDataset } from './training/datasetsRepo.js';
import { patchLabel } from './training/labelStore.js';
import { readSidecar, writeSidecar } from './training/sidecarIO.js';
import { loadSidecarMetadata } from './training/datasetScan.js';
import { createCoverStemCache } from './coverStemCache.js';
import { NoVocalsError, separateOne, YUE2_STEMS_DEFAULT_LEVEL } from './training/yue2Stems.js';
import { yue2Align } from './backends/yue2/client.js';
import type { Yue2AlignWord } from './backends/yue2/align.js';
import { matchSectionsToScore, planScoreFromLyrics, suggestionLyrics, suggestSections } from './backends/yue2/sectionMatch.js';
import { prepareV2Problem, structureHash, type PrepareV2 } from '../contracts/yue2Prepare.js';
import { sidecarPathFor } from './training/paths.js';
import { readAbcSidecar, writeAbcSidecar } from './training/abcSidecar.js';
import { hasSoundingScore } from './backends/yue2/scoreHealth.js';
import { lintScoreSections, scoreSections } from './backends/yue2/scoreSections.js';
import { cachedMp3 } from './audioConvert.js';

const MAX_AUDIO_BYTES = 100 * 1024 * 1024;
const MAX_AUDIO_SECONDS = 10 * 60;
const MAX_ABC_LENGTH = 64 * 1024;
const AUDIO_EXT = new Set(['.wav', '.mp3', '.flac', '.ogg', '.opus', '.m4a', '.aac']);

export class CoverRequestError extends Error {
  constructor(message: string, public readonly status = 400) { super(message); }
}

// audioPath is the chosen file, for dataset matching; workPath is what the
// models read, a cached MP3 when the file is over MAX_AUDIO_BYTES.
export interface CoverSource { sourceId: string; sourceLabel: string; audioPath: string; workPath: string }
type SavedSource = Pick<CoverSource, 'sourceId' | 'sourceLabel'> & { userId: string; name: string };
type CoverInput = { sourceAudioUrl?: unknown; songId?: unknown; sourceLabel?: unknown; abc?: unknown; force?: unknown;
  /** 'sheetsage2' (default) or 'sheetsage2-refined'. */
  transcriber?: unknown;
  /** With the refined transcriber: keys with modes. */
  keyMode?: unknown };

// The source's vocal stem, separated once per audio content: MMS_FA places
// words reliably on a stem, not on the full mix, and a score with no sung
// melody takes its vocal rhythm from it.
const coverStems = createCoverStemCache({
  root: path.join(config.data.dir, 'tmp', 'cover-vocals'),
  level: YUE2_STEMS_DEFAULT_LEVEL,
  separate: (src, out, level) => separateOne(src, out, level, () => {}),
});

export function createYue2CoverService(deps = {
  queue,
  referenceDir: path.join(config.data.dir, 'references'),
  libraryDir: config.data.audioDir,
  jobRoot: jobsDir(),
  duration: readDuration,
  missingModels: missingYue2SheetModels,
  model: resolveYue2SheetModel,
  run: (job: TrainingJob, audioPath: string, dir: string, transcriber?: { refined: boolean; keyMode: boolean }) =>
    runYue2CoverSheetJob(job, audioPath, dir,
      { missingModels: missingYue2SheetModels, run: runYue2AceTrain, vocalStem: coverStems.vocalStem }, transcriber),
  refinedModel: resolveYue2SheetRefinedModel,
  datasetMatch: resolveCoverDatasetSource,
  toMp3: (file: string) => cachedMp3(file, path.join(config.data.dir, 'tmp', 'cover-mp3')),
  vocalStem: coverStems.vocalStem,
  align: (audio: Buffer, lyrics: string): Promise<{ words: Yue2AlignWord[] }> => yue2Align(audio, lyrics),
  // The same merged read as a sample edit, so split caption/lyrics files
  // are folded into <stem>.txt rather than shadowing the new lyrics. Like a
  // sample edit, the label then records these fields as set by the user.
  saveDetails: async (matched: CoverDatasetSource, fields: Record<string, string>) => {
    await writeSidecar(sidecarPathFor(matched.audioPath), { ...loadSidecarMetadata(matched.audioPath), ...fields }, { keepBackups: true });
    const slug = getDataset(matched.datasetId)?.slug;
    if (slug) patchLabel(slug, matched.sampleId, { sources: Object.fromEntries(Object.keys(fields).map(field => [field, 'user' as const])) });
  },
  // Keeps the score it replaces, like saveDetails does for the .txt.
  saveAbc: (matched: CoverDatasetSource, abc: string) => writeAbcSidecar(matched.audioPath, abc, { keepBackups: true, anySounding: true }),
  song: (id: string, userId: string) => getDb().prepare('SELECT audio_url, title FROM songs WHERE id = ? AND user_id = ?').get(id, userId) as { audio_url: string; title: string } | undefined,
}) {
  function readiness() {
    const missing = deps.missingModels();
    const refinedModel = deps.refinedModel?.() ?? '';
    return { ready: missing.length === 0, model: missing.length ? null : path.basename(deps.model()),
      refined: missing.length === 0 && !!refinedModel,
      message: missing.length ? 'SheetSage2 is missing. Download the SheetSage2 Lead-Sheet Transcriber (yue2-sheetsage2-f16) in Model Manager, or supply an approved ABC score.' : null };
  }

  function reviewScore(input: { abc?: unknown; lyrics?: unknown }) {
    if (typeof input.abc !== 'string' || input.abc.length > MAX_ABC_LENGTH ||
        typeof input.lyrics !== 'string' || input.lyrics.length > MAX_ABC_LENGTH) {
      throw new CoverRequestError('ABC and lyrics must be strings of at most 64 KB each.');
    }
    const sections = scoreSections(input.abc);
    return { sections, lint: lintScoreSections(sections, input.lyrics) };
  }

  async function source(input: CoverInput, userId: string): Promise<CoverSource> {
    if (input.songId !== undefined && input.sourceAudioUrl !== undefined) throw new CoverRequestError('Choose a library song or an uploaded source, not both.');
    let sourceId: string;
    let sourceLabel: string;
    let url: string;
    let root: string;
    if (typeof input.songId === 'string' && input.songId.length > 0 && input.songId.length <= 128) {
      const song = deps.song(input.songId, userId);
      if (!song) throw new CoverRequestError('Library song not found', 404);
      sourceId = input.songId;
      sourceLabel = song.title || input.songId;
      url = song.audio_url;
      root = deps.libraryDir;
      if (!url.startsWith('/audio/')) throw new CoverRequestError('Library song has no local audio source.');
    } else if (typeof input.sourceAudioUrl === 'string') {
      sourceId = input.sourceAudioUrl;
      sourceLabel = input.sourceAudioUrl.split('/').at(-1) || '';
      url = input.sourceAudioUrl;
      root = deps.referenceDir;
      if (!url.startsWith('/references/')) throw new CoverRequestError('Use an uploaded /references/ source or a library songId.');
    } else {
      throw new CoverRequestError('sourceAudioUrl or songId is required.');
    }
    const name = url.slice(url.lastIndexOf('/') + 1);
    if (!/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,180}$/.test(name) || url !== `${root === deps.libraryDir ? '/audio' : '/references'}/${name}` || !AUDIO_EXT.has(path.extname(name).toLowerCase())) {
      throw new CoverRequestError('Invalid audio source.');
    }
    const audioPath = path.join(root, name);
    let stat: fs.Stats;
    try {
      stat = fs.lstatSync(audioPath);
      if (!stat.isFile()) throw new Error('not a file');
    } catch { throw new CoverRequestError('Audio source not found. Re-upload or choose another library song.', 404); }
    if (stat.size === 0) throw new CoverRequestError('Audio source is empty.');
    let workPath = audioPath;
    if (stat.size > MAX_AUDIO_BYTES) {
      try { workPath = await deps.toMp3(audioPath); }
      catch (err) { throw new CoverRequestError(`Audio source is over 100 MB and could not be converted: ${(err as Error).message}`); }
    }
    const duration = await deps.duration(workPath);
    if (!Number.isFinite(duration) || duration <= 0 || duration > MAX_AUDIO_SECONDS) throw new CoverRequestError('Audio source must have a readable duration of at most 10 minutes.');
    if (input.sourceLabel !== undefined) {
      if (typeof input.sourceLabel !== 'string' || input.sourceLabel.trim().length > 120) throw new CoverRequestError('sourceLabel must be at most 120 characters.');
      if (input.sourceLabel.trim()) sourceLabel = input.sourceLabel.trim();
    }
    return { sourceId, sourceLabel, audioPath, workPath };
  }

  async function start(input: CoverInput, userId: string) {
    const selected = await source(input, userId);
    if (input.abc !== undefined) {
      if (typeof input.abc !== 'string' || !input.abc.trim() || input.abc.length > MAX_ABC_LENGTH) throw new CoverRequestError('ABC must be non-empty and at most 64 KB.');
      const abc = input.abc.trim();
      return { status: 'done' as const, abc, sections: scoreSections(abc),
        sourceId: selected.sourceId, sourceLabel: selected.sourceLabel };
    }
    if (input.force !== undefined && typeof input.force !== 'boolean') throw new CoverRequestError('force must be a boolean.');
    const matched = deps.datasetMatch(selected.audioPath);
    if (matched && !input.force) {
      const abc = readAbcSidecar(matched.audioPath);
      if (abc) return { status: 'done' as const, abc, sections: scoreSections(abc), sourceId: selected.sourceId,
        sourceLabel: selected.sourceLabel, scoreSource: 'dataset' as const };
    }
    const ready = readiness();
    if (!ready.ready) throw new CoverRequestError(ready.message!, 400);
    if (input.transcriber !== undefined && input.transcriber !== 'sheetsage2' && input.transcriber !== 'sheetsage2-refined') {
      throw new CoverRequestError("transcriber must be 'sheetsage2' or 'sheetsage2-refined'.");
    }
    if (input.keyMode !== undefined && typeof input.keyMode !== 'boolean') throw new CoverRequestError('keyMode must be a boolean.');
    const refined = input.transcriber === 'sheetsage2-refined';
    if (refined && !ready.refined) {
      throw new CoverRequestError('SheetSage2-Refined is missing. Download it (yue2-sheetsage2-refined-f16) in Model Manager, or choose SheetSage2.', 400);
    }
    const transcriber = { refined, keyMode: refined && input.keyMode === true };
    const job = deps.queue.createJob('yue2-sheet', `cover:${userId}`, [], { sourceId: selected.sourceId });
    const dir = path.join(deps.jobRoot, job.id);
    try {
      fs.mkdirSync(dir, { recursive: true });
      const saved: SavedSource = { userId, sourceId: selected.sourceId, sourceLabel: selected.sourceLabel, name: path.basename(selected.workPath) };
      fs.writeFileSync(path.join(dir, 'cover-source.json'), JSON.stringify(saved), { mode: 0o600 });
    } catch (err) {
      deps.queue.cancelJob(job.id);
      throw err;
    }
    deps.queue.enqueue(job, async j => {
      if (j.controller.signal.aborted) return;
      try { await deps.run(j, selected.workPath, dir, transcriber); } catch { /* runner records failed/cancelled status */ }
    });
    return { status: 'queued' as const, jobId: job.id, sourceId: selected.sourceId, sourceLabel: selected.sourceLabel };
  }

  async function lookup(input: CoverInput, userId: string) {
    const selected = await source(input, userId);
    const matched = deps.datasetMatch(selected.audioPath);
    if (!matched) return { matched: false as const };
    const fields = readSidecar(sidecarPathFor(matched.audioPath));
    const bpm = Number(fields.bpm);
    const key = fields.key?.trim() || '';
    const instrumental = fields.is_instrumental?.trim().toLowerCase();
    const hasLyrics = typeof fields.lyrics === 'string' && !!fields.lyrics.trim();
    // A blank flag (as many dataset sidecars carry) is read from the lyrics, as training reads it.
    const validInstrumental = !instrumental || ['true', 'false', '1', '0', 'yes', 'no'].includes(instrumental);
    const isInstrumental = instrumental ? ['true', '1', 'yes'].includes(instrumental) : !hasLyrics;
    const validBpm = Number.isFinite(bpm) && bpm >= 20 && bpm <= 300;
    const abc = readAbcSidecar(matched.audioPath);
    if (!validBpm || !key || !validInstrumental || (!hasLyrics && !isInstrumental)) {
      return abc ? { matched: true as const, metadataAvailable: false as const, abc, sections: scoreSections(abc) } : { matched: false as const };
    }
    return { matched: true as const, datasetId: matched.datasetId, sampleId: matched.sampleId,
      metadataAvailable: true as const,
      lyrics: fields.lyrics || '', bpm, key,
      isInstrumental,
      abc, ...(abc ? { sections: scoreSections(abc) } : {}), lyricsSource: 'dataset-sidecar' as const };
  }

  function find(jobId: string, userId: string) {
    if (!/^[0-9a-f-]{36}$/.test(jobId)) throw new CoverRequestError('Cover job not found', 404);
    let saved: SavedSource;
    try { saved = JSON.parse(fs.readFileSync(path.join(deps.jobRoot, jobId, 'cover-source.json'), 'utf8')) as SavedSource; }
    catch { throw new CoverRequestError('Cover job not found', 404); }
    if (saved.userId !== userId) throw new CoverRequestError('Cover job not found', 404);
    const job = deps.queue.getJob(jobId);
    const summary = job ? deps.queue.toSummary(job) : deps.queue.listJobs().find(j => j.id === jobId);
    if (!summary) throw new CoverRequestError('Cover job not found', 404);
    const abc = summary.status === 'done' ? readYue2CoverAbc(path.join(deps.jobRoot, jobId, 'cover-sheet.json'), saved.name) : undefined;
    return { job: summary, sourceId: saved.sourceId, sourceLabel: saved.sourceLabel,
      ...(abc ? { abc, sections: scoreSections(abc) } : {}) };
  }

  function cancel(jobId: string, userId: string) {
    const current = find(jobId, userId);
    if (current.job.status === 'queued' || current.job.status === 'running') deps.queue.cancelJob(jobId);
    return find(jobId, userId);
  }

  /** What a match proposal was made against: the dataset song and the bytes of
   *  its merged metadata and score. A save refuses when either has moved on. */
  function datasetRevision(matched: CoverDatasetSource) {
    return crypto.createHash('sha256').update(JSON.stringify([
      matched.datasetId, matched.sampleId, loadSidecarMetadata(matched.audioPath), readAbcSidecar(matched.audioPath),
    ])).digest('hex');
  }

  /** The source vocal's word times for these lyrics, on the cached stem. */
  async function alignSource(input: CoverInput & { lyrics?: unknown }, userId: string) {
    if (typeof input.abc !== 'string' || !input.abc.trim() || input.abc.length > MAX_ABC_LENGTH ||
        typeof input.lyrics !== 'string' || !input.lyrics.trim() || input.lyrics.length > MAX_ABC_LENGTH) {
      throw new CoverRequestError('ABC and lyrics must be non-empty strings of at most 64 KB each.');
    }
    const { abc: _abc, lyrics: _lyrics, ...rest } = input;
    const selected = await source(rest, userId);
    const matched = deps.datasetMatch(selected.audioPath);
    const revision = matched ? datasetRevision(matched) : undefined;
    let stem: string;
    try { stem = await deps.vocalStem(selected.workPath); }
    catch (err) {
      if (err instanceof NoVocalsError) throw new CoverRequestError('The source has no vocals to align.', 422);
      throw err;
    }
    const lyrics = input.lyrics.replace(/\r\n?/g, '\n');
    const aligned = await deps.align(fs.readFileSync(stem), lyrics);
    return { abc: input.abc, lyrics, words: aligned.words, dataset: matched ? { revision: revision! } : null };
  }

  /** Retag the lyrics with the score's sections where the source vocal sings
   *  each block. Nothing is saved; the client shows the diff first. */
  async function matchSections(input: CoverInput & { lyrics?: unknown }, userId: string) {
    const { abc, lyrics, words, dataset } = await alignSource(input, userId);
    let result;
    try { result = matchSectionsToScore(abc, lyrics, words); }
    catch (err) { throw new CoverRequestError((err as Error).message, 422); }
    return { ...result, datasetSong: !!dataset, ...dataset };
  }

  /** The other direction: the score's sections set from where the source
   *  vocal sings each lyric block, with the lyrics retagged to match. Nothing
   *  is saved; the client rewrites its own copy of the score. */
  async function planScoreSections(input: CoverInput & { lyrics?: unknown }, userId: string) {
    const { abc, lyrics, words, dataset } = await alignSource(input, userId);
    try { return { ...planScoreFromLyrics(abc, lyrics, words), datasetSong: !!dataset, ...dataset }; }
    catch (err) { throw new CoverRequestError((err as Error).message, 422); }
  }

  /** Proposals for which lyric blocks belong to which section of a Prepare
   *  document, as section id -> block ids. Nothing is stored or rewritten: the
   *  reply names the {sourceKey, structureHash, lyricRevision} it was made for,
   *  so the client can drop it if the document moved on. With `useRecording`
   *  the source vocal is aligned to place blocks by where they are sung; if
   *  that fails the instant (tag and order) proposals are returned instead,
   *  with `recordingError`. */
  async function suggestAssignments(input: CoverInput & { prepare?: unknown; useRecording?: unknown }, userId: string) {
    const { prepare, useRecording, abc: _abc, ...rest } = input;
    const problem = prepareV2Problem(prepare);
    if (problem) throw new CoverRequestError(problem);
    const p = prepare as PrepareV2;
    if (!p.sectionIds.length) throw new CoverRequestError('The score has no sections.');
    const blocks = p.blocks.map(({ id, tag, text }) => ({ id, tag, text }));
    const stamp = { sourceKey: p.sourceKey, structureHash: structureHash(p.abc), lyricRevision: p.lyricRevision };
    const make = (words?: Yue2AlignWord[]) => {
      try { return suggestSections(p.abc, p.sectionIds, blocks, words); }
      catch (err) { throw new CoverRequestError((err as Error).message, 422); }
    };
    if (useRecording !== true) return { ...stamp, ...make(), recordingError: undefined as string | undefined };
    let words: Yue2AlignWord[];
    try { words = (await alignSource({ ...rest, abc: p.abc, lyrics: suggestionLyrics(blocks) }, userId)).words; }
    catch (err) { return { ...stamp, ...make(), recordingError: (err as Error).message }; }
    return { ...stamp, ...make(words), recordingError: undefined };
  }

  /** Write lyrics and/or the revised score, or the score's tempo and key, into
   *  the matched dataset song. Lyrics and tempo go in the .txt, the score in
   *  the .abc; each writer keeps the previous file as a .bak (a timestamped
   *  .bak when one is already there). Lyrics and score need the `revision` a
   *  match returned, so a song that changed since matching is refused. The
   *  two files are separate writes: the .txt goes first, and if the .abc then
   *  fails the result says so rather than reporting a save. */
  async function saveDatasetDetails(input: CoverInput & { lyrics?: unknown; abc?: unknown; revision?: unknown; bpm?: unknown; key?: unknown }, userId: string) {
    const { lyrics, abc, revision, bpm, key, ...rest } = input;
    const fields: Record<string, string> = {};
    if (lyrics !== undefined) {
      if (typeof lyrics !== 'string' || !lyrics.trim() || lyrics.length > MAX_ABC_LENGTH) {
        throw new CoverRequestError('Lyrics must be a non-empty string of at most 64 KB.');
      }
      fields.lyrics = lyrics.replace(/\r\n?/g, '\n');
    }
    if (abc !== undefined && (typeof abc !== 'string' || abc.length > MAX_ABC_LENGTH || !hasSoundingScore(abc))) {
      throw new CoverRequestError('The score must be ABC of at most 64 KB with a sounding vocal bar.');
    }
    if (bpm !== undefined || key !== undefined) {
      if (typeof bpm !== 'number' || !Number.isFinite(bpm) || bpm < 20 || bpm > 300 ||
          typeof key !== 'string' || !/^[A-G][#b]? (Major|Minor)$/.test(key)) {
        throw new CoverRequestError('Tempo must be 20–300 BPM and key like "E Major" or "B Minor".');
      }
      fields.bpm = String(Math.round(bpm));
      fields.key = key;
    }
    if (!Object.keys(fields).length && abc === undefined) throw new CoverRequestError('Nothing to save.');
    const selected = await source(rest, userId);
    const matched = deps.datasetMatch(selected.audioPath);
    if (!matched) throw new CoverRequestError('This source is not a dataset song.', 404);
    if ((fields.lyrics !== undefined || abc !== undefined) && revision !== datasetRevision(matched)) {
      throw new CoverRequestError('The dataset song changed since matching. Match again.', 409);
    }
    const saved = { saved: true as const, datasetId: matched.datasetId, sampleId: matched.sampleId,
      file: path.basename(sidecarPathFor(matched.audioPath)) };
    if (Object.keys(fields).length) await deps.saveDetails(matched, fields);
    if (abc === undefined) return saved;
    try {
      if (!deps.saveAbc(matched, abc as string)) throw new Error('the score was not written');
      return { ...saved, abcSaved: true as const };
    } catch (err) {
      return { ...saved, abcSaved: false as const, abcError: (err as Error).message };
    }
  }

  return { readiness, source, lookup, start, find, cancel, reviewScore, matchSections, planScoreSections, suggestAssignments, saveDatasetDetails };
}

export const yue2CoverService = createYue2CoverService();
