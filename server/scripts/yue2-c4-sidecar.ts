import abcjs from 'abcjs';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { scoreBarClock, scoreSecondsAtWhole } from '../src/services/backends/yue2/scoreClock.js';

export interface TimedWord {
  start: number;
  end: number;
  score: number;
  char0: number;
  char1: number;
}

export interface VocalNote {
  index: number;
  start: number;
  end: number;
  pitch: string;
  bar: number;
  abcStartChar: number;
  abcEndChar: number;
  tiedFromPrevious: boolean;
  tiedToNext: boolean;
}

export interface WordNoteRow {
  wordIndex: number;
  text: string;
  start: number;
  end: number;
  score: number;
  char0: number;
  char1: number;
  noteIndices: number[];
  possibleMelisma: boolean;
  hasTie: boolean;
  status: 'clean' | 'uncertain';
  reason?: string;
}

export interface C4Sidecar {
  version: 1;
  clock: 'abc-meter-tempo-fields';
  bpm: number;
  beatWholeNotes: number;
  notes: VocalNote[];
  words: WordNoteRow[];
  summary: { total: number; clean: number; uncertain: number; possibleMelismas: number; tiedWords: number };
}

/** The cache stores ABC and words5, but not the transcriber's variable beat timestamps. */
export function vocalNotes(abc: string): { notes: VocalNote[]; bpm: number; beatWholeNotes: number } {
  if (!/^V:\s*Vocal\b/im.test(abc)) throw new Error('ABC has no explicit Vocal voice');
  const bars = scoreBarClock(abc);
  const tune = abcjs.parseOnly(abc)[0];
  if (!tune) throw new Error('ABC parser returned no tune');
  const { bpm, beatWholeNotes } = bars[0];
  const notes: VocalNote[] = [];
  let wholeNotes = 0;
  for (const line of tune.lines) {
    // SheetSage declares Vocal first, then Ins.
    const staff = line.staff?.[0];
    for (const event of staff?.voices?.[0] ?? []) {
      if (event.el_type === 'bar') continue;
      if (event.el_type !== 'note') continue;
      const duration = Number(event.duration);
      if (!(duration > 0) || !Number.isFinite(duration)) throw new Error('Invalid Vocal duration');
      if ((event.pitches?.length ?? 0) > 1) throw new Error('Polyphonic Vocal event is unsupported');
      if (event.pitches?.length === 1) {
        const pitch = event.pitches[0];
        const barIndex = bars.findIndex(item => wholeNotes < item.wholeEnd - 1e-7);
        if (barIndex < 0) throw new Error('ABC notes extend beyond the Vocal bar grid');
        notes.push({ index: notes.length, start: scoreSecondsAtWhole(bars, wholeNotes),
          end: scoreSecondsAtWhole(bars, wholeNotes + duration), pitch: pitch.name ?? '',
          bar: barIndex + 1,
          abcStartChar: event.startChar, abcEndChar: event.endChar,
          tiedFromPrevious: Boolean(pitch.endTie), tiedToNext: Boolean(pitch.startTie) });
      }
      wholeNotes += duration;
    }
  }
  if (Math.abs(wholeNotes - bars.at(-1)!.wholeEnd) > 1e-4) throw new Error('ABC notes and Vocal bar grid disagree');
  if (!notes.length) throw new Error('ABC Vocal voice has no notes');
  return { notes, bpm, beatWholeNotes };
}

export function readWords5(bytes: Buffer, lyrics: string): TimedWord[] {
  if (!bytes.length || bytes.length % 20) throw new Error('words5 must contain complete f32 rows');
  const chars = Array.from(lyrics);
  const words: TimedWord[] = [];
  for (let offset = 0; offset < bytes.length; offset += 20) {
    const [start, end, score, char0, char1] = [0, 4, 8, 12, 16].map(i => bytes.readFloatLE(offset + i));
    if (![start, end, score, char0, char1].every(Number.isFinite) || start < 0 || end < start ||
        (words.length > 0 && start < words[words.length - 1].start) ||
        !Number.isInteger(char0) || !Number.isInteger(char1) || char0 < 0 || char1 <= char0 ||
        char1 > chars.length) throw new Error(`Malformed words5 row ${words.length}`);
    words.push({ start, end, score, char0, char1 });
  }
  return words;
}

/** Conservative overlap mapping. A shared note or a timing gap is uncertain. */
export function buildC4Sidecar(abc: string, lyrics: string, timedWords: TimedWord[]): C4Sidecar {
  const { notes, bpm, beatWholeNotes } = vocalNotes(abc);
  const chars = Array.from(lyrics);
  const rows: WordNoteRow[] = timedWords.map((word, wordIndex) => {
    const text = chars.slice(word.char0, word.char1).join('');
    const matched = notes.filter(note => Math.min(note.end, word.end) > Math.max(note.start, word.start));
    const covered = matched.reduce((sum, note) => sum + Math.max(0, Math.min(note.end, word.end) - Math.max(note.start, word.start)), 0);
    const coverage = word.end > word.start ? covered / (word.end - word.start) : 0;
    const reason = !matched.length ? 'no-overlapping-vocal-note' : coverage < 0.8 ? 'low-time-overlap' : undefined;
    return { wordIndex, text, start: word.start, end: word.end, score: word.score,
      char0: word.char0, char1: word.char1, noteIndices: matched.map(note => note.index),
      possibleMelisma: matched.filter(note => !note.tiedFromPrevious).length > 1,
      hasTie: matched.some(note => note.tiedFromPrevious || note.tiedToNext),
      status: reason ? 'uncertain' : 'clean', ...(reason ? { reason } : {}) };
  });
  const noteOwners = new Map<number, number>();
  for (const row of rows) for (const index of row.noteIndices) noteOwners.set(index, (noteOwners.get(index) ?? 0) + 1);
  for (const row of rows) {
    if (row.noteIndices.some(index => (noteOwners.get(index) ?? 0) > 1)) {
      row.status = 'uncertain';
      row.reason = 'note-shared-by-words';
    }
  }
  return { version: 1, clock: 'abc-meter-tempo-fields', bpm, beatWholeNotes, notes, words: rows,
    summary: { total: rows.length, clean: rows.filter(row => row.status === 'clean').length,
      uncertain: rows.filter(row => row.status === 'uncertain').length,
      possibleMelismas: rows.filter(row => row.possibleMelisma).length,
      tiedWords: rows.filter(row => row.hasTie).length } };
}

export function writeC4Sidecar(manifestPath: string, sourceIndex: number, outputPath: string): C4Sidecar {
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  const source = manifest.sources?.[sourceIndex];
  if (!source || typeof source.abc !== 'string' || typeof source.lyrics !== 'string' ||
      typeof source.cursor_words !== 'string') throw new Error('Source lacks ABC, lyrics, or cursor_words');
  const cursorPath = path.resolve(path.dirname(manifestPath), source.cursor_words);
  if (!cursorPath.startsWith(path.resolve(path.dirname(manifestPath)) + path.sep)) throw new Error('cursor_words escapes cache');
  const sidecar = buildC4Sidecar(source.abc, source.lyrics, readWords5(fs.readFileSync(cursorPath), source.lyrics));
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(sidecar, null, 2) + '\n');
  return sidecar;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [, , manifestPath, sourceIndex, outputPath] = process.argv;
  if (!manifestPath || !/^\d+$/.test(sourceIndex ?? '') || !outputPath) {
    console.error('Usage: tsx scripts/yue2-c4-sidecar.ts <manifest> <source-index> <output.json>');
    process.exitCode = 2;
  } else {
    const result = writeC4Sidecar(manifestPath, Number(sourceIndex), outputPath);
    console.log(JSON.stringify(result.summary));
  }
}
