import abcjs from 'abcjs';
import { coverScoreSnapshot } from './coverScore.js';

export type CoverVoices = 'vocal' | 'both';
export type CoverTempo = 'source' | 'free' | number;
export type CoverKey = 'source' | string;

export interface CoverScoreChoices {
  voices: CoverVoices;
  keepChords: boolean;
  tempo: CoverTempo;
  key: CoverKey;
  cfgScale: number;
}

/** Submit-time defaults; every captured cover writes all five choices. */
export const COVER_SCORE_DEFAULTS: CoverScoreChoices = {
  voices: 'vocal', keepChords: false, tempo: 'free', key: 'source', cfgScale: 1,
};

export function coverScoreChoices(raw: Partial<CoverScoreChoices>): CoverScoreChoices {
  const choices = { ...COVER_SCORE_DEFAULTS, ...raw };
  if (choices.voices !== 'vocal' && choices.voices !== 'both') throw new Error('yue2Cover.voices must be vocal or both.');
  if (typeof choices.keepChords !== 'boolean') throw new Error('yue2Cover.keepChords must be a boolean.');
  if (choices.tempo !== 'source' && choices.tempo !== 'free' &&
      (typeof choices.tempo !== 'number' || !Number.isFinite(choices.tempo) || choices.tempo < 20 || choices.tempo > 300)) {
    throw new Error('yue2Cover.tempo must be source, free or 20–300 BPM.');
  }
  if (typeof choices.key !== 'string' || !choices.key.trim()) throw new Error('yue2Cover.key must be source or a target key.');
  if (typeof choices.cfgScale !== 'number' || !Number.isFinite(choices.cfgScale) || choices.cfgScale <= 0 || choices.cfgScale > 2) {
    throw new Error('yue2Cover.cfgScale must be greater than 0 and at most 2.');
  }
  return choices;
}

function eachLine(abc: string, map: (line: string, newline: string) => string): string {
  const parts = abc.split(/(\r?\n)/);
  let result = '';
  for (let i = 0; i < parts.length; i += 2) result += map(parts[i], parts[i + 1] ?? '');
  return result;
}

/** SheetSage declares both voices before the first K:, then switches voice for each body block. */
export function coverVocalOnly(abc: string): string {
  let inHeader = true;
  let inIns = false;
  return eachLine(abc, (line, newline) => {
    if (/^K:/.test(line)) inHeader = false;
    if (/^V:\s*Ins\b/i.test(line)) { if (!inHeader) inIns = true; return ''; }
    if (/^V:\s*/.test(line) || /^\s*%/.test(line)) inIns = false;
    return inIns ? '' : line + newline;
  });
}

export function coverTempo(abc: string, tempo: CoverTempo): string {
  if (tempo === 'source') return abc;
  let found = false;
  let result = eachLine(abc, (line, newline) => {
    if (!/^Q:/.test(line)) return line + newline;
    found = true;
    if (tempo === 'free') return '';
    const match = line.match(/^(Q:\s*[^=\r\n]+?=\s*)(\d+(?:\.\d+)?)(.*)$/);
    if (!match) throw new Error('Cannot set tempo: unsupported Q: field.');
    return `${match[1]}${tempo}${match[3]}${newline}`;
  });
  if (tempo !== 'free' && !found) {
    const nl = abc.includes('\r\n') ? '\r\n' : '\n';
    result = result.replace(/^K:/m, `Q:1/4=${tempo}${nl}K:`);
    if (result === abc) throw new Error('Cannot set tempo: score has no K: field.');
  }
  return result;
}

const PITCH_CLASS: Record<string, number> = {
  C: 0, 'B#': 0, 'C#': 1, Db: 1, D: 2, 'D#': 3, Eb: 3, E: 4, Fb: 4,
  'E#': 5, F: 5, 'F#': 6, Gb: 6, G: 7, 'G#': 8, Ab: 8, A: 9,
  'A#': 10, Bb: 10, B: 11, Cb: 11,
};

function parsedKey(value: string): { pitch: number; minor: boolean } {
  const match = value.trim().match(/^([A-Ga-g])([#b]?)(?:\s*(minor|major|maj|m))?$/i);
  if (!match) throw new Error(`Unsupported cover key: ${value}`);
  const root = match[1].toUpperCase() + match[2];
  return { pitch: PITCH_CLASS[root], minor: ['m', 'minor'].includes((match[3] || '').toLowerCase()) };
}

/** abcjs parses the tune and moves keys, inline keys, notes and chord roots/bass together. */
export function coverTranspose(abc: string, targetKey: CoverKey): string {
  if (targetKey === 'source') return abc;
  const sourceField = abc.match(/^K:\s*([^\r\n]+)/m);
  if (!sourceField) throw new Error('Cannot transpose a cover score without K:.');
  const source = parsedKey(sourceField[1]);
  const target = parsedKey(targetKey);
  if (source.minor !== target.minor) throw new Error('Cover transposition keeps the source key mode.');
  let steps = (target.pitch - source.pitch + 12) % 12;
  if (steps > 6) steps -= 12;
  if (steps === 0) return abc;
  const transposed = abcjs.strTranspose(abc, abcjs.parseOnly(abc), steps);
  const changedField = transposed.match(/^K:\s*([^\r\n]+)/m);
  if (!changedField || parsedKey(changedField[1]).pitch !== target.pitch) {
    throw new Error('Could not transpose the cover score to the requested key.');
  }
  return transposed;
}

/** The order is part of the cover contract: voices, chords, tempo, key. */
export function transformCoverScore(fullScore: string, choices: CoverScoreChoices) {
  const voices = choices.voices === 'vocal' ? coverVocalOnly(fullScore) : fullScore;
  const chords = coverScoreSnapshot(voices, choices.keepChords).renderedAbc;
  const tempo = coverTempo(chords, choices.tempo);
  return { fullScore, renderedAbc: coverTranspose(tempo, choices.key) };
}
