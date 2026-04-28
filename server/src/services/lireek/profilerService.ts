import * as cmuDictRaw from 'cmu-pronouncing-dictionary';
import type { ChunkCallback } from './llmService.js';

export interface SongLyrics {
  title: string;
  album?: string;
  lyrics: string;
}

export interface LyricsProfile {
  artist_id?: number;
  artist: string;
  album?: string;
  themes: string[];
  common_subjects: string[];
  rhyme_schemes: string[];
  avg_verse_lines: number;
  avg_chorus_lines: number;
  vocabulary_notes?: string;
  tone_and_mood?: string;
  structural_patterns?: string;
  additional_notes?: string;
  raw_summary?: string;
  song_subjects?: Record<string, string>;
  subject_categories?: string[];
  repetition_stats?: {
    chorus_repetition_pct?: number;
    verse_repetition_pct?: number;
    cross_section_repeats?: number;
    pattern?: string;
    hook_examples?: string[];
  };
  structure_blueprints?: string[];
  perspective?: string;
  meter_stats?: {
    avg_syllables_per_line?: number;
    syllable_std_dev?: number;
    avg_words_per_line?: number;
    line_length_range?: string;
    line_length_variation?: {
      histogram?: Record<string, number>;
      per_section?: Record<string, any>;
      short_line_examples?: string[];
      long_line_examples?: string[];
    };
  };
  vocabulary_stats?: {
    type_token_ratio?: number;
    total_words?: number;
    unique_words?: number;
    contraction_pct?: number;
    profanity_pct?: number;
    distinctive_words?: string[];
  };
  representative_excerpts?: string[];
  narrative_techniques?: string;
  imagery_patterns?: string;
  signature_devices?: string;
  emotional_arc?: string;
  rhyme_quality?: Record<string, number>;
  examples?: any[];
  [key: string]: any;
}

import * as llmService from './llmService.js';
import { 
  PROFILE_PROMPT_1,
  PROFILE_PROMPT_2,
  PROFILE_PROMPT_3
} from './prompts.js';

// The imported dict is a default export depending on interop.
const CMU_DICT: Record<string, string> = (cmuDictRaw as any).default || cmuDictRaw;

// ── Robust JSON extraction ────────────────────────────────────────────────────

function repairJson(text: string): string {
  // Fix missing commas between string array elements
  let fixed = text.replace(/"\s*\n(\s*")/g, '",\n$1');
  // Fix missing commas after ] when followed by " (next key)
  fixed = fixed.replace(/\]\s*\n(\s*")/g, '],\n$1');
  // Fix stray } after ]
  fixed = fixed.replace(/\]\s*\n\s*},?\s*\n(\s*")/g, '],\n$1');
  // Fix missing commas between } and " or {
  fixed = fixed.replace(/}\s*\n(\s*["{])/g, '},\n$1');
  // Fix trailing commas before ] or }
  fixed = fixed.replace(/,(\s*[}\]])/g, '$1');
  return fixed;
}

function extractJson(text: string): Record<string, any> | null {
  // Strategy 0: strip reasoning model <think> blocks
  let stripped = text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
  if (stripped.includes('<think>')) {
    stripped = stripped.replace(/<think>[\s\S]*/g, '').trim();
  }

  // Strategy 1: direct parse
  try { return JSON.parse(stripped); } catch (e) {}

  // Strategy 2: strip code fences
  let clean = stripped.replace(/^```(?:json)?\s*/m, '').replace(/\s*```$/m, '').trim();
  try { return JSON.parse(clean); } catch (e) {}

  // Strategy 3: find outermost { ... }
  const firstBrace = clean.indexOf('{');
  const lastBrace = clean.lastIndexOf('}');
  let candidate: string | null = null;
  if (firstBrace !== -1 && lastBrace > firstBrace) {
    candidate = clean.slice(firstBrace, lastBrace + 1);
    try { return JSON.parse(candidate); } catch (e) {}
  }

  // Strategy 4: repair
  const toRepair = candidate || clean;
  const repaired = repairJson(toRepair);
  try { return JSON.parse(repaired); } catch (e) {}

  // Strategy 5: brute-force
  let lines = repaired.split('\n');
  for (let attempt = 0; attempt < 3; attempt++) {
    let found = false;
    for (let i = 0; i < lines.length; i++) {
      const s = lines[i].trim();
      if ((s === '}' || s === '},') && i > 0 && i < lines.length - 1) {
        let trialLines = [...lines];
        trialLines.splice(i, 1);
        try {
          return JSON.parse(trialLines.join('\n'));
        } catch (e) {
          lines = trialLines;
          found = true;
          break;
        }
      }
    }
    if (!found) break;
  }

  console.warn("All JSON extraction strategies failed for response");
  return null;
}

// ── CMU Pronouncing Dictionary ────────────────────────────────────────────────

function getPhones(word: string): string[] | null {
  const cleanWord = word.toLowerCase().replace(/['".,!?;:-]/g, '');
  const entry = CMU_DICT[cleanWord];
  if (entry) return entry.split(' ');
  return null;
}

function getVowelTail(phones: string[], n: int = 3): string[] {
  let result: string[] = [];
  for (let i = phones.length - 1; i >= 0; i--) {
    let clean = phones[i].replace(/\d/g, ''); // strip stress marker
    result.push(clean);
    if (result.length >= n) break;
  }
  return result.reverse();
}

function rhymeQuality(wordA: string, wordB: string): string {
  if (wordA === wordB) return 'perfect';

  const phonesA = getPhones(wordA);
  const phonesB = getPhones(wordB);

  if (!phonesA || !phonesB) {
    const a = wordA.toLowerCase(), b = wordB.toLowerCase();
    if (a.length >= 2 && b.length >= 2 && a.slice(-2) === b.slice(-2)) return 'slant';
    return 'none';
  }

  const tailA = getVowelTail(phonesA, 3);
  const tailB = getVowelTail(phonesB, 3);
  
  if (tailA.join('-') === tailB.join('-')) return 'perfect';

  const tailA2 = getVowelTail(phonesA, 2);
  const tailB2 = getVowelTail(phonesB, 2);
  if (tailA2.join('-') === tailB2.join('-')) return 'perfect';

  const vowelsA = phonesA.filter(p => /\d/.test(p)).map(p => p.replace(/\d/g, ''));
  const vowelsB = phonesB.filter(p => /\d/.test(p)).map(p => p.replace(/\d/g, ''));

  if (vowelsA.length && vowelsB.length && vowelsA[vowelsA.length - 1] === vowelsB[vowelsB.length - 1]) {
    if (tailA2[0] === tailB2[0] || tailA.some(p => tailB.includes(p))) return 'slant';
    return 'assonance';
  }

  if (tailA.filter(p => tailB.includes(p)).length >= 2) return 'slant';
  return 'none';
}

function countSyllablesHeuristic(word: string): int {
  let w = word.toLowerCase().replace(/['".,!?;:-]/g, '');
  if (!w) return 0;
  const matches = w.match(/[aeiouy]+/g);
  let count = matches ? matches.length : 0;
  if (w.endsWith('e') && count > 1) count--;
  return Math.max(count, 1);
}

function countSyllables(word: string): int {
  const phones = getPhones(word);
  if (phones) {
    const cmu = phones.filter(p => /\d/.test(p)).length;
    if (cmu > 0) return cmu;
  }
  return countSyllablesHeuristic(word);
}

// ── Section parsing ───────────────────────────────────────────────────────────

const SECTION_HEADER_RE = /^\[(.+?)\]$/i;

const SECTION_LABEL_MAP: Record<string, string> = {
  verse: 'V', chorus: 'C', hook: 'C', bridge: 'B', 
  'pre-chorus': 'PC', prechorus: 'PC', 'post-chorus': 'POC',
  intro: 'I', outro: 'O', interlude: 'IL', refrain: 'C'
};

function normaliseSectionLabel(rawLabel: string): string {
  const lower = rawLabel.toLowerCase().trim();
  for (const [key, code] of Object.entries(SECTION_LABEL_MAP)) {
    if (lower.includes(key)) return code;
  }
  return 'X';
}

function splitIntoSections(lyrics: string): { label: string, lines: string[] }[] {
  const sections: { label: string, lines: string[] }[] = [];
  let currentLabel = 'X';
  let currentLines: string[] = [];

  const lines = lyrics.split('\n');
  for (const line of lines) {
    const stripped = line.trim();
    const match = SECTION_HEADER_RE.exec(stripped);
    if (match) {
      if (currentLines.length) sections.push({ label: currentLabel, lines: currentLines });
      currentLabel = normaliseSectionLabel(match[1]);
      currentLines = [];
    } else if (stripped === '') {
      if (currentLines.length) {
         sections.push({ label: currentLabel, lines: currentLines });
         currentLines = [];
      }
    } else {
      currentLines.push(stripped);
    }
  }
  if (currentLines.length) sections.push({ label: currentLabel, lines: currentLines });
  return sections;
}

// ── Analysis functions ────────────────────────────────────────────────────────

function getLastWord(line: string): string {
  const words = line.match(/[a-zA-Z']+/g);
  return words ? words[words.length - 1].toLowerCase() : '';
}

function detectRhymeScheme(sectionLines: string[]): { scheme: string, quality: Record<string, int> } {
  const lines = sectionLines.slice(0, 8);
  const endings = lines.map(getLastWord);
  
  const mapping: Record<string, string> = {};
  let letterIdx = 0;
  const scheme: string[] = [];
  const qualityCounts: Record<string, int> = { perfect: 0, slant: 0, assonance: 0 };

  for (const word of endings) {
    if (!word) {
      scheme.push('X');
      continue;
    }
    let matched: string | null = null;
    let bestQuality = 'none';

    for (const [existingWord, letter] of Object.entries(mapping)) {
      const q = rhymeQuality(word, existingWord);
      if (q === 'perfect' || q === 'slant' || q === 'assonance') {
        if (!matched || q === 'perfect') {
          matched = letter;
          bestQuality = q;
          if (q === 'perfect') break;
        }
      }
    }

    if (matched) {
      scheme.push(matched);
      if (qualityCounts[bestQuality] !== undefined) {
          qualityCounts[bestQuality]++;
      }
    } else {
      const newLetter = String.fromCharCode(65 + Math.min(letterIdx, 25));
      mapping[word] = newLetter;
      scheme.push(newLetter);
      letterIdx++;
    }
  }

  return { scheme: scheme.join(''), quality: qualityCounts };
}

function analyseRhymes(allSongs: SongLyrics[]): { schemes: string[], quality: Record<string, int> } {
  const schemeFreq: Record<string, int> = {};
  const totalQuality: Record<string, int> = { perfect: 0, slant: 0, assonance: 0 };

  for (const song of allSongs) {
    const sections = splitIntoSections(song.lyrics);
    for (const sec of sections) {
      if ((sec.label === 'V' || sec.label === 'C') && sec.lines.length >= 2) {
        const res = detectRhymeScheme(sec.lines);
        schemeFreq[res.scheme] = (schemeFreq[res.scheme] || 0) + 1;
        totalQuality.perfect += res.quality.perfect;
        totalQuality.slant += res.quality.slant;
        totalQuality.assonance += res.quality.assonance;
      }
    }
  }

  const topSchemes = Object.entries(schemeFreq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(x => x[0]);

  return { schemes: topSchemes, quality: totalQuality };
}

function analyseStructure(allSongs: SongLyrics[]): { v: number, c: number, blueprints: string[] } {
  let vCount = 0, vSum = 0;
  let cCount = 0, cSum = 0;
  const blueprintsFreq: Record<string, int> = {};

  for (const song of allSongs) {
    const sections = splitIntoSections(song.lyrics);
    const labels: string[] = [];
    
    for (const sec of sections) {
      if (sec.label === 'V') { vCount++; vSum += sec.lines.length; }
      else if (sec.label === 'C') { cCount++; cSum += sec.lines.length; }
      
      if (!labels.length || labels[labels.length - 1] !== sec.label) {
        labels.push(sec.label);
      }
    }
    const bp = labels.join('-');
    if (bp) blueprintsFreq[bp] = (blueprintsFreq[bp] || 0) + 1;
  }

  const topBps = Object.entries(blueprintsFreq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(x => x[0]);

  return { 
    v: vCount ? parseFloat((vSum / vCount).toFixed(1)) : 0, 
    c: cCount ? parseFloat((cSum / cCount).toFixed(1)) : 0, 
    blueprints: topBps 
  };
}

function analysePerspective(allSongs: SongLyrics[]): string {
  let p1 = 0, p2 = 0, p3 = 0;
  const fw = new Set(['i','me','my','mine','myself',"i'm","i've","i'll","i'd","im"]);
  const sw = new Set(['you','your','yours','yourself',"you're","you've","you'll",'ya']);
  const tw = new Set(['he','she','they','him','her','them','his','their','hers','theirs']);

  for (const song of allSongs) {
    const words = song.lyrics.toLowerCase().match(/[a-zA-Z']+/g) || [];
    for (const w of words) {
      if (fw.has(w)) p1++;
      else if (sw.has(w)) p2++;
      else if (tw.has(w)) p3++;
    }
  }

  const total = p1 + p2 + p3;
  if (!total) return "Indeterminate (no clear pronoun pattern)";

  const pct1 = Math.round(100 * p1 / total);
  const pct2 = Math.round(100 * p2 / total);
  const pct3 = Math.round(100 * p3 / total);

  const parts: string[] = [];
  if (pct1 >= 50) parts.push(`First-person dominant (${pct1}% I/me/my)`);
  if (pct2 >= 30) parts.push(`Second-person address (${pct2}% you/your)`);
  if (pct3 >= 30) parts.push(`Third-person narrative (${pct3}% he/she/they)`);

  if (!parts.length) parts.push(`Mixed voice (${pct1}% first / ${pct2}% second / ${pct3}% third)`);

  if (pct1 >= 70) parts.push("— confessional / introspective style");
  else if (pct1 >= 50 && pct2 >= 20) parts.push("— conversational / direct address style");
  else if (pct2 >= 50) parts.push("— confrontational / accusatory style");
  else if (pct3 >= 40) parts.push("— storytelling / observational style");

  return parts.join(' ');
}

function analyseMeter(allSongs: SongLyrics[]): Record<string, any> {
  const sylCounts: int[] = [];
  const wordCounts: int[] = [];
  const charCounts: int[] = [];

  for (const song of allSongs) {
    const lines = song.lyrics.split('\n');
    for (let line of lines) {
      line = line.trim();
      if (!line || SECTION_HEADER_RE.test(line)) continue;
      const words = line.match(/[a-zA-Z']+/g);
      if (!words) continue;
      
      const syl = words.reduce((acc: number, w: string) => acc + countSyllables(w), 0);
      sylCounts.push(syl);
      wordCounts.push(words.length);
      charCounts.push(line.length);
    }
  }

  if (!sylCounts.length) {
    return { avg_syllables_per_line: 0, syllable_std_dev: 0, avg_words_per_line: 0, line_length_range: "0-0" };
  }

  const avgSyl = sylCounts.reduce((a, b) => a + b, 0) / sylCounts.length;
  const variance = sylCounts.reduce((a, b) => a + Math.pow(b - avgSyl, 2), 0) / sylCounts.length;
  const avgWords = wordCounts.reduce((a, b) => a + b, 0) / wordCounts.length;

  return {
    avg_syllables_per_line: parseFloat(avgSyl.toFixed(1)),
    syllable_std_dev: parseFloat(Math.sqrt(variance).toFixed(1)),
    avg_words_per_line: parseFloat(avgWords.toFixed(1)),
    line_length_range: `${Math.min(...charCounts)}-${Math.max(...charCounts)} chars`
  };
}

const COMMON_WORDS = new Set("the a an and or but if in on at to for of is it its that this with from by as are was were be been being have has had do does did will would shall should can could may might must not no nor so than too very just all each every both few more most other some such any only same also how when where why what which who whom i me my mine we us our they them their he him his she her you your about after again against between into through during before up down out off over under there here then now get got like go going know want need make take come think say tell give see feel find keep let put seem still try call ask look show turn move live help start run write set play hold bring happen begin walk talk love well back even new way day man right old big long little much good great first last time thing part work world life hand oh yeah hey ah oh uh ooh la da na hoo hey".split(' '));

function analyseVocabulary(allSongs: SongLyrics[]): Record<string, any> {
  const allWords: string[] = [];
  let contractions = 0;
  let profanity = 0;

  const contRe = /\b(?:i[''']m|i[''']ve|i[''']ll|i[''']d|don[''']t|doesn[''']t|didn[''']t|won[''']t|wouldn[''']t|can[''']t|couldn[''']t|shouldn[''']t|isn[''']t|aren[''']t|wasn[''']t|weren[''']t|haven[''']t|hasn[''']t|hadn[''']t|ain[''']t|it[''']s|that[''']s|what[''']s|there[''']s|here[''']s|who[''']s|let[''']s|you[''']re|you[''']ve|you[''']ll|you[''']d|we[''']re|we[''']ve|we[''']ll|we[''']d|they[''']re|they[''']ve|they[''']ll|they[''']d|he[''']s|she[''']s|gonna|wanna|gotta|kinda|sorta|nothin[''']|somethin[''']|burnin[''']|growin[''']|draggin[''']|shaggin[''']|feelin[''']|whinin['''])\b/gi;
  const profRe = new Set("shit fuck fucking fucked damn damn ass hell bitch bastard crap piss dick".split(' '));

  for (const song of allSongs) {
    const text = song.lyrics.toLowerCase();
    const words = text.match(/[a-zA-Z']+/g) || [];
    allWords.push(...words);
    
    const contMatches = text.match(contRe);
    if (contMatches) contractions += contMatches.length;

    for (const w of words) {
      if (profRe.has(w)) profanity++;
    }
  }

  const total = allWords.length;
  if (!total) return { type_token_ratio: 0, total_words: 0, unique_words: 0, contraction_pct: 0, profanity_pct: 0, distinctive_words: [] };

  const unique = new Set(allWords);
  const ttr = unique.size / total;

  const wordFreq: Record<string, int> = {};
  for (const w of allWords) wordFreq[w] = (wordFreq[w] || 0) + 1;

  const distinctive = Object.entries(wordFreq)
    .filter(([w]) => !COMMON_WORDS.has(w) && w.length > 2)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 15)
    .map(x => x[0]);

  return {
    type_token_ratio: parseFloat(ttr.toFixed(3)),
    total_words: total,
    unique_words: unique.size,
    contraction_pct: parseFloat((100 * contractions / total).toFixed(1)),
    profanity_pct: parseFloat((100 * profanity / total).toFixed(1)),
    distinctive_words: distinctive
  };
}

function analyseLineLengthVariation(allSongs: SongLyrics[]): Record<string, any> {
  const sectionSyl: Record<string, int[]> = { V: [], C: [], B: [] };
  const allSyl: int[] = [];
  const examples: { short: {s:int, l:string}[], long: {s:int, l:string}[] } = { short: [], long: [] };

  for (const song of allSongs) {
    const sections = splitIntoSections(song.lyrics);
    for (const sec of sections) {
      for (const line of sec.lines) {
        const words = line.match(/[a-zA-Z']+/g);
        if (!words) continue;
        const syl = words.reduce((acc: number, w: string) => acc + countSyllables(w), 0);
        allSyl.push(syl);
        if (sectionSyl[sec.label]) sectionSyl[sec.label].push(syl);
        
        if (syl <= 4) examples.short.push({ s: syl, l: line.trim() });
        else examples.long.push({ s: syl, l: line.trim() });
      }
    }
  }

  if (!allSyl.length) return {};

  const buckets = { '1-4': 0, '5-7': 0, '8-10': 0, '11-14': 0, '15+': 0 };
  for (const s of allSyl) {
    if (s <= 4) buckets['1-4']++;
    else if (s <= 7) buckets['5-7']++;
    else if (s <= 10) buckets['8-10']++;
    else if (s <= 14) buckets['11-14']++;
    else buckets['15+']++;
  }

  const hist: Record<string, int> = {};
  for (const [k, v] of Object.entries(buckets)) hist[k] = Math.round(100 * v / allSyl.length);

  const perSection: Record<string, any> = {};
  for (const [lbl, counts] of Object.entries(sectionSyl)) {
    if (counts.length) {
      const avg = counts.reduce((a,b)=>a+b,0)/counts.length;
      const std = Math.sqrt(counts.reduce((a,b)=>a+Math.pow(b-avg,2),0)/counts.length);
      perSection[lbl] = {
        min: Math.min(...counts), max: Math.max(...counts),
        avg: parseFloat(avg.toFixed(1)), std: parseFloat(std.toFixed(1))
      };
    }
  }

  const shortEx = examples.short.sort((a,b) => a.s - b.s).slice(0,3).map(x => `(${x.s} syl) ${x.l}`);
  const longEx = examples.long.sort((a,b) => b.s - a.s).slice(0,3).map(x => `(${x.s} syl) ${x.l}`);

  return { histogram: hist, per_section: perSection, short_line_examples: shortEx, long_line_examples: longEx };
}

function analyseRepetition(allSongs: SongLyrics[]): Record<string, any> {
  let cTotal = 0, cRepeat = 0;
  let vTotal = 0, vRepeat = 0;
  const hookEx: string[] = [];
  const globalLines: string[] = [];

  for (const song of allSongs) {
    const sections = splitIntoSections(song.lyrics);
    for (const sec of sections) {
      if (sec.label === 'C' && sec.lines.length >= 2) {
        cTotal += sec.lines.length;
        const counts: Record<string, int> = {};
        sec.lines.forEach((l: string) => { const s = l.trim().toLowerCase(); counts[s] = (counts[s]||0)+1; });
        for (const [l, c] of Object.entries(counts)) {
          if (c > 1) {
            cRepeat += c;
            if (hookEx.length < 5 && l) hookEx.push(l);
          }
        }
      } else if (sec.label === 'V' && sec.lines.length >= 2) {
        vTotal += sec.lines.length;
        const counts: Record<string, int> = {};
        sec.lines.forEach((l: string) => { const s = l.trim().toLowerCase(); counts[s] = (counts[s]||0)+1; });
        for (const c of Object.values(counts)) if (c > 1) vRepeat += c;
      }
    }
    song.lyrics.split('\n').filter((l: string) => l.trim() && !SECTION_HEADER_RE.test(l)).forEach((l: string) => globalLines.push(l.trim().toLowerCase()));
  }

  const gCounts: Record<string, int> = {};
  globalLines.forEach(l => gCounts[l] = (gCounts[l]||0)+1);
  const crossRepeat = Object.values(gCounts).filter(c => c >= 3).length;

  const cPct = Math.round(100 * cRepeat / Math.max(cTotal, 1));
  const vPct = Math.round(100 * vRepeat / Math.max(vTotal, 1));
  
  let pattern = "low-repetition: choruses mostly unique lines";
  if (cPct >= 50) pattern = "heavy-hook: choruses built around repeated lines";
  else if (cPct >= 20) pattern = "moderate-hook: choruses use some repeated lines";

  return { chorus_repetition_pct: cPct, verse_repetition_pct: vPct, cross_section_repeats: crossRepeat, pattern, hook_examples: hookEx.slice(0,5) };
}

function selectRepresentativeExcerpts(allSongs: SongLyrics[], maxExcerpts = 5): string[] {
  const candidates: { score: number, title: string, text: string }[] = [];
  
  for (const song of allSongs) {
    const sections = splitIntoSections(song.lyrics);
    for (const sec of sections) {
      if ((sec.label !== 'V' && sec.label !== 'C') || sec.lines.length < 2) continue;
      const text = sec.lines.join('\n');
      const lengthScore = 1.0 - Math.abs(sec.lines.length - 6) * 0.15;
      const res = detectRhymeScheme(sec.lines);
      const rhymeScore = res.quality.perfect * 1.0 + res.quality.slant * 0.6 + res.quality.assonance * 0.3;
      const sectionName = sec.label === 'V' ? 'Verse' : 'Chorus';
      candidates.push({ score: lengthScore + rhymeScore, title: `${song.title} (${sectionName})`, text });
    }
  }

  candidates.sort((a,b) => b.score - a.score);
  const excerpts: string[] = [];
  const seen = new Set<string>();
  for (const c of candidates) {
    if (!seen.has(c.text)) {
      seen.add(c.text);
      excerpts.push(`[${c.title}]\n${c.text}`);
      if (excerpts.length >= maxExcerpts) break;
    }
  }
  return excerpts;
}

// ── LLM Caller ────────────────────────────────────────────────────────────────

const MAX_RETRIES = 2;

async function llmCallWithRetry(providerName: string, modelName: string, sysPrompt: string, usrPrompt: string, label: string, onPhase?: (p:string)=>void, onChunk?: ChunkCallback): Promise<{ raw: string, data: any }> {
  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    if (attempt > 1 && onPhase) onPhase(`${label} (retry ${attempt}/${MAX_RETRIES})…`);
    try {
      const provider = llmService.getProvider(providerName);
      const raw = await provider.call(sysPrompt, usrPrompt, modelName, onChunk);
      const data = extractJson(raw);
      if (data) return { raw, data };
    } catch (e) {
      console.warn(`${label}: Attempt ${attempt} failed`, e);
    }
  }
  return { raw: '', data: {} };
}

function buildProfilePrompt(artist: string, album: string | null, songs: SongLyrics[], ruleStats: any): string {
  let header = `Artist: ${artist}\n`;
  if (album) header += `Album: ${album}\n`;
  header += `Songs analysed: ${songs.length}\n\n=== RULE-BASED ANALYSIS ===\n`;
  
  header += `Average verse length: ${ruleStats.avg_verse_lines} lines\n`;
  header += `Average chorus length: ${ruleStats.avg_chorus_lines} lines\n`;
  header += `Top rhyme schemes: ${ruleStats.rhyme_schemes.join(', ')}\n`;
  const rq = ruleStats.rhyme_quality;
  header += `Rhyme quality breakdown: ${rq.perfect} perfect, ${rq.slant} slant, ${rq.assonance} assonance\n`;
  header += `Structure blueprints: ${ruleStats.structure_blueprints.join(', ')}\n`;
  header += `Perspective: ${ruleStats.perspective}\n`;
  
  const ms = ruleStats.meter_stats;
  header += `Meter: avg ${ms.avg_syllables_per_line} syllables/line (σ=${ms.syllable_std_dev}), ${ms.avg_words_per_line} words/line, range ${ms.line_length_range}\n`;
  
  const vs = ruleStats.vocabulary_stats;
  header += `Vocabulary: ${vs.total_words} total words, ${vs.unique_words} unique, TTR=${vs.type_token_ratio}\n`;
  header += `Contractions: ${vs.contraction_pct}% of words\nProfanity: ${vs.profanity_pct}% of words\n`;
  header += `Distinctive words: ${vs.distinctive_words.join(', ')}\n`;

  const llv = ms.line_length_variation || {};
  if (llv.histogram) {
    header += `Syllable distribution: ${Object.entries(llv.histogram).map(([k,v]) => `${k}: ${v}%`).join(', ')}\n`;
  }

  const rs = ruleStats.repetition_stats;
  if (rs) {
    header += `Chorus repetition: ${rs.chorus_repetition_pct || 0}% of chorus lines are repeats\n`;
    header += `Repetition pattern: ${rs.pattern || 'unknown'}\n`;
    if (rs.hook_examples?.length) header += `Hook examples: ${rs.hook_examples.slice(0,3).join('; ')}\n`;
  }

  let lyricsSection = "\n=== COMPLETE LYRICS ===\n\n";
  for (const s of songs) lyricsSection += `--- ${s.title} ---\n${s.lyrics}\n\n`;

  return header + lyricsSection;
}

const SUBJECT_SYSTEM_PROMPT = `You are a music analyst. For each song provided, write a ONE-SENTENCE summary of what the song is about — its core subject, not its style.

Then group all the subjects into 5-10 thematic categories that describe the range of topics this artist writes about.

Return JSON in exactly this format:
{
  "song_subjects": {
    "Song Title": "one sentence about what this specific song is about"
  },
  "subject_categories": ["category1", "category2"]
}

Be specific and concrete. Do NOT include any text outside the JSON object.`;

async function analyseSongSubjects(songs: SongLyrics[], providerName: string, modelName: string, onChunk?: ChunkCallback): Promise<{song_subjects: Record<string, string>, subject_categories: string[]}> {
  const songList = songs.map(s => `--- ${s.title} ---\n${s.lyrics.substring(0, 500)}`).join('\n\n');
  const usrPrompt = `Analyse the subjects of these ${songs.length} songs:\n\n${songList}`;
  
  const provider = llmService.getProvider(providerName);
  try {
    const raw = await provider.call(SUBJECT_SYSTEM_PROMPT, usrPrompt, modelName, onChunk);
    const data = extractJson(raw);
    if (data) return { song_subjects: data.song_subjects || {}, subject_categories: data.subject_categories || [] };
  } catch (e) {
    console.warn("Subject LLM call failed", e);
  }
  return { song_subjects: {}, subject_categories: [] };
}

function coerceStr(val: any): string {
  if (Array.isArray(val)) return val.join('\n');
  return val ? String(val) : "";
}

type int = number;

export async function buildProfile(
  artist: string,
  album: string | null,
  songs: SongLyrics[],
  providerName: string,
  modelName?: string,
  onPhase?: (phase: string) => void,
  onChunk?: ChunkCallback
): Promise<Partial<LyricsProfile>> {
  const effModel = modelName || llmService.getProvider(providerName).defaultModel;

  const struct = analyseStructure(songs);
  const rhyme = analyseRhymes(songs);
  const perspective = analysePerspective(songs);
  const meter = analyseMeter(songs);
  const vocab = analyseVocabulary(songs);
  const llv = analyseLineLengthVariation(songs);
  const rep = analyseRepetition(songs);
  const excerpts = selectRepresentativeExcerpts(songs);

  meter.line_length_variation = llv;

  const ruleStats = {
    avg_verse_lines: struct.v,
    avg_chorus_lines: struct.c,
    rhyme_schemes: rhyme.schemes,
    rhyme_quality: rhyme.quality,
    structure_blueprints: struct.blueprints,
    perspective,
    meter_stats: meter,
    vocabulary_stats: vocab,
    repetition_stats: rep
  };

  const usrPrompt = buildProfilePrompt(artist, album, songs, ruleStats);
  const merged: Record<string, any> = {};
  const raws: string[] = [];

  if (onPhase) onPhase("Analysing themes & vocabulary… (1/4)");
  const res1 = await llmCallWithRetry(providerName, effModel, PROFILE_PROMPT_1, usrPrompt, "Call 1/4", onPhase, onChunk);
  raws.push(res1.raw); Object.assign(merged, res1.data);

  if (onPhase) onPhase("Analysing tone & structure… (2/4)");
  const res2 = await llmCallWithRetry(providerName, effModel, PROFILE_PROMPT_2, usrPrompt, "Call 2/4", onPhase, onChunk);
  raws.push(res2.raw); Object.assign(merged, res2.data);

  if (onPhase) onPhase("Analysing imagery & signature… (3/4)");
  const res3 = await llmCallWithRetry(providerName, effModel, PROFILE_PROMPT_3, usrPrompt, "Call 3/4", onPhase, onChunk);
  raws.push(res3.raw); Object.assign(merged, res3.data);

  if (onPhase) onPhase("Analysing song subjects… (4/4)");
  const subjects = await analyseSongSubjects(songs, providerName, effModel, onChunk);

  const rawCombined = raws.join('\n\n---\n\n');

  return {
    artist,
    album: album || undefined,
    themes: merged.themes || [],
    common_subjects: merged.common_subjects || [],
    rhyme_schemes: merged.rhyme_schemes || rhyme.schemes,
    avg_verse_lines: merged.avg_verse_lines || struct.v,
    avg_chorus_lines: merged.avg_chorus_lines || struct.c,
    vocabulary_notes: coerceStr(merged.vocabulary_notes),
    tone_and_mood: coerceStr(merged.tone_and_mood),
    structural_patterns: coerceStr(merged.structural_patterns),
    additional_notes: coerceStr(merged.additional_notes),
    raw_summary: coerceStr(merged.raw_summary || rawCombined),
    // Extra parsed data
    structure_blueprints: struct.blueprints,
    perspective,
    meter_stats: meter,
    vocabulary_stats: vocab,
    representative_excerpts: excerpts,
    narrative_techniques: coerceStr(merged.narrative_techniques),
    imagery_patterns: coerceStr(merged.imagery_patterns),
    signature_devices: coerceStr(merged.signature_devices),
    emotional_arc: coerceStr(merged.emotional_arc),
    rhyme_quality: rhyme.quality as any, // bypassing strict TS checking for complex types
    song_subjects: subjects.song_subjects as any,
    subject_categories: subjects.subject_categories,
    repetition_stats: rep as any,
  };
}

/**
 * Re-runs all local (non-LLM) statistical analysis on an existing profile's
 * lyrics set and patches the profile_data in place. This is fast — no LLM calls.
 */
export function recalculateProfileStats(songs: SongLyrics[], profileData: any): any {
  const vocab = analyseVocabulary(songs);
  const meter = analyseMeter(songs);
  const rhyme = analyseRhymes(songs);
  const struct = analyseStructure(songs);
  const rep = analyseRepetition(songs);
  const lineVar = analyseLineLengthVariation(songs);
  const perspective = analysePerspective(songs);

  return {
    ...profileData,
    // Overwrite computed stats
    vocabulary_stats: vocab,
    meter_stats: meter,
    rhyme_schemes: rhyme.schemes,
    rhyme_quality: rhyme.quality as any,
    avg_verse_lines: struct.v,
    avg_chorus_lines: struct.c,
    repetition_stats: rep as any,
    line_length_variation: lineVar,
    perspective,
  };
}
