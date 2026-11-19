// The Prepare document's editing model (contract: server/src/contracts/
// yue2Prepare.ts). Structure edits change the score alone and carry section
// ids across; lyric edits change words and the assignment map alone. Both
// are pure: each returns a new document or an error, so the caller owns undo.
//
// One rule joins the two steps: a structure edit that changes a section
// (label, music or boundary) clears that section's words and flags it. The
// words go back to Unassigned, never away. Every change to words, the map,
// instrumental or excluded status bumps lyricRevision, which is what makes an
// older lyric confirmation stale.

import {
  PREPARE_V2_VERSION, assignmentProblem, normalizeAbc, preparedPair, prepareSectionLabels, structureHash, structureProblem,
  type PrepareAssignment, type PrepareLyricBlock, type PrepareV2, type PreparedPair,
} from '../../../../server/src/contracts/yue2Prepare';
import { mergeScoreSections, moveSectionStart, parseScore, relabelScoreSection, splitScoreSection } from '@/utils/abc/abcSectionMove';

export type { PrepareAssignment, PrepareLyricBlock, PrepareV2, PreparedPair };
export { structureHash };

type Result = PrepareV2 | { error: string };

// ── Lyric blocks from pasted text ──

const TAG_ONLY = /^\s*\[([^\]\n]+)\]\s*$/;

/** The paste as blocks: each `[Tag]` line opens one, words above the first
 *  tag form an untagged one. A tag with no words makes no block. */
function pasteBlocks(paste: string, nextId: number): { blocks: Array<PrepareLyricBlock & { tagIndex: number }>; nextId: number; tags: string[] } {
  const blocks: Array<PrepareLyricBlock & { tagIndex: number }> = [], tags: string[] = [];
  let tag: string | null = null, lines: string[] = [];
  const close = () => {
    const text = lines.join('\n').replace(/^\n+|\s+$/g, '');
    if (text) blocks.push({ id: `b${nextId++}`, text, tag, tagIndex: tags.length - 1 });
    lines = [];
  };
  for (const line of paste.replace(/\r\n?/g, '\n').split('\n')) {
    const match = TAG_ONLY.exec(line);
    if (!match) { lines.push(line); continue; }
    close();
    tag = match[1].trim();
    tags.push(tag);
  }
  close();
  return { blocks, nextId, tags };
}

const strip = ({ tagIndex: _, ...block }: PrepareLyricBlock & { tagIndex: number }): PrepareLyricBlock => block;

/** A new preparation of `abc` with `paste` as unassigned blocks. */
export function createPrepare(sourceKey: string, abc: string, paste = ''): PrepareV2 {
  const count = prepareSectionLabels(abc).length;
  const sectionIds = Array.from({ length: count }, (_, i) => `s${i + 1}`);
  const pasted = pasteBlocks(paste, count + 1);
  return {
    version: PREPARE_V2_VERSION, sourceKey, abc: normalizeAbc(abc), sectionIds, paste,
    blocks: pasted.blocks.map(strip), assignments: {}, excluded: [], lyricRevision: 0, flagged: [],
    nextId: pasted.nextId, confirmed: { structure: null, lyrics: null },
  };
}

/** A version 2 preparation from a legacy draft's saved score and lyrics; it
 *  never transcribes. When the lyric tags match the score's sections one for
 *  one, block i is proposed for section i (a tag with no words proposes
 *  Instrumental); otherwise every block starts Unassigned. Nothing comes
 *  confirmed: the first V2 confirmation is always the user's. */
export function migrateLegacyPrepare(sourceKey: string, sheetAbc: string, lyrics: string): PrepareV2 {
  const p = createPrepare(sourceKey, sheetAbc, lyrics);
  const labels = prepareSectionLabels(p.abc);
  // The same parse createPrepare made, so block k here is p.blocks[k].
  const pasted = pasteBlocks(lyrics, 0);
  const matches = pasted.tags.length === labels.length && pasted.blocks.every(b => b.tagIndex >= 0) &&
    labels.every((label, i) => label.toLowerCase() === pasted.tags[i].toLowerCase());
  if (!matches) return p;
  const assignments: Record<string, PrepareAssignment> = {};
  p.sectionIds.forEach((id, i) => {
    const k = pasted.blocks.findIndex(b => b.tagIndex === i);
    assignments[id] = k >= 0 ? { kind: 'words', blockIds: [p.blocks[k].id] } : { kind: 'instrumental' };
  });
  return { ...p, assignments };
}

// ── Structure edits ──

export type StructureEdit =
  | { kind: 'merge'; index: number; keep: 'first' | 'second' }
  | { kind: 'relabel'; index: number; label: string }
  | { kind: 'split'; index: number; bar: number; label: string }
  | { kind: 'move'; index: number; bar: number }
  /** Any other rewrite of the ABC: the raw editor, bar regrouping. */
  | { kind: 'replace'; abc: string };

/** Each section's text, marker line to the line before the next marker. */
function sectionTexts(abc: string): string[] {
  const lines = normalizeAbc(abc).split('\n');
  const starts = lines.flatMap((line, i) => line.trim().startsWith('%') && line.trim().slice(1).trim() ? [i] : []);
  return starts.map((start, k) => lines.slice(start, starts[k + 1] ?? lines.length).join('\n'));
}

/** The document after its score became `abc` with sections `ids`: a section
 *  whose text changed, or that is new, loses its words and is flagged if it
 *  had any; sections gone take their assignments with them. */
function withStructure(p: PrepareV2, abc: string, ids: string[], nextId = p.nextId): PrepareV2 {
  const before = new Map(p.sectionIds.map((id, i) => [id, sectionTexts(p.abc)[i]]));
  const after = sectionTexts(abc);
  const assignments: Record<string, PrepareAssignment> = {};
  const flagged = new Set(p.flagged.filter(id => ids.includes(id)));
  let changed = Object.keys(p.assignments).some(id => !ids.includes(id));
  ids.forEach((id, i) => {
    const a = p.assignments[id];
    if (!a) return;
    if (before.get(id) === after[i]) { assignments[id] = a; return; }
    flagged.add(id);
    changed = true;
  });
  return { ...p, abc: normalizeAbc(abc), sectionIds: ids, assignments, flagged: [...flagged], nextId,
    lyricRevision: p.lyricRevision + (changed ? 1 : 0) };
}

/** `edit` applied to the score. Lyric text is never read or changed. */
export function editStructure(p: PrepareV2, edit: StructureEdit): Result {
  const ids = p.sectionIds;
  switch (edit.kind) {
    case 'merge': {
      const out = mergeScoreSections(p.abc, edit.index, edit.keep);
      return 'error' in out ? out : withStructure(p, out.abc, ids.filter((_, i) => i !== edit.index + 1));
    }
    case 'relabel': {
      const out = relabelScoreSection(p.abc, edit.index, edit.label);
      return 'error' in out ? out : withStructure(p, out.abc, ids);
    }
    case 'split': {
      const out = splitScoreSection(p.abc, edit.index, edit.bar, edit.label);
      return 'error' in out ? out
        : withStructure(p, out.abc, [...ids.slice(0, edit.index + 1), `s${p.nextId}`, ...ids.slice(edit.index + 1)], p.nextId + 1);
    }
    case 'move': {
      const out = moveSectionStart(p.abc, edit.index, edit.bar);
      return 'error' in out ? out : withStructure(p, out.abc, ids);
    }
    case 'replace': {
      // Ids follow position only while the section count holds; otherwise
      // which section is which is a guess, so every section is new.
      const count = prepareSectionLabels(edit.abc).length;
      if (count === ids.length) return withStructure(p, edit.abc, ids);
      return withStructure(p, edit.abc, Array.from({ length: count }, (_, i) => `s${p.nextId + i}`), p.nextId + count);
    }
  }
}

/** Whether the structure confirmation matches the score as it stands. */
export const structureConfirmed = (p: PrepareV2) => p.confirmed.structure?.structureHash === structureHash(p.abc);

/** The document with its structure confirmed, or why it cannot be. */
export function confirmStructure(p: PrepareV2): Result {
  const problem = structureProblem(p);
  if (problem) return { error: problem };
  const parsed = parseScore(p.abc);
  if (typeof parsed === 'string') return { error: parsed };
  if (parsed.items.filter(item => item.kind === 'marker').length !== p.sectionIds.length)
    return { error: 'The score has a section marker above its K: line; fix it in the ABC text.' };
  return { ...p, confirmed: { ...p.confirmed, structure: { structureHash: structureHash(p.abc) } } };
}

// ── Lyric edits ──

export type LyricEdit =
  /** Puts a block in a section, taking it from wherever it was. A section
   *  that already has words needs `mode`: append after them, or swap them
   *  into the block's old place. */
  | { kind: 'assign'; blockId: string; sectionId: string; mode?: 'append' | 'swap' }
  | { kind: 'unassign'; blockId: string }
  | { kind: 'instrumental'; sectionId: string }
  /** Clears a section; its words return to Unassigned. */
  | { kind: 'clear'; sectionId: string }
  | { kind: 'exclude'; blockId: string }
  | { kind: 'include'; blockId: string }
  | { kind: 'text'; blockId: string; text: string }
  /** Splits a block before its `line`-th line (1 to its line count - 1). */
  | { kind: 'splitBlock'; blockId: string; line: number }
  /** Joins a block with the one after it in block order. */
  | { kind: 'joinNext'; blockId: string }
  /** Accepted repeat: independent copies of a block for empty sections. */
  | { kind: 'reuse'; blockId: string; sectionIds: string[] }
  /** A new paste. Blocks whose text is unchanged, and unique on both sides,
   *  keep their place; the rest come in Unassigned. */
  | { kind: 'paste'; paste: string };

const wordsOf = (a: PrepareAssignment | undefined) => a?.kind === 'words' ? a.blockIds : [];

/** The map with `blockId` taken out of every section; emptied sections lose
 *  their entry. Returns where it was. */
function takeOut(assignments: Record<string, PrepareAssignment>, blockId: string): { map: Record<string, PrepareAssignment>; from?: string; at: number } {
  const map: Record<string, PrepareAssignment> = {};
  let from: string | undefined, at = -1;
  for (const [id, a] of Object.entries(assignments)) {
    const k = wordsOf(a).indexOf(blockId);
    if (k < 0) { map[id] = a; continue; }
    from = id; at = k;
    const rest = wordsOf(a).filter(b => b !== blockId);
    if (rest.length) map[id] = { kind: 'words', blockIds: rest };
  }
  return { map, from, at };
}

const tagLike = (text: string) => text.split(/\r?\n/).some(line => /^\s*\[[^\]\n]*\]/.test(line));

/** `edit` applied to the words and their map. The score is never changed. */
export function editLyrics(p: PrepareV2, edit: LyricEdit): Result {
  const block = 'blockId' in edit ? p.blocks.find(b => b.id === edit.blockId) : undefined;
  if ('blockId' in edit && !block) return { error: 'That lyric block is gone.' };
  const sections = 'sectionId' in edit ? [edit.sectionId] : edit.kind === 'reuse' ? edit.sectionIds : [];
  if (sections.some(id => !p.sectionIds.includes(id))) return { error: 'That section is not in the score.' };
  const done = (next: Partial<PrepareV2>, touched: string[] = []): PrepareV2 =>
    ({ ...p, ...next, flagged: p.flagged.filter(id => !touched.includes(id)), lyricRevision: p.lyricRevision + 1 });
  const excludedWithout = (id: string) => p.excluded.filter(x => x !== id);
  switch (edit.kind) {
    case 'assign': {
      const target = p.assignments[edit.sectionId];
      if (wordsOf(target).includes(edit.blockId)) return p;
      if (target?.kind === 'words' && !edit.mode) return { error: 'That section already has words; append or swap.' };
      const { map, from, at } = takeOut(p.assignments, edit.blockId);
      const current = wordsOf(map[edit.sectionId]);
      if (edit.mode === 'append' && current.length) map[edit.sectionId] = { kind: 'words', blockIds: [...current, edit.blockId] };
      else {
        map[edit.sectionId] = { kind: 'words', blockIds: [edit.blockId] };
        if (edit.mode === 'swap' && from && current.length) {
          const left = wordsOf(map[from]);
          map[from] = { kind: 'words', blockIds: [...left.slice(0, at), ...current, ...left.slice(at)] };
        }
      }
      return done({ assignments: map, excluded: excludedWithout(edit.blockId) }, [edit.sectionId]);
    }
    case 'unassign': return done({ assignments: takeOut(p.assignments, edit.blockId).map });
    case 'instrumental': {
      if (wordsOf(p.assignments[edit.sectionId]).length) return { error: 'Clear the section\'s words first.' };
      return done({ assignments: { ...p.assignments, [edit.sectionId]: { kind: 'instrumental' } } }, [edit.sectionId]);
    }
    case 'clear': {
      const { [edit.sectionId]: _, ...map } = p.assignments;
      return done({ assignments: map });
    }
    case 'exclude': return done({ assignments: takeOut(p.assignments, edit.blockId).map, excluded: [...excludedWithout(edit.blockId), edit.blockId] });
    case 'include': return done({ excluded: excludedWithout(edit.blockId) });
    case 'text': {
      const text = edit.text.replace(/\r\n?/g, '\n');
      if (tagLike(text)) return { error: 'A line in brackets reads as a section tag; remove the brackets.' };
      if (!text.trim()) return { error: 'A block needs words; exclude it instead.' };
      return done({ blocks: p.blocks.map(b => b.id === edit.blockId ? { ...b, text } : b) });
    }
    case 'splitBlock': {
      const lines = block!.text.split('\n');
      if (!Number.isInteger(edit.line) || edit.line < 1 || edit.line >= lines.length) return { error: 'Split between two lines of the block.' };
      const head = lines.slice(0, edit.line).join('\n').trimEnd(), tail = lines.slice(edit.line).join('\n').trim();
      if (!head.trim() || !tail) return { error: 'Both halves need words.' };
      const added: PrepareLyricBlock = { id: `b${p.nextId}`, text: tail, tag: block!.tag, ...(block!.copyOf ? { copyOf: block!.copyOf } : {}) };
      const k = p.blocks.indexOf(block!);
      const map = Object.fromEntries(Object.entries(p.assignments).map(([id, a]) => [id, a.kind === 'words' && a.blockIds.includes(block!.id)
        ? { kind: 'words' as const, blockIds: a.blockIds.flatMap(b => b === block!.id ? [b, added.id] : [b]) } : a]));
      return done({ blocks: [...p.blocks.slice(0, k), { ...block!, text: head }, added, ...p.blocks.slice(k + 1)], assignments: map,
        excluded: p.excluded.includes(block!.id) ? [...p.excluded, added.id] : p.excluded, nextId: p.nextId + 1 });
    }
    case 'joinNext': {
      const k = p.blocks.indexOf(block!), second = p.blocks[k + 1];
      if (!second) return { error: 'There is no block after this one.' };
      const blocks = p.blocks.filter(b => b !== second).map(b => b === block ? { ...b, text: `${b.text}\n${second.text}` } : b);
      return done({ blocks, assignments: takeOut(p.assignments, second.id).map, excluded: p.excluded.filter(x => x !== second.id) });
    }
    case 'reuse': {
      if (edit.sectionIds.some(id => p.assignments[id])) return { error: 'Reuse words only in sections that have none.' };
      let nextId = p.nextId;
      const copies = edit.sectionIds.map(() => ({ id: `b${nextId++}`, text: block!.text, tag: block!.tag, copyOf: block!.copyOf ?? block!.id }));
      const map = { ...p.assignments };
      edit.sectionIds.forEach((id, i) => { map[id] = { kind: 'words', blockIds: [copies[i].id] }; });
      return done({ blocks: [...p.blocks, ...copies], assignments: map, nextId }, edit.sectionIds);
    }
    case 'paste': {
      const pasted = pasteBlocks(edit.paste, p.nextId);
      const count = (list: Array<{ text: string }>, text: string) => list.filter(b => b.text === text).length;
      const kept = new Map<string, string>();
      for (const b of pasted.blocks) {
        const old = p.blocks.find(o => o.text === b.text);
        if (old && count(p.blocks, b.text) === 1 && count(pasted.blocks, b.text) === 1) kept.set(b.id, old.id);
      }
      const blocks = pasted.blocks.map(b => kept.has(b.id) ? { ...p.blocks.find(o => o.id === kept.get(b.id))!, tag: b.tag } : strip(b));
      const live = new Set(blocks.map(b => b.id));
      const map: Record<string, PrepareAssignment> = {}, flagged = new Set(p.flagged);
      for (const [id, a] of Object.entries(p.assignments)) {
        if (a.kind === 'instrumental') { map[id] = a; continue; }
        const left = a.blockIds.filter(b => live.has(b));
        if (left.length) map[id] = { kind: 'words', blockIds: left };
        if (left.length !== a.blockIds.length) flagged.add(id);
      }
      return { ...p, paste: edit.paste, blocks, assignments: map, excluded: p.excluded.filter(id => live.has(id)),
        flagged: [...flagged], nextId: pasted.nextId, lyricRevision: p.lyricRevision + 1 };
    }
  }
}

/** The document with its lyrics confirmed, or why they cannot be. */
export function confirmLyrics(p: PrepareV2): Result {
  if (!structureConfirmed(p)) return { error: 'Confirm the score structure first.' };
  const problem = assignmentProblem(p);
  if (problem) return { error: problem };
  return { ...p, confirmed: { ...p.confirmed, lyrics: { structureHash: structureHash(p.abc), lyricRevision: p.lyricRevision } } };
}

/** The score and tagged lyrics Generate and Build send, through the same
 *  check the server runs on them. */
export const exportPrepare = (p: PrepareV2): PreparedPair | { error: string } => preparedPair(p);
