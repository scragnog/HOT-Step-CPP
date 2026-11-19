import { useEffect, useState, useSyncExternalStore } from 'react';
import { prepareV2Problem } from '../../../../../server/src/contracts/yue2Prepare';
import {
  confirmLyrics, confirmStructure, editLyrics, editStructure, exportPrepare, migrateLegacyPrepare,
  structureConfirmed, structureHash, type LyricEdit, type PrepareV2, type StructureEdit,
} from '@/utils/lyrics/yue2Prepare';
import { persist, restore } from '@/components/cover-studio/source/coverStudioUtils';

export const PREPARE_DRAFT_KEY = 'cover-studio-prepareV2';
const CUTS_KEY = 'cover-studio-scoreCuts';
export type PrepareStep = 'structure' | 'lyrics' | 'ready';
export type PrepareStructuralOp = StructureEdit | { kind: 'cut'; abc: string };
export type PrepareSuggestions = Record<string, string[] | 'instrumental'>;
export interface PrepareGuard { sourceKey: string; structureHash: string; lyricRevision: number }
export interface PrepareCut { abc: string; lyrics: string; after: string; afterLyrics: string; removedMusic?: boolean }
export interface PrepareInitial {
  sourceKey: string;
  abc: string;
  lyrics?: string;
  prepareV2?: unknown;
  cuts?: PrepareCut[];
}
interface Snapshot {
  doc: PrepareV2;
  cuts: PrepareCut[];
  step: PrepareStep;
  editingStructure: boolean;
  error: string;
}

function seed(initial: PrepareInitial): Snapshot {
  const saved = initial.prepareV2;
  const savedMatches = !prepareV2Problem(saved) && (saved as PrepareV2).sourceKey === initial.sourceKey;
  const doc = savedMatches
    ? structuredClone(saved as PrepareV2)
    : migrateLegacyPrepare(initial.sourceKey, initial.abc, initial.lyrics ?? '');
  const confirmed = structureConfirmed(doc);
  // Persisted cuts accompany the persisted score. A different source must
  // never inherit their disabled recording controls.
  return { doc, cuts: structuredClone(saved != null && !savedMatches ? [] : initial.cuts ?? []), step: !confirmed ? 'structure'
    : 'error' in exportPrepare(doc) ? 'lyrics' : 'ready', editingStructure: !confirmed, error: '' };
}

/** The hook's synchronous owner. Each accepted operation is one history entry,
 * including a whole suggestion batch. Failed operations leave the document intact. */
export function createYue2Prepare(initial: PrepareInitial) {
  let state = seed(initial);
  let past: Snapshot[] = [], future: Snapshot[] = [];
  const listeners = new Set<() => void>();
  const publish = () => listeners.forEach(listener => listener());
  const fail = (error: string) => { state = { ...state, error }; publish(); return false; };
  const commit = (next: Snapshot) => {
    if (JSON.stringify(next) === JSON.stringify(state)) return true;
    past.push(state); future = []; state = next; publish(); return true;
  };
  const accept = (result: PrepareV2 | { error: string }, step: PrepareStep, editingStructure: boolean,
    cuts = state.cuts) => 'error' in result ? fail(result.error)
      : commit({ doc: result, cuts, step, editingStructure, error: '' });
  const guard = (): PrepareGuard => ({ sourceKey: state.doc.sourceKey,
    structureHash: structureHash(state.doc.abc), lyricRevision: state.doc.lyricRevision });
  const matches = (captured: PrepareGuard) => {
    const current = guard();
    return captured.sourceKey === current.sourceKey && captured.structureHash === current.structureHash
      && captured.lyricRevision === current.lyricRevision;
  };

  return {
    getSnapshot: () => state,
    subscribe(listener: () => void) { listeners.add(listener); return () => { listeners.delete(listener); }; },
    get canUndo() { return past.length > 0; },
    get canRedo() { return future.length > 0; },
    captureGuard: guard,
    matchesGuard: matches,
    /** Step navigation alone never unlocks a confirmed score. */
    showStep(step: PrepareStep) {
      if (step !== 'structure' && (state.editingStructure || !structureConfirmed(state.doc)))
        return fail('Confirm the score structure first.');
      if (step === 'ready' && 'error' in exportPrepare(state.doc)) return fail('Confirm the lyrics first.');
      state = { ...state, step, error: '' }; publish(); return true;
    },
    beginStructureEdit() {
      if (state.editingStructure) return true;
      const doc = { ...state.doc, confirmed: { structure: null, lyrics: null } };
      return accept(doc, 'structure', true);
    },
    applyStructuralOp(op: PrepareStructuralOp, captured?: PrepareGuard) {
      if (captured && !matches(captured)) return fail('The score or lyrics changed; request a new result.');
      if (!state.editingStructure || state.step !== 'structure') return fail('Choose Edit structure first.');
      const result = editStructure(state.doc, op.kind === 'cut' ? { kind: 'replace', abc: op.abc } : op);
      if ('error' in result) return fail(result.error);
      if (structureHash(result.abc) === structureHash(state.doc.abc)) return true;
      const cuts = op.kind === 'cut' ? [...state.cuts, { abc: state.doc.abc, lyrics: state.doc.paste,
        after: result.abc, afterLyrics: result.paste, removedMusic: true }] : state.cuts;
      return accept(result, 'structure', true, cuts);
    },
    applyLyricOp(op: LyricEdit) {
      if (state.editingStructure || !structureConfirmed(state.doc)) return fail('Confirm the score structure first.');
      return accept(editLyrics(state.doc, op), 'lyrics', false);
    },
    confirmStructure() { return accept(confirmStructure(state.doc), 'lyrics', false); },
    confirmLyrics() {
      if (state.editingStructure) return fail('Confirm the score structure first.');
      return accept(confirmLyrics(state.doc), 'ready', false);
    },
    /** Accept only the selected proposals. Repeated block ids become independent
     * copies through the reducer, and the entire batch can be undone at once. */
    applySuggestions(captured: PrepareGuard, proposals: PrepareSuggestions, fromRecording = false) {
      if (!matches(captured)) return fail('The source, score or lyrics changed; request new suggestions.');
      if (fromRecording && state.cuts.some(cut => cut.removedMusic !== false))
        return fail('Recording alignment is unavailable after a cut.');
      if (state.editingStructure || !structureConfirmed(state.doc)) return fail('Confirm the score structure first.');
      let doc = state.doc;
      const apply = (op: LyricEdit) => {
        const next = editLyrics(doc, op);
        if ('error' in next) return next.error;
        doc = next; return '';
      };
      // Clear all accepted targets before assigning, so swapping two proposals
      // cannot silently discard words or depend on object iteration order.
      for (const sectionId of Object.keys(proposals)) {
        const error = apply({ kind: 'clear', sectionId });
        if (error) return fail(error);
      }
      const used = new Set<string>();
      for (const [sectionId, blocks] of Object.entries(proposals)) {
        if (blocks === 'instrumental') {
          const error = apply({ kind: 'instrumental', sectionId });
          if (error) return fail(error);
          continue;
        }
        if (!blocks.length || new Set(blocks).size !== blocks.length) return fail('A suggestion needs distinct lyric blocks.');
        for (const blockId of blocks) {
          // Reuse needs an empty target. Make the copy there temporarily, then
          // append it to the ordered blocks already collected for that row.
          if (used.has(blockId)) {
            const previous = doc.assignments[sectionId];
            let error = apply({ kind: 'clear', sectionId });
            if (!error) error = apply({ kind: 'reuse', blockId, sectionIds: [sectionId] });
            if (error) return fail(error);
            const copy = doc.assignments[sectionId];
            if (previous?.kind === 'words' && copy?.kind === 'words') {
              const copyId = copy.blockIds[0];
              error = apply({ kind: 'clear', sectionId });
              for (const id of [...previous.blockIds, copyId]) {
                if (!error) error = apply({ kind: 'assign', blockId: id, sectionId, mode: 'append' });
              }
              if (error) return fail(error);
            }
          } else {
            const error = apply({ kind: 'assign', blockId, sectionId, mode: 'append' });
            if (error) return fail(error);
            used.add(blockId);
          }
        }
      }
      return accept(doc, 'lyrics', false);
    },
    undo() {
      const previous = past.pop();
      if (!previous) return false;
      future.push(state); state = { ...previous, error: '' }; publish(); return true;
    },
    redo() {
      const next = future.pop();
      if (!next) return false;
      past.push(state); state = { ...next, error: '' }; publish(); return true;
    },
    /** Call from the existing Cover studio draft picker's apply callback.
     * A loaded draft starts a fresh history, so undo cannot cross sources. */
    restoreDraft(fields: Record<string, unknown>, source: Omit<PrepareInitial, 'prepareV2' | 'cuts'>) {
      const saved = fields[PREPARE_DRAFT_KEY];
      if (saved != null && prepareV2Problem(saved)) return fail(prepareV2Problem(saved)!);
      const rawCuts = fields[CUTS_KEY];
      if (rawCuts !== undefined && (!Array.isArray(rawCuts) || rawCuts.some(cut => !cut ||
        !['abc', 'lyrics', 'after', 'afterLyrics'].every(key => typeof cut[key] === 'string'))))
        return fail('The saved cut history is malformed.');
      state = seed({ ...source, abc: typeof fields['cover-studio-sheetAbc'] === 'string'
        ? fields['cover-studio-sheetAbc'] : source.abc,
        lyrics: typeof fields['cover-studio-lyrics'] === 'string' ? fields['cover-studio-lyrics'] : source.lyrics,
        prepareV2: saved, cuts: rawCuts as PrepareCut[] | undefined });
      past = []; future = []; publish(); return true;
    },
  };
}

/** Merge draftFields into CoverStudio's existing useStudioDraftMirror call;
 * pass its loaded fields to restoreDraft. Do not create a second Cover mirror.
 * ABC changes for the same source are explicit operations, never prop effects. */
export function useYue2Prepare(initial: PrepareInitial & { hasRecording?: boolean }) {
  const [owner, setOwner] = useState(() => ({ sourceKey: initial.sourceKey, controller: createYue2Prepare({ ...initial,
    prepareV2: initial.prepareV2 ?? restore('prepareV2', null),
    cuts: initial.cuts ?? (initial.prepareV2 === undefined ? restore('scoreCuts', []) : []) }) }));
  let active = owner;
  if (owner.sourceKey !== initial.sourceKey) {
    active = { sourceKey: initial.sourceKey, controller: createYue2Prepare(initial) };
    setOwner(active);
  }
  const controller = active.controller;
  const snapshot = useSyncExternalStore(controller.subscribe, controller.getSnapshot, controller.getSnapshot);
  useEffect(() => {
    persist('prepareV2', snapshot.doc);
    persist('scoreCuts', snapshot.cuts);
  }, [snapshot.doc, snapshot.cuts]);
  const recordingAvailable = !!initial.hasRecording && !snapshot.cuts.some(cut => cut.removedMusic !== false);
  return { ...snapshot, ...controller, canUndo: controller.canUndo, canRedo: controller.canRedo,
    recordingSeekAvailable: recordingAvailable, recordingAlignmentAvailable: recordingAvailable,
    draftFields: { [PREPARE_DRAFT_KEY]: snapshot.doc, [CUTS_KEY]: snapshot.cuts } };
}
