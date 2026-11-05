import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import ts from 'typescript';
import { snapshotOptions } from '../../queries/cover-lyrics/reads';
import { queryClient } from '../../queries/client';
import { leadSheetReadinessPath, leadSheetScoreUpdate } from './leadSheetSource';

// Execute the actual orchestration handlers without importing the studio's audio/CSS dependencies.
function handler(name: string, state: Record<string, unknown>): () => Promise<void> {
  const source = ts.createSourceFile('CoverStudio.tsx', readFileSync(new URL('./CoverStudio.tsx', import.meta.url), 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  let arrow: ts.Expression | undefined;
  const visit = (node: ts.Node) => {
    if (ts.isVariableDeclaration(node) && node.name.getText(source) === name) arrow = node.initializer;
    ts.forEachChild(node, visit);
  };
  visit(source);
  assert.ok(arrow, name);
  const body = ts.transpileModule(`const run = ${arrow.getText(source)}; return run;`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None },
  }).outputText;
  return new Function('state', `with (state) { ${body} }`)(state);
}

test('readiness queries reach the complete mounted API route for default and explicit sources', async () => {
  const original = globalThis.fetch;
  const urls: string[] = [];
  globalThis.fetch = async url => {
    urls.push(String(url));
    return new Response(JSON.stringify({ source: 'sheetsage2', ready: true }));
  };
  queryClient.clear();
  try {
    for (const source of [undefined, 'sheetsage2', 'sheetsage2-refined', 'muscriptor', 'yourmt3-chordnet'] as const) {
      await queryClient.query(snapshotOptions('cover', leadSheetReadinessPath(source)));
      assert.equal(urls.at(-1), `/api/yue2-cover/readiness${source ? `?source=${source}` : ''}`);
    }
  } finally { globalThis.fetch = original; queryClient.clear(); }
});

test('a library transcription retains its draft, then approval and source change use successive revisions', async () => {
  const calls: unknown[][] = [];
  const state: Record<string, any> = {
    token: 'session', sourceAudioUrl: '/audio/library.wav', sourceAssetId: '', sourceSongId: 'library',
    coverDraftId: '', coverDraftRevision: 0, sourceChangeRef: { current: false },
    sourceKey: 'library-source', sourceKeyRef: { current: 'library-source' }, sourceLookupRef: { current: 1 },
    sheetRequestRef: { current: 0 }, coverWorkflowJobRef: { current: '' }, sourceHydratedRef: { current: '' },
    sheetAbc: '', selectedSource: 'sheetsage2', advancedMode: false, heartmulaMode: false,
    sourceMode: 'audio', sourceFileName: 'library.wav', sheetPreparing: false, approvedSheet: null,
    prepareYue2Source: async () => ({ audioUrl: '/audio/library.wav', songId: 'library' }),
    apiClient: { get: async () => new Blob(['audio']) },
    uploadMut: { mutateAsync: async () => ({ asset_id: 'library-asset' }) },
    leadSheetScoreUpdate,
    coverWorkflowApi: {
      open: async () => ({ job: { id: 'open' } }),
      transcribe: async (...args: unknown[]) => { calls.push(['transcribe', ...args]); return { job: { id: 'transcribe' } }; },
      approveScore: async (...args: unknown[]) => {
        calls.push(['approve', ...args]); return { document: { revision: 3 }, abc: 'score', sourceId: 'asset', sourceLabel: 'library' };
      },
      source: async (...args: unknown[]) => { calls.push(['source', ...args]); return { job: { id: 'source' } }; },
    },
    waitForCoverJob: async (id: string) => ({ result: id === 'open'
      ? { documentId: 'library-draft', revision: 1, leadSheetSource: 'sheetsage2' }
      : id === 'transcribe' ? { documentId: 'library-draft', revision: 2, abc: 'score' }
      : { documentId: 'library-draft', revision: 4, leadSheetSource: 'muscriptor' } }),
  };
  for (const name of ['SheetError', 'SheetPreparing', 'CoverDraftId', 'CoverDraftRevision', 'ApprovedSheet', 'SheetAbc',
    'ScoreSections', 'SheetScoreSource', 'ChordQuality', 'LeadSheetSource', 'SheetAudioUrl', 'SourceChanging']) {
    const key = name[0].toLowerCase() + name.slice(1);
    state[`set${name}`] = (value: unknown) => { state[key] = value; };
  }
  state.acceptTranscript = (abc: string) => { state.sheetAbc = abc; };
  await handler('handleTranscribe', state)();
  assert.equal(state.sheetError, '');
  assert.equal(state.coverDraftId, 'library-draft');
  assert.equal(state.coverDraftRevision, 2);
  await handler('handleApproveSheet', state)();
  assert.equal(state.sheetError, '');
  assert.equal(state.coverDraftRevision, 3);
  assert.ok(state.approvedSheet);
  await (handler('changeLeadSheetSource', state) as (choice: string) => Promise<void>)('muscriptor');
  assert.equal(state.sheetError, '');
  assert.deepEqual(calls, [
    ['transcribe', 'library-draft', 1, false], ['approve', 'library-draft', 2, 'score'],
    ['source', 'library-draft', 3, 'muscriptor'],
  ]);
  assert.equal(state.coverDraftRevision, 4);
  assert.equal(state.leadSheetSource, 'muscriptor');
  assert.equal(state.approvedSheet, null);
  assert.equal(state.sheetAbc, '');
});
