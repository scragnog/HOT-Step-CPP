import assert from 'node:assert/strict';
import test from 'node:test';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { LeadSheetSourcePicker } from './LeadSheetSourcePicker';
import { coverWorkflowApi } from '../../services/coverWorkflowApi';
import { leadSheetReadinessPath, leadSheetScoreUpdate, LEAD_SHEET_SOURCES, selectedLeadSheetSource } from './leadSheetSource';

test('the shared picker displays every stored source, including unavailable native models', () => {
  for (const [index, source] of LEAD_SHEET_SOURCES.entries()) {
    const html = renderToStaticMarkup(createElement(LeadSheetSourcePicker, {
      value: source, disabled: false, onChange: () => {},
    }));
    assert.ok(html.includes(['SheetSage2', 'SheetSage2-Refined', 'MuScriptor', 'YourMT3+ / ChordNet'][index]));
    assert.ok(html.includes('Lead-sheet source'));
    assert.ok(!html.includes('<select'));
    assert.ok(!html.includes('disabled=""'), 'readiness does not disable a source choice');
  }
});

test('only an unset draft follows the Models default; saved choices survive default changes', () => {
  assert.equal(selectedLeadSheetSource(undefined, 'sheetsage2-refined'), 'sheetsage2-refined');
  assert.equal(selectedLeadSheetSource(undefined, 'sheetsage2'), 'sheetsage2');
  for (const source of LEAD_SHEET_SOURCES) {
    assert.equal(selectedLeadSheetSource(source, 'sheetsage2'), source);
    assert.equal(selectedLeadSheetSource(source, 'muscriptor'), source);
    assert.equal(new URLSearchParams(leadSheetReadinessPath(source).split('?')[1]).get('source'), source);
  }
  assert.equal(selectedLeadSheetSource('obsolete-source', 'muscriptor'), 'muscriptor');
});

test('source changes drop approval and score provenance; incomplete results keep editable ABC', () => {
  const previous = { abc: 'old score', approved: { abc: 'old score' }, scoreSource: 'dataset', chordQuality: 'provided' };
  assert.deepEqual({ ...previous, ...leadSheetScoreUpdate({}) },
    { abc: '', approved: null, scoreSource: null, chordQuality: null });
  const incomplete = leadSheetScoreUpdate({ abc: 'X:1\nK:C\nCDEF|', scoreSource: 'muscriptor' });
  assert.equal(incomplete.abc, 'X:1\nK:C\nCDEF|');
  assert.equal(incomplete.approved, null);
  assert.equal(incomplete.scoreSource, 'muscriptor');
  assert.equal(incomplete.chordQuality, null);
});

test('MuScriptor chord quality and native or dataset provenance survive the result projection', () => {
  for (const scoreSource of ['muscriptor', 'yourmt3-chordnet', 'dataset'] as const) {
    const update = leadSheetScoreUpdate({ abc: 'score', leadSheet: {
      abc: 'score', scoreSource, leadSheetSource: 'muscriptor', chordQuality: 'estimated',
      timing: { conversionVersion: 1, path: 'abc', bpm: 96, rawEvents: { melody: [], chords: [] } },
    } });
    assert.equal(update.chordQuality, 'estimated');
    assert.equal(update.scoreSource, scoreSource);
  }
});

test('workflow wire requests preserve source IDs and revisions; transcribe has no source override', async () => {
  const original = globalThis.fetch;
  const sent: Array<{ kind: string; input: Record<string, unknown> }> = [];
  globalThis.fetch = async (_url, init) => {
    sent.push(JSON.parse(String(init?.body)));
    return new Response(JSON.stringify({ job: { id: 'test-job' }, created: true }));
  };
  try {
    await coverWorkflowApi.open('asset');
    assert.equal(sent.at(-1)?.input.leadSheetSource, undefined, 'server snapshots Models when omitted');
    for (const source of LEAD_SHEET_SOURCES) {
      await coverWorkflowApi.open('asset', undefined, source);
      assert.equal(sent.at(-1)?.input.leadSheetSource, source);
      await coverWorkflowApi.source('draft', 7, source, 'replacement-asset');
      assert.deepEqual(sent.at(-1), { ...sent.at(-1), kind: 'cover-source', input: {
        documentId: 'draft', revision: 7, leadSheetSource: source, assetId: 'replacement-asset',
      } });
    }
    await coverWorkflowApi.source('draft', 8, 'muscriptor');
    assert.deepEqual(sent.at(-1)?.input, { documentId: 'draft', revision: 8, leadSheetSource: 'muscriptor' });
    await coverWorkflowApi.transcribe('draft', 9, true);
    assert.deepEqual(sent.at(-1)?.input, { documentId: 'draft', revision: 9, force: true });
  } finally { globalThis.fetch = original; }
});
