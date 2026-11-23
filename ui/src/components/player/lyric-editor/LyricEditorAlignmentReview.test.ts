import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { moveSummary } from './LyricEditorAlignmentReview';

// Stand-in for i18next: default text with {{count}} filled in.
const t = (_key: string, fallback: string, opts?: Record<string, unknown>) =>
  fallback.replace('{{count}}', String(opts?.count));

describe('moveSummary', () => {
  it('counts words when words moved', () => {
    assert.equal(moveSummary(3, 4, false, t), '3 would move');
  });
  it('falls back to lines when only line starts change', () => {
    assert.equal(moveSummary(0, 4, false, t), '4 lines would move');
  });
  it('says nothing would move only when no row changes', () => {
    assert.equal(moveSummary(0, 0, false, t), 'nothing would move');
  });
  it('uses past tense once applied', () => {
    assert.equal(moveSummary(0, 4, true, t), '4 lines moved');
    assert.equal(moveSummary(2, 4, true, t), '2 moved');
    assert.equal(moveSummary(0, 0, true, t), 'nothing moved');
  });
});
