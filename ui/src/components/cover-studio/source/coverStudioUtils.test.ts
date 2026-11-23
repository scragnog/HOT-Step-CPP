import assert from 'node:assert/strict';
import test from 'node:test';
import { renderOwnsBusy, sourceDraftPending, yue2CoverContentMissing } from './coverStudioUtils';

test('a YuE2 cover needs a style caption, and lyrics unless it is instrumental', () => {
  assert.equal(yue2CoverContentMissing({ caption: '', lyrics: '', instrumental: true }), 'Write a YuE2 style caption');
  assert.equal(yue2CoverContentMissing({ caption: '  ', lyrics: 'la la', instrumental: false }), 'Write a YuE2 style caption');
  assert.equal(yue2CoverContentMissing({ caption: 'piano', lyrics: ' ', instrumental: false }), 'Enter lyrics or enable Instrumental mode');
  assert.equal(yue2CoverContentMissing({ caption: 'piano', lyrics: '', instrumental: true }), null);
  assert.equal(yue2CoverContentMissing({ caption: 'piano', lyrics: 'la la', instrumental: false }), null);
});

test('a library send (no asset, no draft) is not pending, so Generate stays enabled and registers it on start', () => {
  const none = { sourceMode: 'audio', sourceAssetId: '', coverDraftId: '', coverDraftRevision: 0 };
  assert.equal(sourceDraftPending(none), false);
  assert.equal(sourceDraftPending({ ...none, sourceAssetId: 'asset' }), true, 'an upload waits for its draft');
  assert.equal(sourceDraftPending({ ...none, sourceAssetId: 'asset', coverDraftId: 'doc', coverDraftRevision: 1 }), false);
  assert.equal(sourceDraftPending({ ...none, sourceMode: 'musicxml', sourceAssetId: 'asset' }), false);
});

test('a render that failed to start clears the busy state unless a newer job owns it', () => {
  assert.equal(renderOwnsBusy('', ''), true, 'failed start, nothing else running');
  assert.equal(renderOwnsBusy('', 'newer'), false, 'failed start, a newer job is running');
  assert.equal(renderOwnsBusy('job', 'job'), true);
  assert.equal(renderOwnsBusy('job', 'newer'), false);
  assert.equal(renderOwnsBusy('job', ''), false);
});
