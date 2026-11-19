import assert from 'node:assert/strict';
import test from 'node:test';
import { yue2CoverContentMissing } from './coverStudioUtils';

test('a YuE2 cover needs a style caption, and lyrics unless it is instrumental', () => {
  assert.equal(yue2CoverContentMissing({ caption: '', lyrics: '', instrumental: true }), 'Write a YuE2 style caption');
  assert.equal(yue2CoverContentMissing({ caption: '  ', lyrics: 'la la', instrumental: false }), 'Write a YuE2 style caption');
  assert.equal(yue2CoverContentMissing({ caption: 'piano', lyrics: ' ', instrumental: false }), 'Enter lyrics or enable Instrumental mode');
  assert.equal(yue2CoverContentMissing({ caption: 'piano', lyrics: '', instrumental: true }), null);
  assert.equal(yue2CoverContentMissing({ caption: 'piano', lyrics: 'la la', instrumental: false }), null);
});
