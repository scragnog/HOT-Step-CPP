import { test } from 'node:test';
import assert from 'node:assert/strict';
import { normalizeSeed, parseSeed, seedText } from './seed';

test('uint64 seeds keep every digit', () => {
  assert.equal(parseSeed('16302813008879570944'), '16302813008879570944');
  assert.equal(parseSeed(' 42 '), 42);
  assert.equal(parseSeed('-1'), -1);
  assert.equal(parseSeed('18446744073709551615'), '18446744073709551615');
  assert.equal(parseSeed('18446744073709551616'), undefined);
  assert.equal(parseSeed('-2'), undefined);
  assert.equal(parseSeed('1e5'), undefined);
});

test('a seed number already past 2^53 is spelled exactly', () => {
  // The double the reporter's song was rendered with.
  assert.equal(normalizeSeed(16302813008879570944), '16302813008879570944');
  assert.equal(seedText(16302813008879570944), '16302813008879570944');
  assert.equal(normalizeSeed('9007199254740991'), 9007199254740991);
  assert.equal(normalizeSeed(7), 7);
});
