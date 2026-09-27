import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import test from 'node:test';
import { DiscussionStore } from '../src/collaboration.js';
import { renderDiscussionExport } from '../src/discussion-export.js';

test('exports keep every plan revision and revealed position, never sealed positions', () => {
  const store = new DiscussionStore(':memory:');
  try {
    const room = 'export-test';
    const a = store.join(room, 'Codex', 'A brief with £195, café and 中文.', { blind_positions: true, role: 'Reviewer' });
    const b = store.join(room, 'Claude');
    assert.match(renderDiscussionExport(store.exportTranscript(room)), /No messages recorded yet/);
    store.submitPosition(room, a.participant_id, 'position-a', 'SEALED A', ['https://example.com/a']);
    const sealed = store.exportTranscript(room);
    assert.equal(sealed.discussion.phase, 'positions');
    assert.doesNotMatch(renderDiscussionExport(sealed), /SEALED A/);
    store.submitPosition(room, b.participant_id, 'position-b', 'SEALED B', ['https://example.com/b']);
    const longBody = 'Long message\n'.repeat(1600) + 'LAST LINE';
    const human = store.joinViewer(room, randomUUID());
    store.post(room, human, 'long-message', 'user_direction', longBody);
    const plan1 = store.decide(room, a.participant_id, 'plan1', 0, 'ORIGINAL PLAN', '', undefined,
      [{ issue: 'Check this source', owner: 'Claude' }]);
    store.post(room, b.participant_id, 'critique', 'critique', 'Change the plan. https://example.com/source?q=a&b=c');
    store.decide(room, a.participant_id, 'plan2', 1, 'REVISED PLAN', '');
    store.status(room, human, 'close', 'closed', 'User ended this room.');
    store.recordOutcome(room, a.participant_id, 'outcome', 'abandoned', 'Kept for reference.');
    const snapshot = store.exportTranscript(room);
    const html = renderDiscussionExport(snapshot);
    for (const expected of ['ORIGINAL PLAN', 'REVISED PLAN', 'SEALED A', 'SEALED B', 'LAST LINE', 'Reviewer', '£195, café and 中文.', 'Kept for reference.', 'owner: Claude']) {
      assert.ok(html.includes(expected), expected);
    }
    assert.ok(html.indexOf('ORIGINAL PLAN') < html.indexOf('REVISED PLAN'));
    assert.match(html, /https:\/\/example.com\/source\?q=a&amp;b=c/);
    assert.ok(snapshot.messages.some(message => message.id === plan1.message_id));
    assert.equal(store.read(room, 0, 100).participants.some(p => p.name === 'Codex'), false);
    assert.ok(snapshot.participants.some(p => p.name === 'Codex'));
  } finally { store.close(); }
});

test('HTML content and historical event bodies remain safe and intact', () => {
  const store = new DiscussionStore(':memory:');
  try {
    const a = store.join('escaping', '<img src=x onerror=alert(1)>', 'A <b>brief</b> & a quote "');
    const snapshot = store.exportTranscript('escaping');
    snapshot.messages.push({
      id: 1, room: 'escaping', author: a.discussion.id, participant_id: a.participant_id,
      request_id: 'legacy', reply_to: null, created_at: '2026-09-27T12:00:00Z', kind: 'decision',
      body: '<script>bad()</script> historical event https://example.com/" onclick="bad()',
    });
    const html = renderDiscussionExport(snapshot);
    assert.match(html, /&lt;img src=x onerror=alert\(1\)&gt;/);
    assert.match(html, /&lt;b&gt;brief&lt;\/b&gt;/);
    assert.match(html, /&lt;script&gt;bad\(\)&lt;\/script&gt; historical event/);
    assert.doesNotMatch(html, /<img|<b>|<script>bad|" onclick="bad/);
  } finally { store.close(); }
});
