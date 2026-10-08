import test from 'node:test';
import assert from 'node:assert/strict';
import Database from 'better-sqlite3';
import { z } from 'zod/v4';
import { WorkflowError } from './workflowJobs.js';
import { TypedDocuments, WorkflowDocuments, isTypedDocumentKind, registerDocumentKind, type TypedDocumentKind } from './revisions.js';
import { startWorkflowTestServer } from './testServer.js';
import { INSTALLATION_OWNER, MAX_TYPED_DOCUMENT_BYTES } from '../../contracts/workflow.js';

const bodySchema = z.object({ name: z.string().max(100), value: z.unknown().optional() }).passthrough();
type Body = z.infer<typeof bodySchema>;
const kindDef = (kind: string, extra: Partial<TypedDocumentKind<Body>> = {}): TypedDocumentKind<Body> =>
  ({ kind, scope: 'user', schemaVersion: 1, schema: bodySchema, ...extra });
const store = (def: TypedDocumentKind<Body>, db = new Database(':memory:')) => new TypedDocuments(new WorkflowDocuments(db), db, def);
const client = { origin: 'client' } as const;

function failsWith(fn: () => unknown, status: number, extra: Record<string, unknown> = {}): WorkflowError {
  try { fn(); } catch (err) {
    assert.ok(err instanceof WorkflowError, String(err));
    assert.equal(err.status, status, err.message);
    for (const [k, v] of Object.entries(extra)) assert.deepEqual(err.extra[k], v);
    return err;
  }
  assert.fail(`expected a ${status}`);
}

test('typed documents: a user reads only their own; installation scope is one shared owner', () => {
  const db = new Database(':memory:');
  const mine = store(kindDef('t.user'), db);
  const doc = mine.create('alice', { name: 'a' }, client);
  assert.equal(mine.get(doc.id, 'alice').body.name, 'a');
  failsWith(() => mine.get(doc.id, 'bob'), 404);
  failsWith(() => mine.update(doc.id, 'bob', 1, { name: 'b' }, client), 404);
  failsWith(() => mine.remove(doc.id, 'bob', 1), 404);
  assert.deepEqual(mine.list('bob'), []);

  const shared = new TypedDocuments(new WorkflowDocuments(db), db, kindDef('t.machine', { scope: 'installation' }));
  const setting = shared.create('alice', { name: 'gpu' }, client);
  assert.equal(shared.get(setting.id, 'bob').body.name, 'gpu');
  assert.equal(shared.owner('anyone'), INSTALLATION_OWNER);
  // The same id through the wrong kind is not found, not misread.
  failsWith(() => mine.get(setting.id, INSTALLATION_OWNER), 404);
});

test('typed documents: of two writes from one revision exactly one lands; the other gets the current revision', () => {
  const docs = store(kindDef('t.race'));
  const doc = docs.create('u', { name: 'v1' }, client);
  assert.equal(docs.update(doc.id, 'u', 1, { name: 'first' }, client).revision, 2);
  failsWith(() => docs.update(doc.id, 'u', 1, { name: 'second' }, client), 409, { currentRevision: 2 });
  assert.equal(docs.get(doc.id, 'u').body.name, 'first');
  failsWith(() => docs.remove(doc.id, 'u', 1), 409, { currentRevision: 2 });
});

test('typed documents: missing, null, zero and false survive a round trip; provenance is stamped', () => {
  const docs = store(kindDef('t.values'));
  const body = { name: '', value: { zero: 0, off: false, none: null, list: [] }, unknownExtension: { x: 1 } };
  const doc = docs.create('u', body, { origin: 'client', clientId: 'browser-1' });
  assert.deepEqual(docs.get(doc.id, 'u').body, body);
  assert.equal(doc.provenance.origin, 'client');
  assert.equal(doc.provenance.clientId, 'browser-1');
  assert.equal(typeof doc.provenance.at, 'number');
  assert.equal(doc.schemaVersion, 1);
});

test('typed documents: malformed, oversized and badly sourced bodies are refused before anything is written', () => {
  const docs = store(kindDef('t.invalid'));
  const bad = failsWith(() => docs.create('u', { name: 42 }, client), 400);
  assert.ok(Array.isArray(bad.extra.issues));
  failsWith(() => docs.create('u', { name: 'x', value: 'y'.repeat(MAX_TYPED_DOCUMENT_BYTES) }, client), 400);
  failsWith(() => docs.create('u', { name: 'x' }, { origin: 'somewhere' }), 400);
  assert.deepEqual(docs.list('u'), []);
});

test('typed documents: an older body is upgraded on read; a newer one is refused and never overwritten', () => {
  const db = new Database(':memory:');
  const v1 = store(kindDef('t.versions'), db);
  const old = v1.create('u', { name: 'legacy' }, client);
  const v2 = new TypedDocuments(new WorkflowDocuments(db), db, kindDef('t.versions', {
    schemaVersion: 2, upgrade: (from, body) => ({ ...(body as object), name: `${(body as Body).name}@v${from}` }),
  }));
  const read = v2.get(old.id, 'u');
  assert.equal(read.body.name, 'legacy@v1');
  assert.equal(read.schemaVersion, 2);
  assert.equal(read.storedSchemaVersion, 1);
  const newer = v2.create('u', { name: 'from the future' }, client);

  // A build that only knows v1 cannot read v2, cannot list it, and cannot
  // write or delete over it; the stored content stays.
  failsWith(() => v1.get(newer.id, 'u'), 409, { reason: 'unsupported-version', schemaVersion: 2, supportedVersion: 1 });
  assert.deepEqual(v1.list('u').map(d => d.id), [old.id]);
  failsWith(() => v1.update(newer.id, 'u', 1, { name: 'clobber' }, client), 409, { reason: 'unsupported-version' });
  assert.equal(v2.get(newer.id, 'u').body.name, 'from the future');
  assert.equal(v2.get(newer.id, 'u').revision, 1);

  // No upgrade function: an older version is refused the same way.
  const strict = new TypedDocuments(new WorkflowDocuments(db), db, kindDef('t.versions', { schemaVersion: 2 }));
  failsWith(() => strict.get(old.id, 'u'), 409, { reason: 'unsupported-version', schemaVersion: 1 });
});

test('typed documents: an import happens once per key and value, a failed one leaves nothing, a deleted one is not revived', () => {
  const docs = store(kindDef('t.import'));
  const source = { storageKey: 'hs-example-presets', sourceHash: 'sha256:abc' };

  failsWith(() => docs.importOnce('u', source, { name: 7 }), 400);
  assert.deepEqual(docs.receipts('u'), []);
  assert.deepEqual(docs.list('u'), []);

  const first = docs.importOnce('u', source, { name: 'imported' });
  assert.equal(first.created, true);
  assert.equal(first.document?.provenance.origin, 'import');
  assert.deepEqual(first.document?.provenance.importedFrom, source);
  const again = docs.importOnce('u', source, { name: 'imported twice' });
  assert.equal(again.created, false);
  assert.equal(again.document?.id, first.document?.id);
  assert.equal(docs.list('u').length, 1);

  // Another user imports their own copy of the same browser value.
  assert.equal(docs.importOnce('v', source, { name: 'theirs' }).created, true);
  // A changed value under the same key is a new import.
  assert.equal(docs.importOnce('u', { ...source, sourceHash: 'sha256:def' }, { name: 'edited' }).created, true);

  docs.remove(first.document!.id, 'u', 1);
  const afterDelete = docs.importOnce('u', source, { name: 'imported' });
  assert.equal(afterDelete.created, false);
  assert.equal(afterDelete.document, null);
  assert.equal(afterDelete.receipt.documentId, first.document!.id);
  assert.equal(docs.list('u').length, 1);
});

test('typed kinds: registration is checked and once only; the generic document routes refuse them', async () => {
  assert.throws(() => registerDocumentKind(kindDef('Bad Kind')), /Bad document kind/);
  assert.throws(() => registerDocumentKind(kindDef('t.zero', { schemaVersion: 0 })), /Bad schemaVersion/);
  registerDocumentKind(kindDef('t.http'));
  assert.throws(() => registerDocumentKind(kindDef('t.http')), /already registered/);
  assert.equal(isTypedDocumentKind('t.http'), true);
  assert.equal(isTypedDocumentKind('cover-draft'), false);

  const http = await startWorkflowTestServer();
  try {
    const h = { Authorization: 'Bearer t', 'Content-Type': 'application/json' };
    const typed = new TypedDocuments(http.docs, http.db, kindDef('t.http')).create('u', { name: 'x' }, client);
    const post = await fetch(`${http.origin}/api/workflows/documents`, { method: 'POST', headers: h, body: JSON.stringify({ kind: 't.http', data: { a: 1 } }) });
    assert.equal(post.status, 400);
    assert.equal((await fetch(`${http.origin}/api/workflows/documents?kind=t.http`, { headers: h })).status, 400);
    assert.equal((await fetch(`${http.origin}/api/workflows/documents/${typed.id}`, { headers: h })).status, 400);
    const put = await fetch(`${http.origin}/api/workflows/documents/${typed.id}`, { method: 'PUT', headers: h, body: JSON.stringify({ expectedRevision: 1, data: { a: 1 } }) });
    assert.equal(put.status, 400);
    assert.equal((await fetch(`${http.origin}/api/workflows/documents/${typed.id}?expectedRevision=1`, { method: 'DELETE', headers: h })).status, 400);
    assert.equal(http.docs.get(typed.id, 'u').revision, 1);
    // Untyped kinds behave as before.
    const plain = await fetch(`${http.origin}/api/workflows/documents`, { method: 'POST', headers: h, body: JSON.stringify({ kind: 'cover-draft', data: { a: 1 } }) });
    assert.equal(plain.status, 201);
  } finally {
    await http.close();
  }
});
