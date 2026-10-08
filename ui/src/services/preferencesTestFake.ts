// preferencesTestFake.ts — an in-memory /api/preferences for the 7a client
// tests (presetCollection, settingDocument and the components built on them).
// Not imported by the app.
//
// Same rules as server/src/routes/preferences.ts: revision-checked writes
// (409 with currentRevision), name-conflict imports, idempotent imports by
// storage key and hash. Each request is answered from the state at the time
// it arrives; `hold()` delays the response, as a slow network would.

interface Doc { id: string; family: string; revision: number; body: Record<string, unknown> }
interface Call { method: string; path: string; body: any }

export class PreferencesFake {
  docs: Doc[] = [];
  calls: Call[] = [];
  private seq = 0;
  private receipts = new Map<string, string>();
  private holds: Array<{ match: (c: Call) => boolean; gate: Promise<void> }> = [];
  private failures: Array<{ match: (c: Call) => boolean; status: number }> = [];
  chain: unknown[] = [];

  /** Seed a stored document, as another client would have saved it. */
  seed(family: string, body: Record<string, unknown>): Doc {
    const doc = { id: `doc-${++this.seq}`, family, revision: 1, body };
    this.docs.push(doc);
    return doc;
  }

  /** Another client's write. */
  touch(id: string, body: Record<string, unknown>): void {
    const doc = this.docs.find(d => d.id === id)!;
    doc.body = body;
    doc.revision++;
  }

  /** Delay the response of the next matching request until release(). */
  hold(match: (c: Call) => boolean): () => void {
    let release!: () => void;
    const gate = new Promise<void>(r => { release = r; });
    this.holds.push({ match, gate });
    return release;
  }

  /** Fail the next matching request with `status`. */
  failNext(match: (c: Call) => boolean, status = 500): void {
    this.failures.push({ match, status });
  }

  count(method: string, pathPart = ''): number {
    return this.calls.filter(c => c.method === method && c.path.includes(pathPart)).length;
  }

  install(): () => void {
    const saved = globalThis.fetch;
    globalThis.fetch = (async (url: string, init?: RequestInit) => {
      const u = new URL(url, 'http://local');
      const call: Call = { method: init?.method ?? 'GET', path: u.pathname + u.search, body: init?.body ? JSON.parse(String(init.body)) : undefined };
      this.calls.push(call);
      const fi = this.failures.findIndex(f => f.match(call));
      let reply: [number, unknown];
      if (fi >= 0) reply = [this.failures.splice(fi, 1)[0]!.status, { error: 'Injected failure' }];
      else reply = this.answer(call, u);
      const hi = this.holds.findIndex(h => h.match(call));
      if (hi >= 0) await this.holds.splice(hi, 1)[0]!.gate;
      return new Response(JSON.stringify(reply[1]), { status: reply[0] });
    }) as typeof fetch;
    return () => { globalThis.fetch = saved; };
  }

  private typed(d: Doc) {
    return { id: d.id, kind: d.family, revision: d.revision, schemaVersion: 1, provenance: { origin: 'client', at: 0 }, body: structuredClone(d.body), createdAt: 0, updatedAt: 0 };
  }

  private answer(c: Call, u: URL): [number, unknown] {
    if (u.pathname === '/api/vst/chain') { if (c.method === 'PUT' || c.method === 'POST') this.chain = c.body?.plugins ?? c.body; return [200, { plugins: this.chain }]; }
    const m = u.pathname.match(/^\/api\/preferences\/(presets|settings)\/([a-z0-9-]+)(?:\/([^/]+))?$/);
    if (!m) return [404, { error: `No route ${c.method} ${c.path}` }];
    const [, area, family, rest] = m;
    const mine = this.docs.filter(d => d.family === family);
    const find = (id: string) => mine.find(d => d.id === id);
    const create = (body: Record<string, unknown>) => { const d = { id: `doc-${++this.seq}`, family: family!, revision: 1, body }; this.docs.push(d); return d; };
    const conflict = (d: Doc) => [409, { error: 'Stale revision', currentRevision: d.revision }] as [number, unknown];

    if (rest === 'import') {
      const results = (c.body.items as any[]).map(item => {
        const name = item.name ?? item.body?.name;
        const existing = area === 'settings' ? mine[0] : mine.find(d => (d.body.name ?? d.body.label) === name);
        const same = existing && JSON.stringify(existing.body) === JSON.stringify(item.body);
        if (existing && !same) {
          if (!item.resolution || (area === 'settings' && item.resolution !== 'replace'))
            return { storageKey: item.storageKey, outcome: 'name-conflict', documentId: existing.id, storedName: name };
          if (item.resolution === 'replace') { existing.body = item.body; existing.revision++; return { storageKey: item.storageKey, outcome: 'replaced', documentId: existing.id, storedName: name }; }
        }
        const rkey = `${family}|${item.storageKey}|${item.sourceHash}`;
        const known = this.receipts.get(rkey);
        if (known) return { storageKey: item.storageKey, outcome: 'unchanged', documentId: known, storedName: name };
        const d = create(item.body);
        this.receipts.set(rkey, d.id);
        return { storageKey: item.storageKey, outcome: 'imported', documentId: d.id, storedName: name };
      });
      return [200, { results }];
    }

    if (area === 'settings') {
      const existing = mine[0];
      if (c.method === 'GET') return [200, { document: existing ? this.typed(existing) : null }];
      if (!existing) return [200, { document: this.typed(create(c.body.body)) }];
      if (c.body.expectedRevision === undefined) return [400, { error: 'expectedRevision is required to update an existing document' }];
      if (c.body.expectedRevision !== existing.revision) return conflict(existing);
      existing.body = c.body.body; existing.revision++;
      return [200, { document: this.typed(existing) }];
    }

    if (!rest) {
      if (c.method === 'GET') return [200, { documents: [...mine].reverse().map(d => this.typed(d)) }];
      return [201, { document: this.typed(create(c.body.body)) }];
    }
    const id = decodeURIComponent(rest);
    const doc = find(id);
    if (!doc) return [404, { error: `Document ${id} not found` }];
    if (c.method === 'PUT') {
      if (c.body.expectedRevision !== doc.revision) return conflict(doc);
      doc.body = c.body.body; doc.revision++;
      return [200, { document: this.typed(doc) }];
    }
    if (c.method === 'DELETE') {
      if (Number(u.searchParams.get('expectedRevision')) !== doc.revision) return conflict(doc);
      this.docs = this.docs.filter(d => d !== doc);
      return [200, { removed: true }];
    }
    return [404, { error: 'No route' }];
  }
}

export class MemoryStorage {
  private store = new Map<string, string>();
  getItem(key: string): string | null { return this.store.has(key) ? this.store.get(key)! : null; }
  setItem(key: string, value: string): void { this.store.set(key, value); }
  removeItem(key: string): void { this.store.delete(key); }
  clear(): void { this.store.clear(); }
}

/** A fresh browser (localStorage) and server for one test. */
export async function withBrowser<T>(run: (fake: PreferencesFake, storage: MemoryStorage) => Promise<T>): Promise<T> {
  const fake = new PreferencesFake();
  const storage = new MemoryStorage();
  const g = globalThis as { localStorage?: unknown };
  const savedStorage = g.localStorage;
  g.localStorage = storage;
  const uninstall = fake.install();
  try { return await run(fake, storage); } finally { uninstall(); g.localStorage = savedStorage; }
}

/** Let pending promise callbacks run. */
export const tick = () => new Promise(r => setTimeout(r, 0));
