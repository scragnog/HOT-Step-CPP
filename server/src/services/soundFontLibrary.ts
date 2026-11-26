import { createHash, randomUUID } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const BANK_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const METADATA = 'bank.json';
const BANK_FILE = 'bank.sfbk';

export interface SoundFontEntry {
  id: string;
  name: string;
  size: number;
  format: 'sf2' | 'sf3';
  addedAt: string;
}

type StoredEntry = SoundFontEntry & { sha256: string };

export class SoundFontLibraryError extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}

export function soundFontFormat(filename: string): SoundFontEntry['format'] | null {
  const suffix = path.extname(filename).toLowerCase();
  return suffix === '.sf2' ? 'sf2' : suffix === '.sf3' ? 'sf3' : null;
}

function displayName(value: unknown): string {
  if (typeof value !== 'string' || !value.trim() || value.trim().length > 255 || /[\x00-\x1f\x7f]/.test(value)) {
    throw new SoundFontLibraryError(400, 'Enter a SoundFont name between 1 and 255 characters.');
  }
  return value.trim();
}

function publicEntry({ sha256: _sha256, ...entry }: StoredEntry): SoundFontEntry {
  return entry;
}

export class SoundFontLibrary {
  constructor(private readonly root: string) {}

  private ready(create = false): boolean {
    if (create) fs.mkdirSync(this.root, { recursive: true });
    const stat = fs.lstatSync(this.root, { throwIfNoEntry: false });
    if (!stat) return false;
    if (!stat.isDirectory()) throw new SoundFontLibraryError(500, 'SoundFont library folder is unavailable.');
    return true;
  }

  uploadFolder(): string {
    this.ready(true);
    return this.root;
  }

  private find(id: string): StoredEntry | null {
    if (!BANK_ID.test(id) || !this.ready()) return null;
    const folder = path.join(this.root, id);
    if (!fs.lstatSync(folder, { throwIfNoEntry: false })?.isDirectory()) return null;
    const metadata = path.join(folder, METADATA);
    if (!fs.lstatSync(metadata, { throwIfNoEntry: false })?.isFile()) return null;
    const stat = fs.lstatSync(path.join(folder, BANK_FILE), { throwIfNoEntry: false });
    if (!stat?.isFile()) return null;

    const entry = JSON.parse(fs.readFileSync(metadata, 'utf8')) as StoredEntry;
    if (entry.id !== id || entry.size !== stat.size || !['sf2', 'sf3'].includes(entry.format)) return null;
    return entry;
  }

  private entries(): StoredEntry[] {
    if (!this.ready()) return [];
    return fs.readdirSync(this.root).flatMap(id => {
      const entry = this.find(id);
      return entry ? [entry] : [];
    });
  }

  list(): SoundFontEntry[] {
    return this.entries().map(publicEntry).sort((a, b) => a.name.localeCompare(b.name) || a.id.localeCompare(b.id));
  }

  private require(id: string): StoredEntry {
    const entry = this.find(id);
    if (!entry) throw new SoundFontLibraryError(404, 'SoundFont not found.');
    return entry;
  }

  file(id: string): { path: string; entry: SoundFontEntry } {
    const entry = publicEntry(this.require(id));
    return { path: path.join(this.root, entry.id, BANK_FILE), entry };
  }

  async add(stagedFile: string, filename: string, label: unknown, size: number): Promise<{ entry: SoundFontEntry; created: boolean }> {
    const format = soundFontFormat(filename);
    if (!format) throw new SoundFontLibraryError(400, 'Choose an .sf2 or .sf3 SoundFont.');
    const name = displayName(label ?? filename);
    const file = await fs.promises.open(stagedFile, 'r');
    const hash = createHash('sha256');
    try {
      const header = Buffer.alloc(12);
      const { bytesRead } = await file.read(header, 0, header.length, 0);
      if (bytesRead !== header.length || header.toString('ascii', 0, 4) !== 'RIFF'
          || header.toString('ascii', 8, 12) !== 'sfbk' || header.readUInt32LE(4) !== size - 8 || size <= 12) {
        throw new SoundFontLibraryError(400, 'The file does not have a valid SoundFont header.');
      }
      for await (const chunk of file.createReadStream({ start: 0, autoClose: false })) hash.update(chunk);
    } finally {
      await file.close();
    }

    // No async work between checking the hash and publishing the bank.
    const sha256 = hash.digest('hex');
    const existing = this.entries().find(entry => entry.sha256 === sha256);
    if (existing) return { entry: publicEntry(existing), created: false };

    const id = randomUUID();
    const entry = { id, name, size, format, addedAt: new Date().toISOString(), sha256 };
    const folder = fs.mkdtempSync(path.join(this.root, '.adding-'));
    try {
      fs.renameSync(stagedFile, path.join(folder, BANK_FILE));
      fs.writeFileSync(path.join(folder, METADATA), JSON.stringify(entry), { flag: 'wx' });
      fs.renameSync(folder, path.join(this.root, id));
    } catch (error) {
      this.clean(folder);
      throw error;
    }
    return { entry: publicEntry(entry), created: true };
  }

  rename(id: string, value: unknown): SoundFontEntry {
    const entry = { ...this.require(id), name: displayName(value) };
    const metadata = path.join(this.root, entry.id, METADATA);
    const temp = path.join(this.root, `.renaming-${randomUUID()}.tmp`);
    try {
      fs.writeFileSync(temp, JSON.stringify(entry), { flag: 'wx' });
      fs.renameSync(temp, metadata);
    } finally {
      fs.rmSync(temp, { force: true });
    }
    return publicEntry(entry);
  }

  remove(id: string): void {
    const entry = this.require(id);
    const folder = path.join(this.root, `.removed-${randomUUID()}`);
    fs.renameSync(path.join(this.root, entry.id), folder);
    this.clean(folder);
  }

  private clean(folder: string): void {
    try {
      for (const name of [BANK_FILE, METADATA]) fs.rmSync(path.join(folder, name), { force: true });
      fs.rmdirSync(folder);
    } catch (error) {
      console.warn('[SoundFontLibrary] Could not clean a staging folder', error);
    }
  }
}
