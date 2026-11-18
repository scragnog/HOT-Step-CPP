import fs from 'node:fs';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import type { CaptionReplaceRequest, CaptionReplacePreview } from '../../contracts/captionReplace.js';
import { KNOWN_SIDECAR_KEYS } from './sidecarIO.js';

type Sample = { sampleId: string; audioPath: string; sidecarPath: string; fileMissing?: boolean };
export class CaptionReplaceError extends Error {
  constructor(message: string, public status = 400) { super(message); }
}

/** Preserve field headers, line endings and everything outside caption values. */
function replaceCaption(text: string, find: string, replacement: string): string {
  let field = '';
  return text.split(/(\r\n|\r|\n)/).map((line, i) => {
    if (i % 2) return line;
    if (field === 'lyrics') return line;
    const match = /^(\s*([^:\[]+?)\s*:\s*)(.*)$/.exec(line);
    if (match && KNOWN_SIDECAR_KEYS.has(match[2].trim().toLowerCase())) {
      field = match[2].trim().toLowerCase();
      return field === 'caption' ? match[1] + match[3].split(find).join(replacement) : line;
    }
    return field === 'caption' ? line.split(find).join(replacement) : line;
  }).join('');
}

/** Resolve only scan-owned paths; links may otherwise reach unselected files. */
function readCaption(root: string, file: string): string | null {
  const relative = path.relative(root, path.resolve(file));
  if (!relative || relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
    throw new CaptionReplaceError('Caption is outside the dataset');
  }
  let stat: fs.Stats;
  try { stat = fs.lstatSync(file); } catch (err: any) {
    if (err.code === 'ENOENT') return null;
    throw err;
  }
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink > 1
    || path.resolve(fs.realpathSync(file)) !== path.resolve(file)) {
    throw new CaptionReplaceError('Linked caption files cannot be replaced');
  }
  return fs.readFileSync(file, 'utf8');
}

export function replaceSelectedCaptions(sourceDir: string, samples: Sample[], input: CaptionReplaceRequest): CaptionReplacePreview {
  if (!Array.isArray(input.sampleIds) || !input.sampleIds.length || input.sampleIds.some(id => typeof id !== 'string')
    || typeof input.find !== 'string' || !input.find.length || typeof input.replacement !== 'string'
    || /[\r\n]/.test(input.find + input.replacement)
    || (input.previewToken !== undefined && typeof input.previewToken !== 'string')) {
    throw new CaptionReplaceError('Select tracks and enter single-line find and replacement text');
  }
  const root = fs.realpathSync(sourceDir);
  const byId = new Map(samples.map(s => [s.sampleId, s]));
  const changes: Array<{ sampleId: string; file: string; before: string; after: string }> = [];
  const seen = new Set<string>();
  const fingerprint = createHash('sha256').update(JSON.stringify([root, input.find, input.replacement]));
  for (const id of [...new Set(input.sampleIds)].sort()) {
    const sample = byId.get(id);
    if (!sample || sample.fileMissing) throw new CaptionReplaceError('A selected track is missing; rescan the dataset', 409);
    const stem = sample.audioPath.slice(0, -path.extname(sample.audioPath).length);
    for (const [file, generic] of [[sample.sidecarPath, true], [`${stem}.yue2.txt`, false]] as const) {
      if (seen.has(file)) continue;
      seen.add(file);
      const before = readCaption(root, file);
      fingerprint.update(JSON.stringify([id, file, before]));
      if (before === null) continue;
      const after = generic ? replaceCaption(before, input.find, input.replacement) : before.split(input.find).join(input.replacement);
      if (after !== before) changes.push({ sampleId: id, file, before, after });
    }
  }
  const preview: CaptionReplacePreview = {
    previewToken: fingerprint.digest('hex'),
    tracks: new Set(changes.map(c => c.sampleId)).size,
    files: changes.map(c => ({ sampleId: c.sampleId, file: path.relative(root, c.file) })),
  };
  if (input.previewToken !== undefined) {
    if (input.previewToken !== preview.previewToken) throw new CaptionReplaceError('Captions changed since preview. Preview again before applying.', 409);
    // No awaits between the snapshot check and writes. Each file is replaced atomically.
    for (const change of changes) {
      const tmp = path.join(path.dirname(change.file), `.caption-replace-${randomUUID()}.tmp`);
      try {
        fs.writeFileSync(tmp, change.after, { encoding: 'utf8', flag: 'wx' });
        fs.renameSync(tmp, change.file);
      } finally {
        if (fs.existsSync(tmp)) fs.unlinkSync(tmp);
      }
    }
  }
  return preview;
}
