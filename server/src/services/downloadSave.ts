// downloadSave.ts — write a finished download into the user's chosen folder.
//
// The folder only ever comes from the DOWNLOAD_DIR setting, never from a
// request. What a request does control is the filename, so that is the
// untrusted input: it is reduced to one safe path segment, and the file is
// created with an exclusive open so an existing entry (including a symlink or
// junction planted in the folder) is never followed or overwritten.

import crypto from 'crypto';
import fs from 'fs';
import path from 'path';

const RESERVED_DEVICE = /^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/i;

/** Reduce an arbitrary name to one filename segment that is safe on Windows, macOS and Linux. */
export function safeFilename(name: string): string {
  // Keep only what follows the last separator of either flavour, so `..\x`, `/etc/x` and `C:\x` lose their directories.
  let base = name.split(/[\\/]/).pop() ?? '';
  base = base.replace(/[<>:"|?*\u0000-\u001f]/g, '').replace(/[. ]+$/, '').trim();
  if (!base || base === '.' || base === '..') return 'download';
  const stem = base.slice(0, base.length - path.extname(base).length);
  return RESERVED_DEVICE.test(stem) ? `_${base}` : base;
}

/** Check the setting: absolute, creatable and writable. Returns the real folder path; throws a user-readable Error. */
export function validateDownloadDir(dir: string): string {
  if (!path.isAbsolute(dir)) throw new Error('Download folder must be an absolute path');
  try {
    fs.mkdirSync(dir, { recursive: true });
    const real = fs.realpathSync(dir);
    // Exclusive open of a unique name: never follows or truncates anything already there.
    const probe = path.join(real, `.hotstep-write-test-${process.pid}-${crypto.randomBytes(6).toString('hex')}`);
    fs.closeSync(fs.openSync(probe, 'wx'));
    fs.unlinkSync(probe);
    return real;
  } catch (err: any) {
    throw new Error(`Download folder is not usable: ${err.message}`);
  }
}

function lstatExists(p: string): boolean {
  try { fs.lstatSync(p); return true; } catch { return false; }
}

/** Copy `sourcePath` into `dir` as `filename`, renaming to `name (2).ext` etc. on collision. Returns the written path. */
export function saveIntoDir(dir: string, filename: string, sourcePath: string): string {
  const real = validateDownloadDir(dir);
  const safe = safeFilename(filename);
  const ext = path.extname(safe);
  const stem = ext ? safe.slice(0, -ext.length) : safe;
  for (let n = 1; n < 1000; n++) {
    const candidate = path.join(real, n === 1 ? safe : `${stem} (${n})${ext}`);
    if (path.dirname(candidate) !== real) throw new Error('Refusing to write outside the download folder');
    // lstat counts a symlink or junction as taken; the exclusive open below closes the race after it.
    if (lstatExists(candidate)) continue;
    let fd: number;
    try { fd = fs.openSync(candidate, 'wx'); } catch (err: any) {
      if (err.code === 'EEXIST' || err.code === 'EPERM') continue;
      throw err;
    }
    try {
      // ponytail: whole-file read, tracks are tens of MB; chunk the copy if that stops being true.
      fs.writeSync(fd, fs.readFileSync(sourcePath));
    } finally { fs.closeSync(fd); }
    return candidate;
  }
  throw new Error('Too many files with the same name in the download folder');
}

/**
 * The OS command that shows the download folder, with `file` selected when it
 * is an existing file directly inside that folder. `file` comes from a request,
 * so anything else (another folder, a missing file) just opens the folder.
 */
export function revealCommand(
  dir: string, file: string | undefined, platform: NodeJS.Platform = process.platform,
): { cmd: string; args: string[] } {
  const real = fs.realpathSync(dir);
  let target: string | undefined;
  if (file && path.isAbsolute(file)) {
    try {
      const realFile = fs.realpathSync(file);
      if (path.dirname(realFile) === real && fs.statSync(realFile).isFile()) target = realFile;
    } catch { /* missing: fall back to the folder */ }
  }
  if (platform === 'win32') return { cmd: 'explorer.exe', args: target ? ['/select,', target] : [real] };
  if (platform === 'darwin') return { cmd: 'open', args: target ? ['-R', target] : [real] };
  return { cmd: 'xdg-open', args: [real] };
}
