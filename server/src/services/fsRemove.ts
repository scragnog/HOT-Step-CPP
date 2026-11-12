import fs from 'node:fs';
import path from 'node:path';

function stat(file: string): fs.Stats | undefined {
  try { return fs.lstatSync(file); }
  catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return undefined;
    throw err;
  }
}

/** Verify rmSync's result: Windows Node 24.11 can silently skip non-ASCII paths.
 * Fall back to unlink/rmdir without following symbolic links or junctions.
 * Keep rmSync's force and recursive semantics, including errors it raises. */
export function removeSync(file: string, options: fs.RmOptions = {}): void {
  fs.rmSync(file, options);
  try {
    const entry = stat(file);
    if (!entry) return;
    if (entry.isDirectory() && !entry.isSymbolicLink()) {
      if (!options.recursive) throw new Error('recursive removal required');
      for (const name of fs.readdirSync(file)) removeSync(path.join(file, name), options);
      fs.rmdirSync(file);
    } else {
      fs.unlinkSync(file);
    }
    if (stat(file)) throw new Error('path still exists');
  } catch (err) {
    throw new Error(`Could not remove ${file}: ${(err as Error).message}`, { cause: err });
  }
}
