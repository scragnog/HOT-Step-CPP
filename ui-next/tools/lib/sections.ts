// A test file's sections, each about definitions it names by id (a module's own behaviour). A section that names a
// definition no longer in the index (removed the documented way, docs/06 8b) is skipped and said, never a crash: the
// maintainer who removes a module sees which sections stopped applying, and the rest still run.

export class Retired extends Error {}
/** A definition the section needs: the found one, else the section stops here and is counted as skipped. */
export function need<T>(found: T | undefined, id: string): T {
  if (found === undefined) throw new Retired(id);
  return found;
}
const skipped: string[] = [];
export function section(fn: () => void): void {
  try { fn(); } catch (e) { if (e instanceof Retired) skipped.push(e.message); else throw e; }
}
/** What was skipped, for the file's last line ('' when nothing was). */
export const skips = (): string => (skipped.length ? `; ${skipped.length} section${skipped.length > 1 ? 's' : ''} skipped: ${skipped.length > 1 ? 'they name' : 'it names'} ${[...new Set(skipped)].join(', ')}, which no definition is` : '');
