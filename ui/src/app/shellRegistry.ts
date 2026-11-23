// shellRegistry.ts — discovers shell skins (docs/dev/skins.md#shell-skins). A shell is a
// folder `src/shells/<id>/` with `manifest.json` and an `index.tsx` whose default export is
// the root component (no props). The globs find it, so adding a shell never edits a central list.
// Only the chosen shell's code is downloaded; the manifests are tiny and load eagerly.

import type { ComponentType } from 'react';

export interface ShellManifest {
  id: string;
  name: string;
  author: string;
  version: string;
  /** The skin-api major this shell was written against. */
  api: 0;
  kind: 'shell';
  license?: string;
}

const manifests = import.meta.glob<ShellManifest>('../shells/*/manifest.json', { eager: true, import: 'default' });
const loaders = import.meta.glob<{ default: ComponentType }>('../shells/*/index.tsx');

export const SHELLS: ShellManifest[] = Object.values(manifests).sort((a, b) => a.name.localeCompare(b.name));

export const isRegisteredShell = (id: unknown): id is string => typeof id === 'string' && SHELLS.some(s => s.id === id);

export function loadShell(id: string): Promise<{ default: ComponentType }> {
  const load = loaders[`../shells/${id}/index.tsx`];
  if (!load) return Promise.reject(new Error(`No shell "${id}"`));
  return load();
}
