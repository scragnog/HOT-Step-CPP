// cssStubLoader.mjs — a Node ESM loader hook (node:module register()) that
// resolves `.css` imports to an empty module. Needed only to import
// audioGenQueueStore.ts under plain `node:test`: its module graph pulls in a
// component (playlistStore -> ... ) that imports a stylesheet, which Vite
// handles but plain Node cannot. No bundler, no new dependency — just the
// built-in loader-hooks API. Not used by the app or the Vite build.
export async function resolve(specifier, context, nextResolve) {
  if (specifier.endsWith('.css')) return { url: 'css-stub:' + specifier, shortCircuit: true };
  return nextResolve(specifier, context);
}

export async function load(url, context, nextLoad) {
  if (url.startsWith('css-stub:')) return { format: 'module', source: 'export default {};', shortCircuit: true };
  return nextLoad(url, context);
}
