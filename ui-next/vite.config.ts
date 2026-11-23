// Three builds of the page. `vite` and `vite build` (dist/): the development page and its build. `--mode file`
// (dist-file/index.html): one self-contained file that opens from disk with no server, on the simulated engine, which
// the owner publishes. `--mode test` (dist-test/index.html): the same file with the test-only compatibility layer the
// concept's two tools drive (src/app/testing/compat.ts), which no other build contains.

import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';

/** Inlines the script and the stylesheet into the page, so the build is one file. They stand after the head's meta
 *  lines, so the charset is read first. */
function oneFile(): Plugin {
  return {
    name: 'hotstep-one-file', apply: 'build', enforce: 'post',
    generateBundle: { order: 'post', handler(_o, bundle) {
      const html = Object.values(bundle).find(a => a.type === 'asset' && a.fileName.endsWith('.html'));
      if (!html || html.type !== 'asset') return;
      const blocks: string[] = [], drop: string[] = [];
      for (const [name, a] of Object.entries(bundle)) {
        if (a.type === 'chunk' && a.isEntry) { blocks.push(`<script type="module">${a.code.split('</script').join('<\\/script')}</script>`); drop.push(a.fileName); delete bundle[name]; }
        else if (a.type === 'asset' && a.fileName.endsWith('.css')) { blocks.unshift(`<style>${String(a.source)}</style>`); drop.push(a.fileName); delete bundle[name]; }
      }
      const lines = String(html.source).split('\n').filter(l => !drop.some(f => l.includes(f)));
      const at = lines.findIndex(l => l.startsWith('<meta name="viewport"'));
      lines.splice(at + 1, 0, ...blocks);
      html.source = lines.join('\n');
    } },
  };
}

export default defineConfig(({ mode }) => {
  const single = mode === 'file' || mode === 'test';
  return {
    base: './',
    plugins: [react(), ...(single ? [oneFile()] : [])],
    build: {
      outDir: mode === 'file' ? 'dist-file' : mode === 'test' ? 'dist-test' : 'dist',
      emptyOutDir: true,
      /* the concept's stylesheets are used as they are: never minified or rewritten */
      cssMinify: false,
      cssCodeSplit: false,
      assetsInlineLimit: Number.MAX_SAFE_INTEGER,
      rollupOptions: single ? { output: { inlineDynamicImports: true } } : {},
    },
  };
});
