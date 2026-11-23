// Three builds of the page. `vite` and `vite build` (dist/): the development page and its build. `--mode file`
// (dist-file/index.html): one self-contained file that opens from disk with no server, on the simulated engine, which
// the owner publishes. `--mode test` (dist-test/index.html): the same file with the test-only compatibility layer the
// concept's two tools drive (src/app/testing/compat.ts), which no other build contains. With HS_SWITCH=<feature>
// the test build has that feature switched off (tools/behaviour/switches/*.json) and goes to dist-switch/, for
// tools/behaviour/switches.mjs.

import fs from 'node:fs';
import path from 'node:path';
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

const LF = String.fromCharCode(10), CRLF = String.fromCharCode(13, 10);
/** Switches one feature off: each of its changes must apply exactly once, or the build fails. */
function switchOff(feature: string): Plugin {
  /* the features switched off (tools/behaviour/switches) and the defects planted (tools/visual-baseline/plants) */
  const all = ['tools/behaviour/switches', 'tools/visual-baseline/plants'].map(d => path.resolve(d)).filter(d => fs.existsSync(d))
    .flatMap(dir => fs.readdirSync(dir).filter(f => f.endsWith('.json')).flatMap(f => (JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8')) as { switches: { feature: string; file: string; find: string; replace: string }[] }).switches));
  const mine = all.filter(x => x.feature === feature), done = new Set<number>();
  if (!mine.length) throw new Error(`no switch for the feature "${feature}"`);
  return {
    name: 'hotstep-switch-off', enforce: 'pre',
    transform(code, id) {
      let out = code;
      mine.forEach((x, i) => {
        if (!id.split(String.fromCharCode(92)).join('/').endsWith(x.file)) return;
        const crlf = out.includes(CRLF), find = crlf ? x.find.split(LF).join(CRLF) : x.find, n = out.split(find).length - 1;
        if (n !== 1) throw new Error(`switch "${feature}": "${x.find}" is found ${n} times in ${x.file}`);
        out = out.replace(find, x.replace); done.add(i);
      });
      return out === code ? null : out;
    },
    buildEnd() { mine.forEach((x, i) => { if (!done.has(i)) throw new Error(`switch "${feature}": ${x.file} was not in the build`); }); },
  };
}

export default defineConfig(({ mode }) => {
  const single = mode === 'file' || mode === 'test', off = mode === 'test' ? process.env.HS_SWITCH : undefined;
  return {
    base: './',
    plugins: [...(off ? [switchOff(off)] : []), react(), ...(single ? [oneFile()] : [])],
    build: {
      outDir: off ? 'dist-switch' : mode === 'file' ? 'dist-file' : mode === 'test' ? 'dist-test' : 'dist',
      emptyOutDir: true,
      /* the concept's stylesheets are used as they are: never minified or rewritten */
      cssMinify: false,
      cssCodeSplit: false,
      assetsInlineLimit: Number.MAX_SAFE_INTEGER,
      rollupOptions: single ? { output: { inlineDynamicImports: true } } : {},
    },
  };
});
