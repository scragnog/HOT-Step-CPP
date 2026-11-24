// The instrumented build the attribution tools read (profile.cjs, load-profile.cjs, trace.cjs's stack lines): the
// product page (--mode file) or the test page (--mode test), unminified, with its source map and React's profiling
// build, into tools/perf/results/instr/<dist-file|dist-test>/.
//   npx vite build --mode file --config tools/perf/vite.instr.mjs
// (JavaScript, so the type check of tools/, which a planted-mistake copy runs without vite.config.ts, leaves it out)
import { defineConfig, mergeConfig } from 'vite';
import base from '../../vite.config.ts';

export default defineConfig(env => {
  const b = typeof base === 'function' ? base(env) : base;
  return mergeConfig(b, {
    resolve: { alias: { 'react-dom/client': 'react-dom/profiling' } },
    define: { 'import.meta.env.VITE_TALLY': '"1"' },
    build: { minify: false, sourcemap: true, outDir: `tools/perf/results/instr/${env.mode === 'test' ? 'dist-test' : 'dist-file'}` },
  });
});
