// The instrumented build the attribution tools read (profile.cjs, load-profile.cjs, trace.cjs's stack lines): the
// product page (--mode file) or the test page (--mode test), unminified, with its source map and React's profiling
// build, into tools/perf/results/instr/<dist-file|dist-test>/.
//   npx vite build --mode file --config tools/perf/vite.instr.ts
import { defineConfig, mergeConfig, type ConfigEnv, type UserConfig } from 'vite';
import base from '../../vite.config.ts';

export default defineConfig((env: ConfigEnv) => {
  const b = (typeof base === 'function' ? base(env) : base) as UserConfig;
  return mergeConfig(b, {
    resolve: { alias: { 'react-dom/client': 'react-dom/profiling' } },
    define: { 'import.meta.env.VITE_TALLY': '"1"' },
    build: { minify: false, sourcemap: true, outDir: `tools/perf/results/instr/${env.mode === 'test' ? 'dist-test' : 'dist-file'}` },
  });
});
