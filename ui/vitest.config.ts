// vitest.config.ts — the UI test runner's config.
//
// Kept separate from vite.config.ts so the build config stays about the app.
// `node:test` resolves to test/nodeTestShim.ts because most of the UI suite is
// written for node:test; without the alias vitest reports "No test suite found"
// for those files and `npm test` would cover only 2 of 84.

import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  resolve: {
    alias: {
      'node:test': fileURLToPath(new URL('./test/nodeTestShim.ts', import.meta.url)),
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  test: {
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
    // The suite stubs its own window/localStorage/indexedDB on globalThis, so a
    // real DOM would make those assignments fail on jsdom's read-only getters.
    // A test that genuinely needs one can opt in with `// @vitest-environment jsdom`
    // after adding jsdom as a dev dependency.
    environment: 'node',
    // node:test has no default timeout. These tests wait on real timers (queue
    // polls, settle windows), so vitest's 5 s default is too tight.
    testTimeout: 60_000,
    hookTimeout: 60_000,
  },
});
