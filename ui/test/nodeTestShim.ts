// nodeTestShim.ts — lets the UI's `node:test` files run under vitest.
//
// The UI suite is mostly written for node:test (82 files) with two written for
// vitest. Rather than keep two runners — and an `npm test` that silently skips
// most of the suite — `vitest.config.ts` resolves `node:test` to this module,
// so one command runs every UI test.
//
// Only the surface the tests use is exported. node:test's top-level `before`
// and `after` are suite hooks, which vitest spells `beforeAll`/`afterAll`; they
// are re-exported under the node names so existing imports keep working.
// A node:test file that reaches for something else (mock, run, snapshot) should
// add it here rather than switch the whole file to vitest.

import { test } from 'vitest';

export default test;

export {
  afterAll as after,
  afterEach,
  beforeAll as before,
  beforeEach,
  describe,
  it,
  test,
} from 'vitest';
