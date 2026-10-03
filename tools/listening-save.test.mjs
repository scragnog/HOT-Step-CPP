// Regression test for the autosave race Reviewer found (fd76ecae): an edit
// made before the initial GET of scores.json resolves must not POST until
// that GET has merged in, or the POST overwrites the file with a snapshot
// that's missing whatever the GET would have added.
//
// No DOM here — PERSISTENCE_SCRIPT is inlined into a page's own <script>
// scope (see the comment in listening-save.mjs for the identifiers it
// expects). This stubs that scope well enough to exercise the load-gate
// sequencing with real timers and a fetch whose GET resolves on a delay we
// control.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { PERSISTENCE_SCRIPT } from './listening-save.mjs';

function deferred() {
  let resolve;
  const promise = new Promise((r) => { resolve = r; });
  return { promise, resolve };
}

test('autosave waits for the initial load before POSTing, and merges it in', async () => {
  const getGate = deferred();
  const posts = [];
  let renderCalls = 0;

  const scope = {
    STUDY: { id: 'race-test' },
    LS: 'scoresheet:race-test',
    scores: {},
    picks: {},
    fileHandle: null,
    say: () => {},
    el: () => ({}),
    render: () => { renderCalls++; },
    location: { protocol: 'http:', pathname: '/listening/study-a/index.html' },
    localStorage: { getItem: () => null, setItem: () => {} },
    document: { getElementById: () => ({ set hidden(_v) {}, set textContent(_v) {}, onclick: null }) },
    fetch: async (url, opts) => {
      if (opts?.method === 'POST') {
        posts.push(JSON.parse(opts.body));
        return { ok: true, status: 200 };
      }
      await getGate.promise; // the GET that loads the existing scores.json
      return { ok: true, status: 200, json: async () => ({ scores: { 'remote-row': { v: 1 } } }) };
    },
  };
  scope.merge = (data) => {
    for (const [k, v] of Object.entries(data?.scores || {})) if (!scope.scores[k]) scope.scores[k] = v;
  };
  scope.snapshot = () => ({ study: scope.STUDY.id, scores: scope.scores, picks: scope.picks });

  // persist()/loadState are local to the script body; expose them the same
  // way a <script> would let outer code reach its own top-level functions.
  const fn = new Function(...Object.keys(scope), PERSISTENCE_SCRIPT + '\nreturn {persist, get loadState(){return loadState;}};');
  const api = fn(...Object.values(scope));

  // Simulates the user scoring a render before the GET above has resolved.
  scope.scores['local-row'] = { v: 2 };
  api.persist();
  assert.equal(posts.length, 0, 'no POST before the edit is even made');
  assert.equal(api.loadState, 'pending', 'load is still in flight at this point');

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  // The GET resolves at 500ms — after persist()'s 400ms debounce has already
  // fired once — so the 450ms check below only passes if the pending-load
  // gate, not just the debounce, is what's holding the POST back.
  setTimeout(() => getGate.resolve(), 500);
  await sleep(450);
  assert.equal(posts.length, 0, 'debounce elapsed but load is still pending — must not have posted yet');
  await sleep(600);

  assert.ok(posts.length >= 1, 'expected at least one POST once loading finished');
  const last = posts[posts.length - 1];
  assert.deepEqual(last.scores['remote-row'], { v: 1 }, 'remote row must survive the save');
  assert.deepEqual(last.scores['local-row'], { v: 2 }, 'local edit must survive the save');
  assert.ok(renderCalls >= 1, 'render() runs once the merge lands');
});
