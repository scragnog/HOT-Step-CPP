import test from 'node:test';
import assert from 'node:assert/strict';
import { followSection } from './followSection';
import type { followJob } from '../../services/workflowApi';
import type { WorkflowJob, WorkflowEvent } from '../../../../server/src/contracts/workflow';

/** A followJob that plays a snapshot, then the given events, then ends. */
function fakeFollow(events: Array<Pick<WorkflowEvent, 'type' | 'data'>>, gate?: Promise<void>): typeof followJob {
  return async (_token, _jobId, h) => {
    h.onSnapshot?.({ input: { variants: 2 } } as unknown as WorkflowJob, false);
    for (const [i, e] of events.entries()) {
      if (i === 1 && gate) await gate;
      h.onEvent({ ...e, seq: i + 1, at: 0 });
      await new Promise(r => setTimeout(r, 0));
    }
    return 'succeeded';
  };
}

const variant = (songIds: string[]) => ({ type: 'variant', data: { songIds } });

test('reopened while running: every variant and the end reload the opened project, before any re-render', async () => {
  // The project was just opened from the list: React state still says no
  // project, only the id handed to the follow knows which one it is.
  let open: string | null = 'p2';
  const loads: string[] = [];
  const views: string[] = [];
  let ended: [string | null, number] | null = null;
  await followSection('t', 'p2', 'job', {
    follow: fakeFollow([variant(['a']), variant([])]),
    load: async id => { loads.push(id); return `view-${id}-${loads.length}`; },
    isCurrent: id => open === id,
    onView: v => views.push(v),
    onProgress: () => {},
    onEnd: (status, landed) => { ended = [status, landed]; },
    onError: e => { throw e; },
  }, new AbortController().signal);
  assert.deepEqual(loads, ['p2', 'p2', 'p2']);
  assert.equal(views.length, 3);
  assert.deepEqual(ended, ['succeeded', 1]);
  open = null;
});

test('after the user moves to another project, a late result changes nothing', async () => {
  let open: string | null = 'p1';
  let release!: () => void;
  const gate = new Promise<void>(r => { release = r; });
  const views: string[] = [];
  let ended = false;
  const done = followSection('t', 'p1', 'job', {
    follow: fakeFollow([variant(['a']), variant(['b'])], gate),
    load: async id => `view-${id}`,
    isCurrent: id => open === id,
    onView: v => views.push(v),
    onProgress: () => {},
    onEnd: () => { ended = true; },
    onError: e => { throw e; },
  }, new AbortController().signal);
  await new Promise(r => setTimeout(r, 5));
  assert.deepEqual(views, ['view-p1']);
  open = 'p3';  // navigated away
  release();
  await done;
  assert.deepEqual(views, ['view-p1']);
  assert.equal(ended, false);
});
