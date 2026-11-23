import assert from 'node:assert/strict';
import test from 'node:test';
import { SerialQueue } from './serialQueue.js';

/** A promise resolved by hand, so a task can be held mid-flight. */
function gate(): { promise: Promise<void>; open: () => void } {
  let open = () => {};
  const promise = new Promise<void>(resolve => { open = resolve; });
  return { promise, open };
}

test('work runs in arrival order, one task at a time', async () => {
  const queue = new SerialQueue();
  const events: string[] = [];
  const held = gate();

  const first = queue.run(async () => {
    events.push('first start');
    await held.promise;
    events.push('first end');
  });
  const second = queue.run(async () => { events.push('second start'); });

  await Promise.resolve();
  assert.deepEqual(events, ['first start'], 'the second task has not started');
  assert.equal(queue.busy, true);

  held.open();
  await Promise.all([first, second]);
  assert.deepEqual(events, ['first start', 'first end', 'second start']);
  assert.equal(queue.busy, false);
  assert.equal(queue.queued, 0);
});

test('a queued task is told its place, and told again as the queue moves', async () => {
  const queue = new SerialQueue();
  const held = gate();
  const seen: number[] = [];

  const first = queue.run(() => held.promise);
  const second = queue.run(async () => {}, position => seen.push(position));
  const third = queue.run(async () => {}, position => seen.push(position));

  await Promise.resolve();
  assert.deepEqual(seen, [2, 3], 'second is 2nd in line and third is 3rd');

  held.open();
  await Promise.all([first, second, third]);
  assert.equal(seen.at(-1), 2, 'the third task counts down when the first finishes');
});

test('a failing task does not stall the queue', async () => {
  const queue = new SerialQueue();
  await assert.rejects(queue.run(async () => { throw new Error('boom'); }), /boom/);
  assert.equal(await queue.run(async () => 7), 7);
  assert.equal(queue.busy, false);
  assert.equal(queue.queued, 0);
});
