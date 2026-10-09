// fakeTrainingRunner.ts — stand-ins for the trainer runners, installed with
// labelingQueue.ts's overrideTrainingRunner(kind, run). They do what a real
// runner does to the job record (mark it running, report progress, finish
// it, honour cancel) without spawning a trainer. Everything around them is
// real: the start route's checks, the queue's lanes, the job routes, the SSE
// stream and the _meta.json a finished job leaves behind.

import type { TrainingJob } from '../../src/services/training/labelingQueue.js';

type Queue = typeof import('../../src/services/training/labelingQueue.js');

/** Runs `steps` progress steps `stepMs` apart, then finishes 'done'; a cancel
 *  ends it at the next step (cancelJob has already closed the record). */
export function completingRunner(q: Queue, steps = 3, stepMs = 50) {
  return async (job: TrainingJob) => {
    job.status = 'running';
    job.startedAt = Date.now();
    job.total = steps;
    job.phase = 'train';
    q.emitJob(job);
    for (let i = 1; i <= steps; i++) {
      await new Promise(r => setTimeout(r, stepMs));
      if (q.isCancelled(job)) return;
      job.done = i;
      q.emitProgress(job);
    }
    q.finishJob(job, 'done');
  };
}

/** Marks the job running and never settles: the job is mid-run until the
 *  process dies (restart tests) or the job is cancelled. */
export function heldRunner(q: Queue) {
  return async (job: TrainingJob) => {
    job.status = 'running';
    job.startedAt = Date.now();
    job.phase = 'train';
    q.emitJob(job);
    await new Promise<void>(resolve => job.controller.signal.addEventListener('abort', () => resolve(), { once: true }));
  };
}
