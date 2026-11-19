// serialQueue.ts — one thing at a time, with a visible place in line.
//
// The MMS_FA aligner wants about 5 GB of compute buffer and the engine holds a
// single mutex around it, so a second track's alignment cannot make progress
// while the first is running — it just waits inside a 900 s HTTP timeout with no
// sign of why, and a slow first track surfaces as a failed second one. This
// queue makes the wait visible: work runs in arrival order, and the caller is
// told its position so the stage can say "2nd in line" instead of looking stuck.
//
// It is deliberately not the GPU lane. Whisper and the aligner are CPU work, and
// sharing a lane with renders would put a lyric run in front of a song. Polish
// runs never queue at all: they load no model.

export class SerialQueue {
  private readonly waiting: Array<{ start: () => void; notify: (position: number) => void }> = [];
  private running = false;

  /** True while a task holds the slot. */
  get busy(): boolean { return this.running; }
  /** Tasks waiting behind the one that is running. */
  get queued(): number { return this.waiting.length; }

  /**
   * Run `task` after everything queued before it.
   *
   * `onQueued` receives the task's 1-based place in line — 2 means one task is
   * running — and is called again whenever that place changes, so a stage can
   * count down. It is not called at all when the task starts immediately.
   */
  async run<T>(task: () => Promise<T>, onQueued?: (position: number) => void): Promise<T> {
    if (this.running || this.waiting.length > 0) {
      await new Promise<void>(resolve => {
        this.waiting.push({ start: resolve, notify: onQueued ?? (() => {}) });
        onQueued?.(this.placeOf(this.waiting.length));
      });
    }
    this.running = true;
    try {
      return await task();
    } finally {
      const next = this.waiting.shift();
      if (next) {
        // Hand the slot over before the continuation can run, so a task queued
        // in the same tick cannot slip in beside it.
        this.running = true;
        this.waiting.forEach((entry, index) => entry.notify(this.placeOf(index + 1)));
        next.start();
      } else {
        this.running = false;
      }
    }
  }

  /** Place in line of the nth waiting task, counting the one that is running. */
  private placeOf(waitingIndexOneBased: number): number {
    return waitingIndexOneBased + (this.running ? 1 : 0);
  }
}
