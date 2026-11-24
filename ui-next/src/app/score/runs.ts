// How a planned score is rendered with its seed, for that run only (E-08): generation registers how
// (features/score-preview.tsx, with generate/actions.ts generate); until it does, Continue only closes the preview.

let runner: ((seed: number) => boolean) | null = null;
export const scoreRuns = {
  use(run: (seed: number) => boolean): void { runner = run; },
  /** Renders the score with its seed; false while nothing renders scores (or the run was refused). */
  run(seed: number): boolean { return runner ? runner(seed) : false; },
};
