// Counts of the page's costly steps, for the performance guards (tools/behaviour/scenarios/39-perf.mjs) and the
// measuring tools (tools/perf): a view's model worked out again (`model`) and found changed (`model.changed`), the desk
// laid out (`pack`), tiles measured (`measure`, `measure.tiles`) and the layouts that forced (`measure.layouts`), the
// layout solved (`solve`). Counted in the test build and the instrumented one (tools/perf/vite.instr.ts); in the
// others `tally` does nothing.

export const TALLY: { [name: string]: number } = {};
const on = import.meta.env.MODE === 'test' || !!import.meta.env.VITE_TALLY;
if (on) (globalThis as { __TALLY?: typeof TALLY }).__TALLY = TALLY;
export const tally: (name: string, n?: number) => void = on ? (name, n = 1) => { TALLY[name] = (TALLY[name] ?? 0) + n; } : () => {};
