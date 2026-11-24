// Counts of the page's costly steps, for the performance guards (tools/perf/guards.mjs, in the test build only): a
// view's model worked out again (`model`) and found changed (`model.changed`), the desk laid out (`pack`), tiles
// measured (`measure`, `measure.tiles`) and layouts it forced (`measure.layouts`), the desk store's notifications
// (`notify`). In the other builds `tally` does nothing.

export const TALLY: { [name: string]: number } = {};
export const tally: (name: string, n?: number) => void = import.meta.env.MODE === 'test' ? (name, n = 1) => { TALLY[name] = (TALLY[name] ?? 0) + n; } : () => {};
