// Types for react.mjs, the React for Node the existing page's components run on (tools/real-app/page.ts).
export function useState<T>(init: T | (() => T)): [T, (next: T | ((prev: T) => T)) => void];
export function within<T>(key: string, fn: () => T): T;
export function forget(): void;
export function flushEffects(): number;
