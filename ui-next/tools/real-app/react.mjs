// A React for Node, enough to run the existing Create page's components as functions and read what they draw: every
// element with its props (a control's label, its value, its onChange), so tools/real-app/page.ts can find a control by
// what the page shows and drive it through its own handler. Nothing is mounted: a render calls each function
// component with its props and returns the tree. Hooks give their initial values; a setter does nothing, because the
// page's values live in its stores, which the components read on every render.
//
// Served for `react`, `react/jsx-runtime` and `react-dom` by hooks.mjs. Used only by tools/compare-request.ts.

export const Fragment = 'Fragment';
export const StrictMode = 'StrictMode';
export const Suspense = 'Suspense';
const element = (type, props, key) => ({ $el: true, type, props: props ?? {}, key });
export const jsx = element;
export const jsxs = element;
export const jsxDEV = element;
export function createElement(type, props, ...children) {
  return element(type, { ...(props ?? {}), ...(children.length ? { children: children.length === 1 ? children[0] : children } : {}) });
}
export const isValidElement = x => !!x && typeof x === 'object' && x.$el === true;
export const cloneElement = (el, props, ...children) => element(el.type, { ...el.props, ...(props ?? {}), ...(children.length ? { children } : {}) });
const flat = c => (c == null || typeof c === 'boolean' ? [] : Array.isArray(c) ? c.flatMap(flat) : [c]);
export const Children = {
  toArray: flat, map: (c, f) => flat(c).map(f), forEach: (c, f) => flat(c).forEach(f), count: c => flat(c).length,
  only: c => flat(c)[0],
};

/* hooks: each component's slots, by its place in the tree and the order of its hooks */
const slots = new Map();
let at = null, n = 0;
/** Runs a component's render with its place in the tree, so its hooks find their slots. */
export function within(key, fn) { const was = [at, n]; at = key; n = 0; try { return fn(); } finally { [at, n] = was; } }
/** Forgets every component's state: a fresh page. */
export function forget() { slots.clear(); pending.length = 0; }
function slot(init) {
  const k = `${at}|${n++}`;
  if (!slots.has(k)) slots.set(k, { v: init() });
  return slots.get(k);
}
export function useState(init) {
  const s = slot(() => (typeof init === 'function' ? init() : init));
  return [s.v, next => { s.v = typeof next === 'function' ? next(s.v) : next; }];
}
export function useReducer(r, init, f) {
  const s = slot(() => (f ? f(init) : init));
  return [s.v, action => { s.v = r(s.v, action); }];
}
/* an effect runs after the render it was asked for in, when its dependencies changed (flushEffects) */
const pending = [];
const same = (a, b) => !!a && !!b && a.length === b.length && a.every((x, i) => Object.is(x, b[i]));
export function useEffect(fn, deps) {
  const s = slot(() => ({ deps: undefined }));
  if (deps && same(deps, s.deps)) return;
  s.deps = deps; pending.push(fn);
}
export const useLayoutEffect = useEffect;
/** Runs the effects the last render asked for; their cleanups are not kept. Returns how many ran. */
export function flushEffects() {
  const run = pending.splice(0);
  for (const fn of run) { try { fn(); } catch { /* an effect that needs the browser or the server does nothing here */ } }
  return run.length;
}
export const useInsertionEffect = () => {};
export const useMemo = f => f();
export const useCallback = f => f;
export const useRef = v => slot(() => ({ current: v })).v;
export const useImperativeHandle = () => {};
export const useId = () => 'id';
export const useTransition = () => [false, f => f()];
export const useDeferredValue = v => v;
export const useSyncExternalStore = (_sub, get) => get();
export const useDebugValue = () => {};
export function createContext(value) { const ctx = { $ctx: true, value }; ctx.Provider = 'Provider'; ctx.Consumer = 'Consumer'; return ctx; }
export const useContext = ctx => (ctx && ctx.$ctx ? ctx.value : undefined);
export const forwardRef = render => Object.assign(props => render(props, null), { displayName: render.name });
export const memo = c => c;
export const lazy = () => 'Lazy';
export const startTransition = f => f();
export const createPortal = children => children;
export const flushSync = f => f();

const React = {
  Fragment, StrictMode, Suspense, createElement, isValidElement, cloneElement, Children, useState, useReducer, useEffect,
  useLayoutEffect, useInsertionEffect, useMemo, useCallback, useRef, useImperativeHandle, useId, useTransition,
  useDeferredValue, useSyncExternalStore, useDebugValue, createContext, useContext, forwardRef, memo, lazy,
  startTransition, createPortal, flushSync, within, forget, flushEffects,
};
export default React;
