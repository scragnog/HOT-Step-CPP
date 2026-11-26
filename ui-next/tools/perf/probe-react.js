// Injected before the page's scripts in the instrumented build: a stand-in for the React DevTools hook, so every
// commit is seen. For each commit it walks the fiber tree and counts the components that rendered in it (mounted, or
// whose props or hook state changed since the last commit), by name, with their own render time (selfBaseDuration,
// filled in by the profiling build of react-dom).
(() => {
  const R = window.__rx = { commits: 0, rendered: {}, self: {}, mounts: 0, total: 0, walkMs: 0 };
  const COMP = new Set([0, 1, 11, 14, 15]);
  const seen = new WeakMap();
  const nameOf = f => { const t = f.type; if (!t || typeof t === 'string') return null; return t.displayName || t.name || (t.render && (t.render.displayName || t.render.name)) || (t.type && (t.type.displayName || t.type.name)) || 'anon'; };
  function walk(root) {
    const t0 = performance.now(), stack = [root];
    while (stack.length) {
      const f = stack.pop();
      if (COMP.has(f.tag)) {
        const prev = seen.get(f);
        const rendered = !prev || prev.s !== f.memoizedState || prev.p !== f.memoizedProps;
        if (rendered) {
          const n = nameOf(f) || 'anon';
          R.rendered[n] = (R.rendered[n] || 0) + 1; R.self[n] = (R.self[n] || 0) + (f.selfBaseDuration || 0); R.total++;
          if (!f.alternate) R.mounts++;
          seen.set(f, { s: f.memoizedState, p: f.memoizedProps });
          if (f.alternate) seen.set(f.alternate, { s: f.memoizedState, p: f.memoizedProps });
        }
      }
      if (f.sibling) stack.push(f.sibling);
      if (f.child) stack.push(f.child);
    }
    R.walkMs += performance.now() - t0;
  }
  window.__REACT_DEVTOOLS_GLOBAL_HOOK__ = {
    supportsFiber: true, renderers: new Map(), isDisabled: false,
    inject(r) { const id = this.renderers.size + 1; this.renderers.set(id, r); return id; },
    checkDCE() {}, onScheduleFiberRoot() {}, onCommitFiberUnmount() {}, onPostCommitFiberRoot() {}, setStrictMode() {},
    onCommitFiberRoot(id, root) { R.commits++; try { walk(root.current); } catch (e) { R.err = String(e); } },
  };
  window.__rxReset = () => { R.commits = 0; R.rendered = {}; R.self = {}; R.mounts = 0; R.total = 0; R.walkMs = 0; };
})();
