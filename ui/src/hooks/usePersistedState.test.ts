import assert from 'node:assert/strict';
import test from 'node:test';
import { act, createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { parseHTML } from 'linkedom';

const { window, document } = parseHTML('<html><body><div id="root"></div></body></html>');
const stored = new Map<string, string>();
Object.assign(globalThis, { window, document, IS_REACT_ACT_ENVIRONMENT: true, localStorage: {
  getItem: (key: string) => stored.get(key) ?? null,
  setItem: (key: string, value: string) => { stored.set(key, value); },
  removeItem: (key: string) => { stored.delete(key); },
} });
const { usePersistedState } = await import('./usePersistedState');

test('a saved value is revived on load, and a new key reads its own value instead of inheriting', async () => {
  stored.set('form:a', JSON.stringify({ rank: 32, only: 'one.flac' }));
  stored.set('form:b', JSON.stringify({ rank: 16 }));
  const seen: Array<Record<string, unknown>> = [];
  let set: (v: Record<string, unknown>) => void = () => {};
  const Probe = ({ id }: { id: string }) => {
    const [value, setValue] = usePersistedState<Record<string, unknown>>(`form:${id}`, {},
      ({ only: _only, ...rest }) => rest);
    seen.push(value); set = setValue;
    return null;
  };
  const root = createRoot(document.getElementById('root')!);
  await act(async () => root.render(createElement(Probe, { id: 'a' })));
  assert.deepEqual(seen.at(-1), { rank: 32 });
  await act(async () => set({ rank: 64 }));
  assert.deepEqual(JSON.parse(stored.get('form:a')!), { rank: 64 });
  await act(async () => root.render(createElement(Probe, { id: 'b' })));
  assert.deepEqual(seen.at(-1), { rank: 16 });
  assert.deepEqual(JSON.parse(stored.get('form:b')!), { rank: 16 }, 'switching keys must not write a\'s value into b');
  assert.deepEqual(JSON.parse(stored.get('form:a')!), { rank: 64 });
  await act(async () => root.unmount());
});
