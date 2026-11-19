import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import * as hotkeys from '@/utils/hotkeys';

test('global handler respects editors, dialogs and Compare, and dispatches cancellable list actions', () => {
  let effect: () => (() => void);
  let handler: (e: any) => void;
  let modal = false;
  let handled = false;
  let palette = 0;
  let played = 0;
  const dispatched: any[] = [];
  const window = {
    location: { pathname: '/library' },
    addEventListener: (_: string, fn: any) => { handler = fn; },
    removeEventListener: (_: string, fn: any) => { assert.equal(fn, handler); },
    dispatchEvent: (event: any) => { dispatched.push(event); if (handled) event.preventDefault(); },
  };
  const exports: any = {};
  const source = readFileSync(new URL('./useGlobalHotkeys.ts', import.meta.url), 'utf8');
  vm.runInNewContext(ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText, {
    exports, window, document: { querySelector: () => modal ? {} : null }, CustomEvent: class extends Event {
      detail: any;
      constructor(type: string, init: any) { super(type, init); this.detail = init.detail; }
    },
    require: (id: string) => id === 'react' ? { useEffect: (fn: any) => { effect = fn; } }
      : id.includes('hotkeys') ? hotkeys : { togglePlay: () => { played++; } },
  });
  exports.useGlobalHotkeys(() => {}, () => { palette++; });
  const cleanup = effect!();
  const key = (key: string, target: any = {}, extra = {}) => {
    const event = { key, target, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, ...extra };
    handler!(event);
    return event;
  };
  for (const target of [{ tagName: 'INPUT' }, { tagName: 'TEXTAREA' }, { tagName: 'SELECT' }, { isContentEditable: true }]) {
    assert.equal(key(' ', target).defaultPrevented, false);
    key('Delete', target);
  }
  modal = true; key(' '); key('Delete'); key('k', {}, { ctrlKey: true });
  assert.equal(palette, 0);
  modal = false;
  window.location.pathname = '/compare'; key(' '); key('Delete');
  assert.equal(played, 0); assert.equal(dispatched.length, 0);
  window.location.pathname = '/library';
  key('k', { tagName: 'INPUT' }, { metaKey: true }); assert.equal(palette, 1);
  assert.equal(key('ArrowDown').defaultPrevented, false, 'unhandled list keys retain native behavior');
  handled = true;
  for (const [keyName, action] of [['ArrowUp', 'focusPrev'], ['Enter', 'playFocused'], ['Delete', 'deleteSelection'], ['Escape', 'clearSelection']]) {
    assert.equal(key(keyName).defaultPrevented, true);
    assert.equal(dispatched.at(-1).detail.action, action);
    assert.equal(dispatched.at(-1).cancelable, true);
  }
  key(' ', { tagName: 'BUTTON' }); key('Enter', { tagName: 'A' });
  assert.equal(played, 0);
  key(' '); assert.equal(played, 1);
  cleanup();
});
