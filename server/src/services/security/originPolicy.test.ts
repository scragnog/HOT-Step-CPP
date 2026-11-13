import { test } from 'node:test';
import assert from 'node:assert/strict';
import { isAllowedOrigin, originRejection, parseCorsOrigins, isLoopbackHost, defaultServerHost } from './originPolicy.js';
import { dawAuthError } from '../daw/dawFacade.js';

test('default bind host is loopback', () => {
  assert.equal(defaultServerHost(undefined), '127.0.0.1');
  assert.equal(defaultServerHost(''), '127.0.0.1');
  assert.equal(defaultServerHost('0.0.0.0'), '0.0.0.0');
  assert.equal(defaultServerHost('100.64.1.2'), '100.64.1.2');
});

test('isLoopbackHost', () => {
  for (const h of ['127.0.0.1', 'localhost', '::1', '[::1]']) assert.equal(isLoopbackHost(h), true, h);
  for (const h of ['0.0.0.0', '::', '192.168.1.5']) assert.equal(isLoopbackHost(h), false, h);
});

test('loopback and absent origins are allowed', () => {
  assert.equal(isAllowedOrigin(undefined, []), true);
  for (const o of ['http://localhost:3000', 'http://127.0.0.1:3001', 'http://[::1]:3001'])
    assert.equal(isAllowedOrigin(o, []), true, o);
});

test('foreign origins are refused unless listed', () => {
  for (const o of ['https://evil.example', 'http://localhost.evil.example', 'http://127.0.0.1.evil.example:3001', 'null'])
    assert.equal(isAllowedOrigin(o, []), false, o);
  const list = parseCorsOrigins(' http://100.64.1.2:3001 , https://studio.lan/ ');
  assert.deepEqual(list, ['http://100.64.1.2:3001', 'https://studio.lan']);
  assert.equal(isAllowedOrigin('http://100.64.1.2:3001', list), true);
  assert.equal(isAllowedOrigin('https://studio.lan', list), true);
  assert.equal(isAllowedOrigin('https://evil.example', list), false);
});

test('state-changing and DAW requests with a foreign Origin get 403', () => {
  const evil = 'https://evil.example';
  assert.equal(originRejection('POST', '/api/generate', evil, []), 403);
  assert.equal(originRejection('DELETE', '/api/songs/1', evil, []), 403);
  assert.equal(originRejection('GET', '/api/v1/daw/jobs', evil, []), 403);
  assert.equal(originRejection('GET', '/api/songs', evil, []), null);
  assert.equal(originRejection('OPTIONS', '/api/generate', evil, []), null);
  assert.equal(originRejection('POST', '/api/generate', undefined, []), null);
  assert.equal(originRejection('POST', '/api/v1/daw/generate', undefined, []), null);
  assert.equal(originRejection('POST', '/api/generate', 'http://localhost:3000', []), null);
});

test('same-origin by IP literal is allowed, by hostname is not', () => {
  assert.equal(isAllowedOrigin('http://192.168.1.5:3001', [], '192.168.1.5:3001'), true);
  assert.equal(isAllowedOrigin('http://evil.example:3001', [], 'evil.example:3001'), false);
  assert.equal(isAllowedOrigin('http://192.168.1.5:3001', [], '192.168.1.6:3001'), false);
  assert.equal(originRejection('POST', '/api/generate', 'http://100.64.1.2:3001', [], '100.64.1.2:3001'), null);
});

test('DAW bridge stays loopback-only with no key even when bound to LAN', () => {
  assert.notEqual(dawAuthError('', undefined, '192.168.1.20'), null);
  assert.notEqual(dawAuthError('', undefined, '100.64.1.2'), null);
  assert.equal(dawAuthError('', undefined, '127.0.0.1'), null);
});

test('host allowlist', async () => {
  const { buildAllowedHosts, isAllowedHost } = await import('./originPolicy.js');
  const allowed = buildAllowedHosts('studio.lan', ['http://Box.Tail1234.ts.net:3001'], ' my-tailnet-name ,Other.LAN ');
  for (const h of [undefined, '', 'localhost:3001', '127.0.0.5', '[::1]:3001', '[::1]', '192.168.1.5:3001', '[FE80::1]:3001', 'studio.lan:3001', 'box.tail1234.ts.net', 'MY-TAILNET-NAME:3001', 'other.lan'])
    assert.equal(isAllowedHost(h, allowed), true, String(h));
  for (const h of ['evil.example', 'evil.example:3001', 'localhost.evil.example', '127.0.0.1.evil.example'])
    assert.equal(isAllowedHost(h, allowed), false, h);
});

test('IPv6 same-origin compare is case-insensitive', () => {
  assert.equal(isAllowedOrigin('http://[FE80::1]:3001', [], '[fe80::1]:3001'), true);
  assert.equal(isAllowedOrigin('http://[fe80::1]:3001', [], '[FE80::1]:3001'), true);
});

test('loopback origins are trusted only on the server port or the Vite dev port', () => {
  const ports = [3001, 3000];
  for (const o of ['http://localhost:3001', 'http://127.0.0.1:3000', 'http://[::1]:3001'])
    assert.equal(originRejection('POST', '/api/generate', o, [], undefined, ports), null, o);
  for (const o of ['http://localhost:8080', 'http://127.0.0.1:5173', 'http://localhost'])
    assert.equal(originRejection('POST', '/api/generate', o, [], undefined, ports), 403, o);
  assert.equal(originRejection('POST', '/api/generate', 'http://localhost:8080', ['http://localhost:8080'], undefined, ports), null);
});

test('DAW path check is case-insensitive', () => {
  assert.equal(originRejection('GET', '/API/V1/DAW/status', 'https://evil.example', []), 403);
  assert.equal(originRejection('GET', '/Api/v1/Daw', 'https://evil.example', []), 403);
});
