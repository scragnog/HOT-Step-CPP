// The behaviour suite's page driver: one fresh, deterministic page per scenario, and the helpers scenarios use (t.click,
// t.press, t.drag, t.ok, ...). See README.md in this folder.
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';

const T0 = Date.UTC(2031, 0, 1, 12, 0, 0);
/* the same determinism as tools/visual-baseline: seeded Math.random, instant smooth scrolling; plus downloads are
   captured instead of saved (exports are read back as text), and a small library for settling and snapshots */
const INIT = `(() => { let s = 0x2545F491; Math.random = () => { s |= 0; s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const inst = o => (o && typeof o === 'object' && o.behavior === 'smooth') ? { ...o, behavior: 'instant' } : o;
  for (const P of [Element.prototype, window]) for (const m of ['scrollTo', 'scrollBy', 'scroll']) { const f = P[m]; if (f) P[m] = function (a, b) { return typeof a === 'object' ? f.call(this, inst(a)) : f.call(this, a, b); }; }
  const siv = Element.prototype.scrollIntoView; Element.prototype.scrollIntoView = function (o) { return siv.call(this, inst(o)); };
  const bh = window.__bh = { downloads: [], blobs: {}, cov: {} };
  /* coverage: the data-* keys carried by what each click landed on (the delegated handlers dispatch on these) */
  document.addEventListener('click', e => { for (let el = e.target; el && el !== document; el = el.parentElement) if (el.dataset) for (const k of Object.keys(el.dataset)) bh.cov[k] = (bh.cov[k] || 0) + 1; if (e.target.id) bh.cov['#' + e.target.id] = 1; }, true);
  const cou = URL.createObjectURL; URL.createObjectURL = function (b) { const u = cou.call(this, b); bh.blobs[u] = b; return u; };
  const ac = HTMLAnchorElement.prototype.click; HTMLAnchorElement.prototype.click = function () { if (this.download) { bh.downloads.push({ name: this.download, blob: bh.blobs[this.href] }); return; } return ac.call(this); };
  bh.finish = () => { for (let round = 0; round < 4; round++) { document.body && document.body.offsetWidth; let moved = 0;
      for (const a of document.getAnimations()) { try { const t = a.effect && a.effect.getComputedTiming(); if (!t) continue;
        if (t.iterations === Infinity || t.endTime === Infinity) { if (a.playState !== 'paused' || a.currentTime !== 0) { a.pause(); a.currentTime = 0; moved++; } }
        else if (a.playState !== 'finished') { a.finish(); moved++; } } catch (e) { try { a.cancel(); moved++; } catch (e2) {} } }
      if (!moved) break; } };
  const hash = s => { let x = 0x811c9dc5; for (let i = 0; i < s.length; i++) { x ^= s.charCodeAt(i); x = Math.imul(x, 16777619); } return (x >>> 0).toString(16); };
  const desc = a => !a ? '' : a.tagName.toLowerCase() + (a.id ? '#' + a.id : '') + [...a.attributes].filter(x => x.name.startsWith('data-') || x.name === 'aria-label').map(x => '[' + x.name + '=' + x.value + ']').join('');
  /* a digest of everything a step can change: each top-level element's markup, the state object, storage, focus, title */
  /* the toast and the dropdown float, and the page re-places them on scroll events, which the browser sends on its own
     frames: their inline position is left out (their content is not; where they sit is the visual baseline's job) */
  const markup = e => { if (e.id !== 'toast' && e.id !== 'dd') return e.outerHTML; const c = e.cloneNode(true); c.removeAttribute('style'); return c.outerHTML; };
  bh.snap = full => { const kids = [...document.body.children].filter(e => e.tagName !== 'SCRIPT').map(e => { const h = markup(e); return [e.id || e.getAttribute('class') || e.tagName, hash(h), full ? h : h.length]; });
    /* the state object with its top-level keys sorted (which file adds a key first is not behaviour), and a key that holds
       null or an empty object treated as absent (an empty build is no build) */
    let st = ''; try { const S = window.S, o = {}; Object.keys(S).sort().forEach(k => { const v = S[k]; if (v === null || v === undefined || (v.constructor === Object && !Object.keys(v).length)) return; o[k] = v; }); st = JSON.stringify(o, (k, v) => v instanceof Set ? [...v] : v); } catch (e) { st = 'unserializable: ' + e.message; }
    const ls = {}; for (let i = 0; i < localStorage.length; i++) { const k = localStorage.key(i); ls[k] = localStorage.getItem(k); }
    return { kids, S: st, ls, act: desc(document.activeElement), title: document.title, root: document.documentElement.className + '|' + (document.documentElement.dataset.theme || ''), scroll: [scrollX, scrollY] }; };
})();`;

/* the ports Chromium refuses to load from (net::ERR_UNSAFE_PORT, its kRestrictedPorts): the system can hand one out
   for port 0 (6669 was, and every scenario of that run failed at page.goto), so the server listens again elsewhere */
const UNSAFE_PORTS = new Set([1, 7, 9, 11, 13, 15, 17, 19, 20, 21, 22, 23, 25, 37, 42, 43, 53, 69, 77, 79, 87, 95, 101, 102, 103, 104, 109, 110, 111, 113, 115, 117, 119, 123, 135, 137, 139, 143, 161, 179, 389, 427, 465, 512, 513, 514, 515, 526, 530, 531, 532, 540, 548, 554, 556, 563, 587, 601, 636, 989, 990, 993, 995, 1719, 1720, 1723, 2049, 3659, 4045, 4190, 5060, 5061, 6000, 6566, 6665, 6666, 6667, 6668, 6669, 6679, 6697, 10080]);
const listenSafe = srv => new Promise(res => { const go = () => srv.listen(0, '127.0.0.1', () => { if (!UNSAFE_PORTS.has(srv.address().port)) return res(srv.address().port); srv.close(() => go()); }); go(); });
export function serve(file) {
  const body = fs.readFileSync(file);
  const srv = http.createServer((q, r) => { r.writeHead(200, { 'content-type': 'text/html; charset=utf-8' }); r.end(body); });
  return listenSafe(srv).then(port => ({ url: `http://127.0.0.1:${port}/hotstep-tiers.html`, close: () => srv.close() }));
}

/* Runs one scenario; returns { id, results: [[ok, label, detail]], errs, trace } */
export async function runScenario(browser, sc, url, o = {}) {
  const ctx = await browser.newContext({ viewport: { width: sc.w || 1920, height: sc.h || 1080 }, colorScheme: sc.scheme || 'light', reducedMotion: sc.reduced ? 'reduce' : 'no-preference', deviceScaleFactor: 1, hasTouch: !!sc.touch });
  await ctx.route(u => !u.href.startsWith(url.replace(/hotstep-tiers\.html$/, '')), r => r.abort());
  const page = await ctx.newPage();
  const results = [], errs = [], trace = [];
  page.on('pageerror', e => errs.push('pageerror: ' + e.message));
  page.on('console', m => { if ((m.type() === 'error' || m.type() === 'warning') && !/net::ERR|Failed to load resource/.test(m.text())) errs.push(m.type() + ': ' + m.text()); });
  await page.addInitScript(INIT);
  await page.clock.pauseAt(T0);
  /* Animations run on the browser's own timeline, in real time; the page measures layout at moments the fake clock sets
     (a toast's width, a button's place), so how far an animation had run would depend on the machine's load. The
     timeline is frozen: every animation stays at its start until the harness finishes it, at the same points every run. */
  const cdp = await ctx.newCDPSession(page); const freeze = async () => { await cdp.send('Animation.enable'); await cdp.send('Animation.setPlaybackRate', { playbackRate: 0 }); };
  await freeze();
  const fin = () => page.evaluate(() => window.__bh?.finish());
  let mouse = null;
  /* Fake time runs timers and animation frames; ResizeObserver and IntersectionObserver are delivered by the browser's own
     rendering updates, so after each slice of fake time the harness waits for two real ones (seen by a throwaway
     ResizeObserver), whatever the machine's load. The fake time spent is the same on every run. */
  const frame = () => page.evaluate(() => new Promise(r => { let n = 0; const go = () => { const ro = new ResizeObserver(() => { ro.disconnect(); if (++n < 2) go(); else r(); }); ro.observe(document.documentElement); }; go(); }));
  const settle = async (ms = 600) => { const step = Math.max(50, Math.round(ms / 3)); for (let i = 0; i < 3; i++) { await fin(); await page.clock.runFor(step); await fin(); await frame(); }
    for (let i = 0; i < 3; i++) { await page.clock.runFor(16); await fin(); await frame(); }
    /* when what is under a still mouse changes (an overlay closes), the browser sends mouseout and mouseover with a mouse
       move of its own, on its own clock: the harness moves the mouse to where it is, so they arrive now, every run */
    if (mouse) { await page.mouse.move(mouse.x, mouse.y); await frame(); } };
  /* every undo step the page makes is written down, so the end of the scenario can check how each one is named */
  const boot = async () => { await page.evaluate(() => { const raf = window.requestAnimationFrame; window.requestAnimationFrame = cb => raf(() => cb(performance.now()));
    const pu = window.pushUndo; if (pu) window.pushUndo = function (label, ...rest) { (window.__bh.labels = window.__bh.labels || []).push(label); return pu.call(this, label, ...rest); }; }); await settle(1500); };
  if (sc.storage) await page.addInitScript(st => { if (!sessionStorage.getItem('__bh_seeded')) { sessionStorage.setItem('__bh_seeded', '1'); for (const [k, v] of Object.entries(st)) localStorage.setItem(k, v); } }, sc.storage);
  await page.goto(url);
  await boot();
  let step = 0; const cov = [];
  const snap = async label => { if (!o.trace) return; trace.push({ step: step++, label, ...(await page.evaluate(f => window.__bh.snap(f), !!o.dump)) }); };
  /* the page scrolls the element into view itself; the scroll events that sends arrive with the browser's next frames, so
     they are let in before the pointer acts, every time */
  const where = async sel => { const r = await page.evaluate(sel => { const el = document.querySelector(sel); if (!el) return null; el.scrollIntoView({ block: 'center', inline: 'nearest', behavior: 'instant' }); const r = el.getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, h: r.height }; }, sel); if (r) await frame(); return r; };
  const t = {
    page, settle,
    /* a query (no snapshot) and a change (snapshot after) */
    js: (fn, arg) => page.evaluate(fn, arg),
    /* a change the scenario makes from inside the page sets the scene: it is not a person's gesture, so the page's undo
       history takes it as where things stand (any event it dispatched is a gesture, and is recorded as one) */
    do: async (label, fn, arg, ms) => { await page.evaluate(() => { if (typeof vsG === 'number') { if (typeof vsSoonT !== 'undefined' && vsSoonT) vsCatch(); vsG++; } }); await page.evaluate(fn, arg); await page.evaluate(() => { if (typeof vsCatch !== 'function') { if (typeof vsSettle === 'function') vsSettle(); return; } if (vsSoonT) vsCatch(); else vsSettle(); }); await settle(ms); await snap(label); },
    /* a real mouse click at the element's centre (or at fx, fy of its box), after the page scrolled it into view itself */
    click: async (sel, ms, at) => { let r = await where(sel); if (!r) { results.push([false, 'click ' + sel, 'no such element']); return; }
      /* an action that shows only while its row is hovered (a workflow row's buttons): point at the row first, as a hand would */
      /* (the row's keys are laid out but take the pointer only while it is over the row) */
      const asleep = r.w && await page.evaluate(([sel, x, y]) => { const el = document.querySelector(sel), h = document.elementFromPoint(x, y); return !!el.closest('.pre .pacts') && !(h && (el === h || el.contains(h))); }, [sel, r.x + r.w / 2, r.y + r.h / 2]);
      if (!r.w || asleep) { const row = await page.evaluate(sel => { const p = document.querySelector(sel).closest('.pre'); if (!p) return null; const b = p.getBoundingClientRect(); return { x: b.left + 8, y: b.top + b.height / 2 }; }, sel); if (row) { await page.mouse.move(row.x, row.y); r = await where(sel); } }
      const [fx, fy] = at || [0.5, 0.5], x = r.x + r.w * fx, y = r.y + r.h * fy;
      /* a click that would land on something else (a toast over the button, say) is a mistake in the scenario: say so */
      const cover = await page.evaluate(([sel, x, y]) => { const el = document.querySelector(sel), h = document.elementFromPoint(x, y); return !h || el === h || el.contains(h) ? null : (h.id ? '#' + h.id : h.tagName.toLowerCase() + '.' + [...h.classList].join('.')); }, [sel, x, y]);
      if (cover) { results.push([false, 'click ' + sel, 'covered by ' + cover]); return; }
      await page.mouse.click(x, y); mouse = { x, y }; await settle(ms); await snap('click ' + sel); },
    dblclick: async (sel, ms) => { const r = await where(sel); if (!r) { results.push([false, 'dblclick ' + sel, 'no such element']); return; } await page.mouse.dblclick(r.x + r.w / 2, r.y + r.h / 2); mouse = { x: r.x + r.w / 2, y: r.y + r.h / 2 }; await settle(ms); await snap('dblclick ' + sel); },
    hover: async (sel, ms) => { const r = await where(sel); if (!r) { results.push([false, 'hover ' + sel, 'no such element']); return; } await page.mouse.move(r.x + r.w / 2, r.y + r.h / 2); mouse = { x: r.x + r.w / 2, y: r.y + r.h / 2 }; await settle(ms); await snap('hover ' + sel); },
    press: async (key, ms) => { await page.keyboard.press(key); await settle(ms); await snap('press ' + key); },
    type: async (text, ms) => { await page.keyboard.type(text); await settle(ms); await snap('type ' + text.slice(0, 20)); },
    focus: async (sel, ms) => { const ok = await page.evaluate(sel => { const el = document.querySelector(sel); if (!el) return false; el.scrollIntoView({ block: 'center', behavior: 'instant' }); el.focus(); return true; }, sel); if (!ok) results.push([false, 'focus ' + sel, 'no such element']); await settle(ms); await snap('focus ' + sel); },
    /* a pointer drag: press at (fx, fy) of the element's box, move by each [dx, dy] in turn, release */
    mouseDrag: async (sel, moves, at, ms) => { const r = await where(sel); if (!r) { results.push([false, 'drag ' + sel, 'no such element']); return; }
      const [fx, fy] = at || [0.5, 0.5]; let x = r.x + r.w * fx, y = r.y + r.h * fy; await page.mouse.move(x, y); await page.mouse.down();
      mouse = { x, y }; for (const [dx, dy] of moves) { x += dx; y += dy; await page.mouse.move(x, y, { steps: 4 }); mouse = { x, y }; await settle(100); }
      await page.mouse.up(); await settle(ms); await snap('drag ' + sel); },
    /* an HTML5 drag: dragstart on the source, dragover on the target at (fx, fy), then drop and dragend (or only dragend) */
    drag: async (src, dst, o2 = {}) => {
      const okStart = await page.evaluate(src => { const el = document.querySelector(src); if (!el) return false; window.__dt = new DataTransfer(); window.__dsrc = el; el.dispatchEvent(new DragEvent('dragstart', { bubbles: true, cancelable: true, dataTransfer: window.__dt })); return true; }, src);
      if (!okStart) { results.push([false, 'drag ' + src, 'no such element']); return; }
      await settle(200);
      const okOver = await page.evaluate(([dst, at]) => { const t = document.querySelector(dst); if (!t) return false; const r = t.getBoundingClientRect(), x = r.left + r.width * at[0], y = r.top + r.height * at[1];
        const o = { bubbles: true, cancelable: true, dataTransfer: window.__dt, clientX: x, clientY: y }; const el = document.elementFromPoint(x, y) || t; el.dispatchEvent(new DragEvent('dragenter', o)); el.dispatchEvent(new DragEvent('dragover', o)); window.__dpt = [el, o]; return true; }, [dst, o2.at || [0.3, 0.3]]);
      if (!okOver) results.push([false, 'drag to ' + dst, 'no such element']);
      await settle(300);
      await page.evaluate(drop => { const [el, o] = window.__dpt || []; if (drop && el) el.dispatchEvent(new DragEvent('drop', o)); const s = window.__dsrc; if (s && s.isConnected) s.dispatchEvent(new DragEvent('dragend', { bubbles: true, dataTransfer: window.__dt })); }, o2.drop !== false);
      await settle(o2.ms); await snap('drag ' + src + ' -> ' + dst);
    },
    /* a file dropped on an element (a drop zone, or anywhere) */
    dropFile: async (sel, name, type, ms) => { await page.evaluate(([sel, name, type]) => { const el = document.querySelector(sel), dt = new DataTransfer(); dt.items.add(new File(['x'], name, { type }));
      const r = el.getBoundingClientRect(), o = { bubbles: true, cancelable: true, dataTransfer: dt, clientX: r.left + 4, clientY: r.top + 4 }; el.dispatchEvent(new DragEvent('dragover', o)); el.dispatchEvent(new DragEvent('drop', o)); }, [sel, name, type]); await settle(ms); await snap('drop ' + name); },
    file: async (sel, name, mime, text, ms) => { await page.setInputFiles(sel, { name, mimeType: mime, buffer: Buffer.from(text ?? 'x') }); await settle(ms); await snap('file ' + name); },
    resize: async (w, h, ms) => { await page.setViewportSize({ width: w, height: h || 1080 }); await settle(ms); await snap('resize ' + w); },
    reload: async () => { cov.push(await page.evaluate(() => window.__bh.cov)); await page.reload(); await freeze(); await boot(); await snap('reload'); },
    toast: () => page.evaluate(() => document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg').textContent),
    download: () => page.evaluate(async () => { const d = window.__bh.downloads.pop(); return d ? { name: d.name, text: d.blob ? await d.blob.text() : '' } : null; }),
    ok: async (label, fn, arg) => { let v, detail = ''; try { v = await page.evaluate(fn, arg); } catch (e) { v = false; detail = String(e.message).split('\n')[0]; }
      const pass = v === true || (v && typeof v === 'object' && v.ok === true); if (!pass && !detail) detail = typeof v === 'object' ? JSON.stringify(v).slice(0, 300) : String(v).slice(0, 300); results.push([pass, label, pass ? '' : detail]); },
    /* a value the page computes must equal what is expected */
    eq: async (label, fn, want, arg) => { let v; try { v = await page.evaluate(fn, arg); } catch (e) { v = 'threw: ' + String(e.message).split('\n')[0]; } const pass = JSON.stringify(v) === JSON.stringify(want); results.push([pass, label, pass ? '' : `got ${JSON.stringify(v)?.slice(0, 300)}, want ${JSON.stringify(want)?.slice(0, 300)}`]); },
    note: label => snap(label),
  };
  await snap('boot');
  try { await sc.run(t); } catch (e) { results.push([false, 'scenario threw', String(e.stack || e).split('\n').slice(0, 3).join(' | ')]); }
  await settle(600);
  await snap('end');
  cov.push(await page.evaluate(() => window.__bh.cov));
  /* Two checks on every scenario (L-01, L-09, Q-04). Every change a person makes to the desk is an undo step: the page
     writes down, in vsStray, any change it found between gestures that is not one it makes itself, so a control that
     changes the desk without a step fails here wherever a scenario uses it. And every step is named in the page's own
     words: never a control's internal key, a bare "On", or the vague words the page used before it named them. */
  const und = await page.evaluate(() => ({ stray: window.vsStray || [], labels: window.__bh.labels || [] })).catch(() => ({ stray: [], labels: [] }));
  results.push([!und.stray.length, 'every change to the desk was an undo step', und.stray.join(' | ').slice(0, 400)]);
  const BAD_LABEL = /\b(?:lm|mm|yp|yc|yue|hm|samp|pp|ad|lat|sch|sa3|rp|tempo|lyrics|takes|seed|model|song|cap|apg|perf|dcw|den|lss|timbre|avoid|wild|ag)\.[a-z]|^On (?:on|off) →|\b(?:on|off) (?:on|off) →|\bChange wild\b|Songs per Generate|^Slot \d|^Change the (?:timbre reference|mastering reference|repaint region|recording|adapters)(?: and |$)|^Edit Lyrics$|^Clear file$|\bundefined\b|\bnull\b|\bNaN\b|\b[a-z]{2}[A-Z]{2}/;
  const bad = [...new Set(und.labels.filter(l => BAD_LABEL.test(l)))];
  results.push([!bad.length, "every undo step is named in the page's words", bad.join(' | ').slice(0, 400)]);
  results.push([!errs.length, 'no console errors or warnings', errs.join(' | ').slice(0, 600)]);
  await ctx.close();
  return { id: sc.id, results, trace, cov: Object.assign({}, ...cov) };
}
