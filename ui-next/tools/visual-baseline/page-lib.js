/* Runs inside the page (evaluated by vb.mjs after load). Defines window.__vb: freezing animations, stable element keys,
   computed-style records and the pseudo-state forcing plan. Plain script, no imports. */
(() => {
  const PSEUDOS = ['::before', '::after'];
  const STATE = ['hover', 'focus', 'focus-visible', 'focus-within', 'active'];
  const vb = window.__vb = {};

  /* finite animations and transitions jump to their end; infinite ones stop at time 0 */
  vb.finish = () => {
    let n = 0;
    for (let round = 0; round < 4; round++) {
      document.body && document.body.offsetWidth;
      const as = document.getAnimations();
      let moved = 0;
      for (const a of as) {
        try {
          const t = a.effect && a.effect.getComputedTiming();
          if (!t) continue;
          if (t.iterations === Infinity || t.endTime === Infinity) { if (a.playState !== 'paused' || a.currentTime !== 0) { a.pause(); a.currentTime = 0; moved++; } }
          else if (a.playState !== 'finished') { a.finish(); moved++; }
        } catch (e) { try { a.cancel(); moved++; } catch (e2) {} }
      }
      n += moved; if (!moved) break;
    }
    return n;
  };

  /* Lay out every box again, from the root, before a reading. Chromium answers getComputedStyle for an `auto` margin
     from the margin it stored on the box at its last layout, and that depends on how the box was last laid out, not
     only on the layout it has now: when a box was last placed by its parent the stored margin is the used one, but some
     incremental layouts (a style change elsewhere, such as a forced :hover in a sibling subtree) store the margin as
     computed, `auto` as 0, and leave the box where it was. Which of the two came last depends on what the page did
     before and on when frames happened, so the same layout read 12px in one run and 0px in another (#bento in the
     Quick tier). A constructable stylesheet that sets contain-intrinsic-size on every element and is then removed
     marks every box for layout twice and changes nothing else: contain-intrinsic-size only acts on a size-contained box,
     where `0px` and `none` (the page sets no other value) give the same size, `none` to `0px` cannot transition, and an
     adopted stylesheet is not a DOM mutation. After the second layout every box has been placed by its parent. */
  const relayoutSheet = new CSSStyleSheet();
  relayoutSheet.replaceSync('*, *::before, *::after { contain-intrinsic-size: 0px !important; }');
  let cisChecked = false;
  vb.relayout = () => {
    if (!cisChecked) { cisChecked = true;
      const css = [...document.styleSheets].map(sh => { try { return [...sh.cssRules].map(r => r.cssText).join('\n'); } catch (e) { return ''; } }).join('\n');
      if (/contain-intrinsic/.test(css + [...document.querySelectorAll('[style]')].map(e => e.getAttribute('style')).join(';')))
        throw new Error('the page sets contain-intrinsic-size, which vb.relayout() in page-lib.js overrides: give relayout another property'); }
    const html = document.documentElement, before = document.getAnimations().length;
    document.adoptedStyleSheets = [...document.adoptedStyleSheets, relayoutSheet]; html.getBoundingClientRect();
    document.adoptedStyleSheets = document.adoptedStyleSheets.filter(s => s !== relayoutSheet); html.getBoundingClientRect();
    if (document.getAnimations().length !== before) vb.finish();
  };

  /* resolves after n rendering updates of the real browser (the page's timers and frame callbacks run on the fake clock,
     but ResizeObserver and IntersectionObserver are delivered in real rendering updates). A new ResizeObserver reports
     its first observation in the next rendering update; the next one is started from a task, after that update. */
  const RO = window.ResizeObserver;
  vb.frames = (n = 2) => new Promise(res => {
    const ch = new MessageChannel();
    const one = () => { const ro = new RO(() => { ro.disconnect(); if (--n <= 0) res(true); else ch.port1.postMessage(0); }); ro.observe(document.documentElement); };
    ch.port2.onmessage = one; one();
  });

  vb.useProps = list => { vb.props = list; return list.length; };
  vb.setProps = custom => {
    const cs = getComputedStyle(document.documentElement), p = [];
    for (let i = 0; i < cs.length; i++) if (!cs[i].startsWith('--')) p.push(cs[i]);
    p.sort();
    vb.props = p.concat(custom.filter(c => !p.includes(c)).sort());
    return vb.props.length;
  };

  /* stable key: a unique id, else the parent's key and the element's index among its element siblings */
  vb.keys = () => {
    const ids = {}, map = new Map();
    document.querySelectorAll('[id]').forEach(e => { ids[e.id] = (ids[e.id] || 0) + 1; });
    const walk = (el, key) => {
      map.set(el, key);
      let i = 0;
      for (const c of el.children) { const k = c.id && ids[c.id] === 1 && !/^f-/.test(c.id) ? '#' + c.id : key + '>' + c.localName + ':' + i; walk(c, k); i++; }
    };
    walk(document.documentElement, 'html');
    vb.map = map;
    return map;
  };

  const read = (cs, props) => { const v = new Array(props.length); for (let i = 0; i < props.length; i++) v[i] = cs.getPropertyValue(props[i]); return v; };

  /* records for a list of elements: [key, pseudo, values[]] */
  vb.collect = (els, props = vb.props) => {
    const out = [];
    for (const el of els) {
      const key = vb.map.get(el); if (!key) continue;
      out.push([key, '', read(getComputedStyle(el), props)]);
      for (const ps of PSEUDOS) { const c = getComputedStyle(el, ps), ct = c.content; if (ct && ct !== 'none' && ct !== 'normal') out.push([key, ps, read(c, props)]); }
      if ((el.localName === 'input' || el.localName === 'textarea') && el.hasAttribute('placeholder')) out.push([key, '::placeholder', read(getComputedStyle(el, '::placeholder'), props)]);
    }
    return out;
  };

  /* sparse encoding against the most common value of each property */
  vb.encode = recs => {
    const props = vb.props, mode = [];
    for (let i = 0; i < props.length; i++) {
      const cnt = new Map(); let best = '', bn = -1;
      for (const r of recs) { const v = r[2][i], n = (cnt.get(v) || 0) + 1; cnt.set(v, n); if (n > bn) { bn = n; best = v; } }
      mode.push(best);
    }
    const els = recs.map(([k, ps, v]) => { const d = {}; for (let i = 0; i < v.length; i++) if (v[i] !== mode[i]) d[i] = v[i]; return [k, ps, d]; });
    return { mode, els };
  };

  vb.recordAll = () => { vb.finish(); vb.relayout(); vb.keys(); const els = [...document.querySelectorAll('*')]; return vb.encode(vb.collect(els)); };

  /* ---- pseudo-state plan ---- */
  const splitTop = (s, sepRe) => { const out = []; let depth = 0, q = null, cur = ''; for (let i = 0; i < s.length; i++) { const c = s[i];
    if (q) { cur += c; if (c === '\\') { cur += s[++i] || ''; continue; } if (c === q) q = null; continue; }
    if (c === '"' || c === "'") { q = c; cur += c; continue; }
    if (c === '(' || c === '[') depth++; else if (c === ')' || c === ']') depth--;
    if (!depth && sepRe.test(c)) { out.push(cur); cur = ''; continue; } cur += c; } out.push(cur); return out; };
  /* split a complex selector into compounds; combinators are dropped (only ancestry is used to pick elements) */
  const compounds = sel => { const out = []; let depth = 0, q = null, cur = ''; const s = sel.trim();
    for (let i = 0; i < s.length; i++) { const c = s[i];
      if (q) { cur += c; if (c === '\\') { cur += s[++i] || ''; continue; } if (c === q) q = null; continue; }
      if (c === '"' || c === "'") { q = c; cur += c; continue; }
      if (c === '(' || c === '[') depth++; else if (c === ')' || c === ']') depth--;
      if (!depth && (c === ' ' || c === '>' || c === '+' || c === '~')) { if (cur.trim()) out.push(cur.trim()); cur = ''; continue; }
      cur += c; }
    if (cur.trim()) out.push(cur.trim()); return out; };
  const stRe = new RegExp(':(' + STATE.join('|') + ')(?![\\w-])', 'g');
  /* remove state pseudo-classes from a compound; return the stripped compound and the ones to force */
  const strip = comp => {
    const force = new Set();
    let s = comp.replace(/::?(before|after|placeholder|selection|marker|-webkit-[\w-]+)(?![\w-])/g, m => /^:(before|after)$/.test(m) || m.startsWith('::') ? '' : m);
    /* :not(...) whose arguments are only state pseudos: the default state, nothing to force */
    s = s.replace(/:not\(((?:\s*,?\s*:(?:hover|focus-visible|focus-within|focus|active))+)\s*\)/g, '');
    s = s.replace(/:(is|where)\(((?:\s*,?\s*:(?:hover|focus-visible|focus-within|focus|active))+)\s*\)/g, (m, f, args) => { (args.match(stRe) || []).forEach(x => force.add(x.slice(1))); return ''; });
    if (/:has\(/.test(s) && stRe.test(s)) { stRe.lastIndex = 0; return null; }
    stRe.lastIndex = 0;
    s = s.replace(/:not\([^()]*\)/g, m => m); // leave other :not() as is
    s = s.replace(stRe, (m, p) => { force.add(p); return ''; });
    if (!s || /^[>+~]/.test(s)) s = '*';
    return { s, force: [...force] };
  };
  const rulesOf = list => { const out = []; for (const r of list) { if (r.selectorText) out.push(r.selectorText); if (r.cssRules) out.push(...rulesOf(r.cssRules)); } return out; };
  const visible = e => e.getClientRects().length > 0;

  /* The properties a forced state can change: every longhand set by a rule whose selector uses a state pseudo-class,
     every property of a rule that reads a custom property such a rule sets, and the box geometry (layout follows). */
  const GEOM = ['width', 'height', 'top', 'right', 'bottom', 'left', 'inline-size', 'block-size', 'transform-origin', 'perspective-origin',
    'margin-top', 'margin-right', 'margin-bottom', 'margin-left', 'padding-top', 'padding-right', 'padding-bottom', 'padding-left',
    'inset-block-start', 'inset-block-end', 'inset-inline-start', 'inset-inline-end', 'margin-block-start', 'margin-block-end', 'margin-inline-start', 'margin-inline-end',
    'padding-block-start', 'padding-block-end', 'padding-inline-start', 'padding-inline-end'];
  const styleRules = list => { const out = []; for (const r of list) { if (r.selectorText !== undefined) out.push(r); if (r.cssRules) out.push(...styleRules(r.cssRules)); } return out; };
  vb.forcedPropList = () => {
    const all = []; for (const sh of document.styleSheets) { try { all.push(...styleRules(sh.cssRules)); } catch (e) {} }
    const set = new Set(GEOM), vars = new Set();
    for (const r of all) { stRe.lastIndex = 0; if (!stRe.test(r.selectorText)) continue; stRe.lastIndex = 0;
      for (let i = 0; i < r.style.length; i++) { set.add(r.style[i]); if (r.style[i].startsWith('--')) vars.add(r.style[i]); }
      for (const m of r.style.cssText.matchAll(/(^|;)\s*([a-z-]+)\s*:/g)) set.add(m[2]); }
    for (const r of all) { const t = r.style.cssText; if (![...vars].some(v => new RegExp('var\\(\\s*' + v + '[\\s,)]').test(t))) continue;
      for (let i = 0; i < r.style.length; i++) set.add(r.style[i]); for (const m of t.matchAll(/(^|;)\s*([a-z-]+)\s*:/g)) set.add(m[2]); }
    const el = document.createElement('div'), lh = new Set();
    for (const p of set) { if (p.startsWith('--')) { lh.add(p); continue; } el.style.cssText = ''; el.style.setProperty(p, 'initial'); if (!el.style.length) lh.add(p); for (let i = 0; i < el.style.length; i++) lh.add(el.style[i]); }
    return [...lh].filter(p => vb.props.includes(p)).sort();
  };

  /* targets: [{ els: [key...], force: [[pseudos]...], region: key, sel }] */
  vb.plan = () => {
    vb.keys();
    const sels = new Set(); for (const sh of document.styleSheets) { try { rulesOf(sh.cssRules).forEach(t => splitTop(t, /,/).forEach(x => { x = x.trim(); stRe.lastIndex = 0; if (stRe.test(x)) sels.add(x); stRe.lastIndex = 0; })); } catch (e) {} }
    const seen = new Set(), targets = [], skipped = [];
    for (const sel of sels) {
      const comps = compounds(sel).map(strip);
      if (comps.some(c => !c)) { skipped.push(sel); continue; }
      const prefixes = comps.map((c, i) => comps.slice(0, i + 1).map(x => x.s).join(' '));
      let subs; try { subs = [...document.querySelectorAll(prefixes[prefixes.length - 1])]; } catch (e) { skipped.push(sel); continue; }
      if (!subs.length) continue;
      const s = subs.find(visible) || subs[0];
      const chosen = []; let ok = true;
      comps.forEach((c, i) => { if (!c.force.length) return;
        if (i === comps.length - 1) { chosen.push([s, c.force]); return; }
        let found = null; for (let a = s; a && !found; a = a.parentElement) { for (let x = a; x && !found; x = x.previousElementSibling) { try { if (x.matches(prefixes[i])) found = x; } catch (e) {} } }
        if (found) chosen.push([found, c.force]); else ok = false; });
      if (!ok || !chosen.length) continue;
      const top = chosen.map(c => c[0]).reduce((a, b) => a.contains(b) ? a : b.contains(a) ? b : (a.compareDocumentPosition(b) & 2 ? b : a));
      const t = { els: chosen.map(c => vb.map.get(c[0])), force: chosen.map(c => c[1].slice().sort()), region: vb.map.get(top), sib: /[+~]/.test(sel.replace(/\([^()]*\)|\[[^\]]*\]/g, '')), sel };
      const id = JSON.stringify([t.els, t.force]); if (seen.has(id)) continue; seen.add(id); targets.push(t);
    }
    /* batches of targets whose regions cannot touch: one style pass per batch, not per target */
    const tops = targets.map(t => vb.byKey(t.region)), batches = [];
    targets.forEach((t, i) => { const a = tops[i];
      let b = batches.find(b => b.length < 25 && b.every(j => { const c = tops[j]; return !a.contains(c) && !c.contains(a) && a.parentElement !== c.parentElement; }));
      if (!b) batches.push(b = []); b.push(i); });
    return { targets, skipped, batches: batches.map(b => b.map(i => targets[i])) };
  };

  /* selectors (one per selector-list entry, state pseudo-classes and pseudo-elements removed) that match something now */
  vb.used = () => {
    const out = [];
    for (const sh of document.styleSheets) { let rs; try { rs = rulesOf(sh.cssRules); } catch (e) { continue; }
      for (const t of rs) for (const x of splitTop(t, /,/)) { const sel = x.trim(); const cs = compounds(sel).map(strip); if (cs.some(c => !c)) { out.push(sel); continue; }
        try { if (document.querySelector(cs.map(c => c.s).join(' '))) out.push(sel); } catch (e) { out.push(sel); } } }
    return [...new Set(out)];
  };

  /* for analysis: every selector (state pseudo-classes and pseudo-elements removed) and the elements it matches now,
     as indexes into keys */
  vb.matches = () => {
    vb.keys(); const all = [...document.querySelectorAll('*')], idx = new Map(all.map((e, i) => [e, i])), out = {};
    for (const sh of document.styleSheets) { let rs; try { rs = rulesOf(sh.cssRules); } catch (e) { continue; }
      for (const t of rs) for (const x of splitTop(t, /,/)) { const sel = x.trim(); if (sel in out) continue; const cs = compounds(sel).map(c => strip(c) || { s: '*' });
        let q = sel.replace(/::?(before|after|placeholder|selection|marker|-webkit-[\w-]+)(?![\w-])/g, m => /^:(before|after)$/.test(m) || m.startsWith('::') ? '' : m);
        q = q.replace(/:not\(((?:\s*,?\s*:(?:hover|focus-visible|focus-within|focus|active))+)\s*\)/g, '').replace(/:(is|where)\(((?:\s*,?\s*:(?:hover|focus-visible|focus-within|focus|active))+)\s*\)/g, '');
        stRe.lastIndex = 0; q = q.replace(stRe, ''); stRe.lastIndex = 0;
        q = q.replace(/:has\(\s*\)/g, ':has(*)').replace(/([>+~])\s*\)/g, '$1 *)');
        q = q.trim().replace(/([>+~])\s*$/, '$1 *').replace(/^\s*([>+~])/, '* $1').replace(/([>+~])\s*([>+~])/g, '$1 * $2') || '*';
        let found; try { found = document.querySelectorAll(q); } catch (e) { try { found = document.querySelectorAll(cs.map(c => c.s).join(' ')); } catch (e2) { found = null; } }
        out[sel] = found ? [...found].map(e => idx.get(e)) : null; } }
    return { keys: all.map(e => vb.map.get(e)), sel: out };
  };

  /* for analysis: given named class patterns, which names were seen inside which (ancestor>descendant) */
  vb.nesting = map => {
    const res = map.map(([n, src]) => [n, new RegExp(src)]), memo = new Map(), out = new Set();
    const of = el => { const toks = [...el.classList].concat(el.id ? ['#' + el.id] : []), s = new Set();
      for (const t of toks) { if (!memo.has(t)) { const m = res.find(([, re]) => re.test(t)); memo.set(t, m ? m[0] : null); } const n = memo.get(t); if (n) s.add(n); } return s; };
    const walk = (el, anc) => { const mine = of(el); for (const a of anc) for (const d of mine) if (a !== d) out.add(a + '>' + d);
      const next = mine.size ? new Set([...anc, ...mine]) : anc; for (const c of el.children) walk(c, next); };
    walk(document.documentElement, new Set()); return [...out];
  };

  /* the distinct class lists on elements, for analysis */
  vb.classSets = () => { const set = new Set(); document.querySelectorAll('[class]').forEach(e => { const c = e.getAttribute('class').trim().split(/\s+/).filter(Boolean).sort().join(' '); if (c) set.add(e.localName + '|' + c); }); return [...set]; };

  vb.byKey = k => { for (const [el, key] of vb.map) if (key === k) return el; return null; };
  vb.stash = keys => { vb.keys(); vb.stashed = keys.map(vb.byKey); return vb.stashed.map(Boolean); };
  /* what forcing a state on these elements can restyle: the topmost one's subtree, its following siblings (with their
     subtrees when the selector has a sibling combinator), and every forced element's ancestors (for :has) */
  vb.recordRegion = list => {
    vb.finish(); vb.relayout();
    const set = new Set();
    for (const [regionKey, elKeys, sib] of list) {
      const reg = vb.byKey(regionKey); if (!reg) return null;
      set.add(reg); reg.querySelectorAll('*').forEach(y => set.add(y));
      for (let x = reg.nextElementSibling; x; x = x.nextElementSibling) { set.add(x); if (sib) x.querySelectorAll('*').forEach(y => set.add(y)); }
      for (const k of elKeys) for (let a = vb.byKey(k); a; a = a.parentElement) set.add(a);
    }
    return vb.collect([...set], vb.fprops);
  };
  /* ---- what computed styles do not show (ui-next): each element's role, accessible name, ARIA states, tooltip, own
     words and attributes, as one digest per element, compared like a property. The role is the explicit one or the
     element's implicit one; the name follows the accessible-name order for what the pages use (aria-labelledby,
     aria-label, the text of a button, link, option or heading, an input's label, then the title). Own words are the
     element's own text nodes, spaces collapsed (whitespace between elements is no difference). Attributes are every
     aria-*, data-*, title, tabindex, hidden, disabled, type, placeholder, role, for, href, alt, value and checked; the
     inline style and the class are the computed styles' business. */
  const IMPLICIT = { button: 'button', a: 'link', input: 'textbox', textarea: 'textbox', select: 'combobox', nav: 'navigation', main: 'main', header: 'banner', aside: 'complementary', h1: 'heading', h2: 'heading', h3: 'heading', h4: 'heading', ul: 'list', li: 'listitem', output: 'status', section: 'region', article: 'article', img: 'img', svg: 'graphics-document' };
  const NAMED_BY_TEXT = /^(button|link|option|heading|radio|tab|menuitem|switch|checkbox)$/;
  const squash = s => s.replace(/\s+/g, ' ').trim();
  const roleOf = el => el.getAttribute('role') || (el.localName === 'input' && /^(checkbox|radio|range)$/.test(el.type) ? ({ checkbox: 'checkbox', radio: 'radio', range: 'slider' })[el.type] : IMPLICIT[el.localName] || '');
  const nameOf = el => {
    const by = el.getAttribute('aria-labelledby'); if (by) return squash(by.split(/\s+/).map(id => document.getElementById(id)?.textContent || '').join(' '));
    const al = el.getAttribute('aria-label'); if (al !== null) return squash(al);
    if (NAMED_BY_TEXT.test(roleOf(el))) { const t = squash(el.textContent || ''); if (t) return t; }
    if (el.id) { const l = document.querySelector(`label[for="${CSS.escape(el.id)}"]`); if (l) return squash(l.textContent); }
    return squash(el.getAttribute('title') || '');
  };
  const ATTR = /^(aria-.*|data-.*|title|tabindex|hidden|disabled|type|placeholder|role|for|href|alt|value|checked|open)$/;
  vb.semOf = el => {
    const own = squash([...el.childNodes].filter(n => n.nodeType === 3).map(n => n.textContent).join(''));
    const attrs = [...el.attributes].filter(a => ATTR.test(a.name)).map(a => a.name + '=' + a.value).sort();
    const live = (el.localName === 'input' || el.localName === 'textarea') ? ['value:' + el.value] : [];
    return [roleOf(el), nameOf(el), own, ...attrs, ...live].join('␟');
  };
  vb.semantics = () => { vb.keys(); const out = []; for (const el of document.querySelectorAll('body *')) { const k = vb.map.get(el); if (k) out.push([k, vb.semOf(el)]); } return out; };

  /* ---- a part recorded on its own (ui-next): the first element matching a selector and everything under it, keyed from
     the selector, so the same control compares wherever each page puts it */
  vb.recordPart = sel => {
    const root = document.querySelector(sel); if (!root) return null;
    vb.finish(); vb.relayout();
    const map = new Map(), walk = (el, key) => { map.set(el, key); let i = 0; for (const c of el.children) { walk(c, key + '>' + c.localName + ':' + i); i++; } };
    walk(root, sel);
    const els = [...map.keys()], saved = vb.map; vb.map = map;
    const recs = vb.collect(els); vb.map = saved;
    return { recs, sem: els.map(el => [map.get(el), vb.semOf(el)]) };
  };
})();
