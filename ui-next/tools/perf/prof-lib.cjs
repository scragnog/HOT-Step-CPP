// The source-mapped CPU profile digest shared by profile.cjs and load-profile.cjs (instrumented build only).
const fs = require('fs'), path = require('path');
const { UI, BUILD } = require('./lib.cjs');
const { SourceMapConsumer } = require(path.join(UI, 'node_modules', 'source-map-js'));
const TOP = 25;
/* the page's script is inline: positions in the profile are the HTML document's; the map is the script's */
const html = fs.readFileSync(path.join(BUILD, 'dist-file', 'index.html'), 'utf8').split('\n');
const L0 = html.findIndex(l => l.startsWith('<script type="module">')), C0 = '<script type="module">'.length;
const mapFile = fs.readdirSync(path.join(BUILD, 'dist-file', 'assets')).find(f => f.endsWith('.map'));
const smc = new SourceMapConsumer(JSON.parse(fs.readFileSync(path.join(BUILD, 'dist-file', 'assets', mapFile), 'utf8')));
const where = cf => {
  if (cf.lineNumber < L0 || !cf.url) return cf.url ? '(html)' : '(native)';
  const p = smc.originalPositionFor({ line: cf.lineNumber - L0 + 1, column: cf.lineNumber === L0 ? cf.columnNumber - C0 : cf.columnNumber });
  return p.source ? `${p.source.replace(/^.*?(src\/|node_modules\/)/, '$1')}:${p.line}` : '?';
};

function digest(profile) {
  const byId = new Map(profile.nodes.map(n => [n.id, n])), parent = new Map();
  for (const n of profile.nodes) for (const c of n.children || []) parent.set(c, n.id);
  const self = new Map(), total = new Map(), file = new Map(), natives = new Map();
  const label = n => { const cf = n.callFrame, w = where(cf); return `${cf.functionName || '(anon)'} ${w}`; };
  const labels = new Map(profile.nodes.map(n => [n.id, label(n)]));
  const dts = profile.timeDeltas;
  for (let i = 0; i < profile.samples.length; i++) {
    const dt = (dts[i + 1] ?? dts[i]) / 1000; let id = profile.samples[i];
    const l = labels.get(id); self.set(l, (self.get(l) || 0) + dt);
    /* time in a native call (a forced layout, a style read) is put on the script that made it */
    if (/\(native\)$/.test(l) && !/^\((idle|program|garbage)/.test(l)) { let p = parent.get(id); while (p !== undefined && /\(native\)$/.test(labels.get(p))) p = parent.get(p); const k = `${l.split(' ')[0]} < ${p !== undefined ? labels.get(p) : '?'}`; natives.set(k, (natives.get(k) || 0) + dt); }
    if (!/^\((idle|program)\)/.test(l)) { const f = l.split(' ').pop().replace(/:\d+$/, ''); file.set(f, (file.get(f) || 0) + dt); }
    const onStack = new Set();
    while (id !== undefined) { const x = labels.get(id); if (!onStack.has(x)) { onStack.add(x); total.set(x, (total.get(x) || 0) + dt); } id = parent.get(id); }
  }
  const top = (m, k) => [...m].filter(([l]) => !/^\((idle|program|root)\)/.test(l)).sort((a, b) => b[1] - a[1]).slice(0, k).map(([l, ms]) => [l, Math.round(ms * 10) / 10]);
  const busy = [...self].filter(([l]) => !/^\(idle\)/.test(l)).reduce((s, [, v]) => s + v, 0);
  return { busy: Math.round(busy), self: top(self, TOP), total: top(total, TOP * 2), files: top(file, TOP), natives: top(natives, TOP) };
}

/* a listener's tag is the stack line where it subscribed (index.html:line:col): mapped to the source */
const mapTag = t => t.replace(/file:\/\/\/\S*?index\.html:(\d+):(\d+)/g, (_, l, c) => { const p = smc.originalPositionFor({ line: +l - L0, column: +l - 1 === L0 ? +c - 1 - C0 : +c - 1 }); return p.source ? `${p.source.replace(/^.*?(src\/|node_modules\/)/, '$1')}:${p.line}` : `html:${l}`; });

module.exports = { digest, mapTag, where };
