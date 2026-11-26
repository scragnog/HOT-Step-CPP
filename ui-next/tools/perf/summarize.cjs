// Turns the measurements into tables.
//   node summarize.cjs --dir results/<label>                 writes results/<label>/summary.md (and prints it)
//   node summarize.cjs --compare results/<a> results/<b>     one table of a against b (before and after)
//   node summarize.cjs results/x/perf.json [...]             the per-action table of perf files
// Target: interaction to next paint <= 100 ms; a drag holds 60 fps (no frame gap over 33 ms) with no long task;
// usable within 1 s of opening.
const fs = require('fs'), path = require('path');
const med = a => { const s = a.filter(x => x != null).sort((x, y) => x - y); return s.length ? s[Math.floor((s.length - 1) / 2)] : null; };
const rng = a => { const s = a.filter(x => x != null); return s.length ? `${Math.min(...s)}-${Math.max(...s)}` : '-'; };
const read = f => { try { return JSON.parse(fs.readFileSync(f, 'utf8')); } catch (e) { return null; } };
const ORDER = ['empty', 'typical', 'full'];
const isDrag = a => /drag/.test(a);

/** Per action and desk: the medians over the runs. */
function rows(R) {
  const out = new Map();
  for (const a of [...new Set(R.runs.map(r => r.action))]) for (const d of ORDER) {
    const all = R.runs.filter(r => r.action === a && r.desk === d), rs = all.filter(r => !r.skip && !r.error);
    if (!all.length) continue;
    if (!rs.length) { out.set(`${a} ${d}`, { a, d, none: all[0].skip ? 'n/a (' + all[0].skip + ')' : 'error: ' + (all[0].error || '').slice(0, 60) }); continue; }
    /* INP: Event Timing's figure; below 16 ms it reports nothing, and the next-frame figure stands in */
    const inp = rs.map(r => r.inp ?? r.nf);
    const drag = isDrag(a) && rs.some(r => r.fps !== undefined);
    const o = { a, d, inp: med(inp), inpR: rng(inp), task: med(rs.map(r => r.task)), lt: med(rs.map(r => r.lt)), ltMax: med(rs.map(r => r.ltMax)), layouts: med(rs.map(r => r.layouts)) };
    if (drag) Object.assign(o, { fps: med(rs.map(r => r.fps)), gap: med(rs.map(r => r.gapMax)), over33: med(rs.map(r => r.over33)) });
    o.miss = [o.inp > 100 ? 'INP' : '', drag && (o.over33 > 0 || o.lt > 0) ? 'drag' : ''].filter(Boolean);
    out.set(`${a} ${d}`, o);
  }
  return out;
}
const cell = o => (!o ? '' : o.none ? o.none : `${o.inp} (${o.inpR})`);
const dragCell = o => (o && o.fps !== undefined ? `${o.fps} fps, gap ${o.gap}, LT ${o.lt}/${o.ltMax}` : o && !o.none ? `LT ${o.lt}/${o.ltMax}` : '');

function perfTable(R, title) {
  const L = [`### ${title} (throttle ${R.throttle || 1}x, ${R.priority}, runs ${new Set(R.runs.map(r => r.run)).size})`, '',
    '| Action | Desk | INP ms, median (range) | Main thread ms | Layouts | Drag / long tasks (count/max ms) | Target |', '| --- | --- | --- | --- | --- | --- | --- |'];
  for (const o of rows(R).values()) L.push(o.none ? `| ${o.a} | ${o.d} | ${o.none} | | | | |` : `| ${o.a} | ${o.d} | ${cell(o)} | ${o.task} | ${o.layouts} | ${dragCell(o)} | ${o.miss.length ? '**miss: ' + o.miss.join(', ') + '**' : 'ok'} |`);
  return L.join('\n');
}

function loadTable(L, title) {
  if (!L) return '';
  const out = [`### ${title} (throttle ${L.throttle || 1}x)`, '', '| Desk | Usable ms, median (range) | FCP | Main thread ms (script, style, layout) | Long tasks |', '| --- | --- | --- | --- | --- |'];
  for (const d of ['fresh', ...ORDER]) {
    const rs = L.runs.filter(r => r.desk === d); if (!rs.length) continue;
    out.push(`| ${d} | ${med(rs.map(r => r.usable))} (${rng(rs.map(r => r.usable))}) | ${med(rs.map(r => r.fcp))} | ${med(rs.map(r => r.task))} (${med(rs.map(r => r.script))}, ${med(rs.map(r => r.style))}, ${med(rs.map(r => r.layout))}) | ${med(rs.map(r => r.longTasks.length))} |`);
  }
  return out.join('\n');
}
function idleTable(I) {
  if (!I) return '';
  const out = ['### Idle and during a run', '', '| Desk | Idle: main thread ms/s, renderer CPU % | After a click | During a run |', '| --- | --- | --- | --- |'];
  for (const [d, v] of Object.entries(I)) { const f = x => (x ? `${x.mainThreadMsPerS} ms/s, ${x.rendererCpuPct}%` : ''); out.push(`| ${d} | ${f(v.idle)} | ${f(v.idleAfterClick)} | ${f(v.run)} |`); }
  return out.join('\n');
}
function memTable(M, title) {
  if (!M) return '';
  const r = M.rounds, a = r[0], b = r[Math.min(2, r.length - 1)], z = r[r.length - 1];
  return `### ${title}\n\nheap after load ${a.heapMB} MB, after round 2 ${b.heapMB} MB, after round ${r.length - 1} ${z.heapMB} MB; nodes ${a.nodes} / ${b.nodes} / ${z.nodes}; listeners ${a.listeners} / ${b.listeners} / ${z.listeners}`;
}
function motionTable(M) {
  if (!M) return '';
  const out = ['### Motions (tools/motion/cost.mjs: main thread over 700 ms, software compositing)', '', '| Motion | Main thread ms | Style | Layout | Dropped frames | Longest frame ms |', '| --- | --- | --- | --- | --- | --- |'];
  for (const [k, v] of Object.entries(M)) out.push(`| ${k} | ${v.task} | ${v.style} | ${v.layout} | ${v.dropped} | ${v.longest} |`);
  return out.join('\n');
}

function summary(dir) {
  const P = read(path.join(dir, 'perf.json')), T = read(path.join(dir, 'perf-throttle4.json'));
  const parts = [`# Measurements: ${path.basename(dir)}`];
  if (P) parts.push(perfTable(P, 'Every action'));
  if (T) parts.push(perfTable(T, 'Every action on the full desk, 4x slower CPU'));
  parts.push(loadTable(read(path.join(dir, 'load.json')), 'Load'), loadTable(read(path.join(dir, 'load-throttle4.json')), 'Load, 4x slower CPU'));
  parts.push(idleTable(read(path.join(dir, 'idle.json'))), memTable(read(path.join(dir, 'memory.json')), 'Memory, mixed session with Generate'), memTable(read(path.join(dir, 'memory-nogen.json')), 'Memory, mixed session without Generate'));
  parts.push(motionTable(read(path.join(dir, 'motion.json'))));
  const ml = (() => { try { const xs = fs.readFileSync(path.join(dir, 'machine-load.jsonl'), 'utf8').trim().split('\n').map(l => JSON.parse(l).busy); return xs.length ? `Machine load while measuring: median ${med(xs)} % (${rng(xs)})` : ''; } catch (e) { return ''; } })();
  parts.push(ml);
  return parts.filter(Boolean).join('\n\n') + '\n';
}

function compare(a, b) {
  const L = [`# ${path.basename(a)} against ${path.basename(b)}`, ''];
  const pa = read(path.join(a, 'perf.json')), pb = read(path.join(b, 'perf.json')), ta = read(path.join(a, 'perf-throttle4.json')), tb = read(path.join(b, 'perf-throttle4.json'));
  if (pa && pb) {
    const ra = rows(pa), rb = rows(pb), tra = ta ? rows(ta) : new Map(), trb = tb ? rows(tb) : new Map();
    L.push('| Action | Desk | INP ms before | INP ms after | Main thread ms before / after | Drag / long tasks after | 4x CPU INP before / after | Target after (1x / 4x) |', '| --- | --- | --- | --- | --- | --- | --- | --- |');
    for (const [k, o] of rb) {
      const p = ra.get(k), t0 = tra.get(k), t1 = trb.get(k);
      const tgt = o.none ? '' : `${o.miss.length ? '**' + o.miss.join(', ') + '**' : 'ok'} / ${t1 && !t1.none ? (t1.miss.length ? '**' + t1.miss.join(', ') + '**' : 'ok') : '-'}`;
      L.push(`| ${o.a} | ${o.d} | ${cell(p)} | ${cell(o)} | ${p && !p.none ? p.task : ''} / ${o.none ? '' : o.task} | ${dragCell(o)} | ${t0 && !t0.none ? t0.inp : '-'} / ${t1 && !t1.none ? t1.inp : '-'} | ${tgt} |`);
    }
  }
  for (const [f, t] of [['load.json', 'Load'], ['load-throttle4.json', 'Load, 4x slower CPU']]) { const x = read(path.join(a, f)), y = read(path.join(b, f)); if (x || y) L.push('', loadTable(x, `${t}, ${path.basename(a)}`), '', loadTable(y, `${t}, ${path.basename(b)}`)); }
  L.push('', idleTable(read(path.join(a, 'idle.json'))).replace('### Idle', `### ${path.basename(a)}: idle`), '', idleTable(read(path.join(b, 'idle.json'))).replace('### Idle', `### ${path.basename(b)}: idle`));
  L.push('', memTable(read(path.join(a, 'memory-nogen.json')), `${path.basename(a)}: memory without Generate`), '', memTable(read(path.join(b, 'memory-nogen.json')), `${path.basename(b)}: memory without Generate`));
  const ma = read(path.join(a, 'motion.json')), mb = read(path.join(b, 'motion.json'));
  if (ma && mb) {
    L.push('', '### Motions (main thread ms over 700 ms / dropped frames / longest frame ms)', '', '| Motion | before | after |', '| --- | --- | --- |');
    for (const k of Object.keys(mb)) { const x = ma[k], y = mb[k], f = v => (v ? `${v.task} / ${v.dropped} / ${v.longest}` : '-'); L.push(`| ${k} | ${f(x)} | ${f(y)} |`); }
  }
  return L.join('\n') + '\n';
}

const argv = process.argv.slice(2);
if (argv[0] === '--dir') { const s = summary(path.resolve(argv[1])); fs.writeFileSync(path.join(path.resolve(argv[1]), 'summary.md'), s); console.log(s); }
else if (argv[0] === '--compare') { const s = compare(path.resolve(argv[1]), path.resolve(argv[2])); if (argv[3]) fs.writeFileSync(argv[3], s); console.log(s); }
else for (const f of argv) console.log(perfTable(read(f), f) + '\n');
