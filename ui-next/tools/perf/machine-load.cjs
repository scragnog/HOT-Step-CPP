// Samples the whole machine's CPU use every 2 s while the measurements run (other work on the machine changes the
// numbers): node machine-load.cjs <out.jsonl> [seconds]
const os = require('os'), fs = require('fs');
const out = process.argv[2], secs = +(process.argv[3] || 3600);
const snap = () => os.cpus().reduce((a, c) => { const t = Object.values(c.times).reduce((x, y) => x + y, 0); return { idle: a.idle + c.times.idle, total: a.total + t }; }, { idle: 0, total: 0 });
let prev = snap(); const end = Date.now() + secs * 1000;
const iv = setInterval(() => { const s = snap(), busy = 1 - (s.idle - prev.idle) / (s.total - prev.total); prev = s; fs.appendFileSync(out, JSON.stringify({ t: new Date().toISOString(), busy: Math.round(busy * 100) }) + '\n'); if (Date.now() > end) clearInterval(iv); }, 2000);
