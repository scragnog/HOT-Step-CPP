// Shared helpers for the page's performance measurements (tools/perf/README.md).
//   PERF_BUILD=<folder holding dist-file/ and dist-test/>   the build measured (default: ui-next itself, after
//                                                           `npm run build:file` and `npm run build:test`)
//   PLAYWRIGHT_MODULE=<node_modules/playwright>              Playwright, when it is not installed here
//   PERF_SOFT=1                                              software compositing instead of the GPU
const fs = require('fs'), path = require('path');
const loadPW = () => { for (const m of [process.env.PLAYWRIGHT_MODULE, 'playwright'].filter(Boolean)) { try { return require(m); } catch (e) {} } throw new Error('Playwright not found: run npm install, or set PLAYWRIGHT_MODULE to a playwright folder'); };
const { chromium } = loadPW();
const HERE = __dirname, UI = path.resolve(HERE, '..', '..');
const BUILD = process.env.PERF_BUILD ? path.resolve(process.env.PERF_BUILD) : UI;
const url = which => 'file:///' + path.join(BUILD, which, 'index.html').split(path.sep).join('/');
const PAGES = { file: url('dist-file'), test: url('dist-test') };
/* the saved desks the measurements open (make-desks.cjs writes them) and where results go */
const DESKS = path.join(HERE, 'desks'), RESULTS = path.join(HERE, 'results');
const ROOT = HERE;
module.exports = { chromium, ROOT, UI, BUILD, PAGES, DESKS, RESULTS, url };
/** A fresh context whose localStorage holds a saved desk (empty, typical, full) before the page's scripts run. */
async function openWith(browser, desk, page = 'file', opts = {}) {
  const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 }, ...opts });
  if (desk) {
    const ls = JSON.parse(fs.readFileSync(path.join(DESKS, desk + '.json'), 'utf8'));
    await ctx.addInitScript(ls => { if (!sessionStorage.getItem('__seeded')) { for (const [k, v] of Object.entries(ls)) localStorage.setItem(k, v); sessionStorage.setItem('__seeded', '1'); } }, ls);
  }
  const p = await ctx.newPage();
  return { ctx, page: p, url: PAGES[page] };
}
module.exports.openWith = openWith;
/** Chromium as the owner's browser runs it: on the GPU (headless otherwise composites and rasterises in software,
 *  SwiftShader, which inflates every paint-heavy figure). PERF_SOFT=1 measures software compositing instead. */
const GPU_ARGS = ['--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist'];
const launch = () => chromium.launch(process.env.PERF_SOFT ? {} : { args: GPU_ARGS });
module.exports.launch = launch;
