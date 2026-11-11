import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const local = process.env.LOCALAPPDATA;
function cachedPlaywright() {
  const cache = path.join(local, 'npm-cache', '_npx');
  for (const entry of fs.readdirSync(cache)) {
    const candidate = path.join(cache, entry, 'node_modules', 'playwright');
    if (fs.existsSync(path.join(candidate, 'package.json'))) return candidate;
  }
  throw new Error('Set UPDATE_GATE_PLAYWRIGHT to an installed Playwright package');
}
function cachedChromium() {
  const cache = path.join(local, 'ms-playwright');
  const versions = fs.readdirSync(cache).filter(n => /^chromium-\d+$/.test(n)).sort((a, b) => Number(b.split('-')[1]) - Number(a.split('-')[1]));
  for (const version of versions) {
    const exe = path.join(cache, version, 'chrome-win64', 'chrome.exe');
    if (fs.existsSync(exe)) return exe;
  }
  throw new Error('Set UPDATE_GATE_CHROMIUM to an installed Chromium executable');
}
export const { chromium } = require(process.env.UPDATE_GATE_PLAYWRIGHT || cachedPlaywright());
export const browserOptions = { headless: true, executablePath: process.env.UPDATE_GATE_CHROMIUM || cachedChromium() };
