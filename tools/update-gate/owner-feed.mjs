// Owner fixture: redirect only this pair's release feed and assets.
// No fault injection, process interception, clock changes or payload edits.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Readable } from 'node:stream';
const root = path.dirname(fileURLToPath(import.meta.url));
const cfg = JSON.parse(fs.readFileSync(path.join(root, 'feed.json'), 'utf8'));
const nativeFetch = globalThis.fetch;
globalThis.fetch = async (input, init) => {
  const url = String(input?.url || input);
  const api = 'https://api.github.com/repos/scragnog/HOT-Step-CPP';
  if (url === `${api}/releases/latest`) return Response.json(cfg.release);
  if (url === `${api}/commits/${cfg.release.tag_name}`) return Response.json({ sha: cfg.sha });
  const asset = cfg.release.assets.find(a => a.browser_download_url === url);
  if (asset) return new Response(Readable.toWeb(fs.createReadStream(path.join(root, 'assets', asset.name))), { headers: { 'content-length': String(asset.size) } });
  return nativeFetch(input, init);
};
