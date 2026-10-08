// contracts/adapters.ts — wire shapes for /api/adapters (server/src/routes/adapters.ts).

export interface BrowseEntry {
  name: string;
  path: string;
  type: 'dir' | 'file';
  size?: number;
}

export interface BrowseResponse { current: string; entries: BrowseEntry[] }
/** A 404/500 browse error still carries current/entries so a caller that
 *  doesn't check the status can fall back to an empty listing. */
export interface BrowseErrorResponse extends BrowseResponse { error: string }

export interface AdapterFile {
  name: string;
  path: string;
  size: number;
  /** Trigger word embedded in the adapter's safetensors metadata ('' = none). */
  trigger?: string;
  /** Where that trigger sat in the training captions. 'replace' means the
   *  trigger WAS the whole caption, so inference must drop the caption too. */
  triggerPosition?: 'prepend' | 'append' | 'replace' | '';
}

export interface ScanResponse { files: AdapterFile[] }

export interface LmAdapterEntry {
  name: string;
  path: string;
  kind: 'peft' | 'lokr' | 'safetensors';
  size: number;
  mtime: number;
  /** '0.6B' | '1.7B' | '4B' from the lm-<size> parent folder, else the
   *  legacy -<size> name suffix, else ''. */
  lmSize: string;
  /** Training-run stamp (YYYY-MM-DD_HH-MM-SS subfolder); '' for an
   *  unversioned/legacy adapter. */
  run: string;
  trigger: string;
  triggerPosition: 'prepend' | 'append' | 'replace' | '';
  /** hot_step_eval.json sidecar: marginal+transition JS distance to the
   *  artist's ground truth — LOWER = closer. null = never evaluated. */
  evalScore: number | null;
  /** 'toward' | 'away' | 'inconclusive' | '' */
  evalVerdict: string;
}

export interface LmAdaptersResponse {
  root: string;
  adapters: LmAdapterEntry[];
  /** Set instead of a non-OK status — the route always returns 200 so the
   *  picker can still show whatever root it resolved. */
  error?: string;
}
