import { isPresent } from '@/utils/guards';
// reportIssue.ts — redaction and the prefilled GitHub issue URL for the
// Report an issue tab. Pure, so the redaction and the length cap are tested.

export const ISSUE_URL = 'https://github.com/scragnog/HOT-Step-CPP/issues/new';
export const ISSUE_TEMPLATE = 'in-app-report.md';
/** Browsers and GitHub choke on very long URLs; logs are trimmed to stay under this. */
export const MAX_URL_LENGTH = 7500;

export interface RedactNames { home?: string; user?: string }

const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/** Masks home directory, user name, IPs, emails and token-like strings in log text. */
export function redact(text: string, { home, user }: RedactNames = {}): string {
  let out = text;
  if (home && home.length > 3) {
    const parts = home.replace(/^[A-Za-z]:/, '').split(/[\\/]+/).filter(isPresent).map(esc);
    out = out.replace(new RegExp(`(?:[A-Za-z]:)?[\\\\/]+${parts.join('[\\\\/]+')}`, 'gi'), '~');
  }
  // Any other profile path (another drive, a log from a different account).
  out = out.replace(/((?:[A-Za-z]:)?[\\/](?:Users|home)[\\/])[^\\/\s"':]+/gi, '$1<user>');
  if (user && user.length >= 2) out = out.replace(new RegExp(`\\b${esc(user)}\\b`, 'gi'), '<user>');
  out = out
    .replace(/\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{8,}/gi, '$1 <redacted>')
    // Prefixed keys too (WORKER_TOKEN, HF_TOKEN). Only a numeric count under a plural "tokens" key (max_tokens=512) stays.
    .replace(/([\w-]*(?:api[_-]?key|token|secret|password|passwd|authorization|credential)[\w-]*["']?\s*[:=]\s*["']?)([^\s"',;]+)/gi,
      (m, k, v) => (/tokens["']?\s*[:=]\s*["']?$/i.test(k) && /^\d+$/.test(v) ? m : `${k}<redacted>`))
    .replace(/\b(?:sk|pk|ghp|gho|ghs|github_pat|hf|xox[abp])[-_][A-Za-z0-9_-]{16,}/g, '<redacted>')
    .replace(/\b[A-Za-z0-9_-]{40,}\b/g, m => (/\d/.test(m) && /[A-Za-z]/.test(m) ? '<redacted>' : m))
    .replace(/[\w.+-]+@[\w-]+(?:\.[\w-]+)+/g, '<email>')
    .replace(/\b\d{1,3}(?:\.\d{1,3}){3}\b/g, '<ip>')
    .replace(/(?<![\w:])::1(?![\w:])/g, '<ip>')
    .replace(/(?<![\w:])(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}(?![\w:])/gi, m => ((m.includes('::') || m.split(':').length >= 6) && /[0-9a-f]{2,}/i.test(m) ? '<ip>' : m));
  return out;
}

export interface LogSection { name: string; lines: string[] }

export interface ReportParts {
  title: string;
  description: string;
  system: string;
  sections: LogSection[];
}

const fence = (name: string, lines: string[]) =>
  `<details><summary>${name}</summary>\n\n\`\`\`\n${lines.join('\n')}\n\`\`\`\n</details>`;

export function renderBody({ description, system, sections }: Omit<ReportParts, 'title'>): string {
  const logs = sections.filter(s => s.lines.length).map(s => fence(s.name, s.lines));
  return [
    description.trim() && `## Description\n\n${description.trim()}`,
    `## System\n\n\`\`\`\n${system}\n\`\`\``,
    logs.length && `## Logs\n\n${logs.join('\n\n')}`,
  ].filter(isPresent).join('\n\n');
}

export function buildUrl(title: string, body: string): string {
  const q = new URLSearchParams({ template: ISSUE_TEMPLATE, title, body });
  return `${ISSUE_URL}?${q.toString().replace(/\+/g, '%20')}`;
}

/** Builds the issue URL under `max` characters, always. Cuts in this order: the
 *  oldest log lines (from the longest section), the description, the system
 *  info, the title. Returns the exact title and body the URL carries. */
export function buildReport(parts: ReportParts, max = MAX_URL_LENGTH): { url: string; title: string; body: string; trimmed: boolean } {
  const sections = parts.sections.map(s => ({ ...s, lines: [...s.lines] }));
  let { title, description, system } = parts;
  const make = () => {
    const body = renderBody({ description, system, sections });
    return { title, body, url: buildUrl(title, body) };
  };
  let built = make();
  let trimmed = false;
  let cutDescription = 0;
  while (built.url.length > max) {
    const longest = sections.reduce((a, b) => (b.lines.length > a.lines.length ? b : a), sections[0]);
    if (longest?.lines.length) {
      // Drop half the longest section's oldest lines per pass; the URL is re-encoded each time.
      longest.lines.splice(0, Math.max(1, Math.ceil(longest.lines.length / 2)));
    } else if (description.trim()) {
      description = description.slice(0, Math.floor(description.length * 0.8));
      cutDescription = 1;
    } else if (system) {
      system = system.slice(0, Math.floor(system.length * 0.8));
    } else if (title) {
      title = title.slice(0, Math.floor(title.length * 0.8));
    } else break;
    trimmed = true;
    built = make();
  }
  if (cutDescription && description.trim()) {
    const note = '\n[description cut to fit the link; use Copy full logs and paste the rest]';
    description += note;
    const marked = make();
    if (marked.url.length <= max) built = marked; else description = description.slice(0, -note.length);
  }
  return { ...built, trimmed };
}

/** Model lines for the System section, following the active engine. ACE-Step
 *  models are per-request settings; every other engine reports what it holds
 *  loaded through its model catalogue (`defaults`), so no file names live here.
 *  A model setting can be a file path; the report keeps the file name only. */
export function modelLines(
  engine: string,
  ace: { ditModel?: unknown; lmModel?: unknown; vaeModel?: unknown },
  catalogue: { buckets?: Record<string, unknown>; defaults?: Record<string, unknown>; meta?: Record<string, Record<string, { label?: string }>> } | undefined,
  unknown: string,
): Array<[string, string]> {
  const name = (v: unknown) => String(v || '').split(/[\\/]/).pop() || '';
  if (engine === 'ace') {
    return [['DiT model', name(ace.ditModel) || unknown], ['LM model', name(ace.lmModel) || unknown], ['VAE model', name(ace.vaeModel) || unknown]];
  }
  const d = catalogue?.defaults ?? {};
  const lines: Array<[string, string]> = Object.keys(catalogue?.buckets ?? {}).map(b => [`${b} model`, name(catalogue?.meta?.[b]?.[String(d[b])]?.label) || name(d[b]) || unknown]);
  // YuE2 adapters: a half with no pick is simply not listed.
  for (const [label, key] of [['AR LoRA', 'lmAdapterAr'], ['NAR LoRA', 'lmAdapterNar']] as const) {
    if (name(d[key])) lines.push([label, name(d[key])]);
  }
  return lines;
}
