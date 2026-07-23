// modelName.ts — model-name prettifying + title suffixing ("Song Name - Fable 5")
//
// CANONICAL single source, shared with tools/mcp-lyricstudio (which imports it
// directly — it runs from TS source via tsx). The suffix append is deterministic
// code, NOT part of any LLM prompt, so titles stay clean for validation and the
// suffix can't be hallucinated.

/** Turn a model id like "claude-fable-5" / "claude-opus-4-8" into "Fable 5" / "Opus 4.8",
 *  or an OpenAI-compatible / LM Studio id like
 *  "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF" into "Meta Llama 3.1 8B Instruct".
 *  Friendly names ("Fable 5", "Gemini 3 Pro") pass through unchanged. */
export function prettifyModel(raw: string): string {
  let s = raw.trim();
  if (!s) return s;
  if (/\s/.test(s) && !/[-_/]/.test(s)) return s;    // already a friendly name
  const slash = s.lastIndexOf('/');                  // org/repo paths → repo part
  if (slash >= 0) s = s.slice(slash + 1);
  s = s.replace(/^(us\.)?(anthropic[./])?(claude-)?/i, '');
  s = s.replace(/\.gguf$/i, '');                     // file-style ids
  s = s.replace(/[-_.]gguf$/i, '');                  // "...-GGUF" repo suffix
  s = s.replace(/[-_.](i?q\d+(?:_[a-z0-9]+)*|f16|f32|bf16|fp16)$/i, ''); // quant tag e.g. -Q4_K_M
  s = s.replace(/[-.]?\d{8}$/, '');                  // date suffix e.g. -20251001
  s = s.replace(/[-.]?v\d+(?:[.:]\d+)*$/i, '');      // version suffix: v1, v0.3, v1:0
  const parts = s.split(/[-_]/).filter(Boolean);
  const out: string[] = [];
  for (const part of parts) {
    // Join consecutive single-digit segments as a version: opus 4 8 → opus 4.8
    if (/^\d$/.test(part) && out.length && /^\d+(\.\d+)*$/.test(out[out.length - 1])) {
      out[out.length - 1] += `.${part}`;
    } else {
      out.push(/^\d/.test(part) ? part : part.charAt(0).toUpperCase() + part.slice(1));
    }
  }
  // Community model names can be absurdly long
  // ("Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF");
  // the first few tokens identify the model, the rest is noise in a title suffix.
  return out.slice(0, 4).join(' ') || raw.trim();
}

/** Append " - <Model>" to a title unless it already carries that suffix (or is empty). */
export function withModelSuffix(title: string, model?: string): string {
  const t = title.trim();
  if (!t || !model) return t;
  const pretty = prettifyModel(model);
  if (!pretty || t.toLowerCase().endsWith(`- ${pretty.toLowerCase()}`)) return t;
  return `${t} - ${pretty}`;
}
