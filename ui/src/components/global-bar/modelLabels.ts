// modelLabels.ts — Human-friendly labels for model filenames
//
// Maps raw .gguf filenames from the models directory to short, readable labels.

/** Parse a DiT model filename into a nice label like "Base/Turbo XL-BF16" */
export function formatDitModel(filename: string): string {
  if (!filename) return '—';
  const name = filename.replace(/\.gguf$/i, '');

  // Extract quant suffix (last segment after final dash, e.g. BF16, Q8_0, Q4_K_M)
  const quantMatch = name.match(/-(BF16|MXFP4|NVFP4|Q\d+_?\w*|IQ\d+_?\w*)$/);
  const quant = quantMatch ? quantMatch[1] : '';
  const base = quant ? name.slice(0, -(quant.length + 1)) : name;

  // Map known base patterns to friendly names
  const labelMap: [RegExp, string][] = [
    [/^acestep-v15-merge-base-turbo-xl-ta-[\d.]+$/, 'Merge Base/Turbo XL'],
    [/^acestep-v15-merge-sft-turbo-xl-ta-([\d.]+)$/, 'Merge SFT/Turbo XL $1'],
    [/^acestep-v15-xl-sftturbo50$/, 'XL SFT+Turbo50'],
    [/^acestep-v15-xl-turbo$/, 'XL Turbo'],
    [/^acestep-v15-xl-base$/, 'XL Base'],
    [/^acestep-v15-xl-sft$/, 'XL SFT'],
    [/^acestep-v15-sftturbo50$/, 'SFT+Turbo50'],
    [/^acestep-v15-turbo-continuous$/, 'Turbo Continuous'],
    [/^acestep-v15-turbo-shift(\d)$/, 'Turbo Shift$1'],
    [/^acestep-v15-turbo$/, 'Turbo'],
    [/^acestep-v15-base$/, 'Base'],
    [/^acestep-v15-sft$/, 'SFT'],
  ];

  let label = base;
  for (const [pattern, replacement] of labelMap) {
    if (pattern.test(base)) {
      label = base.replace(pattern, replacement);
      break;
    }
  }

  return quant ? `${label}-${quant}` : label;
}

/** Parse an LM model filename into a nice label like "LM 4B-BF16" */
export function formatLmModel(filename: string): string {
  if (!filename) return '—';
  const name = filename.replace(/\.gguf$/i, '');

  // LM pattern: acestep-5Hz-lm-{size}-{quant}
  const lmMatch = name.match(/^acestep-5Hz-lm-([\d.]+B)-([\w_]+)$/);
  if (lmMatch) return `LM ${lmMatch[1]}-${lmMatch[2]}`;

  // Qwen embedding
  const qwenMatch = name.match(/^Qwen3-Embedding-([\d.]+B)-([\w_]+)$/);
  if (qwenMatch) return `Qwen ${qwenMatch[1]}-${qwenMatch[2]}`;

  return name;
}

/** Parse a VAE model filename into a nice label like "VAE" or "ScragVAE" */
export function formatVaeModel(filename: string): string {
  if (!filename) return '—';
  const isST = /\.safetensors$/i.test(filename);
  const name = filename
    .replace(/\.(gguf|safetensors)$/i, '')
    .replace(/-(BF16|F16|F32)$/i, '');
  const suffix = isST ? ' (ST)' : '';
  if (name === 'vae') return 'VAE' + suffix;
  if (name === 'scragvae') return 'ScragVAE' + suffix;
  return name + suffix;
}

/** Parse an embedding model filename into a nice label like "Qwen3 0.6B-Q8" */
export function formatEmbeddingModel(filename: string): string {
  if (!filename) return '—';
  const name = filename.replace(/\.gguf$/i, '');
  const qwenMatch = name.match(/^Qwen3-Embedding-([\d.]+B)-([\w_]+)$/);
  if (qwenMatch) return `Qwen3 ${qwenMatch[1]}-${qwenMatch[2]}`;
  return name;
}

/** Strip path and extension from a mastering reference filename */
export function formatReferenceName(filename: string): string {
  if (!filename) return '';
  // Strip any path prefix (forward or backslash)
  const basename = filename.replace(/^.*[/\\]/, '');
  // Strip file extension
  return basename.replace(/\.[^.]+$/, '');
}

/** Format a scheduler string into a short display name */
export function formatScheduler(scheduler: string): string {
  if (scheduler.startsWith('composite')) return 'Composite';
  if (scheduler.startsWith('beta:')) return 'Beta';
  if (scheduler.startsWith('power:')) return 'Power';
  const names: Record<string, string> = {
    linear: 'Linear', cosine: 'Cosine', beta57: 'Beta57',
    ddim_uniform: 'DDIM', sgm_uniform: 'SGM',
    bong_tangent: 'Tangent', linear_quadratic: 'Lin-Quad',
  };
  return names[scheduler] || scheduler;
}

/** Get a contextual description for a DiT model based on its filename */
export function getDitModelDescription(filename: string): string {
  if (!filename) return '';
  const name = filename.replace(/\.gguf$/i, '').toLowerCase();

  if (name.includes('turbo') && name.includes('xl')) return 'Extended architecture with turbo training. Fast inference at 8–15 steps with enriched audio quality.';
  if (name.includes('merge') && name.includes('xl')) return 'Merged XL checkpoint combining base stability with turbo speed. Best of both worlds.';
  if (name.includes('sftturbo') || (name.includes('sft') && name.includes('turbo'))) return 'Supervised fine-tuned + turbo-distilled. Consistent quality at reduced step counts.';
  if (name.includes('xl') && name.includes('sft')) return 'Extended architecture, supervised fine-tuned. Maximum consistency for complex prompts.';
  if (name.includes('xl') && name.includes('base')) return 'Extended architecture baseline. Full quality with 30–50 steps recommended.';
  if (name.includes('turbo-continuous')) return 'Turbo variant trained for continuous flow. Works well across all step ranges.';
  if (name.includes('turbo-shift')) return 'Turbo variant with pre-baked shift calibration.';
  if (name.includes('turbo')) return 'Distilled for fast inference. Best at 8–15 steps. The speed workhorse.';
  if (name.includes('sft')) return 'Supervised fine-tuned for consistent, prompt-adherent output.';
  if (name.includes('base')) return 'Full quality baseline model. Best with 30–50 steps.';
  if (name.includes('merge')) return 'Merged checkpoint blending multiple training paradigms.';
  return '';
}

/** Get a contextual description for an LM model based on its filename */
export function getLmModelDescription(filename: string): string {
  if (!filename) return '';
  const name = filename.toLowerCase();
  if (name.includes('4b')) return 'Large language model (4B params). Richer lyric generation and metadata planning.';
  if (name.includes('1.5b') || name.includes('1b')) return 'Compact language model. Faster inference, lighter VRAM usage.';
  return 'Language model for lyric and metadata generation.';
}

/** Get a contextual description for a VAE model */
export function getVaeModelDescription(filename: string): string {
  if (!filename) return '';
  const isST = /\.safetensors$/i.test(filename);
  const name = filename.replace(/\.(gguf|safetensors)$/i, '').toLowerCase();
  const stNote = isST ? ' Full F32 precision — best quality for cover/restyle.' : '';
  if (name.startsWith('vae')) return 'Stock ACE-Step VAE. Standard latent-to-audio decoding.' + stNote;
  if (name.includes('scragvae') || name.includes('scrag')) return 'ScragVAE — custom-trained for reduced artifacts and improved clarity.' + stNote;
  return 'Variational autoencoder for latent-to-audio decoding.' + stNote;
}
