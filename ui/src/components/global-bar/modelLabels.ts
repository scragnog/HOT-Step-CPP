// modelLabels.ts — Human-friendly labels for model filenames
//
// Maps raw model names (GGUF filenames or safetensors directory names)
// to short, readable labels. Format badges are handled by ModelSelect.

/** Parse a DiT model filename into a nice label like "Base/Turbo XL-BF16" */
export function formatDitModel(filename: string): string {
  if (!filename) return '—';

  // ONNX models: special labels
  if (filename === 'dit_fp8.onnx') return 'DiT XL FP8 (TensorRT)';
  if (filename === 'dit_bf16.onnx') return 'DiT XL BF16 (TensorRT)';

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

  // LM pattern: acestep-5Hz-lm-{size}-{quant} (GGUF)
  const lmMatch = name.match(/^acestep-5Hz-lm-([\d.]+B)-([\w_]+)$/);
  if (lmMatch) return `LM ${lmMatch[1]}-${lmMatch[2]}`;

  // LM safetensors dir: acestep-5Hz-lm-{size} (no quant)
  const lmStMatch = name.match(/^acestep-5Hz-lm-([\d.]+B)$/);
  if (lmStMatch) return `LM ${lmStMatch[1]}`;

  // ONNX LM directory: lm-{size} (e.g. "lm-4B", "lm-0.6B", "lm-1.7B")
  const lmOnnxMatch = name.match(/^lm-([\d.]+B)$/i);
  if (lmOnnxMatch) return `LM ${lmOnnxMatch[1]}`;

  // Qwen embedding
  const qwenMatch = name.match(/^Qwen3-Embedding-([\d.]+B)-([\w_]+)$/);
  if (qwenMatch) return `Qwen ${qwenMatch[1]}-${qwenMatch[2]}`;

  return name;
}

/** Parse a VAE model filename into a nice label like "VAE" or "ScragVAE" */
export function formatVaeModel(filename: string): string {
  if (!filename) return '—';
  const name = filename
    .replace(/\.(gguf|safetensors|onnx)$/i, '')
    .replace(/-(BF16|F16|F32)$/i, '');
  // Format badge in ModelSelect handles GGUF/ST indication — no suffix needed
  if (name === 'vae') return 'VAE';
  if (name === 'scragvae') return 'ScragVAE';
  if (name === 'vae-DreamVAE') return 'DreamVAE';
  if (name === 'vae-Regrind-V9b') return 'Regrind V9b';
  if (name === 'vae-Regrind-V9b-Blend50') return 'Regrind V9b Blend50';
  return name;
}

/** Parse an embedding model filename into a nice label like "Qwen3 0.6B-Q8" */
export function formatEmbeddingModel(filename: string): string {
  if (!filename) return '—';
  const name = filename.replace(/\.gguf$/i, '');
  // GGUF: Qwen3-Embedding-0.6B-Q8_0
  const qwenMatch = name.match(/^Qwen3-Embedding-([\d.]+B)-([\w_]+)$/);
  if (qwenMatch) return `Qwen3 ${qwenMatch[1]}-${qwenMatch[2]}`;
  // Safetensors dir: Qwen3-Embedding-0.6B
  const qwenStMatch = name.match(/^Qwen3-Embedding-([\d.]+B)$/);
  if (qwenStMatch) return `Qwen3 ${qwenStMatch[1]}`;
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

/**
 * Format a scheduler string into a short display name for badge display.
 * NOTE: The dropdown now gets primary display names from the plugin registry
 * (registry.schedulers[].display). This function is the fallback used by
 * GenerationBadge and other non-dropdown contexts. Unknown scheduler names
 * fall through to `|| scheduler` which returns the raw plugin name.
 */
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
  // Strip both extensions — works for GGUF filenames and safetensors dirs
  const name = filename.replace(/\.gguf$/i, '').toLowerCase();

  // ONNX / TRT models
  if (name.includes('fp8')) return 'FP8 quantized DiT — fastest inference with FP8 tensor cores. LoRA adapters not yet supported.';
  if (name === 'dit_bf16.onnx') return 'BF16 DiT via TensorRT. Equivalent quality to GGUF BF16 with TRT acceleration.';

  if (name.includes('turbo') && name.includes('xl')) return 'Extended architecture with turbo training. Fast inference at 8–15 steps with enriched audio quality.';
  if (name.includes('merge') && name.includes('xl')) return 'Merged XL checkpoint combining base stability with turbo speed. Best of both worlds.';
  if (name.includes('sftturbo') || (name.includes('sft') && name.includes('turbo'))) return 'Supervised fine-tuned + turbo-distilled. Consistent quality at reduced step counts.';
  if (name.includes('xl') && name.includes('sft')) return 'Extended architecture, supervised fine-tuned. Maximum consistency for complex prompts.';
  if (name.includes('xl') && name.includes('base')) return 'Extended architecture baseline. Full quality with 30–50 steps recommended.';
  if (name.includes('turbo-continuous')) return 'Turbo variant trained for continuous flow. Works well across all step ranges.';
  if (name.includes('turbo-shift')) return 'Turbo variant with pre-baked shift calibration.';
  // Merge checkpoints must be matched BEFORE the bare turbo/sft/base substrings,
  // else a "merge-base-turbo" blend wrongly gets the pure-Turbo description (#68).
  if (name.includes('merge')) {
    if (name.includes('sft') && name.includes('turbo')) return 'Merged SFT+Turbo checkpoint — fine-tuned consistency with turbo speed. ~8–20 steps.';
    if (name.includes('base') && name.includes('turbo')) return 'Merged Base+Turbo checkpoint — baseline quality blended with turbo speed. ~8–20 steps.';
    if (name.includes('base') && name.includes('sft')) return 'Merged Base+SFT checkpoint — baseline quality with fine-tuned consistency.';
    return 'Merged checkpoint blending multiple training paradigms.';
  }
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
  const name = filename.replace(/\.(gguf|safetensors|onnx)$/i, '').toLowerCase();
  if (name.includes('dreamvae')) return 'DreamVAE — alternative decoder architecture with smoother output characteristics.';
  if (name.includes('regrind') && name.includes('blend50')) return 'Regrind V9b Blend50 — 50/50 blend of Regrind V9b with stock weights for balanced clarity.';
  if (name.includes('regrind-v9b') || name.includes('regrind_v9b')) return 'Regrind V9b — latest iteration with refined high-frequency response and reduced artifacts.';
  if (name.includes('regrind-v7') || name.includes('regrind_v7')) return 'Regrind V7 — retrained decoder for improved spectral fidelity.';
  if (name.includes('regrind')) return 'Regrind VAE — retrained decoder for improved audio clarity.';
  if (name.includes('scragvae') || name.includes('scrag')) return 'ScragVAE — custom-trained for reduced artifacts and improved clarity.';
  if (name.startsWith('vae')) return 'Stock ACE-Step VAE. Standard latent-to-audio decoding.';
  return 'Variational autoencoder for latent-to-audio decoding.';
}
