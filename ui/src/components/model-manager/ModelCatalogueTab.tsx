// ModelCatalogueTab.tsx — Tabbed model catalogue browser
//
// Two-level tabs: a top-level family bar (ACE-Step 1.5 / MiniMax-Music3 /
// YuE2 / Shared) selects which backend's models to browse, and — for
// families with more than one role — a role sub-tab bar underneath picks
// the specific component (DiT, LM, VAE, ...). The family selection is
// controlled by the parent (ModelManagerModal) so the Starter Packs section
// above the catalogue can filter to the same family.

import React, { useState, useEffect, useMemo } from 'react';
import { ChevronDown, ChevronRight, Download, ExternalLink, Info, KeyRound, Mail, ShieldCheck } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { ModelRow } from './ModelRow';
import { Toggle } from '../shared/Toggle';
import { ParamLabel } from '../shared/ParamLabel';
import { usePersistedState } from '../../hooks/usePersistedState';
import { useCapabilities } from '../../hooks/useCapabilities';
import type { RegistryFile, DownloadJob } from '../../types';

export type FamilyTab = 'as1.5' | 'mm3' | 'yue2' | 'shared';

export const FAMILY_TABS: { id: FamilyTab; label: string }[] = [
  { id: 'as1.5', label: 'ACE-Step 1.5' },
  { id: 'mm3', label: 'MiniMax-Music3' },
  { id: 'yue2', label: 'YuE2' },
  { id: 'shared', label: 'Shared' },
];

type RoleTab = 'dit' | 'lm' | 'embedding' | 'vae' | 'pp-vae' | 'stablestep' | 'supersep' | 'whisper' | 'mm3' | 'moss' | 'yue2';

/** Role → family fallback, for entries with no `family` field of their own
 *  (a hand-edited catalogue, or one written before this field existed). Every
 *  entry in the shipped registry carries `family` directly — this only
 *  matters as a safety net. */
const ROLE_FAMILY_FALLBACK: Record<RoleTab, FamilyTab> = {
  dit: 'as1.5', lm: 'as1.5', embedding: 'as1.5', vae: 'as1.5', 'pp-vae': 'as1.5',
  stablestep: 'shared', supersep: 'shared', whisper: 'shared', moss: 'shared',
  mm3: 'mm3', yue2: 'yue2',
};

/** Family for one registry file, honoring an explicit `family` first. The
 *  'runtime' role has no tab of its own and splits across two families: the
 *  TensorRT DLLs are MM3-only, the CUDA/ORT runtime DLLs are shared. */
export function familyForFile(f: RegistryFile): FamilyTab {
  if (f.family) return f.family;
  if (f.role === 'runtime') return f.id.startsWith('trt-rt-') ? 'mm3' : 'shared';
  return ROLE_FAMILY_FALLBACK[f.role as RoleTab] ?? 'shared';
}

/** Family for a starter pack: its own `family` field, or (fallback) the
 *  family of its most common constituent file. */
export function familyForPack(pack: { family?: FamilyTab; fileIds?: string[] }, files: RegistryFile[]): FamilyTab {
  if (pack.family) return pack.family;
  const counts: Partial<Record<FamilyTab, number>> = {};
  for (const id of pack.fileIds ?? []) {
    const f = files.find(x => x.id === id);
    if (!f) continue;
    const fam = familyForFile(f);
    counts[fam] = (counts[fam] ?? 0) + 1;
  }
  let best: FamilyTab = 'shared';
  let bestCount = -1;
  for (const [fam, count] of Object.entries(counts) as [FamilyTab, number][]) {
    if (count > bestCount) { best = fam; bestCount = count; }
  }
  return best;
}

interface Props {
  files: RegistryFile[];
  downloadJobs: DownloadJob[];
  onDownload: (fileId: string) => void;
  onCancel: (jobId: string) => void;
  onResume: (jobId: string) => void;
  onDelete: (filename: string) => void;
  /** Selected by the family tab bar in ModelManagerModal (which also filters
   *  the Starter Packs section to the same family) — the sole reason this
   *  is a prop rather than local state. */
  activeFamily: FamilyTab;
}

/** Role sections shown under each family tab, in display order. */
const FAMILY_SECTIONS: Record<FamilyTab, { id: RoleTab; label: string }[]> = {
  'as1.5': [
    { id: 'dit', label: 'DiT Models' },
    { id: 'lm', label: 'Language Models' },
    { id: 'embedding', label: 'Text Encoder' },
    { id: 'vae', label: 'VAE' },
    { id: 'pp-vae', label: 'PP-VAE' },
  ],
  mm3: [
    { id: 'mm3', label: 'MiniMax-Music3' },
  ],
  yue2: [
    { id: 'yue2', label: 'YuE2' },
  ],
  shared: [
    { id: 'stablestep', label: 'StableStep' },
    { id: 'supersep', label: 'Stem Separation' },
    { id: 'whisper', label: 'Whisper' },
    { id: 'moss', label: 'Captioning (MOSS)' },
  ],
};

// ── Info blocks per category ────────────────────────────────

const DIT_INFO: Record<string, string> = {
  'Standard (2B)': 'Standard 2-billion parameter DiT models. Seven variants with different speed/quality trade-offs. Turbo is the fastest (8 steps), SFT has best lyric adherence (32-50 steps), Base offers maximum creative range (60-100 steps).',
  'XL (4B)': 'XL 4-billion parameter DiT models — double the parameters for richer, more detailed audio. Same variant structure as Standard but with noticeably better quality.',
  'XL Merges (Task Arithmetic)': 'Custom blended XL models created by merging two parent checkpoints using task arithmetic. The λ value controls the blend ratio. These use base-mode scheduling (60-100 steps).',
  'MXFP4 (Blackwell Optimized)': 'Microscaling FP4 quantised models — 3.7x compression with native FP4 Tensor Core acceleration on RTX 5000 series GPUs. On older GPUs, they still work via software fallback with the same quality and compression.',
};

const ROLE_INFO: Record<string, string> = {
  lm: 'The Language Model generates audio codes and musical structure from your text prompt. Larger models (4B) produce better quality but use more VRAM. The 4B Q8 is recommended for most users.',
  embedding: 'The text encoder (Qwen3 Embedding) converts your caption and lyrics into embeddings for the DiT. It is architecturally locked — all DiT models were trained with this exact encoder. You need exactly one.',
  vae: 'The VAE (Variational Autoencoder) decodes the DiT\'s latent output into audio waveforms. The standard VAE is required for all generation. ScragVAE is a fine-tuned decoder with improved high-frequency response — it\'s a drop-in replacement.',
  'pp-vae': 'The Post-Processing VAE performs a neural audio polish pass — running generated audio through an encode→decode round-trip to smooth artifacts and improve tonal coherence. Optional but recommended. Use F32 for best quality.',
  stablestep: 'Stable Audio 3 refiner models for the StableStep post-processing feature. StableStep re-renders the instrumental through Stable Audio 3 to replace VAE fizz with real detail; vocals are split out, cleaned with PP-VAE, and remixed. Two engine backends are available — install either (or both): the GGML backend (4 GGUF files, ~5.8 GB) runs on CUDA, Vulkan or CPU and is the fastest option on NVIDIA in current testing; the ONNX backend (~12 GB, fp32) runs via TensorRT on NVIDIA only and is slow on first use while the TensorRT engine builds (one-time per length bucket). The tokenizer files from the ONNX set are required by BOTH backends. Powered by Stability AI.',
  supersep: 'Stem separation models for Cover Studio. Uses a 4-stage ONNX pipeline: BS-Roformer splits audio into 6 stems, Mel-Band RoFormer separates lead/backing vocals, MDX23C isolates drum components, and HTDemucs refines the "other" stem. All 4 models are required for full separation. Models run via ONNX Runtime GPU — no Python needed.',
  whisper: 'OpenAI Whisper models for transcribing actual sung lyrics with word-level timestamps. Enable Whisper Lyrics in Post-Processing to use.',
  moss: 'MOSS-Music-8B — the only model here that ANALYSES audio rather than generating it. It captions your own tracks locally in the Training Studio, writing what it actually hears instead of rewriting a text analysis, and emits both the ACE-Step caption format and MM3 Structured Captions from a single pass. Pick one LM (Q8_0 recommended) plus the audio tower, which is required and never quantised. Nothing else in the app depends on these — they are only used when you choose MOSS as the caption provider.',
  mm3: 'MiniMax-Music3 — a separate generation backend with its own models: a language model plus a 5-way split flow stack (depth decoder, condition encoder, DiT, vocoder — the LM and DiT are the two you must pick; the rest default to auto). All required roles load together, needing ~24 GB of VRAM. Switch to it via the Backend toggle in the top bar. A LICENSE file is fetched alongside automatically once a GGUF finishes downloading.',
  yue2: 'YuE2 — a third generation backend: a language model plus a VAE decoder, producing 48 kHz stereo audio from a freeform style + lyrics prompt (no headed caption template). Pick the LM plus exactly one VAE (Standard or Legacy — not both). Switch to it via the Backend toggle in the top bar. YuE2 is licensed CC BY-NC 4.0, but its authors have clarified that individuals — creators, musicians, researchers — may use the model and its outputs freely, including commercially; only companies need a commercial licence (gezhang@umich.edu). A LICENSE file is fetched alongside automatically once a GGUF finishes downloading.',
  // mm3Trt is NOT here — unlike every other entry in this record, it's new
  // text (the reviewer flagged it for i18n), so it's translated at its one
  // call site via t('models.mm3Trt.info') instead of joining this
  // not-yet-translated table.
};

// ── Grouping logic ──────────────────────────────────────────

interface ModelGroup {
  name: string;
  info?: string;
  files: RegistryFile[];
}

function groupDitFiles(files: RegistryFile[]): ModelGroup[] {
  const ditFiles = files.filter(f => f.role === 'dit');

  const standard = ditFiles.filter(f => f.scale === 'standard' && f.quant !== 'MXFP4');
  const xl = ditFiles.filter(f => f.scale === 'xl' && !f.variant?.startsWith('merge-') && f.quant !== 'MXFP4');
  const xlMerges = ditFiles.filter(f => f.scale === 'xl' && f.variant?.startsWith('merge-') && f.quant !== 'MXFP4');
  const mxfp4 = ditFiles.filter(f => f.quant === 'MXFP4');

  const groups: ModelGroup[] = [];
  if (standard.length) groups.push({ name: 'Standard (2B)', info: DIT_INFO['Standard (2B)'], files: standard });
  if (xl.length) groups.push({ name: 'XL (4B)', info: DIT_INFO['XL (4B)'], files: xl });
  if (xlMerges.length) groups.push({ name: 'XL Merges (Task Arithmetic)', info: DIT_INFO['XL Merges (Task Arithmetic)'], files: xlMerges });
  if (mxfp4.length) groups.push({ name: 'MXFP4 (Blackwell Optimized)', info: DIT_INFO['MXFP4 (Blackwell Optimized)'], files: mxfp4 });
  return groups;
}

function groupLmFiles(files: RegistryFile[]): ModelGroup[] {
  const lmFiles = files.filter(f => f.role === 'lm');
  const sizes = ['4B', '1.7B', '0.6B'];
  return sizes
    .map(s => ({
      name: `${s} Parameters`,
      files: lmFiles.filter(f => f.variant === s),
    }))
    .filter(g => g.files.length > 0);
}

// ── YuE2 quant ladder ─────────────────────────────────────────
//
// The registry only ships a BF16 LM today — the quant ladder entries land in
// a later step. Grouping is data-driven from role 'yue2' files by quant so
// each new quant just appears here once its registry entry exists, with no
// UI change required.

const YUE2_QUANT_ORDER = ['BF16', 'Q8_0', 'Q6_K', 'Q5_K_M', 'Q4_K_M-imat', 'Q4_K_S-imat', 'Q3_K_M-imat', 'NVFP4', 'MXFP4'];

const YUE2_QUANT_NOTES: Record<string, string> = {
  BF16: 'Reference precision — the source weights, no quantisation.',
  Q8_0: 'Near-lossless — indistinguishable from BF16 in practice, about half the size.',
  Q6_K: 'Close to source quality with a real size saving.',
  Q5_K_M: 'Close to source quality with a real size saving, smaller than Q6_K.',
  'Q4_K_M-imat': 'Imatrix-guided — importance-weighted rounding keeps quality up at a noticeably smaller size.',
  'Q4_K_S-imat': 'Imatrix-guided — importance-weighted rounding keeps quality up at a noticeably smaller size.',
  'Q3_K_M-imat': 'Imatrix-guided — the smallest quant considered usable.',
  NVFP4: 'Experimental — scores below the k-quants above in testing so far.',
  MXFP4: 'Experimental — scores below the k-quants above in testing so far.',
};

function sortByQuantOrder(quants: string[]): string[] {
  return [...quants].sort((a, b) => {
    const ia = YUE2_QUANT_ORDER.indexOf(a);
    const ib = YUE2_QUANT_ORDER.indexOf(b);
    if (ia === -1 && ib === -1) return a.localeCompare(b);
    if (ia === -1) return 1;
    if (ib === -1) return -1;
    return ia - ib;
  });
}

const Yue2Tab: React.FC<{
  files: RegistryFile[];
  downloadJobs: DownloadJob[];
  onDownload: (fileId: string) => void;
  onCancel: (jobId: string) => void;
  onResume: (jobId: string) => void;
  onDelete: (filename: string) => void;
}> = ({ files, downloadJobs, onDownload, onCancel, onResume, onDelete }) => {
  const lmFiles = useMemo(() => files.filter(f => f.variant === 'lm' || (f.variant ?? '').startsWith('lm-')), [files]);
  const imatrixFiles = useMemo(
    () => files.filter(f => f.variant === 'imatrix' || f.id.includes('imatrix') || f.filename.includes('imatrix')),
    [files],
  );
  const vaeFiles = useMemo(() => files.filter(f => (f.variant ?? '').startsWith('vae-')), [files]);
  const otherFiles = useMemo(
    () => files.filter(f => !lmFiles.includes(f) && !imatrixFiles.includes(f) && !vaeFiles.includes(f)),
    [files, lmFiles, imatrixFiles, vaeFiles],
  );

  const quantGroups = useMemo(() => {
    const byQuant = new Map<string, RegistryFile[]>();
    for (const f of lmFiles) {
      const list = byQuant.get(f.quant) ?? [];
      list.push(f);
      byQuant.set(f.quant, list);
    }
    return sortByQuantOrder([...byQuant.keys()]).map(quant => ({
      quant,
      note: YUE2_QUANT_NOTES[quant],
      files: byQuant.get(quant)!,
    }));
  }, [lmFiles]);

  const rowProps = { downloadJobs, onDownload, onCancel, onResume, onDelete };
  const findJob = (id: string) => downloadJobs.find(j => j.fileId === id && j.status !== 'completed' && j.status !== 'cancelled');

  return (
    <div className="space-y-4">
      <div className="rounded-xl bg-zinc-100/50 dark:bg-zinc-800/50 border border-zinc-200 dark:border-white/5 px-4 py-3 text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
        {ROLE_INFO.yue2}
      </div>

      {/* Language model — grouped by quant */}
      <div>
        <h4 className="px-1 text-xs font-semibold text-zinc-700 dark:text-zinc-300 mb-1.5">Language Model — pick one quant</h4>
        {quantGroups.length === 0 ? (
          <p className="px-1 text-xs text-zinc-500">No YuE2 language model files in the catalogue yet.</p>
        ) : (
          <div className="space-y-2">
            {quantGroups.map(g => (
              <div key={g.quant} className="rounded-xl border border-zinc-200 dark:border-white/5 bg-zinc-50/80 dark:bg-zinc-900/50 p-3">
                <div className="flex items-baseline gap-2 mb-1.5">
                  <span className="text-xs font-mono font-semibold text-zinc-700 dark:text-zinc-300">{g.quant}</span>
                  {g.note && <span className="text-[10px] text-zinc-500 leading-relaxed">{g.note}</span>}
                </div>
                <div className="space-y-1.5">
                  {g.files.map(f => (
                    <ModelRow key={f.id} file={f} downloadJob={findJob(f.id)} {...rowProps} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Imatrix (quant-authoring only) */}
      {imatrixFiles.length > 0 && (
        <div>
          <h4 className="px-1 text-xs font-semibold text-zinc-700 dark:text-zinc-300 mb-1">Imatrix data</h4>
          <p className="px-1 text-[10px] text-zinc-500 mb-1.5 leading-relaxed">
            Only needed to produce new quants yourself — not required to run any quant above.
          </p>
          <div className="space-y-1.5">
            {imatrixFiles.map(f => (
              <ModelRow key={f.id} file={f} downloadJob={findJob(f.id)} {...rowProps} />
            ))}
          </div>
        </div>
      )}

      {/* VAE decoder */}
      <div>
        <h4 className="px-1 text-xs font-semibold text-zinc-700 dark:text-zinc-300 mb-1">VAE decoder — pick one</h4>
        <p className="px-1 text-[10px] text-zinc-500 mb-1.5 leading-relaxed">
          Standard is recommended; Legacy is kept for parity with earlier YuE checkpoints.
        </p>
        <div className="space-y-1.5">
          {vaeFiles.map(f => (
            <ModelRow key={f.id} file={f} downloadJob={findJob(f.id)} {...rowProps} />
          ))}
        </div>
      </div>

      {otherFiles.length > 0 && (
        <div className="space-y-1.5">
          {otherFiles.map(f => (
            <ModelRow key={f.id} file={f} downloadJob={findJob(f.id)} {...rowProps} />
          ))}
        </div>
      )}

      {/* Licence notice */}
      <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 px-4 py-3 flex items-start gap-2.5">
        <ShieldCheck size={15} className="mt-0.5 flex-shrink-0 text-amber-500" />
        <p className="text-xs text-amber-700 dark:text-amber-400 leading-relaxed">
          YuE2's weights are licensed{' '}
          <a href="https://creativecommons.org/licenses/by-nc/4.0/" target="_blank" rel="noopener noreferrer"
            className="inline-flex items-center gap-1 underline hover:no-underline">
            CC BY-NC 4.0<ExternalLink size={10} />
          </a>, but the upstream authors have clarified that individual creators, musicians
          and researchers may use the model and its outputs freely, including commercially.
          Only companies need a commercial licence — contact{' '}
          <a href="mailto:gezhang@umich.edu" className="inline-flex items-center gap-1 underline hover:no-underline">
            gezhang@umich.edu<Mail size={10} />
          </a>.
        </p>
      </div>
    </div>
  );
};

// ── Collapsible group component ─────────────────────────────

const CollapsibleGroup: React.FC<{
  group: ModelGroup;
  downloadJobs: DownloadJob[];
  onDownload: (fileId: string) => void;
  onCancel: (jobId: string) => void;
  onResume: (jobId: string) => void;
  onDelete: (filename: string) => void;
  defaultOpen?: boolean;
}> = ({ group, downloadJobs, onDownload, onCancel, onResume, onDelete, defaultOpen = false }) => {
  const { t } = useTranslation();
  const [open, setOpen] = useState(defaultOpen);
  const [showInfo, setShowInfo] = useState(false);
  const installed = group.files.filter(f => f.installed).length;

  return (
    <div className="rounded-xl border border-zinc-200 dark:border-white/5 bg-zinc-50/80 dark:bg-zinc-900/50 overflow-hidden">
      {/* Group header */}
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-4 py-3 hover:bg-white/[0.02] transition-colors"
      >
        {open ? <ChevronDown size={14} className="text-zinc-500" /> : <ChevronRight size={14} className="text-zinc-500" />}
        <span className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">{group.name}</span>
        <span className="text-[10px] text-zinc-600 font-mono">
          {installed}/{group.files.length} installed
        </span>
        {group.info && (
          <button
            onClick={(e) => { e.stopPropagation(); setShowInfo(!showInfo); }}
            className="ml-auto p-1 rounded-lg hover:bg-white/5 text-zinc-600 hover:text-zinc-600 dark:text-zinc-400 transition-colors"
            title={t('models.aboutCategory')}
          >
            <Info size={13} />
          </button>
        )}
      </button>

      {/* Info panel */}
      {showInfo && group.info && (
        <div className="px-4 py-2.5 bg-zinc-100/50 dark:bg-zinc-800/50 border-t border-zinc-200 dark:border-white/5 text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
          {group.info}
        </div>
      )}

      {/* File list */}
      {open && (
        <div className="px-3 pb-3 space-y-1.5">
          {group.files.map(f => (
            <ModelRow
              key={f.id}
              file={f}
              downloadJob={downloadJobs.find(j => j.fileId === f.id && j.status !== 'completed' && j.status !== 'cancelled')}
              onDownload={onDownload}
              onCancel={onCancel}
              onResume={onResume}
              onDelete={onDelete}
            />
          ))}
        </div>
      )}
    </div>
  );
};

// ── TensorRT (MM3 DiT) group ─────────────────────────────────
//
// Not one of the generic role sections — it's a dedicated group within the
// mm3 family, since the ONNX graph is role 'mm3' but the runtime DLLs are
// role 'runtime' (like the SuperSep/cuBLAS runtime entries, which have no
// catalogue UI of their own at all; this is the first role:'runtime' set
// that gets one, because unlike those, a TensorRT builder resource comes in
// several GPU-specific variants a user must choose between, not just
// "install all").

const Mm3TrtGroup: React.FC<{
  onnxFile?: RegistryFile;
  coreFiles: RegistryFile[];
  builderFiles: RegistryFile[];
  downloadJobs: DownloadJob[];
  onDownload: (fileId: string) => void;
  onCancel: (jobId: string) => void;
  onResume: (jobId: string) => void;
  onDelete: (filename: string) => void;
}> = ({ onnxFile, coreFiles, builderFiles, downloadJobs, onDownload, onCancel, onResume, onDelete }) => {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [showInfo, setShowInfo] = useState(false);
  const { capabilities } = useCapabilities();
  const ditRuntime = (capabilities?.core as { dit_runtime?: { supported?: boolean; sm?: number } } | undefined)?.dit_runtime;
  const sm = ditRuntime?.sm;

  // Same rule as cuda-rt-*/supersep-rt-* (modelDownloadService.ts
  // CUDA_ONLY_FILE_PREFIXES): a Vulkan/CPU build never sees these files at
  // all, so this would already render empty there. `supported` catches the
  // narrower CUDA-but-not-built-with-TRT case (no engine/deps/tensorrt SDK at
  // compile time) that the file-prefix filter can't see.
  if (ditRuntime?.supported !== true) return null;

  const allFiles = [...(onnxFile ? [onnxFile] : []), ...coreFiles, ...builderFiles];
  const installed = allFiles.filter(f => f.installed).length;

  // A DLL tagged with `sm` matching the probed device is the one the user
  // actually needs; everything else in the list is for a different GPU. When
  // `sm` is unknown (engine down, no CUDA device, older engine build with no
  // dit_runtime.sm) nothing is marked "required" — a wrong guess is worse
  // than no guess.
  const requiredBuilder = builderFiles.filter(f => sm != null && f.sm === sm);
  const otherBuilder = builderFiles.filter(f => !(sm != null && f.sm === sm));

  const rowProps = { downloadJobs, onDownload, onCancel, onResume, onDelete };

  return (
    <div className="rounded-xl border border-zinc-200 dark:border-white/5 bg-zinc-50/80 dark:bg-zinc-900/50 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-4 py-3 hover:bg-white/[0.02] transition-colors"
      >
        {open ? <ChevronDown size={14} className="text-zinc-500" /> : <ChevronRight size={14} className="text-zinc-500" />}
        <span className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">{t('models.mm3Trt.title')}</span>
        <span className="text-[10px] text-zinc-600 font-mono">
          {t('models.xInstalled', { installed, total: allFiles.length })}
        </span>
        <button
          onClick={(e) => { e.stopPropagation(); setShowInfo(!showInfo); }}
          className="ml-auto p-1 rounded-lg hover:bg-white/5 text-zinc-600 hover:text-zinc-600 dark:text-zinc-400 transition-colors"
          title={t('models.aboutCategory')}
        >
          <Info size={13} />
        </button>
      </button>

      {showInfo && (
        <div className="px-4 py-2.5 bg-zinc-100/50 dark:bg-zinc-800/50 border-t border-zinc-200 dark:border-white/5 text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
          {t('models.mm3Trt.info')}
        </div>
      )}

      {open && (
        <div className="px-3 pb-3 space-y-3">
          {onnxFile && (
            <div className="space-y-1.5">
              <h5 className="px-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                {t('models.mm3Trt.graphHeading')}
              </h5>
              <ModelRow file={onnxFile} {...rowProps}
                downloadJob={downloadJobs.find(j => j.fileId === onnxFile.id && j.status !== 'completed' && j.status !== 'cancelled')} />
            </div>
          )}

          {coreFiles.length > 0 && (
            <div className="space-y-1.5">
              <h5 className="px-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                {t('models.mm3Trt.runtimeHeading')}
              </h5>
              {coreFiles.map(f => (
                <ModelRow key={f.id} file={f} {...rowProps}
                  downloadJob={downloadJobs.find(j => j.fileId === f.id && j.status !== 'completed' && j.status !== 'cancelled')} />
              ))}
            </div>
          )}

          {requiredBuilder.length > 0 && (
            <div className="space-y-1.5">
              <h5 className="px-1 text-[10px] font-semibold uppercase tracking-wider text-emerald-500">
                {t('models.mm3Trt.requiredGpuHeading', { sm })}
              </h5>
              {requiredBuilder.map(f => (
                <ModelRow key={f.id} file={f} {...rowProps}
                  downloadJob={downloadJobs.find(j => j.fileId === f.id && j.status !== 'completed' && j.status !== 'cancelled')} />
              ))}
            </div>
          )}

          {otherBuilder.length > 0 && (
            <div className="space-y-1.5">
              <h5 className="px-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                {t('models.mm3Trt.otherGpuHeading')}
              </h5>
              {otherBuilder.map(f => (
                <ModelRow key={f.id} file={f} {...rowProps}
                  downloadJob={downloadJobs.find(j => j.fileId === f.id && j.status !== 'completed' && j.status !== 'cancelled')} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ── StableStep tab (license gate + optional HF token) ───────

const STABLESTEP_LICENSE_TEXT =
  "These weights are derived from Stability AI's Stable Audio 3 and are licensed under the " +
  'Stability AI Community License (free for individuals and organizations under $1M annual ' +
  'revenue; commercial use above that requires a license from Stability AI).';

const StableStepTab: React.FC<{
  files: RegistryFile[];
  downloadJobs: DownloadJob[];
  onDownload: (fileId: string) => void;
  onCancel: (jobId: string) => void;
  onResume: (jobId: string) => void;
  onDelete: (filename: string) => void;
}> = ({ files, downloadJobs, onDownload, onCancel, onResume, onDelete }) => {
  // License acceptance is persisted so it's asked once.
  const [licenseAccepted, setLicenseAccepted] = usePersistedState('hs-stablestepLicenseAccepted', false);
  // Optional Hugging Face token — forwarded as `Authorization: Bearer <token>`
  // on huggingface.co requests (only needed if the repo is gated).
  const [hfToken, setHfToken] = usePersistedState('hs-hfToken', '');
  const [licenseNudge, setLicenseNudge] = useState(false);

  const missing = files.filter(f => !f.installed);

  // Two engine backends ship under the same repo: the GGUF files (models root)
  // power the GGML backend; everything else is the ONNX/TensorRT set. The
  // tokenizer JSONs in the ONNX set are required by BOTH backends.
  const ggufFiles = files.filter(f => f.filename.endsWith('.gguf'));
  const onnxFiles = files.filter(f => !f.filename.endsWith('.gguf'));

  // Gate every download behind license acceptance.
  const gatedDownload = (fileId: string) => {
    if (!licenseAccepted) {
      setLicenseNudge(true);
      return;
    }
    onDownload(fileId);
  };

  const handleDownloadAll = () => {
    if (!licenseAccepted) {
      setLicenseNudge(true);
      return;
    }
    for (const f of missing) onDownload(f.id);
  };

  return (
    <div className="space-y-3">
      {/* Info block */}
      <div className="rounded-xl bg-zinc-100/50 dark:bg-zinc-800/50 border border-zinc-200 dark:border-white/5 px-4 py-3 text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
        {ROLE_INFO.stablestep}
      </div>

      {/* License acceptance gate */}
      <div className={`rounded-xl border px-4 py-3 transition-colors ${
        licenseAccepted
          ? 'border-emerald-500/20 bg-emerald-500/5'
          : licenseNudge
            ? 'border-amber-500/40 bg-amber-500/10'
            : 'border-zinc-200 dark:border-white/5 bg-zinc-100/50 dark:bg-zinc-800/50'
      }`}>
        <div className="flex items-start gap-2.5">
          <ShieldCheck size={15} className={`mt-0.5 flex-shrink-0 ${licenseAccepted ? 'text-emerald-400' : 'text-zinc-500'}`} />
          <div className="flex-1">
            <Toggle
              accent="pink"
              checked={licenseAccepted}
              onChange={checked => { setLicenseAccepted(checked); setLicenseNudge(false); }}
              label={STABLESTEP_LICENSE_TEXT}
              info="Confirms you accept the Stability AI Community License for these StableStep weights (free for individuals and organizations under $1M annual revenue; commercial use above that needs a license from Stability AI). Off: every download in this tab is blocked until you accept."
            />
            <a
              href="https://stability.ai/community-license-agreement"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 mt-1.5 ml-6 text-[11px] text-sky-400 hover:text-sky-300 transition-colors"
            >
              <ExternalLink size={11} />
              Stability AI Community License Agreement
            </a>
            {licenseNudge && !licenseAccepted && (
              <p className="mt-1.5 ml-6 text-[11px] text-amber-400">
                Please accept the license terms above before downloading.
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Optional Hugging Face token */}
      <div className="rounded-xl border border-zinc-200 dark:border-white/5 bg-zinc-100/50 dark:bg-zinc-800/50 px-4 py-3">
        <div className="flex items-center gap-1.5 mb-1.5">
          <KeyRound size={12} className="text-zinc-500" />
          <ParamLabel
            label="Hugging Face token (optional)"
            className="text-xs font-medium text-zinc-500 uppercase tracking-wider"
            info="Sent as an Authorization header on requests to huggingface.co, so a gated StableStep repo can be downloaded. Leave empty for an anonymous download, which works for any repo that isn't gated. Stored locally and never sent anywhere else."
          />
        </div>
        <input
          type="password"
          value={hfToken}
          onChange={e => setHfToken(e.target.value)}
          placeholder="hf_..."
          autoComplete="off"
          spellCheck={false}
          className="w-full px-3 py-2 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 font-mono placeholder-zinc-500 focus:border-sky-500/50 focus:ring-1 focus:ring-sky-500/20 outline-none transition-colors"
        />
      </div>

      {/* Download all */}
      {missing.length > 0 && (
        <div className="flex items-center justify-between px-1">
          <span className="text-[11px] text-zinc-500">
            {files.length - missing.length}/{files.length} files installed &middot; GGML set ~5.8 GB &middot; ONNX set ~12 GB
          </span>
          <button
            onClick={handleDownloadAll}
            disabled={!licenseAccepted}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              licenseAccepted
                ? 'bg-gradient-to-r from-pink-500 to-pink-600 text-white hover:from-pink-400 hover:to-pink-500 shadow-lg shadow-pink-500/10'
                : 'bg-zinc-200 dark:bg-zinc-800 text-zinc-500 cursor-not-allowed'
            }`}
            title={licenseAccepted ? 'Download all missing StableStep files' : 'Accept the license first'}
          >
            <Download size={12} />
            Download all missing ({missing.length})
          </button>
        </div>
      )}

      {/* File list — grouped by engine backend */}
      <div className={`space-y-3 ${licenseAccepted ? '' : 'opacity-60'}`}>
        {ggufFiles.length > 0 && (
          <div className="space-y-1.5">
            <div className="px-1">
              <h4 className="text-xs font-semibold text-zinc-700 dark:text-zinc-300">
                GGML backend (universal — CUDA/Vulkan/CPU)
              </h4>
              <p className="text-[10px] text-zinc-500 leading-relaxed">
                4 GGUF files (~5.8 GB). Fastest option on NVIDIA in current testing
                and the only backend for Vulkan/CPU builds. Also requires the
                tokenizer files from the ONNX set below.
              </p>
            </div>
            {ggufFiles.map(f => (
              <ModelRow
                key={f.id}
                file={f}
                downloadJob={downloadJobs.find(j => j.fileId === f.id && j.status !== 'completed' && j.status !== 'cancelled')}
                onDownload={gatedDownload}
                onCancel={onCancel}
                onResume={onResume}
                onDelete={onDelete}
              />
            ))}
          </div>
        )}
        {onnxFiles.length > 0 && (
          <div className="space-y-1.5">
            <div className="px-1">
              <h4 className="text-xs font-semibold text-zinc-700 dark:text-zinc-300">
                ONNX backend (NVIDIA TensorRT)
              </h4>
              <p className="text-[10px] text-zinc-500 leading-relaxed">
                fp32 ONNX set (~12 GB), NVIDIA only. The tokenizer files in this
                set are required by BOTH backends.
              </p>
            </div>
            {onnxFiles.map(f => (
              <ModelRow
                key={f.id}
                file={f}
                downloadJob={downloadJobs.find(j => j.fileId === f.id && j.status !== 'completed' && j.status !== 'cancelled')}
                onDownload={gatedDownload}
                onCancel={onCancel}
                onResume={onResume}
                onDelete={onDelete}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// ── Main component ──────────────────────────────────────────

export const ModelCatalogueTab: React.FC<Props> = ({ files, downloadJobs, onDownload, onCancel, onResume, onDelete, activeFamily }) => {
  const sections = FAMILY_SECTIONS[activeFamily];
  const [activeSection, setActiveSection] = useState<RoleTab>(sections[0].id);

  // Switching family can leave activeSection pointing at a section that
  // doesn't exist under the new family — snap back to that family's first.
  useEffect(() => {
    if (!sections.some(s => s.id === activeSection)) {
      setActiveSection(sections[0].id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeFamily]);

  const ditGroups = useMemo(() => groupDitFiles(files), [files]);
  const lmGroups = useMemo(() => groupLmFiles(files), [files]);
  const embeddingFiles = useMemo(() => files.filter(f => f.role === 'embedding'), [files]);
  const vaeFiles = useMemo(() => files.filter(f => f.role === 'vae'), [files]);
  const ppVaeFiles = useMemo(() => files.filter(f => f.role === 'pp-vae'), [files]);
  const stablestepFiles = useMemo(() => files.filter(f => f.role === 'stablestep'), [files]);
  const supersepFiles = useMemo(() => files.filter(f => f.role === 'supersep'), [files]);
  const whisperFiles = useMemo(() => files.filter(f => f.role === 'whisper'), [files]);
  // The TensorRT ONNX graph is role 'mm3' (it lives in models/mm3/ like every
  // other MM3 weight) but renders in its own group below, not the flat list —
  // split it out here rather than filtering inline at every use site.
  const mm3Files = useMemo(() => files.filter(f => f.role === 'mm3' && f.id !== 'mm3-dit-trt-onnx'), [files]);
  const mm3TrtOnnx = useMemo(() => files.find(f => f.id === 'mm3-dit-trt-onnx'), [files]);
  const mm3TrtCore = useMemo(
    () => files.filter(f => f.id === 'trt-rt-nvinfer' || f.id === 'trt-rt-onnxparser'), [files],
  );
  const mm3TrtBuilders = useMemo(
    () => files.filter(f => f.role === 'runtime' && f.id.startsWith('trt-rt-builder-')), [files],
  );
  const mossFiles = useMemo(() => files.filter(f => f.role === 'moss'), [files]);
  const yue2Files = useMemo(() => files.filter(f => f.role === 'yue2'), [files]);

  const renderSimpleGroup = (roleFiles: RegistryFile[], info?: string) => (
    <div className="space-y-3">
      {info && (
        <div className="rounded-xl bg-zinc-100/50 dark:bg-zinc-800/50 border border-zinc-200 dark:border-white/5 px-4 py-3 text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
          {info}
        </div>
      )}
      <div className="space-y-1.5">
        {roleFiles.map(f => (
          <ModelRow
            key={f.id}
            file={f}
            downloadJob={downloadJobs.find(j => j.fileId === f.id && j.status !== 'completed' && j.status !== 'cancelled')}
            onDownload={onDownload}
            onCancel={onCancel}
            onResume={onResume}
            onDelete={onDelete}
          />
        ))}
      </div>
    </div>
  );

  return (
    <div>
      {/* Role sub-tab bar — only when the family has more than one section.
          (The family tab bar itself lives in ModelManagerModal, which also
          uses it to filter the Starter Packs section above this component.) */}
      {sections.length > 1 && (
        <div className="flex gap-1 border-b border-zinc-200 dark:border-white/5 mb-4 overflow-x-auto">
          {sections.map(tab => {
            const count = files.filter(f => f.role === tab.id).length;
            const installedCount = files.filter(f => f.role === tab.id && f.installed).length;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveSection(tab.id)}
                className={`px-4 py-2.5 text-xs font-medium whitespace-nowrap transition-colors border-b-2 -mb-px ${
                  activeSection === tab.id
                    ? 'text-pink-400 border-pink-500'
                    : 'text-zinc-500 border-transparent hover:text-zinc-700 dark:hover:text-zinc-300'
                }`}
              >
                {tab.label}
                <span className="ml-1.5 text-[10px] text-zinc-600 font-mono">{installedCount}/{count}</span>
              </button>
            );
          })}
        </div>
      )}

      <div className={sections.length === 1 ? 'mt-4' : ''}>
        {activeSection === 'dit' && (
          <div className="space-y-3">
            {ditGroups.map(g => (
              <CollapsibleGroup
                key={g.name}
                group={g}
                downloadJobs={downloadJobs}
                onDownload={onDownload}
                onCancel={onCancel}
                onResume={onResume}
                onDelete={onDelete}
              />
            ))}
          </div>
        )}

        {activeSection === 'lm' && (
          <div className="space-y-3">
            <div className="rounded-xl bg-zinc-100/50 dark:bg-zinc-800/50 border border-zinc-200 dark:border-white/5 px-4 py-3 text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
              {ROLE_INFO.lm}
            </div>
            {lmGroups.map(g => (
              <CollapsibleGroup
                key={g.name}
                group={g}
                downloadJobs={downloadJobs}
                onDownload={onDownload}
                onCancel={onCancel}
                onResume={onResume}
                onDelete={onDelete}
                defaultOpen
              />
            ))}
          </div>
        )}

        {activeSection === 'embedding' && renderSimpleGroup(embeddingFiles, ROLE_INFO.embedding)}
        {activeSection === 'vae' && renderSimpleGroup(vaeFiles, ROLE_INFO.vae)}
        {activeSection === 'pp-vae' && renderSimpleGroup(ppVaeFiles, ROLE_INFO['pp-vae'])}
        {activeSection === 'stablestep' && (
          <StableStepTab
            files={stablestepFiles}
            downloadJobs={downloadJobs}
            onDownload={onDownload}
            onCancel={onCancel}
            onResume={onResume}
            onDelete={onDelete}
          />
        )}
        {activeSection === 'supersep' && renderSimpleGroup(supersepFiles, ROLE_INFO.supersep)}
        {activeSection === 'whisper' && renderSimpleGroup(whisperFiles, ROLE_INFO.whisper)}
        {activeSection === 'mm3' && (
          <div className="space-y-3">
            {renderSimpleGroup(mm3Files, ROLE_INFO.mm3)}
            <Mm3TrtGroup
              onnxFile={mm3TrtOnnx}
              coreFiles={mm3TrtCore}
              builderFiles={mm3TrtBuilders}
              downloadJobs={downloadJobs}
              onDownload={onDownload}
              onCancel={onCancel}
              onResume={onResume}
              onDelete={onDelete}
            />
          </div>
        )}
        {activeSection === 'moss' && renderSimpleGroup(mossFiles, ROLE_INFO.moss)}
        {activeSection === 'yue2' && (
          <Yue2Tab
            files={yue2Files}
            downloadJobs={downloadJobs}
            onDownload={onDownload}
            onCancel={onCancel}
            onResume={onResume}
            onDelete={onDelete}
          />
        )}
      </div>
    </div>
  );
};
