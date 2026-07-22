// ModelCatalogueTab.tsx — Tabbed model catalogue browser

import React, { useState, useMemo } from 'react';
import { ChevronDown, ChevronRight, Download, ExternalLink, Info, KeyRound, ShieldCheck } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { ModelRow } from './ModelRow';
import { usePersistedState } from '../../hooks/usePersistedState';
import type { RegistryFile, DownloadJob } from '../../types';

interface Props {
  files: RegistryFile[];
  downloadJobs: DownloadJob[];
  onDownload: (fileId: string) => void;
  onCancel: (jobId: string) => void;
  onResume: (jobId: string) => void;
  onDelete: (filename: string) => void;
}

type RoleTab = 'dit' | 'lm' | 'embedding' | 'vae' | 'pp-vae' | 'stablestep' | 'supersep' | 'whisper';

const TABS: { id: RoleTab; label: string }[] = [
  { id: 'dit', label: 'DiT Models' },
  { id: 'lm', label: 'Language Models' },
  { id: 'embedding', label: 'Text Encoder' },
  { id: 'vae', label: 'VAE' },
  { id: 'pp-vae', label: 'PP-VAE' },
  { id: 'stablestep', label: 'StableStep' },
  { id: 'supersep', label: 'Stem Separation' },
  { id: 'whisper', label: 'Whisper' },
];

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
  stablestep: 'Stable Audio 3 refiner models for the StableStep post-processing feature. StableStep re-renders the instrumental through Stable Audio 3 to replace VAE fizz with real detail; vocals are split out, cleaned with PP-VAE, and remixed. Two engine backends are available — install either (or both): the GGML backend (4 GGUF files, ~5.8 GB) runs on CUDA, Vulkan or CPU and is the fastest option on NVIDIA in current testing; the ONNX backend (~12 GB, fp32) runs via TensorRT on NVIDIA only and is slow on first use while the TensorRT engine builds (one-time per length bucket). The tokenizer files from the ONNX set are required by BOTH backends.',
  supersep: 'Stem separation models for Cover Studio. Uses a 4-stage ONNX pipeline: BS-Roformer splits audio into 6 stems, Mel-Band RoFormer separates lead/backing vocals, MDX23C isolates drum components, and HTDemucs refines the "other" stem. All 4 models are required for full separation. Models run via ONNX Runtime GPU — no Python needed.',
  whisper: 'OpenAI Whisper models for transcribing actual sung lyrics with word-level timestamps. Enable Whisper Lyrics in Post-Processing to use.',
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
            <label className="flex items-start gap-2.5 cursor-pointer">
              <input
                type="checkbox"
                checked={licenseAccepted}
                onChange={e => { setLicenseAccepted(e.target.checked); setLicenseNudge(false); }}
                className="mt-0.5 h-3.5 w-3.5 rounded border-zinc-400 dark:border-zinc-600 accent-emerald-500 flex-shrink-0 cursor-pointer"
              />
              <span className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
                {STABLESTEP_LICENSE_TEXT}
              </span>
            </label>
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
        <label className="flex items-center gap-1.5 text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
          <KeyRound size={12} />
          Hugging Face token (optional)
        </label>
        <input
          type="password"
          value={hfToken}
          onChange={e => setHfToken(e.target.value)}
          placeholder="hf_..."
          autoComplete="off"
          spellCheck={false}
          className="w-full px-3 py-2 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 font-mono placeholder-zinc-500 focus:border-sky-500/50 focus:ring-1 focus:ring-sky-500/20 outline-none transition-colors"
        />
        <p className="mt-1.5 text-[10px] text-zinc-500 leading-relaxed">
          Only needed if the repository is gated on Hugging Face. Leave empty for an
          anonymous download. Stored locally and sent only to huggingface.co.
        </p>
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

export const ModelCatalogueTab: React.FC<Props> = ({ files, downloadJobs, onDownload, onCancel, onResume, onDelete }) => {
  const [activeTab, setActiveTab] = useState<RoleTab>('dit');

  const ditGroups = useMemo(() => groupDitFiles(files), [files]);
  const lmGroups = useMemo(() => groupLmFiles(files), [files]);
  const embeddingFiles = useMemo(() => files.filter(f => f.role === 'embedding'), [files]);
  const vaeFiles = useMemo(() => files.filter(f => f.role === 'vae'), [files]);
  const ppVaeFiles = useMemo(() => files.filter(f => f.role === 'pp-vae'), [files]);
  const stablestepFiles = useMemo(() => files.filter(f => f.role === 'stablestep'), [files]);
  const supersepFiles = useMemo(() => files.filter(f => f.role === 'supersep'), [files]);
  const whisperFiles = useMemo(() => files.filter(f => f.role === 'whisper'), [files]);

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
      {/* Tab bar */}
      <div className="flex gap-1 border-b border-zinc-200 dark:border-white/5 mb-4">
        {TABS.map(tab => {
          const count = files.filter(f => f.role === tab.id).length;
          const installedCount = files.filter(f => f.role === tab.id && f.installed).length;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2.5 text-xs font-medium transition-colors border-b-2 -mb-px ${
                activeTab === tab.id
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

      {/* Tab content */}
      {activeTab === 'dit' && (
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

      {activeTab === 'lm' && (
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

      {activeTab === 'embedding' && renderSimpleGroup(embeddingFiles, ROLE_INFO.embedding)}
      {activeTab === 'vae' && renderSimpleGroup(vaeFiles, ROLE_INFO.vae)}
      {activeTab === 'pp-vae' && renderSimpleGroup(ppVaeFiles, ROLE_INFO['pp-vae'])}
      {activeTab === 'stablestep' && (
        <StableStepTab
          files={stablestepFiles}
          downloadJobs={downloadJobs}
          onDownload={onDownload}
          onCancel={onCancel}
          onResume={onResume}
          onDelete={onDelete}
        />
      )}
      {activeTab === 'supersep' && renderSimpleGroup(supersepFiles, ROLE_INFO.supersep)}
      {activeTab === 'whisper' && renderSimpleGroup(whisperFiles, ROLE_INFO.whisper)}
    </div>
  );
};
