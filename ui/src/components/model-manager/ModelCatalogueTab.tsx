// ModelCatalogueTab.tsx — Tabbed model catalogue browser

import React, { useState, useMemo } from 'react';
import { ChevronDown, ChevronRight, Info } from 'lucide-react';
import { ModelRow } from './ModelRow';
import type { RegistryFile, DownloadJob } from '../../types';

interface Props {
  files: RegistryFile[];
  downloadJobs: DownloadJob[];
  onDownload: (fileId: string) => void;
  onCancel: (jobId: string) => void;
  onResume: (jobId: string) => void;
  onDelete: (filename: string) => void;
}

type RoleTab = 'dit' | 'lm' | 'embedding' | 'vae' | 'pp-vae';

const TABS: { id: RoleTab; label: string }[] = [
  { id: 'dit', label: 'DiT Models' },
  { id: 'lm', label: 'Language Models' },
  { id: 'embedding', label: 'Text Encoder' },
  { id: 'vae', label: 'VAE' },
  { id: 'pp-vae', label: 'PP-VAE' },
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
  const [open, setOpen] = useState(defaultOpen);
  const [showInfo, setShowInfo] = useState(false);
  const installed = group.files.filter(f => f.installed).length;

  return (
    <div className="rounded-xl border border-white/5 bg-zinc-900/50 overflow-hidden">
      {/* Group header */}
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-4 py-3 hover:bg-white/[0.02] transition-colors"
      >
        {open ? <ChevronDown size={14} className="text-zinc-500" /> : <ChevronRight size={14} className="text-zinc-500" />}
        <span className="text-sm font-semibold text-zinc-300">{group.name}</span>
        <span className="text-[10px] text-zinc-600 font-mono">
          {installed}/{group.files.length} installed
        </span>
        {group.info && (
          <button
            onClick={(e) => { e.stopPropagation(); setShowInfo(!showInfo); }}
            className="ml-auto p-1 rounded-lg hover:bg-white/5 text-zinc-600 hover:text-zinc-400 transition-colors"
            title="About this category"
          >
            <Info size={13} />
          </button>
        )}
      </button>

      {/* Info panel */}
      {showInfo && group.info && (
        <div className="px-4 py-2.5 bg-zinc-800/50 border-t border-white/5 text-xs text-zinc-400 leading-relaxed">
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

// ── Main component ──────────────────────────────────────────

export const ModelCatalogueTab: React.FC<Props> = ({ files, downloadJobs, onDownload, onCancel, onResume, onDelete }) => {
  const [activeTab, setActiveTab] = useState<RoleTab>('dit');

  const ditGroups = useMemo(() => groupDitFiles(files), [files]);
  const lmGroups = useMemo(() => groupLmFiles(files), [files]);
  const embeddingFiles = useMemo(() => files.filter(f => f.role === 'embedding'), [files]);
  const vaeFiles = useMemo(() => files.filter(f => f.role === 'vae'), [files]);
  const ppVaeFiles = useMemo(() => files.filter(f => f.role === 'pp-vae'), [files]);

  const renderSimpleGroup = (roleFiles: RegistryFile[], info?: string) => (
    <div className="space-y-3">
      {info && (
        <div className="rounded-xl bg-zinc-800/50 border border-white/5 px-4 py-3 text-xs text-zinc-400 leading-relaxed">
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
      <div className="flex gap-1 border-b border-white/5 mb-4">
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
                  : 'text-zinc-500 border-transparent hover:text-zinc-300'
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
          <div className="rounded-xl bg-zinc-800/50 border border-white/5 px-4 py-3 text-xs text-zinc-400 leading-relaxed">
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
    </div>
  );
};
