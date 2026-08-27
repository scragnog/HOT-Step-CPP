// ModelsDropdown.tsx — Model selection UI for the global param bar
//
// Uses custom ModelSelect dropdown to show GGUF/SafeTensors format badges.

import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Download } from 'lucide-react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { modelApi } from '../../services/api';
import { formatDitModel, formatLmModel, formatVaeModel, formatEmbeddingModel, getDitModelDescription, getLmModelDescription, getVaeModelDescription } from './modelLabels';
import { ModelManagerModal } from '../model-manager/ModelManagerModal';
import { ModelSelect, getModelFormat } from './ModelSelect';
import { ParamLabel } from '../shared/ParamLabel';
import type { AceModels } from '../../types';

type ModelFormat = 'gguf' | 'safetensors' | 'onnx';
const FORMAT_ORDER: ModelFormat[] = ['gguf', 'safetensors', 'onnx'];
const FORMAT_LABELS: Record<ModelFormat, string> = { gguf: 'GGUF', safetensors: 'SafeTensors', onnx: 'ONNX' };
const FORMAT_FILTER_KEY = 'model-format-filter';

function loadFormatFilter(): Record<ModelFormat, boolean> {
  const allOn = { gguf: true, safetensors: true, onnx: true };
  try {
    const raw = localStorage.getItem(FORMAT_FILTER_KEY);
    if (!raw) return allOn;
    const parsed = JSON.parse(raw);
    return {
      gguf: parsed.gguf !== false,
      safetensors: parsed.safetensors !== false,
      onnx: parsed.onnx !== false,
    };
  } catch {
    return allOn;
  }
}

export const ModelsDropdown: React.FC = () => {
  const gp = useGlobalParams();
  const { t } = useTranslation();
  const [models, setModels] = useState<AceModels | null>(null);
  const [showModelManager, setShowModelManager] = useState(false);

  useEffect(() => {
    modelApi.list()
      .then(setModels)
      .catch(() => {});
  }, []);

  // Auto-select first available model when list loads and nothing is selected
  useEffect(() => {
    if (!models?.models) return;
    const dit = models.models.dit || [];
    const lm = models.models.lm || [];
    const vae = models.models.vae || [];
    const emb = models.models.embedding || [];

    if (dit.length > 0 && (!gp.ditModel || !dit.includes(gp.ditModel))) {
      gp.setDitModel(dit[0]);
    }
    if (lm.length > 0 && (!gp.lmModel || !lm.includes(gp.lmModel))) {
      gp.setLmModel(lm[0]);
    }
    if (vae.length > 0 && (!gp.vaeModel || !vae.includes(gp.vaeModel))) {
      gp.setVaeModel(vae[0]);
    }
    if (emb.length > 0 && (!gp.embeddingModel || !emb.includes(gp.embeddingModel))) {
      gp.setEmbeddingModel(emb[0]);
    }
  }, [models]);

  const ditModels = models?.models?.dit || [];
  const lmModels = models?.models?.lm || [];
  const vaeModels = models?.models?.vae || [];
  const embeddingModels = models?.models?.embedding || [];

  // Format-type filter (gguf / safetensors / onnx). Persisted across sessions.
  // Only affects which options the dropdowns show — not auto-selection, so a
  // selected model stays selected even when its format is filtered out.
  const [formatFilter, setFormatFilter] = useState<Record<ModelFormat, boolean>>(loadFormatFilter);
  useEffect(() => {
    try { localStorage.setItem(FORMAT_FILTER_KEY, JSON.stringify(formatFilter)); } catch { /* ignore */ }
  }, [formatFilter]);

  // Only surface chips for formats that actually exist across the available models.
  const presentFormats = new Set<ModelFormat>(
    [...ditModels, ...lmModels, ...vaeModels, ...embeddingModels].map(getModelFormat)
  );
  const chipFormats = FORMAT_ORDER.filter((f) => presentFormats.has(f));

  const byFormat = (list: string[]) => list.filter((m) => formatFilter[getModelFormat(m)]);
  const toggleFormat = (f: ModelFormat) =>
    setFormatFilter((prev) => ({ ...prev, [f]: !prev[f] }));

  return (
    <div className="space-y-3">
      {/* Format-type filter — toggles which model formats appear in the dropdowns */}
      {chipFormats.length > 1 && (
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider mr-0.5">Format</span>
          {chipFormats.map((f) => {
            const active = formatFilter[f];
            return (
              <button
                key={f}
                type="button"
                onClick={() => toggleFormat(f)}
                aria-pressed={active}
                title={active ? `Hide ${FORMAT_LABELS[f]} models` : `Show ${FORMAT_LABELS[f]} models`}
                className={`px-2 py-0.5 rounded-md text-[10px] font-semibold uppercase tracking-wide border transition-colors
                  ${active
                    ? 'bg-pink-500/15 text-pink-400 border-pink-500/30'
                    : 'bg-transparent text-zinc-500 border-zinc-300 dark:border-white/10 hover:text-zinc-400'}`}
              >
                {FORMAT_LABELS[f]}
              </button>
            );
          })}
        </div>
      )}

      {/* DiT Model */}
      <div>
        <ParamLabel label={t('models.ditModel')} info={getDitModelDescription(gp.ditModel)} rootClassName="flex mb-1.5"
          className="text-xs font-medium text-zinc-500 uppercase tracking-wider" />
        <ModelSelect
          id="dit-model-select"
          value={gp.ditModel}
          onChange={gp.setDitModel}
          options={byFormat(ditModels)}
          formatLabel={formatDitModel}
          placeholder={t('common.loading')}
        />
      </div>

      {/* LM Model */}
      <div>
        <ParamLabel label={t('models.lmModel')} info={getLmModelDescription(gp.lmModel)} rootClassName="flex mb-1.5"
          className="text-xs font-medium text-zinc-500 uppercase tracking-wider" />
        <ModelSelect
          id="lm-model-select"
          value={gp.lmModel}
          onChange={gp.setLmModel}
          options={byFormat(lmModels)}
          formatLabel={formatLmModel}
          placeholder={t('common.loading')}
        />
      </div>

      {/* Planner Adapter (LM) moved to the Adapters dropdown, alongside the
          DiT adapters it pairs with. */}

      {/* VAE Model — only show when multiple VAEs are available */}
      {vaeModels.length > 1 && (
        <div>
          <ParamLabel label={t('models.vaeDecoder')} info={getVaeModelDescription(gp.vaeModel)} rootClassName="flex mb-1.5"
            className="text-xs font-medium text-zinc-500 uppercase tracking-wider" />
          <ModelSelect
            id="vae-model-select"
            value={gp.vaeModel}
            onChange={gp.setVaeModel}
            options={byFormat(vaeModels)}
            formatLabel={formatVaeModel}
            placeholder={t('common.loading')}
          />
        </div>
      )}

      {/* Text Encoder — only show when multiple are available */}
      {embeddingModels.length > 1 && (
        <div>
          <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">{t('models.textEncoder')}</label>
          <ModelSelect
            id="embedding-model-select"
            value={gp.embeddingModel}
            onChange={gp.setEmbeddingModel}
            options={byFormat(embeddingModels)}
            formatLabel={formatEmbeddingModel}
            placeholder={t('common.loading')}
          />
        </div>
      )}

      {/* Get More Models */}
      <div className="border-t border-zinc-200 dark:border-white/5 pt-3 mt-1">
        <button
          onClick={() => setShowModelManager(true)}
          className="w-full px-3 py-2 rounded-xl bg-pink-500/10 border border-pink-500/20
                     text-sm text-pink-400 hover:bg-pink-500/20 hover:text-pink-300
                     transition-colors flex items-center justify-center gap-2"
        >
          <Download size={14} />
          {t('models.getMoreModels')}
        </button>
      </div>

      {/* Model Manager Modal */}
      {showModelManager && (
        <ModelManagerModal onClose={() => {
          setShowModelManager(false);
          sessionStorage.setItem('mm-auto-dismissed', '1');
        }} />
      )}
    </div>
  );
};

/** Summary badge for the Models section */
export const ModelsBadge: React.FC = () => {
  const { ditModel, lmModel, vaeModel } = useGlobalParams();

  return (
    <span className="text-[10px] text-zinc-500 font-mono truncate">
      {formatDitModel(ditModel)} · {formatLmModel(lmModel)} · {formatVaeModel(vaeModel)}
    </span>
  );
};
