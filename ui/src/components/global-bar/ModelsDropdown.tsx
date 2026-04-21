// ModelsDropdown.tsx — Model selection UI for the global param bar
//
// Adapted from create/ModelSelector.tsx to read from GlobalParamsContext.

import React, { useEffect, useState } from 'react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { modelApi } from '../../services/api';
import type { AceModels } from '../../types';

const selectClasses = "w-full px-3 py-2 rounded-xl bg-zinc-800 border border-white/10 text-sm text-zinc-200 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors cursor-pointer";

export const ModelsDropdown: React.FC = () => {
  const gp = useGlobalParams();
  const [models, setModels] = useState<AceModels | null>(null);

  useEffect(() => {
    modelApi.list()
      .then(setModels)
      .catch(() => {});
  }, []);

  const ditModels = models?.models?.dit || [];
  const lmModels = models?.models?.lm || [];
  const vaeModels = models?.models?.vae || [];

  return (
    <div className="space-y-3">
      {/* DiT Model */}
      <div>
        <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">DiT Model</label>
        <select className={selectClasses} value={gp.ditModel}
          onChange={e => gp.setDitModel(e.target.value)}>
          {ditModels.length === 0 && <option value="">Loading...</option>}
          {ditModels.map(m => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </div>

      {/* LM Model */}
      <div>
        <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">LM Model</label>
        <select className={selectClasses} value={gp.lmModel}
          onChange={e => gp.setLmModel(e.target.value)}>
          {lmModels.length === 0 && <option value="">Loading...</option>}
          {lmModels.map(m => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </div>

      {/* VAE Model — only show when multiple VAEs are available */}
      {vaeModels.length > 1 && (
        <div>
          <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">VAE Decoder</label>
          <select className={selectClasses} value={gp.vaeModel}
            onChange={e => gp.setVaeModel(e.target.value)}>
            {vaeModels.map(m => (
              <option key={m} value={m}>{m.replace(/-BF16\.gguf$/, '')}</option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
};

/** Summary badge for the Models section */
export const ModelsBadge: React.FC = () => {
  const { ditModel, lmModel } = useGlobalParams();
  const ditShort = ditModel ? ditModel.replace(/\.gguf$/i, '').split(/[-_]/).slice(0, 3).join('-') : '—';
  const lmShort = lmModel ? lmModel.replace(/\.gguf$/i, '').split(/[-_]/).slice(0, 2).join('-') : '—';

  return (
    <span className="text-[10px] text-zinc-500 font-mono truncate">
      {ditShort} · {lmShort}
    </span>
  );
};
