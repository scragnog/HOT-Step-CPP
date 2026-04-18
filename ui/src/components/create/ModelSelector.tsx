// ModelSelector.tsx — Model selection dropdowns + adapter group scale controls
// Ported to Tailwind styling matching hot-step-9000.

import React, { useEffect, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { modelApi } from '../../services/api';
import type { AceModels } from '../../types';

interface AdapterGroupScales {
  self_attn: number;
  cross_attn: number;
  mlp: number;
  cond_embed: number;
}

interface ModelSelectorProps {
  ditModel: string;
  onDitModelChange: (v: string) => void;
  lmModel: string;
  onLmModelChange: (v: string) => void;
  vaeModel: string;
  onVaeModelChange: (v: string) => void;
  adapter: string;
  onAdapterChange: (v: string) => void;
  adapterScale: number;
  onAdapterScaleChange: (v: number) => void;
  adapterGroupScales: AdapterGroupScales;
  onAdapterGroupScalesChange: (v: AdapterGroupScales) => void;
  adapterMode: string;
  onAdapterModeChange: (v: string) => void;
}

const GROUP_INFO = [
  { key: 'self_attn' as const,  label: 'Self-Attn',     help: 'How audio frames relate to each other over time' },
  { key: 'cross_attn' as const, label: 'Cross-Attn',    help: 'How strongly your text prompt shapes the output' },
  { key: 'mlp' as const,        label: 'MLP',           help: 'Timbre, tonal texture, and sonic character' },
  { key: 'cond_embed' as const, label: 'Conditioning',  help: 'How the adapter reshapes text/style interpretation' },
];

const selectClasses = "w-full px-3 py-2 rounded-xl bg-zinc-900 border border-white/10 text-sm text-zinc-200 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors cursor-pointer";

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  ditModel, onDitModelChange, lmModel, onLmModelChange,
  vaeModel, onVaeModelChange,
  adapter, onAdapterChange, adapterScale, onAdapterScaleChange,
  adapterGroupScales, onAdapterGroupScalesChange,
  adapterMode, onAdapterModeChange,
}) => {
  const [models, setModels] = useState<AceModels | null>(null);
  const [showGroupScales, setShowGroupScales] = useState(false);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    modelApi.list()
      .then(setModels)
      .catch(() => {});
  }, []);

  const ditModels = models?.models?.dit || [];
  const lmModels = models?.models?.lm || [];
  const vaeModels = models?.models?.vae || [];
  const adapters = models?.adapters || [];

  const handleGroupScaleChange = (key: keyof AdapterGroupScales, value: number) => {
    onAdapterGroupScalesChange({ ...adapterGroupScales, [key]: value });
  };

  const allDefault = GROUP_INFO.every(g => adapterGroupScales[g.key] === 1.0);

  return (
    <div className="space-y-1 pt-3 border-t border-white/5">
      {/* Accordion header */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-3 py-2.5 rounded-xl hover:bg-white/5 transition-colors"
      >
        <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Models & Adapters</span>
        <ChevronDown size={14} className={`text-zinc-500 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="px-3 pb-3 space-y-3">
          {/* DiT Model */}
          <div>
            <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">DiT Model</label>
            <select className={selectClasses} value={ditModel}
              onChange={e => onDitModelChange(e.target.value)}>
              {ditModels.length === 0 && <option value="">Loading...</option>}
              {ditModels.map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>

          {/* LM Model */}
          <div>
            <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">LM Model</label>
            <select className={selectClasses} value={lmModel}
              onChange={e => onLmModelChange(e.target.value)}>
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
              <select className={selectClasses} value={vaeModel}
                onChange={e => onVaeModelChange(e.target.value)}>
                {vaeModels.map(m => (
                  <option key={m} value={m}>{m.replace(/-BF16\.gguf$/, '')}</option>
                ))}
              </select>
            </div>
          )}

          {/* Adapter */}
          {adapters.length > 0 && (
            <>
              <div>
                <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Adapter (LoRA)</label>
                <select className={selectClasses} value={adapter}
                  onChange={e => onAdapterChange(e.target.value)}>
                  <option value="">None</option>
                  {adapters.map(a => (
                    <option key={a} value={a}>{a}</option>
                  ))}
                </select>
              </div>

              {adapter && (
                <>
                  {/* Adapter Scale */}
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <label className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Adapter Scale</label>
                      <span className="text-xs text-zinc-400 font-mono">{adapterScale.toFixed(2)}</span>
                    </div>
                    <input type="range" value={adapterScale}
                      onChange={e => onAdapterScaleChange(parseFloat(e.target.value))}
                      min={0} max={2} step={0.05} className="w-full" />
                  </div>

                  {/* Adapter Loading Mode */}
                  <div>
                    <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Loading Mode</label>
                    <div className="flex rounded-xl overflow-hidden border border-white/10">
                      <button
                        onClick={() => onAdapterModeChange('merge')}
                        className={`flex-1 px-3 py-1.5 text-xs font-medium transition-colors ${
                          adapterMode === 'merge'
                            ? 'bg-zinc-700 text-white'
                            : 'bg-zinc-900 text-zinc-500 hover:text-zinc-300'
                        }`}
                      >
                        Merge
                      </button>
                      <button
                        onClick={() => onAdapterModeChange('runtime')}
                        className={`flex-1 px-3 py-1.5 text-xs font-medium transition-colors ${
                          adapterMode === 'runtime'
                            ? 'bg-pink-600 text-white'
                            : 'bg-zinc-900 text-zinc-500 hover:text-zinc-300'
                        }`}
                      >
                        Runtime LoRA ⚡
                      </button>
                    </div>
                    <p className="text-[10px] text-zinc-600 mt-1">
                      {adapterMode === 'runtime'
                        ? 'Fast: keeps base weights quantized, applies adapter at inference (~5s load)'
                        : 'Classic: merges adapter into weights (slow for K-quant models, ~127s)'}
                    </p>
                  </div>

                  {/* Group Scales Toggle */}
                  <button
                    onClick={() => setShowGroupScales(!showGroupScales)}
                    className="flex items-center gap-2 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
                  >
                    <ChevronDown size={12} className={`transition-transform duration-200 ${showGroupScales ? 'rotate-180' : ''}`} />
                    Group Scales
                    {!allDefault && (
                      <span className="w-1.5 h-1.5 rounded-full bg-pink-500" title="Group scales modified" />
                    )}
                  </button>

                  {/* Group Scale Sliders */}
                  {showGroupScales && (
                    <div className="rounded-xl bg-zinc-900/50 border border-white/5 p-3 space-y-3">
                      {GROUP_INFO.map(({ key, label, help }) => (
                        <div key={key}>
                          <div className="flex items-center justify-between mb-1">
                            <label className="text-xs text-zinc-500" title={help}>{label}</label>
                            <span className={`text-xs font-mono ${
                              adapterGroupScales[key] === 1.0 ? 'text-zinc-600' : 'text-pink-400'
                            }`}>
                              {adapterGroupScales[key].toFixed(2)}
                            </span>
                          </div>
                          <input
                            type="range"
                            value={adapterGroupScales[key]}
                            onChange={e => handleGroupScaleChange(key, parseFloat(e.target.value))}
                            min={0} max={4} step={0.05}
                            className="w-full"
                          />
                        </div>
                      ))}
                      <div className="text-center text-[10px] text-zinc-600 mt-1">
                        Scale changes apply on next generation (DiT reload)
                      </div>
                    </div>
                  )}
                </>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
};
