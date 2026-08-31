// EnhancePanel.tsx — optional cloud enhancement steps
//
// Genius lyrics + LLM caption rewrite. Each button is replaced by a hint row
// when its capability is missing — nothing leaves the machine unless pressed.

import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Cloud, Loader2, Sparkles } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { StyledSelect } from '../shared/StyledSelect';
import { useTrainingStore } from '../../stores/trainingStore';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';

interface EnhancePanelProps {
  /** Restrict the run to the grid selection when non-empty. */
  selectedSampleIds: string[];
  disabled: boolean;
}

export const EnhancePanel: React.FC<EnhancePanelProps> = ({ selectedSampleIds, disabled }) => {
  const { t } = useTranslation();
  const caps = useTrainingStore(s => s.capabilities);
  const startGenius = useTrainingStore(s => s.startGenius);
  const startCaption = useTrainingStore(s => s.startCaption);

  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState<'genius' | 'caption' | null>(null);
  const [provider, setProvider] = useState('');
  const [model, setModel] = useState('');

  // MOSS is not an LLM provider — no API key, no model list — but it belongs in
  // the same dropdown because it answers the same question the user is asking:
  // who writes the caption? Listed first because it is the only option that
  // actually hears the track rather than rewriting the local analysis.
  //
  // It does NOT become the default just by being available: `defaultProvider`
  // still wins if it resolves, so an existing Gemini setup is left alone. MOSS
  // is only auto-selected when nothing else is configured.
  const mossProvider = caps?.moss.available
    ? { id: 'moss', name: 'MOSS-Music 8B — local, hears the audio', available: true, models: [] as string[], defaultModel: '' }
    : null;
  const providers = [
    ...(mossProvider ? [mossProvider] : []),
    ...(caps?.llm.providers.filter(p => p.available) ?? []),
  ];
  const activeProvider = providers.find(p => p.id === (provider || caps?.llm.defaultProvider)) ?? providers[0];
  const isMoss = activeProvider?.id === 'moss';

  const scoped = selectedSampleIds.length > 0 ? { sampleIds: selectedSampleIds } : {};

  const runGenius = async () => {
    setBusy('genius');
    try { await startGenius({ ...scoped }); } finally { setBusy(null); }
  };

  const runCaption = async () => {
    setBusy('caption');
    try {
      await startCaption({
        ...scoped,
        provider: activeProvider?.id,
        model: model || activeProvider?.defaultModel,
      });
    } finally { setBusy(null); }
  };

  const btn = 'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors disabled:opacity-40 disabled:cursor-not-allowed';

  return (
    <div className={CARD}>
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2 text-left"
      >
        {open ? <ChevronDown size={14} className="text-zinc-500" /> : <ChevronRight size={14} className="text-zinc-500" />}
        <Cloud size={15} className="text-sky-500" />
        <span className="text-sm font-bold text-zinc-900 dark:text-white flex-1">{t('trainingStudio.enhance.title')}</span>
      </button>

      {open && (
        <div className="mt-3 flex flex-col gap-4">
          <p className="text-xs text-zinc-600 dark:text-zinc-400">{t('trainingStudio.enhance.subtitle')}</p>

          {/* Genius */}
          <div className="flex flex-col gap-1.5">
            {caps?.genius.configured ? (
              <>
                <button
                  onClick={() => void runGenius()}
                  disabled={disabled || busy !== null}
                  className={`${btn} self-start bg-pink-500/10 border border-pink-500/20 text-pink-500 hover:bg-pink-500/20`}
                >
                  {busy === 'genius' ? <Loader2 size={13} className="animate-spin" /> : <Sparkles size={13} />}
                  {t('trainingStudio.enhance.genius')}
                </button>
                <span className="text-[11px] text-zinc-500">{t('trainingStudio.enhance.geniusHint')}</span>
              </>
            ) : (
              <div className="text-[11px] text-zinc-500 px-3 py-2 rounded-lg border border-zinc-200 dark:border-white/5 bg-zinc-50 dark:bg-black/20">
                {t('trainingStudio.enhance.geniusMissing')}
              </div>
            )}
          </div>

          {/* LLM caption */}
          <div className="flex flex-col gap-1.5">
            {/* MOSS needs no credentials, so `llm.configured` must not gate the
                whole block — otherwise the one provider that works offline is
                hidden precisely when no API key is set. */}
            {(caps?.llm.configured || caps?.moss.available) && activeProvider ? (
              <>
                <div className="flex items-end gap-2 flex-wrap">
                  <div className="flex flex-col gap-1 min-w-40">
                    <span className="text-[11px] font-semibold text-zinc-600 dark:text-zinc-400">{t('trainingStudio.enhance.provider')}</span>
                    <StyledSelect
                      accent="amber"
                      size="sm"
                      value={activeProvider.id}
                      onChange={(v) => { setProvider(v); setModel(''); }}
                      options={providers.map(p => ({ value: p.id, label: p.name }))}
                    />
                    {/* Same reason as LabelPanel: the server knows whether the
                        binary or the weights are missing, and silently dropping
                        that leaves the user hunting for a provider that cannot
                        appear. */}
                    {!caps?.moss.available && caps?.moss.missing && (
                      <span className="text-[10px] text-zinc-500">
                        MOSS (local) unavailable: {caps.moss.missing}
                      </span>
                    )}
                  </div>
                  {/* MOSS is one fixed local model — an empty picker beside it
                      reads as "still loading", so it is omitted entirely. */}
                  {!isMoss && (
                    <div className="flex flex-col gap-1 min-w-48">
                      <span className="text-[11px] font-semibold text-zinc-600 dark:text-zinc-400">{t('trainingStudio.enhance.model')}</span>
                      <StyledSelect
                        accent="amber"
                        size="sm"
                        value={model || activeProvider.defaultModel}
                        onChange={(v) => setModel(v)}
                        options={activeProvider.models.map(m => ({ value: m, label: m }))}
                        searchPlaceholder="Filter models…"
                      />
                    </div>
                  )}
                  <button
                    onClick={() => void runCaption()}
                    disabled={disabled || busy !== null}
                    className={`${btn} bg-violet-500/10 border border-violet-500/20 text-violet-500 hover:bg-violet-500/20`}
                  >
                    {busy === 'caption' ? <Loader2 size={13} className="animate-spin" /> : <Sparkles size={13} />}
                    {t('trainingStudio.enhance.caption')}
                  </button>
                </div>
                <span className="text-[11px] text-zinc-500">{t('trainingStudio.enhance.captionHint')}</span>
              </>
            ) : (
              <div className="text-[11px] text-zinc-500 px-3 py-2 rounded-lg border border-zinc-200 dark:border-white/5 bg-zinc-50 dark:bg-black/20">
                {t('trainingStudio.enhance.captionMissing')}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default EnhancePanel;
