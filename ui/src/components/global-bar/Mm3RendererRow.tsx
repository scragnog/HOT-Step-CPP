// Mm3RendererRow.tsx — GGML / TensorRT renderer picker for the MM3 Flow DiT.
//
// Lives under the Flow DiT bucket in BackendModelsDropdown (plan
// docs/plans/2026-09-11-mm3-trt-dit-shipping.md, "UI" section): the renderer
// pairs with the DiT quant choice right above it, not with the unrelated
// generation knobs in the Generation dropdown, which is where this control
// used to render (as the now-removed `mm3DitBackend` backend extension —
// see the NOTE in server/src/services/backends/minimax/index.ts).
//
// Source of truth stays `backendParams.mm3DitBackend` (globalParamsStore),
// the same generic bag the removed extension wrote to — so generate.ts and
// mapMinimaxParams() need no change. This component just writes the same key
// from a new spot.

import React from 'react';
import { Download } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { StyledSelect } from '../shared/StyledSelect';
import { ParamLabel } from '../shared/ParamLabel';

/** Mirrors server/src/services/backends/minimax/client.ts Mm3DitRuntime.
 *  Every field beyond the original four is optional: an engine build from
 *  before TensorRT shipped simply omits them. */
export interface Mm3DitRuntime {
  supported: boolean;
  available: boolean;
  backend: 'ggml' | 'tensorrt' | string;
  reason: string;
  gpu_mb?: number;
  sm?: number;
  runtime?: { nvinfer: boolean; parser: boolean; builder_resource: boolean };
  assets?: { onnx: boolean; manifest: boolean; engine: boolean };
  needs_build?: boolean;
}

interface Props {
  ditRuntime: Mm3DitRuntime | undefined;
  value: 'ggml' | 'tensorrt';
  onChange: (v: 'ggml' | 'tensorrt') => void;
  /** TensorRT precision (backendParams.mm3DitPrecision -> request dit_precision). */
  precision: 'bf16' | 'fp8';
  onPrecisionChange: (v: 'bf16' | 'fp8') => void;
  onOpenModelManager: () => void;
}

export const Mm3RendererRow: React.FC<Props> = ({ ditRuntime, value, onChange, precision, onPrecisionChange, onOpenModelManager }) => {
  const { t } = useTranslation();
  const trtAvailable = ditRuntime?.available === true;
  // reason is engine-supplied prose (Mm3DitRuntime.reason) and stays as-is —
  // it names the actual failing tier (missing DLL, no CUDA device, etc.) and
  // translating it would require the engine to speak i18n keys. The fallback
  // below (no reason string at all, e.g. an older engine build) IS ours, so
  // that one goes through t().
  const reason = ditRuntime?.reason || (ditRuntime ? t('globalBar.mm3RendererUnavailableFallback') : '');

  return (
    <div className="mt-2">
      <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
        {t('globalBar.mm3RendererLabel')}
      </label>
      <div className="flex rounded-xl border border-zinc-300 dark:border-white/10 overflow-hidden">
        <button
          type="button"
          onClick={() => onChange('ggml')}
          className={`flex-1 px-3 py-2 text-sm transition-colors ${
            value === 'ggml'
              ? 'bg-pink-500/15 text-pink-400'
              : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200 dark:hover:bg-zinc-700'
          }`}
        >
          {t('globalBar.mm3RendererGgml')}
        </button>
        <button
          type="button"
          onClick={() => trtAvailable && onChange('tensorrt')}
          disabled={!trtAvailable}
          title={trtAvailable ? undefined : reason}
          className={`flex-1 px-3 py-2 text-sm border-l border-zinc-300 dark:border-white/10 transition-colors ${
            value === 'tensorrt' && trtAvailable
              ? 'bg-pink-500/15 text-pink-400'
              : trtAvailable
                ? 'bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200 dark:hover:bg-zinc-700'
                : 'bg-zinc-100/50 dark:bg-zinc-800/50 text-zinc-400 dark:text-zinc-600 cursor-not-allowed'
          }`}
        >
          {t('globalBar.mm3RendererTensorrt')}
        </button>
      </div>

      {trtAvailable ? (
        value === 'tensorrt' && (
          <>
            {ditRuntime?.needs_build && (
              <p className="mt-1.5 text-[10px] text-amber-500 dark:text-amber-400 leading-relaxed">
                {t('globalBar.mm3RendererNeedsBuild')}
              </p>
            )}
            <div className="mt-2">
              <ParamLabel
                label={t('globalBar.mm3PrecisionLabel', 'TensorRT precision')}
                className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5"
                info={t('globalBar.mm3PrecisionInfo')}
              />
              <StyledSelect
                accent="pink"
                value={precision}
                onChange={onPrecisionChange}
                options={[
                  { value: 'fp8', label: t('globalBar.mm3PrecisionFp8', 'FP8 (default)') },
                  { value: 'bf16', label: t('globalBar.mm3PrecisionBf16', 'BF16') },
                ]}
                className="w-full"
              />
            </div>
          </>
        )
      ) : (
        <div className="mt-1.5 space-y-1.5">
          {reason && (
            <p className="text-[10px] text-zinc-500 leading-relaxed">{reason}</p>
          )}
          <button
            type="button"
            onClick={onOpenModelManager}
            className="inline-flex items-center gap-1.5 text-[11px] text-pink-400 hover:text-pink-300 transition-colors"
          >
            <Download size={11} />
            {t('globalBar.mm3RendererGetTensorrt')}
          </button>
        </div>
      )}
    </div>
  );
};
