// DiffTrace.tsx - small canvas line chart of the active take vs the baseline:
// loudness delta (LU) and log-spectral distance (dB) per segment.

import React, { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { CompareDiff } from '@/services/compareApi';
import { ParamLabel } from '@/components/shared/controls/ParamLabel';

interface Props { diff: CompareDiff | null; activeTitle: string; baselineTitle: string; same: boolean; }

export const DiffTrace: React.FC<Props> = ({ diff, activeTitle, baselineTitle, same }) => {
  const { t } = useTranslation();
  const ref = useRef<HTMLCanvasElement | null>(null);
  const box = useRef<HTMLDivElement | null>(null);
  const [w, setW] = useState(0);
  const H = 96;

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    setW(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const c = ref.current;
    if (!c || !w) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.floor(w * dpr); c.height = Math.floor(H * dpr);
    const g = c.getContext('2d');
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, H);
    g.strokeStyle = 'rgba(161,161,170,0.35)';
    g.beginPath(); g.moveTo(0, H / 2); g.lineTo(w, H / 2); g.stroke();
    if (!diff || same) return;
    const series = (vals: (number | null)[], color: string, symmetric: boolean) => {
      const nums = vals.filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
      if (nums.length < 2) return;
      const m = Math.max(1, ...nums.map(Math.abs));
      g.strokeStyle = color; g.lineWidth = 1.5; g.beginPath();
      let started = false;
      vals.forEach((v, i) => {
        if (typeof v !== 'number' || !Number.isFinite(v)) { started = false; return; }
        const x = (i / (vals.length - 1)) * w;
        const y = symmetric ? H / 2 - (v / m) * (H / 2 - 4) : H - 4 - (v / m) * (H - 8);
        if (started) g.lineTo(x, y); else { g.moveTo(x, y); started = true; }
      });
      g.stroke();
    };
    series(diff.logSpectralDistanceDb, 'rgba(56,189,248,0.9)', false);
    series(diff.loudnessDeltaLu, 'rgba(244,114,182,0.9)', true);
  }, [diff, w, same]);

  const maxAbs = (vals: (number | null)[]) => {
    const n = vals.filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
    return n.length ? Math.max(...n.map(Math.abs)) : null;
  };
  const lu = diff ? maxAbs(diff.loudnessDeltaLu) : null;
  const lsd = diff ? maxAbs(diff.logSpectralDistanceDb) : null;

  return (
    <div className="rounded-xl border border-zinc-200 dark:border-white/5 bg-white/60 dark:bg-zinc-900/40 p-3">
      <div className="flex flex-wrap items-center gap-2 mb-2 text-xs">
        <ParamLabel
          label={t('compare.diffTitle', 'Difference trace')}
          info={t('compare.diffInfo', 'Active take against the baseline over time. Pink: loudness difference in LU (centre line is equal). Blue: log-spectral distance in dB, higher means the tone differs more. Flat lines mean the takes sound alike there.')}
        />
        <span className="text-zinc-500 truncate">{activeTitle} {t('compare.vs', 'vs')} {baselineTitle}</span>
        {lu !== null && <span className="text-pink-400">{t('compare.maxLu', 'max {{v}} LU', { v: lu.toFixed(1) })}</span>}
        {lsd !== null && <span className="text-sky-400">{t('compare.maxLsd', 'max {{v}} dB LSD', { v: lsd.toFixed(1) })}</span>}
      </div>
      <div ref={box} className="relative w-full">
        <canvas ref={ref} style={{ width: '100%', height: H }} />
        {(same || !diff) && (
          <div className="absolute inset-0 flex items-center justify-center text-xs text-zinc-500">
            {same ? t('compare.diffPickOther', 'Pick a take other than the baseline to see the trace') : t('compare.loading', 'Loading...')}
          </div>
        )}
      </div>
    </div>
  );
};
