// Yue2StageCard.tsx — the shell every YuE2 preparation stage renders in.
//
// The header folds the stage's explanation, status and params away (always
// collapsed on arrival); the run button (`action`) and live progress
// (`footer`) stay outside the fold so a stage can be run without opening it.

import React, { useState } from 'react';
import { Check, ChevronDown, ListChecks } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { CARD } from '@/components/training-studio/lib/yue2CardUi';


export function TrainingBatchCard({ onOpen, description, disabled }: {
  onOpen?: () => void; description: string; disabled?: boolean;
}) {
  const { t } = useTranslation();
  return <div className={CARD}>
    <div className="flex items-center justify-between gap-3">
      <div>
        <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">{t('trainingStudio.yue2.aitkBatch.title', 'Train several datasets')}</h3>
        <p className="text-[11px] text-zinc-500 mt-1">{description}</p>
      </div>
      <button type="button" onClick={onOpen} disabled={disabled} className={`shrink-0 rounded-lg bg-amber-500 px-3 py-2 text-xs font-semibold text-black hover:bg-amber-400${disabled !== undefined ? ' disabled:opacity-40' : ''}`}>
        <ListChecks size={14} className="inline mr-1" />{t('trainingStudio.yue2.aitkBatch.open', 'Train multiple')}
      </button>
    </div>
  </div>;
}

export function TrainingAllStagesCard({ description, hint, onRun, disabled, label, icon, children }: {
  description: string; hint?: React.ReactNode; onRun: () => void; disabled: boolean;
  label?: React.ReactNode; icon?: React.ReactNode; children?: React.ReactNode;
}) {
  const { t } = useTranslation();
  return <div className={CARD}>
    <p className={`text-[11px] text-zinc-500 leading-relaxed ${hint ? 'mb-1' : 'mb-3'}`}>{description}</p>
    {hint && <p className="text-[10px] text-zinc-500 leading-snug mb-3">{hint}</p>}
    <button onClick={onRun} disabled={disabled}
      className="w-full px-4 py-2.5 rounded-lg text-sm font-semibold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2">
      {icon ?? <ListChecks size={15} />}{label ?? t('trainingStudio.yue2.runAllStart', 'Perform all stages')}
    </button>
    {children}
  </div>;
}

/** YuE2's training form stays visible; only its optional settings fold away. */
export function TrainingRunPanel({ title, description, children }: {
  title: React.ReactNode; description?: React.ReactNode; children: React.ReactNode;
}) {
  return <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-5">
    <h3 className="text-sm font-semibold text-amber-700 dark:text-amber-300">{title}</h3>
    {description && <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-2 leading-relaxed">{description}</p>}
    {children}
  </div>;
}

export const Yue2StageCard: React.FC<{
  icon: React.ReactNode;
  title: string;
  done: boolean;
  action: React.ReactNode;
  footer?: React.ReactNode;
  children: React.ReactNode;
}> = ({ icon, title, done, action, footer, children }) => {
  const [open, setOpen] = useState(false);
  return (
    <div className={CARD}>
      <button type="button" onClick={() => setOpen(!open)} className="w-full flex items-center gap-2 text-left group">
        <ChevronDown size={14} className={`text-zinc-400 transition-transform duration-200 ${open ? 'rotate-180' : ''}`} />
        {icon}
        <h3 className="text-sm font-semibold text-zinc-900 dark:text-white group-hover:underline">{title}</h3>
        {done && <span className="flex items-center gap-1 text-[11px] text-emerald-500"><Check size={12} />done</span>}
      </button>
      {open && <div className="mt-3">{children}</div>}
      <div className="mt-3">{action}</div>
      {footer}
    </div>
  );
};
