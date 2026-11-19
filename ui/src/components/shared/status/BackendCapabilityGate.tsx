// BackendCapabilityGate.tsx — full-body guard for task-mode studios that not
// every backend supports (docs/plans/multi-backend-architecture.md §4.5,
// §2 principle 2: "capabilities, not conditionals").
//
// Gates purely on useCapabilities().capabilities?.features.<flag> — NEVER on
// backend id. While capabilities are undefined/loading, renders children (ACE
// behavior) so a studio never flash-hides its body on mount before the first
// /api/capabilities response lands.

import React from 'react';
import { Layers } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useCapabilities } from '@/hooks/useCapabilities';
import type { BackendFeatureCapabilities } from '@/stores/backendStore';

interface BackendCapabilityGateProps {
  /** Key into BackendCapabilities.features that this studio requires. */
  feature: keyof BackendFeatureCapabilities;
  children: React.ReactNode;
}

export const BackendCapabilityGate: React.FC<BackendCapabilityGateProps> = ({ feature, children }) => {
  const { t } = useTranslation();
  const { capabilities } = useCapabilities();

  // capabilities === null → still loading / never fetched → default to
  // showing the studio body (matches ACE's always-on behavior today).
  if (capabilities && !capabilities.features[feature]) {
    return (
      <div className="h-full flex items-center justify-center p-8">
        <div className="max-w-md flex flex-col items-center gap-3 text-center rounded-xl border border-amber-500/25 bg-amber-500/10 px-6 py-8">
          <Layers size={22} className="text-amber-500 dark:text-amber-400 flex-shrink-0" />
          <p className="text-sm text-amber-700 dark:text-amber-400">
            {t('backendGate.notSupported')}
          </p>
        </div>
      </div>
    );
  }

  return <>{children}</>;
};
