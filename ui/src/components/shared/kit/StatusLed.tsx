// StatusLed.tsx — a small lit indicator plus its label. State is never colour
// alone: every state carries a label, and warn also carries an icon.
// `generating` is the travelling LED (static under prefers-reduced-motion).

import React from 'react';
import { TriangleAlert } from 'lucide-react';

export type LedState = 'idle' | 'ok' | 'warn' | 'error' | 'generating';

export interface StatusLedProps {
  state: LedState;
  label: string;
  className?: string;
}

const LED_CLASS = { idle: 'kit-led-idle', ok: 'kit-led-ok', error: 'kit-led-error' } as const;

export const StatusLed: React.FC<StatusLedProps> = ({ state, label, className = '' }) => (
  <span className={`inline-flex items-center gap-2 text-xs text-ink-muted ${className}`}>
    {state === 'generating' ? <span className="kit-led-track" aria-hidden="true" />
      : state === 'warn' ? <TriangleAlert size={12} className="shrink-0 text-status-warn" aria-hidden="true" />
      : <span className={`kit-led ${LED_CLASS[state]}`} aria-hidden="true" />}
    <span>{label}</span>
  </span>
);
