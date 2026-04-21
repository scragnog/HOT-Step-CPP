// Toast.tsx — Slide-in notification component
// Ported from hot-step-9000.

import React, { useEffect } from 'react';
import { CheckCircle, AlertTriangle, Info, X, XCircle } from 'lucide-react';

export type ToastType = 'success' | 'error' | 'info' | 'warning';

interface ToastProps {
  message: string;
  type: ToastType;
  isVisible: boolean;
  onClose: () => void;
  duration?: number;
}

const icons: Record<ToastType, React.ReactNode> = {
  success: <CheckCircle size={18} className="text-green-400" />,
  error: <XCircle size={18} className="text-red-400" />,
  warning: <AlertTriangle size={18} className="text-yellow-400" />,
  info: <Info size={18} className="text-blue-400" />,
};

const borderColors: Record<ToastType, string> = {
  success: 'border-green-500/30',
  error: 'border-red-500/30',
  warning: 'border-yellow-500/30',
  info: 'border-blue-500/30',
};

export const Toast: React.FC<ToastProps> = ({ message, type, isVisible, onClose, duration = 2500 }) => {
  const onCloseRef = React.useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    if (!isVisible) return;
    const timer = setTimeout(() => onCloseRef.current(), duration);
    return () => clearTimeout(timer);
  }, [isVisible, duration]);

  if (!isVisible) return null;

  return (
    <div className="fixed bottom-24 left-1/2 -translate-x-1/2 z-[100]">
      <div className={`
        toast-in flex items-center gap-3 px-4 py-3 rounded-xl
        bg-zinc-900/95 backdrop-blur-lg border ${borderColors[type]}
        shadow-xl min-w-[300px] max-w-[500px]
      `}>
        {icons[type]}
        <span className="text-sm text-white flex-1">{message}</span>
        <button onClick={onClose} className="text-zinc-400 hover:text-white transition-colors">
          <X size={14} />
        </button>
      </div>
    </div>
  );
};
