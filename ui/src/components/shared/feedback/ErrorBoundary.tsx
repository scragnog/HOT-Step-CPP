// ErrorBoundary.tsx — catches a render-time throw so the app shows the error
// instead of going blank.
//
// Without one, React 19 unmounts the whole root on any uncaught render error
// and the tab is left white with nothing in it. main.tsx wraps the app in one;
// App.tsx wraps the studio area in another, keyed on the active view, so a
// broken studio leaves the sidebar and player working and switching studio
// clears the error.

import React from 'react';
import { useTranslation } from 'react-i18next';

interface ErrorBoundaryProps {
  children: React.ReactNode;
  /** Clears the error whenever it changes (the active view, for the studio area). */
  resetKey?: unknown;
}

interface ErrorBoundaryState {
  error: Error | null;
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    console.error('[ErrorBoundary]', error, info.componentStack);
  }

  componentDidUpdate(prev: ErrorBoundaryProps): void {
    if (this.state.error && prev.resetKey !== this.props.resetKey) this.setState({ error: null });
  }

  render(): React.ReactNode {
    if (!this.state.error) return this.props.children;
    return <ErrorFallback error={this.state.error} onRetry={() => this.setState({ error: null })} />;
  }
}

const ErrorFallback: React.FC<{ error: Error; onRetry: () => void }> = ({ error, onRetry }) => {
  const { t } = useTranslation();
  return (
    <div className="flex-1 w-full h-full min-h-[50vh] flex items-center justify-center p-8 bg-white dark:bg-black">
      <div className="max-w-xl w-full bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 rounded-2xl p-6 shadow-2xl">
        <h3 className="text-lg font-bold text-zinc-900 dark:text-white mb-2">
          {t('errorBoundary.title', 'Something went wrong on this page')}
        </h3>
        <p className="text-sm text-zinc-700 dark:text-zinc-300 mb-3">
          {t('errorBoundary.message', 'The page hit an error and stopped drawing. Jobs running on the server are not affected. The full details are in the browser console (F12).')}
        </p>
        <pre className="text-xs text-red-700 dark:text-red-400 bg-zinc-100 dark:bg-zinc-950 rounded-lg p-3 mb-6 whitespace-pre-wrap break-words max-h-48 overflow-auto">
          {error.name}: {error.message}
        </pre>
        <div className="flex items-center justify-end gap-3">
          <button
            onClick={onRetry}
            className="px-4 py-2 rounded-lg bg-zinc-100 dark:bg-zinc-800 text-zinc-800 dark:text-zinc-200 font-semibold hover:bg-zinc-300 dark:hover:bg-zinc-700 transition-colors"
          >
            {t('errorBoundary.retry', 'Try again')}
          </button>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 rounded-lg font-semibold transition-colors bg-pink-600 text-white hover:bg-pink-500"
          >
            {t('errorBoundary.reload', 'Reload page')}
          </button>
        </div>
      </div>
    </div>
  );
};
