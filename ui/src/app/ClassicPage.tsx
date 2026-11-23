// ClassicPage.tsx — a classic page inside a shell's own frame, for the pages a shell has not
// built yet. Lazy: the page's code downloads only when the shell renders it. Only pages that
// take no props are listed; Library and Create need App.tsx handlers, so a shell uses
// openInStudio() to reload into classic for those.

import React, { Suspense, lazy } from 'react';
import { ErrorBoundary } from '../components/shared/ErrorBoundary';

const PAGES = {
  'training-studio': lazy(() => import('../components/training-studio/TrainingStudio').then(m => ({ default: m.TrainingStudio }))),
} satisfies Record<string, React.ComponentType>;

export type ClassicPageView = keyof typeof PAGES;

export const ClassicPage: React.FC<{ view: ClassicPageView; className?: string }> = ({ view, className }) => {
  const Page = PAGES[view];
  // Classic pages style themselves with Tailwind classes and `dark:`, which follow <html>.
  // `contain` keeps their absolutely positioned bits inside this box.
  return (
    <div className={`classic-page relative flex flex-col overflow-hidden bg-white dark:bg-black text-zinc-900 dark:text-zinc-100 ${className ?? ''}`} style={{ contain: 'layout paint' }}>
      <ErrorBoundary>
        <Suspense fallback={null}><Page /></Suspense>
      </ErrorBoundary>
    </div>
  );
};
