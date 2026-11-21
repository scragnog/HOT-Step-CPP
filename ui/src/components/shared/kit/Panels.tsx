// Panels.tsx — the big containers: SectionPanel, Well, Module (+ Rail), SongItem,
// EmptyState. All styling is the depth recipe in index.css (.kit-section,
// .kit-well, .kit-module, .kit-item); nothing here sets colours.
// See docs/dev/ui-design.md.

import React from 'react';
import { StatusLed, type StatusLedProps } from './StatusLed';

export interface SectionPanelProps extends Omit<React.HTMLAttributes<HTMLElement>, 'title'> {
  title: React.ReactNode;
  /** A StatusLed in the header, after the title. */
  led?: StatusLedProps;
  /** Controls pinned to the right of the header strip. */
  actions?: React.ReactNode;
  footer?: React.ReactNode;
  /** Classes for the body (padding, layout, scrolling). */
  bodyClassName?: string;
}

/** Chassis-level container: header strip, body, optional footer. */
export const SectionPanel: React.FC<SectionPanelProps> = ({ title, led, actions, footer, bodyClassName = '', className = '', children, ...rest }) => (
  <section className={`kit-section ${className}`} {...rest}>
    <header className="kit-section-head">
      <h2 className="kit-section-title">{title}</h2>
      {led && <StatusLed {...led} />}
      {actions && <div className="ml-auto flex items-center gap-2">{actions}</div>}
    </header>
    <div className={`kit-section-body ${bodyClassName}`}>{children}</div>
    {footer && <footer className="kit-section-foot">{footer}</footer>}
  </section>
);

/** Recessed area: lyrics box, waveform, log. Darker at the top edge. */
export const Well: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ className = '', ...rest }) => (
  <div className={`kit-well kit-well-block ${className}`} {...rest} />
);

/** Divider rail: horizontal between rows, vertical between controls. */
export const Rail: React.FC<{ vertical?: boolean; className?: string }> = ({ vertical = false, className = '' }) => (
  <div role="separator" aria-orientation={vertical ? 'vertical' : 'horizontal'} className={`${vertical ? 'kit-rail-v' : 'kit-rail-h'} ${className}`} />
);

export interface ModuleProps extends Omit<React.HTMLAttributes<HTMLElement>, 'title'> {
  /** The engraved label. */
  label: string;
}

/** A group of controls under an engraved label, like a synth front-panel section. */
export const Module: React.FC<ModuleProps> = ({ label, className = '', children, ...rest }) => (
  <section aria-label={label} className={`kit-module ${className}`} {...rest}>
    <div className="kit-module-label">{label}</div>
    <Rail />
    <div className="p-3">{children}</div>
  </section>
);

export interface SongItemProps extends Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, 'title'> {
  title: string;
  meta?: string;
  duration?: string;
  /** Any CSS gradient, for the cover thumbnail. */
  cover?: string;
  /** The current selection. */
  selected?: boolean;
  /** Currently playing: the title takes the tint and the LED lights. */
  playing?: boolean;
}

/** A row in a song list. States: rest, hover, selected, playing. */
export const SongItem: React.FC<SongItemProps> = ({ title, meta, duration, cover, selected = false, playing = false, className = '', ...rest }) => (
  <button
    type="button"
    aria-current={selected || undefined}
    data-playing={playing || undefined}
    className={`kit-item kit-focus ${className}`}
    {...rest}
  >
    <span className="kit-thumb" style={{ background: cover ?? 'linear-gradient(135deg, rgb(var(--tint-hover)), rgb(var(--tint-deep)))' }} aria-hidden="true" />
    <span className="min-w-0 flex-1">
      <span className="kit-item-title block truncate text-sm font-medium">{title}</span>
      {meta && <span className="block truncate text-xs text-ink-muted">{meta}</span>}
    </span>
    {playing && <StatusLed state="generating" label="Playing" className="[&>span:last-child]:sr-only" />}
    {duration && <span className="kit-mono shrink-0 text-xs text-ink-muted">{duration}</span>}
  </button>
);

export interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  children?: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}

/** What a panel shows when it has nothing in it. */
export const EmptyState: React.FC<EmptyStateProps> = ({ icon, title, children, action, className = '' }) => (
  <div className={`flex flex-col items-center justify-center gap-2 px-6 py-10 text-center ${className}`}>
    {icon && <div className="kit-well mb-1 flex h-12 w-12 items-center justify-center rounded-full text-ink-muted">{icon}</div>}
    <div className="kit-heading text-sm font-semibold text-ink">{title}</div>
    {children && <p className="max-w-xs text-xs text-ink-muted">{children}</p>}
    {action && <div className="mt-2">{action}</div>}
  </div>
);
