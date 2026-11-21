// Panel.tsx — Panel (a titled surface) and Card (a raised tile inside one).
// Styles live in index.css (.kit-panel, .kit-card).

import React, { useState } from 'react';
import { ChevronDown } from 'lucide-react';

export interface PanelProps extends Omit<React.HTMLAttributes<HTMLElement>, 'title'> {
  /** Header text. Omit both title and actions for a headerless panel. */
  title?: React.ReactNode;
  /** Controls pinned to the right of the header. */
  actions?: React.ReactNode;
  /** The header toggles the body open and closed. */
  collapsible?: boolean;
  defaultOpen?: boolean;
}

export const Panel: React.FC<PanelProps> = ({ title, actions, collapsible = false, defaultOpen = true, className = '', children, ...rest }) => {
  const [open, setOpen] = useState(defaultOpen);
  const shown = !collapsible || open;
  return (
    <section className={`kit-panel ${className}`} {...rest}>
      {(title || actions) && (
        <header className="kit-panel-header">
          {collapsible ? (
            <button type="button" aria-expanded={open} onClick={() => setOpen(o => !o)} className="kit-focus flex items-center gap-1.5 rounded-ctl uppercase tracking-wider">
              <ChevronDown size={14} className={`transition-transform duration-glow ${open ? '' : '-rotate-90'}`} />
              {title}
            </button>
          ) : <span>{title}</span>}
          {actions}
        </header>
      )}
      {shown && <div className="p-3 text-sm text-ink">{children}</div>}
    </section>
  );
};

export interface CardProps extends React.HTMLAttributes<HTMLElement> {
  /** Makes the card a button: hover glow, press, keyboard focus. */
  onClick?: React.MouseEventHandler<HTMLElement>;
}

export const Card: React.FC<CardProps> = ({ onClick, className = '', ...rest }) =>
  onClick ? (
    <button type="button" onClick={onClick} className={`kit-card kit-raised kit-lift kit-card-interactive kit-focus block w-full text-ink ${className}`} {...(rest as React.ButtonHTMLAttributes<HTMLButtonElement>)} />
  ) : (
    <div className={`kit-card kit-raised text-ink ${className}`} {...(rest as React.HTMLAttributes<HTMLDivElement>)} />
  );
