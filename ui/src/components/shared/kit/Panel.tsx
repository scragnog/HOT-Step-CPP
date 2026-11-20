// Panel.tsx — Panel (a titled surface) and Card (a raised tile inside one).
// Styles live in index.css (.kit-panel, .kit-card).

import React from 'react';

export interface PanelProps extends Omit<React.HTMLAttributes<HTMLElement>, 'title'> {
  /** Header text. Omit both title and actions for a headerless panel. */
  title?: React.ReactNode;
  /** Controls pinned to the right of the header. */
  actions?: React.ReactNode;
}

export const Panel: React.FC<PanelProps> = ({ title, actions, className = '', children, ...rest }) => (
  <section className={`kit-panel ${className}`} {...rest}>
    {(title || actions) && (
      <header className="kit-panel-header">
        <span>{title}</span>
        {actions}
      </header>
    )}
    <div className="p-3 text-sm text-ink">{children}</div>
  </section>
);

export interface CardProps extends React.HTMLAttributes<HTMLElement> {
  /** Makes the card a button: hover glow, press, keyboard focus. */
  onClick?: React.MouseEventHandler<HTMLElement>;
}

export const Card: React.FC<CardProps> = ({ onClick, className = '', ...rest }) =>
  onClick ? (
    <button type="button" onClick={onClick} className={`kit-card kit-card-interactive kit-focus block w-full text-ink ${className}`} {...(rest as React.ButtonHTMLAttributes<HTMLButtonElement>)} />
  ) : (
    <div className={`kit-card text-ink ${className}`} {...(rest as React.HTMLAttributes<HTMLDivElement>)} />
  );
