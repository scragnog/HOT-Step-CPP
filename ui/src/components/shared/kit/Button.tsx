// Button.tsx — the kit's button. One base recipe (.kit-btn in index.css: radius,
// depth, border, hover glow, press); `size` and `variant` ride on top.
// To add a size or variant: add its .kit-btn-<name> class in index.css and its
// name here. See docs/dev/ui-design.md.

import React from 'react';

export type ButtonSize = 'sm' | 'md' | 'lg';
export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';

/** Whole class names: Tailwind only keeps classes it can see in the source. */
const SIZE_CLASS: Record<ButtonSize, string> = { sm: 'kit-btn-sm', md: 'kit-btn-md', lg: 'kit-btn-lg' };
const VARIANT_CLASS: Record<ButtonVariant, string> = {
  primary: 'kit-btn-primary kit-fill', secondary: 'kit-btn-secondary kit-raised', ghost: 'kit-btn-ghost', danger: 'kit-btn-danger kit-raised',
};

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  size?: ButtonSize;
  variant?: ButtonVariant;
  /** Square button for a lone icon. Prefer IconButton, which also demands a label. */
  iconOnly?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ size = 'md', variant = 'secondary', iconOnly = false, className = '', type = 'button', ...rest }, ref) => (
    <button
      ref={ref}
      type={type}
      className={`kit-btn kit-lift kit-focus ${SIZE_CLASS[size]} ${VARIANT_CLASS[variant]}${iconOnly ? ' kit-btn-icon' : ''} ${className}`}
      {...rest}
    />
  ),
);
Button.displayName = 'Button';

export interface IconButtonProps extends Omit<ButtonProps, 'iconOnly' | 'aria-label'> {
  /** Required: an icon alone tells a screen reader nothing. Also the tooltip. */
  'aria-label': string;
}

export const IconButton = React.forwardRef<HTMLButtonElement, IconButtonProps>(
  (props, ref) => <Button ref={ref} iconOnly title={props['aria-label']} {...props} />,
);
IconButton.displayName = 'IconButton';
