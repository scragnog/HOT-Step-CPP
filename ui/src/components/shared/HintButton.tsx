// HintButton.tsx — a button that explains itself with the house hover card.
//
// The rulebook (docs/dev/ui-design.md, rule 3) says an explained control uses
// ParamLabel's card, not a native `title`: the browser's own tooltip looks like
// nothing else in the app and never appears on touch. Buttons were the gap,
// because ParamLabel was written for a label beside a field. `children` mode
// covers that, and this wraps the two so a toolbar does not repeat the pair for
// every icon.

import React from 'react';
import { ParamLabel } from './ParamLabel';

export interface HintButtonProps {
  /** The card's heading, usually the button's visible text. */
  label: string;
  /** What the button does and what changes, in the rulebook's voice. */
  info: string;
  children: React.ReactNode;
  onClick?: (e: React.MouseEvent<HTMLButtonElement>) => void;
  disabled?: boolean;
  /** Button classes. */
  className?: string;
  /** Wrapper classes: positioning, margins, alignment. */
  rootClassName?: string;
  /** For an icon-only button, whose name is not on screen. */
  ariaLabel?: string;
}

export const HintButton: React.FC<HintButtonProps> = ({
  label, info, children, onClick, disabled, className = '', rootClassName = '', ariaLabel,
}) => (
  <ParamLabel label={label} info={info} rootClassName={rootClassName}>
    <button type="button" className={className} onClick={onClick} disabled={disabled} aria-label={ariaLabel}>
      {children}
    </button>
  </ParamLabel>
);

export default HintButton;
