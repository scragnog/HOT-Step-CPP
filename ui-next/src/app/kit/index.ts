// The part kit: everything a part's file may use (eslint.config.js lets a file in src/app/parts import from here, from
// react and react-dom, from its sibling parts and from src/app/services, and from nothing else). docs/09, "How a part
// is written".

export { definePart } from './define.ts';
export type { Gesture, ParamView, PartView, StateView, Slot, Surface, MeterScale } from './types.ts';
export type { Role } from '../../definitions/roles.ts';
export { ui, useSay } from '../i18n/index.ts';
export { Icon } from './icons.tsx';
export { LcdFace } from './LcdFace.tsx';
export { onValueKey, onValueKeyUp, onWheelStep } from './value.ts';
export { Glyph } from './glyphs.tsx';
/* the server boundary (services/index.ts), for a bespoke part responsible for its own conversation with the server
   (parts.ts PARTS `responsible`: Build by section's session) */
export { service } from '../register/services.ts';
