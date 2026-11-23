// Named places in the page's shell that a work package fills by registering a component from its own file, so the
// shell is not edited by every package that draws into it (the concept's defineScreen and its render hooks). An empty
// slot draws nothing; a filled one draws its fills in the order they were registered.
//
//   fill('centre.prompt', PromptCard)          in src/app/features/prompt.tsx
//   <Slot name="centre.prompt" />              in the shell, where the concept puts it
//
// The slots: desk.lead (the lyric sheet's place at the head of the desk), centre.prompt and centre.quickLyrics (the
// prompt card and Quick's lyric sheet), centre.gobar and centre.latest (the go bar's contents and Latest), panel.
// workflows (the Workflows panel's list), engine.panel (the engine panel's models and figures below its cards),
// overlays (dialogs, popovers and menus, drawn at the page's root).

import type { ComponentType } from 'react';

export type SlotName = 'desk.lead' | 'centre.prompt' | 'centre.quickLyrics' | 'centre.gobar' | 'centre.latest' | 'panel.workflows' | 'engine.panel' | 'overlays';
const FILLS = new Map<SlotName, ComponentType[]>();

export function fill(name: SlotName, view: ComponentType): void { FILLS.set(name, [...(FILLS.get(name) ?? []), view]); }
export const fillsOf = (name: SlotName): readonly ComponentType[] => FILLS.get(name) ?? [];
