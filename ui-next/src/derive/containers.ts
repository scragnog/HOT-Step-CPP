// What a container knows of its children (docs/09, package 2): each child's part, whether it is drawn, the words its
// flag or label prints, whether it stands back, the parts it holds in turn, and whether it is one of the lines a higher
// tier adds. A container draws the concept's housing for what it holds from this (a terminal line prints each
// control's label as its flag, a terminal gathers its higher tier's lines, a rack group puts its switch in its header,
// a deck indents a recorder's own lines) and decides nothing about a value: every answer is the core's (nodeShown,
// nodeWord, dimOf). Nothing here has a branch for any module.

import type { Node } from '../definitions/types.ts';
import { type Desk, type Index, editsOf, has, nodeAt } from './core.ts';
import { nodeShown } from './desk.ts';
import { type Msg, type Slot, msg } from './messages.ts';
import { childKeys, forcedOf, nodeWord, paramWords } from './strings.ts';
import { dimOf } from './view.ts';

export interface KidModel {
  part: string;
  /** Drawn here: the renderer draws nothing for a child that is not. */
  shown: boolean;
  /** Its label (the node's, else its first parameter's): a terminal line prints it as the control's flag. */
  label?: Slot;
  /** Why it stands back, when it does. */
  dim: Slot | null;
  /** The parts of its own children that are drawn. */
  inner: string[];
  /** One of the lines a higher tier adds: it shows at fewer tiers than its container. */
  extra: boolean;
  /** The parameters it edits, or its own children edit, on the engine in use (a guard's line: its penalty first). */
  keys: string[];
  /** A choice it edits is run at another value meanwhile (ParamFace.forced): its pairing is marked forced. */
  forced?: true;
}

/** A container's children on the desk, in the order the renderer draws them. `node` is the container's node as the
 *  definition writes it, `key` its translation key. */
export function kidsOf(ix: Index, node: Node, desk: Desk, key: string): KidModel[] {
  const kids = node.children ?? [], keys = childKeys(key, kids);
  return kids.map((c, i) => {
    const shown = nodeShown(ix, c, desk), at = shown ? nodeAt(ix, c, desk) : null, k = keys[i]!;
    const label = at ? nodeWord(ix, c, desk, k, 'label') ?? (at.param && has(ix, at.param, desk.engine) ? paramWords(ix, at.param, desk).label : undefined) : undefined;
    return {
      part: c.part, shown, ...(label !== undefined ? { label } : {}),
      dim: at ? dimOf(ix, c, desk, k) : null,
      inner: at ? (at.children ?? []).filter(g => nodeShown(ix, g, desk)).map(g => g.part) : [],
      extra: !!c.tiers && (!node.tiers || node.tiers.some(t => !c.tiers!.includes(t))),
      keys: at ? [...editsOf(at), ...(at.children ?? []).flatMap(g => editsOf(g))].filter(x => has(ix, x, desk.engine)) : [],
      ...(at && [...editsOf(at), ...(at.children ?? []).flatMap(g => editsOf(g))].some(x => forcedOf(ix, x, desk)) ? { forced: true as const } : {}),
    };
  });
}

/* ---------------------------------------------------------------- a drop zone's file */

/** The files an audio drop zone takes: an audio type, or a name with an audio file's extension (13-files.js useFile). */
const AUDIO_EXT = /\.(wav|flac|mp3|ogg|oga|opus|m4a|aac|aif|aiff|wma)$/i;
/** Why a file offered to a drop zone is refused (it is no audio file), or null: it is taken. */
export const fileRefused = (name: string, type: string): Msg | null => ((type || '').startsWith('audio/') || AUDIO_EXT.test(name) ? null : msg('file.notAudio', { name }));
/** What a drop outside every drop zone says. */
export const dropOutside = (): Msg => msg('file.outside');
