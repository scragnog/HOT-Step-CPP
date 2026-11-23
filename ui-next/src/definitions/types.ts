// The definition format: what a module, pedal or side unit of the Create desk is, as plain data.
//
// Every definition file exports one object that satisfies ModuleDef, PedalDef or UnitDef. The objects hold
// data only: no functions, no class instances, nothing JSON cannot carry. tools/validate-definitions.ts checks
// that each one survives a JSON round trip unchanged, and what the type system cannot check besides.
//
// The parts a definition may name (PartName) are the inventory in parts.ts. Adding a part there makes it
// available to every definition; nothing else needs to change.
//
// docs/02-definition-format.md explains each field with an example from a real definition.

import type { PartName } from './parts.ts';

/** The five engines, by the ids the server uses for them (GET /api/capabilities). */
export type EngineId = 'ace' | 'minimax-m3' | 'yue2' | 'sa3' | 'heartmula';

export type Tier = 'quick' | 'studio' | 'lab';

export type Json = string | number | boolean | null | Json[] | { [key: string]: Json };

/** A value a parameter can hold. */
export type Value = string | number | boolean;

/** A value for some engines, keyed by engine id. An engine left out takes the base value. */
export type PerEngine<T> = { [E in EngineId]?: T };

/** English source text. Its translation key is derived from where it sits (docs/05-what-the-definition-drives.md,
 *  "Strings and translation"), so a definition never names a translation key itself. */
export type Text = string;

/** What a parameter holds. A choice is one of its options' values. */
export type ValueType = 'number' | 'boolean' | 'choice' | 'text';

/** Units a number is shown in. 's' reads as m:ss from a minute up. */
export type Unit = 's' | 'ms' | 'Hz' | 'dB' | 'dBFS' | 'LUFS' | '%';

/** How a number is written wherever it is shown: read-outs, tips, undo steps, "Also set" lines.
 *  Without one, a number shows as many decimals as its step needs, then its unit. */
export interface Format {
  /** Fixed decimals, when the step's own would be too few (YuE2 penalties step by 0.005). */
  decimals?: number;
  /** The value in words: "{v} codes ({s})". {v} is the number, {s} the same in seconds (needs perSecond). */
  template?: Text;
  /** How many of the value's units make a second, for {s} in the template (5 codes a second). */
  perSecond?: number;
  /** Exact values said as words instead: 0 is "whole history", Top-P 0 is "off", false is "bypassed". */
  words?: { value: Value; text: Text }[];
  /** 0..1 written as a percentage. */
  percent?: true;
}

/** One option of a choice. `value` is what is stored and sent; the rest is how it reads. */
export interface Option {
  value: string | number;
  label: Text;
  /** On a chip selector or a segmented row where the label does not fit. */
  short?: Text;
  tip?: Text;
  /** What an LCD picker prints for it ("-14 LUFS", "TILED V2.1"). Default: the label in capitals. */
  lcd?: Text;
  /** The LCD's small second line for it ("refine only"). */
  detail?: Text;
}

/** Where a choice's options come from when the server owns the list. */
export type OptionSource =
  | 'plugins.solver' | 'plugins.scheduler' | 'plugins.guidance' | 'plugins.postprocess'
  | 'models' | 'languages' | 'adapters.lm' | 'mm3.planks' | 'mm3.hiddens' | 'caption.sources' | 'caption.datasets';

export interface OptionsFrom {
  from: OptionSource;
  /** For 'models': the backend's bucket (GET /api/backends/models buckets). */
  bucket?: string;
  /** Options shown before the server's list ("Native": the engine's own sampling). */
  include?: Option[];
}

/** A condition on parameters, desk state or the engine. Every referenced parameter must exist. */
export type Condition =
  | { param: string; is?: Value | Value[]; not?: Value | Value[]; above?: number; below?: number; auto?: boolean }
  | { state: string; set?: boolean; atLeast?: number }
  | { onDesk: string }
  | { engine: EngineId[] }
  | { all: Condition[] }
  | { any: Condition[] }
  | { not: Condition };

/** Where the real application keeps a value. The channel says how it travels:
 *  param    a field of the Create request body, held in ui/src/stores/globalParamsStore.ts (GENERATION_DEFAULTS)
 *  ext      a backend-declared extension knob, sent in backendParams (GET /api/capabilities extensions)
 *  content  a per-song field of the Create form (ui/src/components/create/createContent.ts)
 *  model    engine state chosen through POST /api/backends/models, not part of the request
 *  ui       form state the browser keeps and uses itself; never sent as such
 *  plugin   a Lua plugin's declared parameter, sent in pluginParams as "<plugin>:<key>" */
export type Channel = 'param' | 'ext' | 'content' | 'model' | 'ui' | 'plugin';
export type WireKey = `${Channel}:${string}`;

export type WireRef =
  | WireKey
  | {
    to: WireKey;
    /** The real field means the opposite (lm.on is skipLm). */
    invert?: true;
    /** Multiply by this on the way out (seconds to milliseconds: 1000). */
    scale?: number;
    /** Sent while the parameter is on auto (bpm 0, duration -1, keyScale ''). */
    auto?: Json;
    /** Anything the mapping cannot say in fields: packing, a missing declaration, a doubt. */
    note: Text;
  }
  /** No counterpart in the real application. The text says what the real app does instead. */
  | { none: Text };

/** A parameter that can be left to the engine. Its auto switch is stored beside it as `<key>.auto`. */
export interface Auto {
  /** What auto stands for, said with it: "auto (1.00)". Default: the default value as an estimate, "≈ 3.0". */
  says?: Text;
}

/** A text check the inventory knows (parts.ts VALIDATORS), with the message shown when it fails. */
export interface Check {
  rule: 'seed' | 'seedOrMinusOne' | 'seedOrEmpty' | 'timesteps';
  message: Text;
  /** A field that holds something composed keeps what was typed while it is invalid, and says what is in force. */
  keepInvalid?: Text;
}

/** What may differ per engine. */
export interface ParamFace {
  label: Text;
  /** Where room is short: a pedal knob, a chip, the "Also set" line. */
  short?: Text;
  /** Its name in undo steps when the label alone is ambiguous or names a group. */
  undoName?: Text;
  tip?: Text;
  /** [min, max, step]. */
  range?: [number, number, number];
  unit?: Unit;
  format?: Format;
  default: Value;
  auto?: Auto | false;
  options?: Option[] | OptionsFrom;
  placeholder?: Text;
  /** Text "Insert example" puts in the field. */
  example?: string;
  check?: Check;
  /** The parameter applies only when this holds. Otherwise its control is hidden (a pedal keeps its room), its value
   *  is kept, it never shows as changed, and it is not sent. */
  when?: Condition;
  /** A label that depends on another setting (DCW's first knob is "Low" in Both mode, "Scaler" otherwise). */
  labelWhen?: { when: Condition; label: Text }[];
  wire: WireRef;
}

export interface ParamDef extends ParamFace {
  key: string;
  type: ValueType;
  /** Which engines have it. Default: the definition's engines. */
  engines?: EngineId[];
  /** Who keeps it across an engine switch. Default: the definition's. */
  owner?: Owner;
  /** What saves it. Default: the definition's. */
  stored?: Stored;
  /** A song value whose limit differs per engine (length, takes): an engine that cannot take it reads it at its
   *  limit and holds the value for an engine that can. */
  held?: true;
  /** Stays a step below (or above) another parameter: a range's two ends never cross. */
  below?: string;
  above?: string;
  /** On a pedal, stands beside this switch and shows while it is off (Fixed seed beside Follow the seed). */
  beside?: string;
  /** Presses within this many milliseconds are one undo step (taps on TAP). */
  run?: number;
  per?: PerEngine<Partial<ParamFace>>;
}

/** engine: kept per engine, put aside on a switch and brought back on return.
 *  song: travels with the song to whichever engine is chosen. */
export type Owner = 'engine' | 'song';
/** settings: saved in a revision. content: saved with drafts, and in a revision saved "with content".
 *  none: saved with the desk only (a repaint region's settings belong to the song being repainted). */
export type Stored = 'settings' | 'content' | 'none';

/** Something a module keeps that is not a setting: a file, a seed, a selection, its adapters. */
export interface StateDef {
  key: string;
  kind: 'file' | 'seed' | 'span' | 'slots' | 'record';
  /** How undo and toasts name it: "the timbre reference". */
  noun: Text;
  /** The undo step for loading a file: "Load {file} as the timbre reference". */
  load?: Text;
  owner: Owner;
  stored: Stored;
  wire: WireRef | PerEngine<WireRef>;
}

/** A button a face offers. */
export type Action =
  | { do: 'set'; param: string; value: Value; label: Text; tip?: Text }
  | { do: 'auto'; param: string; label: Text }
  | { do: 'tier'; tier: Tier; ring?: string[]; label: Text }
  | { do: 'example'; param: string; label: Text }
  | { do: 'add'; item: string; label: Text }
  | { do: 'bespoke'; name: string; label: Text; tip?: Text };

/** A Quick choice: the plain word, and the detailed value it sets. */
export type Choice = [Text, Value];

/** One element of a face: a part from the inventory, bound to what it shows or edits. */
export interface Node {
  part: PartName;
  /** Shown only at these tiers. Without it, at every tier the definition shows at. */
  tiers?: Tier[];
  /** The parameter it edits or shows. */
  param?: string;
  /** Parts bound to several: an XY pad's x and y, a terminal line's tokens. */
  params?: string[];
  /** Its words where they are not the parameter's label. */
  label?: Text;
  tip?: Text;
  /** Static words: a note, a hint, a strip's reason. */
  text?: Text;
  /** Options the part takes, as listed for it in the inventory. */
  opts?: { [key: string]: Json };
  children?: Node[];
  /** Shown only while this holds. */
  when?: Condition;
  /** Shown only on these engines. */
  engines?: EngineId[];
  /** Stands back at 45% while this holds, says why, and stays operable. */
  dim?: { when: Condition; says: Text };
  /** A Quick choice row: the plain choices, per engine when they differ. */
  choices?: Choice[] | PerEngine<Choice[]>;
  /** How a choice row names a value between its choices: "{v} steps". */
  between?: Text;
  action?: Action;
}

/** The one-line strip a tile becomes when it has nothing to show. The first that holds wins. */
export interface Strip {
  when: Condition;
  says: Text;
  action?: Action;
}

/** How a tile asks for room. The packer decides; these are hints. */
export interface Size {
  /** Columns at Studio and Lab: 1, 2 or the whole row (4). */
  w: 1 | 2 | 4;
  /** Columns at Quick. Default 1. */
  quickW?: 2 | 4;
  /** The packer may widen it to close a gap. */
  grow?: boolean;
  /** A two-column tile the packer may fold to one column on a three-column desk. */
  fold?: true;
  /** Folds to one column on a three-column desk at Studio and Lab. */
  foldAt3?: true;
  /** The packer may make it taller, up to this many pixels more; controls keep to the top. */
  stretch?: number;
  /** As a strip it keeps its width, so it holds the row it will take. */
  stripKeepsWidth?: true;
}

/** A thumbnail is a drawing in the thumbnail kit's classes (src/css/30-thumbnails.css in the concept), kept as
 *  an SVG fragment in thumbs/. A pedal's or side unit's file holds only its middle: the frame is drawn from the
 *  definition (name, hue, foot switch, width). */
export interface Thumb {
  file: string;
  /** A different drawing on some engines. */
  per?: PerEngine<string>;
}

export type Category = 'Compose' | 'Engine' | 'Chain' | 'Words';

interface DefBase {
  id: string;
  name: Text;
  desc: Text;
  /** Place among its kind: Modules panel, picker and "Everything" order for modules, chain order for pedals. */
  at: number;
  engines: EngineId[];
  /** Defaults for its parameters. */
  owner: Owner;
  stored: Stored;
  params: ParamDef[];
  state?: StateDef[];
  /** Required for everything the Modules panel and the picker offer; built-in and source modules have none. */
  thumb?: Thumb;
}

export interface ModuleDef extends DefBase {
  kind: 'module';
  category: Category;
  /** Shown only at Lab, under the module it extends; its settings apply at every tier. */
  labUnder?: string;
  /** Built into the page, not added from the Modules panel (the prompt, the lyric sheet, the models). */
  builtIn?: true;
  /** Brought by an action on a song (Repaint), not from the panel. */
  source?: true;
  /** What each tier adds, for the picker's description: [Quick, Studio, Lab]. */
  tierSummary?: [Text, Text, Text];
  size: Size;
  per?: PerEngine<{ name?: Text; quickName?: Text; desc?: Text; size?: Partial<Size> }>;
  /** Its name at Quick when Quick shows less of it ("Tempo"). */
  quickName?: Text;
  /** What the tile shows, one list for every tier: a node with `tiers` appears only at those tiers. A Lab-only
   *  module (labUnder) shows at Lab alone, so its nodes need no tiers. */
  face: Node[];
  strip?: Strip[];
}

export interface PedalDef extends DefBase {
  kind: 'pedal';
  /** Short name outside Lab. */
  short?: Text;
  /** The pedal's face colour, a hue in degrees (the --h of the concept's pedals). */
  hue: number;
  /** Its foot switch. null: always on (Pre-VST gain). */
  on: string | null;
  /** The one knob it shows at Studio. null: it shows its Studio face nodes instead. */
  main: string | null;
  /** Its thumbnail's width in pedal units. */
  thumbWidth: 1 | 2;
  /** What it shows beyond what is laid out from its parameters: at Studio, in place of a main knob; at Lab, after
   *  its knobs and selectors and before its switches. Tier each node. */
  face?: Node[];
  /** Engaged with nothing to run: an amber LED and the reason. The first that holds wins. */
  idle?: Strip[];
  /** What the blanking plate over kept-empty knob cells says: which setting fills them. */
  blank?: Strip[];
}

export interface UnitDef extends DefBase {
  kind: 'unit';
  /** Its power switch. */
  on: string;
  /** Takes the board's full width. */
  wide?: true;
  /** Runs after the chain, so it stops while the chain is bypassed. */
  gated?: true;
  /** The object it shows at every tier (a lyric LCD, a status line, an art swatch). At Lab its controls follow. */
  face: Node[];
}

export type Definition = ModuleDef | PedalDef | UnitDef;
