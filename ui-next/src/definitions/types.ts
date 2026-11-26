// The definition format: what a module, pedal or side unit of the Create desk is, as plain data.
//
// Every definition file exports one object that satisfies ModuleDef, PedalDef or UnitDef. The objects hold data
// only: no functions, no class instances, nothing JSON cannot carry. tools/validate-definitions.ts checks each one
// in its JSON form against these types, and then what the types cannot check.
//
// The lists a definition may draw from (parts, units, text checks, option sources, desk facts, named actions,
// wire transforms) are the inventory in parts.ts and transforms.ts, and the engines are engines.ts. Adding an entry
// there makes it available to every definition; nothing here changes.
//
// docs/02-definition-format.md explains each field with an example from a real definition.

import type { OptsOf, PartName, UnitName, CheckName, OptionSourceName, ActionName } from './parts.ts';
import type { Role } from './roles.ts';
import type { TransformName } from './transforms.ts';
import type { EngineId } from './engines.ts';

export type { EngineId };

export type Tier = 'quick' | 'studio' | 'lab';

export type Json = string | number | boolean | null | Json[] | { [key: string]: Json };

/** A value a parameter can hold. */
export type Value = string | number | boolean;

/** English source text. It may name values to fill in (docs/05, "Words that name values"): {engine}, {value:<key>},
 *  {count:<state or fact>}, {file:<state>|<words when empty>}, and in a parameter's own words {min}, {max} and
 *  {default}. Its translation key is derived from where it sits, so a definition never names a key. */
export type Text = string;

/** What a parameter holds. A choice is one of its options' values. */
export type ValueType = 'number' | 'boolean' | 'choice' | 'text';

/** How a number is written wherever it is shown: read-outs, tips, undo steps, "Also set" lines, held notes.
 *  Without one: docs/05 "How a number is written". */
export interface Format {
  /** Fixed decimals, when the step's own would be too few (YuE2 penalties step by 0.005). */
  decimals?: number;
  /** The value in words: "{v} codes ({s})". {v} is the number, {s} the same in seconds (needs perSecond). */
  template?: Text;
  /** How many of the value's units make a second, for {s} in the template (5 codes a second). */
  perSecond?: number;
  /** Exact values said as words instead: 0 is "whole history", Top-P 0 is "off", false is "bypassed". */
  words?: { value: Value; text: Text }[];
  /** A share from 0 to 1 written as a percent: the reference blend's 0.95 is "95%". */
  percent?: true;
}

/** One option of a choice. `value` is what is stored and sent; the rest is how it reads. */
export interface Option {
  value: string | number;
  label: Text;
  /** On a chip selector or a segmented row where the label does not fit. */
  short?: Text;
  tip?: Text;
  /** What an LCD prints for it ("-14 LUFS", "V2.1"): the LCD picker, a pedal's read-out, a side unit's screen.
   *  Default: the label in capitals. */
  lcd?: Text;
  /** What an LCD picker and its list print for it, where that differs from what a read-out prints (`lcd`): the Tiled
   *  decoder's picker says "Tiled V2.1" where its read-out says "V2.1". Default: `lcd`. */
  pick?: Text;
  /** The LCD's small second line for it ("refine only"). */
  detail?: Text;
  /** Values this option was saved under before, read as this one when saved data is loaded (docs/05, "Saved
   *  data"). */
  was?: (string | number)[];
}

/** Where a choice's options come from when the server owns the list. */
export interface OptionsFrom {
  from: OptionSourceName;
  /** For 'models': the backend's bucket (GET /api/backends/models buckets). */
  bucket?: string;
  /** For 'models': a model file the engine loads (the engine panel's): its "" (the engine's default) stands for the
   *  file the server has selected in the bucket, and its size counts in the VRAM figures (derive/vram.ts). */
  loaded?: boolean;
  /** Options the definition knows, merged with the server's list by value: words given here win, and a value the
   *  server does not list ("Native": the engine's own sampling) is offered first. */
  include?: Option[];
  /** Only these of the server's values (a composite schedule's stages). */
  only?: string[];
}

/** A condition on parameters, desk state or the engine. A condition on a parameter the engine does not have is
 *  false. */
export type Condition =
  | { param: string; is?: Value | Value[]; not?: Value | Value[]; above?: number; below?: number; auto?: boolean }
  | { state: string; set?: boolean; atLeast?: number }
  | { onDesk: string }
  | { engine: EngineId[] }
  | { all: Condition[] }
  | { any: Condition[] }
  | { not: Condition };

/** Where a value goes in the request, or in the engine's state. The channel says which:
 *  param    a field of GENERATION_DEFAULTS (server/src/contracts/generationControls.ts), the resolver's input
 *  ext      a backend's extension knob (capabilities().extensions), sent in backendParams
 *  content  a field of CreateContent (ui/src/components/create/createContent.ts), the Create form's content
 *  body     a top-level request field set after the resolver (HeartMuLa's frozen model choices)
 *  model    engine state chosen through POST /api/backends/models, never part of the request
 *  intent   a part of the Create intent beside its params (server/src/contracts/resolution.ts): `compose`, the
 *           caption steps the server runs, and `captionSource`
 *  ui       form state the browser keeps and uses itself; never sent
 *  A name may be a path into its field: `param:adapterGroupScales.self_attn`, `intent:compose.loraTrigger`. */
export type Channel = 'param' | 'ext' | 'content' | 'body' | 'intent' | 'model' | 'ui';
export type WireKey = `${Channel}:${string}`;

/** A wire that does more than copy the value (docs/05, "What is sent"). */
export interface WireMap {
  to: WireKey;
  /** A switch whose real field means the opposite (lm.on is skipLm). */
  invert?: true;
  /** Multiply on the way out (seconds to milliseconds: 1000). */
  scale?: number;
  /** Divide on the way out (seconds to minutes: 60; percent to 0..1: 100). */
  divide?: number;
  /** A choice's value to the real value (Simple is false). An option the map leaves out sends nothing here. */
  map?: { [value: string]: Json };
  /** A named transform (transforms.ts) for what fields cannot say: several parameters packed into one string, text
   *  sent as a number. Every parameter of a packing transform names it. */
  via?: TransformName;
  /** Sent while the parameter is on auto. null: nothing is sent, so the engine's own default applies. */
  auto?: Json;
  /** Sent only while this holds; otherwise `else` is sent, or nothing (the denoiser's strength while it is off). */
  when?: Condition;
  else?: Json;
  /** The checks of the real app this wire deliberately differs from, each explained in `note`: its range or step
   *  (`range`), its options (`options`), its default (`default`), or a field the engine does not read (`unread`).
   *  The request comparison still runs over the whole value space (docs/07). */
  differs?: ('range' | 'options' | 'default' | 'unread')[];
  /** What a reader needs to know: why it differs, a field read but not declared. A note waives no check. */
  note?: Text;
  /** Names the real field had before, for reading back a profile or a song saved under one (fromReal). */
  was?: WireKey[];
}

/** A single wire, several (one control to several fields), or none: no counterpart in the real application, and
 *  the text says what the real app does instead. */
export type Wire = WireKey | WireMap | Wire[] | { none: Text };

/** A parameter that can be left to the engine, beside a number. Its switch is stored in the desk's auto map. */
export interface Auto {
  /** What auto stands for, said with it: "auto (1.00)". Default: the default value as an estimate, "≈ 3.0". */
  says?: Text;
  /** What its auto button's tip says auto does ("Auto lets the planner choose the length"). */
  tip?: Text;
}

/** A text check from the inventory (parts.ts CHECKS), with the message shown when it fails. */
export interface Check {
  rule: CheckName;
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
  unit?: UnitName;
  format?: Format;
  default: Value;
  auto?: Auto | false;
  options?: Option[] | OptionsFrom;
  placeholder?: Text;
  /** The longest text it takes, in characters (the trigger word 24, a title 80). */
  maxLength?: number;
  /** Text "Insert example" puts in the field. */
  example?: string;
  check?: Check;
  /** The parameter applies only when this holds. Otherwise its control is hidden (a pedal keeps its room), its value
   *  is kept, it never shows as changed, and it is not sent. */
  when?: Condition;
  /** While one of these holds the engine runs `value` whatever is chosen (a choice): it is shown pressed, the others
   *  disabled with `tip`, and `says` beside them; the choice is kept and returns when none holds (Apply as is Runtime
   *  while section directives or phase windows are set). The first that holds wins. Nothing sent changes: the server
   *  forces it on its own. */
  forced?: { when: Condition; value: Value; says: Text; tip: Text }[];
  wire: Wire;
}

/** One line of a parameter's per-engine table: these engines take these fields instead of the base ones. */
export type EngineOverride = { engines: EngineId[] } & Partial<ParamFace>;
/** Words that change with another setting (DCW's first knob is "Low" in Both mode, "Scaler" otherwise). A word line
 *  changes words only: what the value is, where it goes and when it applies are the parameter's (an engine line's). */
export type WordsOverride = { when: Condition; engines?: never } & Partial<Pick<ParamFace, 'label' | 'short' | 'tip' | 'auto' | 'placeholder'>>
  & { [K in Exclude<keyof ParamFace, 'label' | 'short' | 'tip' | 'auto' | 'placeholder' | 'when'>]?: never };

/** A song value whose limit differs per engine (length, takes): an engine that cannot take it reads it at its limit
 *  and holds the value for an engine that can. Slots: {value} the held value, {shown} the limit in force, {engine}
 *  the engine it returns on. */
export interface Held {
  /** Said by the control while the value is held: "Your {value} return on {engine}". */
  note: Text;
  /** Said in the engine switch's toast: "Takes capped at {shown}". */
  toast: Text;
  /** How the held value is said in the toast's "your ... waits on <engine>" ("{value} takes"). Default "{value}". */
  value?: Text;
}

/** A named undo step a part asks for (docs/05, "Undo step names"): its words, and the parameters the step may change
 *  besides its own (loading a second adapter stacks them). Slots: {name} (the parameter's name), {noun}, {from}, {to},
 *  {file}, and any the part fills ({n}, {section}, {tag}). */
export type Step = Text | { says: Text; changes: string[] };

export interface ParamDef extends ParamFace {
  key: string;
  type: ValueType;
  /** What the page needs it for, when it needs this one in particular (roles.ts: the main box, the takes count). The
   *  page asks for the role, never the key, so a rename keeps it working. */
  role?: Role;
  /** Keys this parameter was saved under before, read as this one when saved data is loaded. */
  was?: string[];
  /** Which engines have it. Default: the definition's. Required when `per` gives engine lines. */
  engines?: EngineId[];
  /** Who keeps it across an engine switch. Default: the definition's. */
  owner?: Owner;
  /** What saves it. Default: the definition's. */
  stored?: Stored;
  held?: Held;
  /** Stays a step below this parameter, which stays a step above it: a range's two ends never cross. */
  staysBelow?: string;
  /** On a pedal, stands beside this switch and shows while it is off (Fixed seed beside Follow the seed). */
  beside?: string;
  /** Presses within this many milliseconds are one undo step (taps on TAP). */
  run?: number;
  /** The named steps its part takes on it, besides setting it ("Compose caption", "Add {chip} to the {name}"). */
  steps?: { [step: string]: Step };
  /** Engine lines (each engine in at most one), then word lines that hold while their condition does. */
  per?: (EngineOverride | WordsOverride)[];
}

/** engine: kept per engine, put aside on a switch and brought back on return.
 *  song: travels with the song to whichever engine is chosen. */
export type Owner = 'engine' | 'song';
/** settings: saved in a revision. content: saved with drafts, and in a revision saved "with content".
 *  none: saved with the desk only (a repaint region's settings belong to the song being repainted). */
export type Stored = 'settings' | 'content' | 'none';

/** Something a module keeps that is not a setting: a file, the seed, a selection, its adapters, a list. */
export interface StateDef {
  key: string;
  /** What the page needs it for, when it needs this one in particular (roles.ts: the song a repaint works on). */
  role?: Role;
  /** Keys it was saved under before (docs/05, "Saved data"). */
  was?: string[];
  kind: 'file' | 'seed' | 'span' | 'slots' | 'record' | 'list';
  /** How undo and toasts name it: "the timbre reference". */
  noun: Text;
  /** How a step that replaces it begins ("Timbre reference my-track.wav → second.flac"), where its noun would not
   *  (13-files.js fileStep). Default: its noun. */
  title?: Text;
  /** How an empty value reads in a step ("random" for the seed). */
  empty?: Text;
  /** Its undo steps, by the name its part uses: the part asks for a step by name and the words come from here.
   *  Slots: {file}, {from}, {to}, and any the step's caller fills ({n}, {section}). A file has `load` and `clear`
   *  without saying (docs/05, "Undo step names"). */
  steps?: { [step: string]: Step };
  /** The steps the page says in a toast with Undo when the part takes them (E-23: a recording loaded to continue
   *  from); every other step is said only by Undo and Redo. */
  tells?: string[];
  owner: Owner;
  stored: Stored;
  wire: Wire | { engines: EngineId[]; wire: Wire }[];
}

/** A button a face offers. */
export type Action =
  | { do: 'set'; param: string; value: Value; label: Text; tip?: Text }
  | { do: 'auto'; param: string; label: Text }
  /** Go to a tier and ring these controls there: parameter keys, or a state key for its drop zone. */
  | { do: 'tier'; tier: Tier; ring?: string[]; label: Text }
  | { do: 'example'; param: string; label: Text }
  | { do: 'add'; item: string; label: Text }
  /** Reset every parameter the node's children edit (a plugin's control group). */
  | { do: 'reset'; label: Text }
  /** Add a section tag to the lyrics. */
  | { do: 'tag'; tag: string; label: Text; tip?: Text }
  /** A named action from the inventory (parts.ts ACTIONS). */
  | { do: 'bespoke'; name: ActionName; label: Text; tip?: Text };

/** A Quick choice: the plain word, and the detailed value it sets, or several values at once. */
export type Choice = [Text, Value | { [key: string]: Value }];

/** What a node may change per engine, tier or state, shallow like a parameter's lines. */
export type NodeOverride = { engines?: EngineId[]; tiers?: Tier[]; when?: Condition } & Partial<NodeFields>;

/** The fields every node may carry, whatever its part. */
export interface NodeFields {
  /** The parameter it edits. Only a control or bespoke part edits. */
  param: string;
  /** Parts that edit several: an XY pad's x and y, a bespoke part's values. */
  params: string[];
  /** What a read-out or a bespoke drawing shows without editing it. */
  reads: string[];
  /** State it edits: a drop zone's file, the seed. */
  state: string[];
  /** Its words where they are not the parameter's label. */
  label: Text;
  tip: Text;
  /** Static words: a note, a hint. */
  text: Text;
  /** Stands back at 45% while this holds, says why, and stays operable. */
  dim: { when: Condition; says: Text };
  /** A Quick choice row's plain choices. */
  choices: Choice[];
  /** How a choice row names a value between its choices: "{v} steps". */
  between: Text;
  /** Buttons or links it offers. */
  actions: Action[];
}

interface NodeBase extends Partial<NodeFields> {
  /** Shown only at these tiers. Without it, at every tier its parent shows at. */
  tiers?: Tier[];
  /** Shown only while this holds. */
  when?: Condition;
  /** Shown only on these engines. */
  engines?: EngineId[];
  children?: Node[];
}

type NodeFor<P extends PartName> = NodeBase & {
  part: P;
  /** The part's options, as the inventory lists them for it. */
  opts?: OptsOf<P>;
  /** Lines that change its fields on some engines, tiers or states. Later lines win. */
  per?: (NodeOverride & { opts?: OptsOf<P> })[];
};

/** One element of a face: a part from the inventory, bound to what it shows or edits, with that part's options. */
export type Node = { [P in PartName]: NodeFor<P> }[PartName];

/** The one-line strip a tile becomes when it has nothing to show, a pedal's idle reason, or (Strip without
 *  actions) words. The first that holds wins. */
export interface Strip {
  when: Condition;
  says: Text;
  actions?: Action[];
  /** Only at these tiers. */
  tiers?: Tier[];
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
  /** One column at Quick and Studio while this holds (the Adapters with one cartridge: S-14). */
  narrowWhen?: Condition;
}

/** A thumbnail is a drawing in the thumbnail kit's classes (src/css/30-thumbnails.css in the concept), kept as an
 *  SVG fragment in thumbs/. A pedal's or side unit's file holds only its middle: the frame is drawn from the
 *  definition. `node tools/thumb.ts <pedal>` draws a pedal's middle from its parameters. */
export interface Thumb {
  file: string;
  /** A different drawing on some engines. */
  per?: { engines: EngineId[]; file: string }[];
}

/** The groups modules are offered in, in this order (the Modules panel, the picker). A new one is added here. */
export const CATEGORIES = ['Compose', 'Engine', 'Chain', 'Words'] as const;
export type Category = (typeof CATEGORIES)[number];

interface DefBase {
  id: string;
  /** Ids it was saved under before: a workflow or desk naming one has this definition instead. */
  was?: string[];
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

/** What a module may change per engine. */
export type ModuleOverride = { engines: EngineId[] } & Partial<Pick<ModuleDef, 'name' | 'quickName' | 'desc' | 'tierSummary' | 'starter'>> & { size?: Partial<Size> };

export interface ModuleDef extends DefBase {
  kind: 'module';
  category: Category;
  /** Shown only at Lab, under the module it extends; its settings apply at every tier. */
  labUnder?: string;
  /** Built into the page, not added from the Modules panel (the prompt, the lyric sheet, the models). */
  builtIn?: true;
  /** Brought by an action on a song (Repaint), not from the panel. Its values are not part of a Create request. */
  source?: true;
  /** Holds the pedalboard: the pedals and side units on the board are drawn in its tile and answer to it (the
   *  Pedalboard). */
  board?: true;
  /** What each tier adds, for the picker's description: [Quick, Studio, Lab]. */
  tierSummary?: [Text, Text, Text];
  size: Size;
  per?: ModuleOverride[];
  /** Its name at Quick when Quick shows less of it ("Tempo"). */
  quickName?: Text;
  /** Offered on the empty desk: its place among the offers (`at`), and, for the first, the words the empty Quick desk
   *  names it with ("Add tempo, a seed or pedals from Modules"). */
  starter?: { at: number; noun?: Text };
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
  /** Its foot switch. null: always on (Pre-VST gain), drawn with the neutral face. */
  on: string | null;
  /** The one knob it shows at Studio. null: it shows its Studio face nodes instead. */
  main: string | null;
  /** What it shows beyond what is laid out from its parameters: at Studio, in place of a main knob; at Lab, after
   *  its knobs and selectors and before its switches. Tier each node. */
  face?: Node[];
  /** Engaged with nothing to run: an amber LED and the reason. The first that holds wins. */
  idle?: Strip[];
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

/* ---------------------------------------------------------------- saved data
   Every record the page saves carries `v`, the version of its shape. Values are keyed by parameter key, in the
   definition's own units; auto switches are a map of their own. docs/05, "Saved data", says how each is read. */

export const SAVED_VERSION = 1;

/** A parameter's values: key to value, and key to "on auto". */
export interface SavedValues {
  vals: { [key: string]: Value };
  auto?: { [key: string]: boolean };
}

/** A workflow: the desk's layout. No values. */
export interface Workflow {
  v: 1;
  name: string;
  tier: Tier;
  modules: string[];
  /** Pedals and side units, in their order. */
  board: string[];
  note?: string;
}

/** A revision: one engine's settings, saved under a workflow. */
export interface Revision extends SavedValues {
  v: 1;
  workflow: string;
  name: string;
  engine: EngineId;
  /** State saved in a revision (stored: 'settings'): the adapter slots. */
  state?: { [key: string]: Json };
  /** Saved "with content": the content values and the prompt's text are in `vals` too. */
  withContent?: true;
  note?: string;
}

/** A draft: one song's content for one engine. */
export interface Draft extends SavedValues {
  v: 1;
  name: string;
  engine: EngineId;
}

/** The desk autosave: everything, so a reload or another tab finds the desk as it was. */
export interface SavedDesk extends SavedValues {
  v: 1;
  engine: EngineId;
  tier: Tier;
  modules: string[];
  board: string[];
  state: { [key: string]: Json };
  /** Engine-owned values put aside for the engines not in use. */
  stash: { [engine: string]: SavedValues & { state?: { [key: string]: Json } } };
  /** Held values, by parameter key. */
  hold: { [key: string]: { value: number; shown: number; from: EngineId } };
}
