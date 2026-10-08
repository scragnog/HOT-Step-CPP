// services/preferences/kinds.ts — registers this slice's typed document
// kinds. Imported once (by routes/preferences.ts) so registration runs
// before any request; touches no database (registerDocumentKind is pure).
import { registerDocumentKind, type TypedDocumentKind } from '../workflows/revisions.js';
import {
  aiContinuePresetBodySchema, aiContinueTemplateBodySchema,
  stormTuningBodySchema, vstChainPresetBodySchema, yue2JointPresetBodySchema,
  type AiContinuePresetBody, type AiContinueTemplateBody,
  type StormTuningBody, type VstChainPresetBody, type Yue2JointPresetBody,
} from '../../contracts/preferences.js';

export const VST_CHAIN_PRESET = registerDocumentKind<VstChainPresetBody>({
  kind: 'preferences.vst-chain-preset', scope: 'installation', schemaVersion: 1, schema: vstChainPresetBodySchema,
});
export const AI_CONTINUE_STYLE_PRESET = registerDocumentKind<AiContinuePresetBody>({
  kind: 'preferences.ai-continue-style-preset', scope: 'installation', schemaVersion: 1, schema: aiContinuePresetBodySchema,
});
export const AI_CONTINUE_LYRIC_PRESET = registerDocumentKind<AiContinuePresetBody>({
  kind: 'preferences.ai-continue-lyric-preset', scope: 'installation', schemaVersion: 1, schema: aiContinuePresetBodySchema,
});
export const AI_CONTINUE_TEMPLATE = registerDocumentKind<AiContinueTemplateBody>({
  kind: 'preferences.ai-continue-template', scope: 'installation', schemaVersion: 1, schema: aiContinueTemplateBodySchema,
});
export const YUE2_JOINT_PRESET = registerDocumentKind<Yue2JointPresetBody>({
  kind: 'preferences.yue2-joint-preset', scope: 'installation', schemaVersion: 1, schema: yue2JointPresetBodySchema,
});
export const STORM_TUNING = registerDocumentKind<StormTuningBody>({
  kind: 'preferences.storm-tuning', scope: 'installation', schemaVersion: 1, schema: stormTuningBodySchema,
});

/** A family's kind and body are erased to `Record<string, unknown>` here:
 *  the map below holds several unrelated body types side by side, and every
 *  consumer (routes/preferences.ts, services/preferences/presets.ts) only
 *  ever round-trips a body through its own kind's schema, which still runs
 *  at the original type. Only this heterogeneous directory is erased. */
interface ErasedPresetFamily {
  def: TypedDocumentKind<Record<string, unknown>>;
  nameOf: (body: Record<string, unknown>) => string;
}
const erasePreset = <T extends Record<string, unknown>>(def: TypedDocumentKind<T>, nameOf: (body: T) => string): ErasedPresetFamily =>
  ({ def: def as unknown as TypedDocumentKind<Record<string, unknown>>, nameOf: nameOf as unknown as (body: Record<string, unknown>) => string });

/** Named-preset families, keyed by their URL segment. Each is a collection
 *  (list/create/update/remove); `nameOf` finds the browser-side identity used
 *  for import conflict detection. */
export const PRESET_FAMILIES: Record<string, ErasedPresetFamily> = {
  'vst-chain': erasePreset(VST_CHAIN_PRESET, b => b.name),
  'ai-continue-style': erasePreset(AI_CONTINUE_STYLE_PRESET, b => b.label),
  'ai-continue-lyric': erasePreset(AI_CONTINUE_LYRIC_PRESET, b => b.label),
  'yue2-joint': erasePreset(YUE2_JOINT_PRESET, b => b.name),
};
export type PresetFamily = 'vst-chain' | 'ai-continue-style' | 'ai-continue-lyric' | 'yue2-joint';

/** Singleton families: one document per installation, current value only
 *  (no named collection). Same erasure, no `nameOf`. */
export const SINGLETON_FAMILIES: Record<string, TypedDocumentKind<Record<string, unknown>>> = {
  'ai-continue-template': AI_CONTINUE_TEMPLATE as unknown as TypedDocumentKind<Record<string, unknown>>,
  'storm-tuning': STORM_TUNING as unknown as TypedDocumentKind<Record<string, unknown>>,
};
export type SingletonFamily = 'ai-continue-template' | 'storm-tuning';
