// training/languageCodes.ts — dataset language -> the ISO code the inference
// FSM can actually emit (2026-08-11).
//
// THE BUG THIS EXISTS TO PREVENT. Dataset creation used to default
// `default_language` to the literal string 'english'. That value flows
// dataset -> sidecar -> preprocess __metadata__ -> lm_codes.jsonl -> the
// training CoT YAML, so LM adapters learned to emit `language: english`.
// But metadata-fsm.h constrains the `language:` field to 51 ISO codes
// ('en', 'fr', 'zh', ..., 'unknown') — 'english' is NOT in that set and the
// sampler literally cannot produce it. With the language unpinned at
// inference the model's probability mass therefore sits on a forbidden word
// and whichever allowed ISO code survives the mask wins: measured on
// 4yearstrong_someway (6 seeds, unpinned) the adapter emitted sr/pt/fr on 3
// of 6 runs, against 1 of 6 for the base LM. All 2444 rows of the 183-dataset
// corpus were affected; server/scripts/fix-training-language.mjs repaired the
// existing data, and this module keeps new datasets from reintroducing it.
//
// Keep VALID_LANGUAGE_CODES in sync with metadata-fsm.h:203-208.

export const VALID_LANGUAGE_CODES: ReadonlySet<string> = new Set([
  'ar', 'az', 'bg', 'bn', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi',
  'fr', 'he', 'hi', 'hr', 'ht', 'hu', 'id', 'is', 'it', 'ja', 'ko', 'la', 'lt',
  'ms', 'ne', 'nl', 'no', 'pa', 'pl', 'pt', 'ro', 'ru', 'sa', 'sk', 'sr', 'sv',
  'sw', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'yue', 'zh', 'unknown',
]);

const NAME_TO_ISO: Record<string, string> = {
  arabic: 'ar', azerbaijani: 'az', bulgarian: 'bg', bengali: 'bn', catalan: 'ca',
  czech: 'cs', danish: 'da', german: 'de', greek: 'el', english: 'en',
  spanish: 'es', persian: 'fa', farsi: 'fa', finnish: 'fi', french: 'fr',
  hebrew: 'he', hindi: 'hi', croatian: 'hr', haitian: 'ht', hungarian: 'hu',
  indonesian: 'id', icelandic: 'is', italian: 'it', japanese: 'ja', korean: 'ko',
  latin: 'la', lithuanian: 'lt', malay: 'ms', nepali: 'ne', dutch: 'nl',
  norwegian: 'no', punjabi: 'pa', polish: 'pl', portuguese: 'pt', romanian: 'ro',
  russian: 'ru', sanskrit: 'sa', slovak: 'sk', serbian: 'sr', swedish: 'sv',
  swahili: 'sw', tamil: 'ta', telugu: 'te', thai: 'th', tagalog: 'tl',
  turkish: 'tr', ukrainian: 'uk', urdu: 'ur', vietnamese: 'vi', cantonese: 'yue',
  chinese: 'zh', mandarin: 'zh',
};

/**
 * Normalize a user/dataset language to an FSM-emittable ISO code.
 *
 * Already-valid codes pass through. Full names are mapped. An unrecognised
 * value falls back to `fallback` ('en') rather than being passed through —
 * passing it through is exactly how 'english' reached 2444 training rows.
 */
export function normalizeLanguage(value: unknown, fallback = 'en'): string {
  const raw = String(value ?? '').trim().toLowerCase();
  if (!raw) return fallback;
  if (VALID_LANGUAGE_CODES.has(raw)) return raw;
  const mapped = NAME_TO_ISO[raw];
  if (mapped) return mapped;
  // "en-GB" / "en_US" style tags: take the primary subtag when it is valid.
  const primary = raw.split(/[-_]/)[0];
  if (VALID_LANGUAGE_CODES.has(primary)) return primary;
  return fallback;
}

/** A dataset's declared lyric language: an ISO code, or '' for 'auto' — no
 *  override, so each file keeps the local AI's own guess (mixed-language
 *  corpora, #191). The dataset JSON spells '' as 'auto' so a rebuild keeps it. */
export function datasetLanguage(value: unknown): string {
  const raw = String(value ?? '').trim().toLowerCase();
  return raw === 'auto' ? '' : normalizeLanguage(raw);
}
