// Translation: the application's mechanism (i18next, as ui/src/i18n), holding one table per locale, on an instance of
// its own (createInstance), so the page never shares the application's default instance and its language detector
// (SL-16). Every word the page shows is a message (an id and its slots, src/derive/messages.ts); say() fills it from
// the locale's table, so a definition's words, the core's sentences and the page's chrome are translated the same way.
// English is generated from the definitions (stringsOf), the core (coreStrings) and the page's own words, which are
// gathered from i18n/words/*.ts (one file per work package; an id defined twice fails at load).

import i18next from 'i18next';
import { initReactI18next } from 'react-i18next';
import { useCallback, useSyncExternalStore } from 'react';
import { DEFINITIONS } from '../../definitions/index.ts';
import { coreStrings, stringsOf } from '../../derive/strings.ts';
import { msg, say, type Msg, type Slot } from '../../derive/messages.ts';

const WORD_FILES = import.meta.glob<{ WORDS: { [id: string]: string } }>('./words/*.ts', { eager: true });
const PAGE: { [id: string]: string } = {};
for (const [file, m] of Object.entries(WORD_FILES)) for (const [id, text] of Object.entries(m.WORDS)) {
  if (id in PAGE) throw new Error(`the word ${id} is defined twice (${file})`);
  PAGE[id] = text;
}
/** Every id of the page's own words, for the check that each one a component uses exists. */
export const PAGE_WORDS: ReadonlySet<string> = new Set(Object.keys(PAGE));

const EN_TABLE: { [id: string]: string } = {
  ...Object.fromEntries(coreStrings().map(s => [s.key, s.text])),
  ...Object.fromEntries(DEFINITIONS.flatMap(stringsOf).map(s => [s.key, s.text])),
  ...PAGE,
};

const i18n = i18next.createInstance();
void i18n.use(initReactI18next).init({
  resources: { en: { translation: EN_TABLE } },
  lng: 'en',
  fallbackLng: 'en',
  keySeparator: false,
  nsSeparator: false,
  interpolation: { escapeValue: false },
});

/** The locale's table, falling back to English for an id it lacks. */
const tableOf = (lang: string): { [id: string]: string } => {
  const t = (i18n.getResourceBundle(lang, 'translation') ?? {}) as { [id: string]: string };
  return lang === 'en' ? t : { ...EN_TABLE, ...t };
};

/** How words are printed: as written, or in capitals (an LCD's first line, a picker's face) or small letters, in the
 *  page's language. */
export interface SayCase { upper?: boolean; lower?: boolean }
const cased = (t: string, lang: string, o?: SayCase) => (o?.upper ? t.toLocaleUpperCase(lang) : o?.lower ? t.toLocaleLowerCase(lang) : t);

/* the page's language, which a component that says words follows: a subscription of its own to the instance's change of
   language, so each of the page's many components costs no more than that (the tables are all loaded at start) */
const onLanguage = (f: () => void) => { i18n.on('languageChanged', f); return () => i18n.off('languageChanged', f); };
const language = () => i18n.language || 'en';
/** Says a message (or any slot) in the page's language. */
export function useSay(): (s: Slot | undefined, o?: SayCase) => string {
  const lang = useSyncExternalStore(onLanguage, language);
  return useCallback((s: Slot | undefined, o?: SayCase) => (s === undefined ? '' : cased(say(s, tableOf(lang), lang), lang, o)), [lang]);
}
/** The same outside React (a toast queued by the store, the document title). */
export const sayNow = (s: Slot): string => say(s, tableOf(i18n.language || 'en'), i18n.language || 'en');
/** A message of the page's own words. */
export const ui = (id: string, slots?: Msg['slots']): Msg => msg(`ui.${id}`, slots);

export default i18n;
