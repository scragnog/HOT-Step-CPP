// Translation: the application's mechanism (i18next, as ui/src/i18n), holding one table per locale. Every word the
// page shows is a message (an id and its slots, src/derive/messages.ts); say() fills it from the locale's table, so a
// definition's words, the core's sentences and the page's chrome are translated the same way. English is generated
// from the definitions (stringsOf), the core (coreStrings) and the page's own table (ui.ts).

import i18n from 'i18next';
import { initReactI18next, useTranslation } from 'react-i18next';
import { useCallback } from 'react';
import { DEFINITIONS } from '../../definitions/index.ts';
import { coreStrings, stringsOf } from '../../derive/strings.ts';
import { msg, say, type Msg, type Slot } from '../../derive/messages.ts';
import { UI_EN } from './ui.ts';

const EN_TABLE: { [id: string]: string } = {
  ...Object.fromEntries(coreStrings().map(s => [s.key, s.text])),
  ...Object.fromEntries(DEFINITIONS.flatMap(stringsOf).map(s => [s.key, s.text])),
  ...UI_EN,
};

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

/** Says a message (or any slot) in the page's language. */
export function useSay(): (s: Slot | undefined) => string {
  const { i18n: inst } = useTranslation();
  const lang = inst.language || 'en';
  return useCallback((s: Slot | undefined) => (s === undefined ? '' : say(s, tableOf(lang), lang)), [lang]);
}
/** The same outside React (a toast queued by the store, the document title). */
export const sayNow = (s: Slot): string => say(s, tableOf(i18n.language || 'en'), i18n.language || 'en');
/** A message of the page's own table. */
export const ui = (id: string, slots?: Msg['slots']): Msg => msg(`ui.${id}`, slots);

export default i18n;
