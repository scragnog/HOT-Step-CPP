// What saved data may still name that no definition has any more: parameter keys, state keys and definition ids that
// were removed on purpose. A record that names one loads without it and without a word; anything else it names that
// no definition has is counted ("2 settings could not be used"). A rename is not retirement: the new key, id or
// option lists the old one in `was`, and old records read it as the new one.
//
// tools/fixtures/shipped.json lists every key, state and id the definitions have shipped with; the validator fails
// when one of them is neither in a definition (under its name or in a `was`) nor here, so removing or renaming
// something that saved data holds is always a decision written down. `npm run docs` adds new ones to that list.

export const RETIRED: { keys: readonly string[]; states: readonly string[]; ids: readonly string[] } = {
  keys: [],
  states: [],
  ids: [],
};
