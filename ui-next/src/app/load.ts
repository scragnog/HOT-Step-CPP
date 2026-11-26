// Where the page's registration points are filled: every file in these folders is loaded once, in name order, and
// registers what it adds. A work package adds its files here and edits none of these lines (docs/09, "Adding without
// colliding").
//
//   features/*.ts(x)       keyboard shortcuts (register/keys.ts), slots it fills (register/slots.ts), named actions
//                          (register/actions.ts), undoable state (register/undoable.ts)
//   parts/*.tsx            the parts (kit/define.ts definePart)
//   i18n/words/*.ts        the page's own words (i18n/index.ts gathers them)
//   styles/rebuild/*.css   the rebuild's stylesheets, after the concept's (styles/index.ts)

import.meta.glob('./features/*.{ts,tsx}', { eager: true });
import.meta.glob('./parts/*.tsx', { eager: true });
