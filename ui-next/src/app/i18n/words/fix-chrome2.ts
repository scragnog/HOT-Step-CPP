// The second fix round's words around the desk: an Undo or Redo with nothing of its step left to move says so, and
// what changed since (F-24, 14-undo.js); a symbol of a plugin menu shown under a finger says a second tap uses it (S-02).

export const WORDS: { [id: string]: string } = {
  'ui.undone.nothing': 'Nothing to put back for {what}',
  'ui.undone.nothing.kept': 'Nothing to put back for {what}: {kept}',
  'ui.redone.nothing': 'Nothing to redo for {what}',
  'ui.redone.nothing.kept': 'Nothing to redo for {what}: {kept}',
  'ui.dd.tapAgain': '{name}. Tap again to use it',
};
