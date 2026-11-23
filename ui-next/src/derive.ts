// What is derived from the definitions, written once, as pure functions of the definitions and the desk. The page
// is a view over these; the validator and the tools call them too. Nothing here has a branch for any one module.
// docs/05-what-the-definition-drives.md says what each one does.
//
//   derive/core.ts      a parameter on an engine, conditions, values, how a number is written, words, faces
//   derive/request.ts   the request: the desk in real names, and the Create intent

export * from './derive/core.ts';
export * from './derive/request.ts';
export * from './derive/values.ts';
export * from './derive/steps.ts';
export * from './derive/engine.ts';
export * from './derive/tiers.ts';
export * from './derive/records.ts';
export * from './derive/parts.ts';
export * from './derive/strings.ts';
export * from './derive/messages.ts';
export * from './derive/view.ts';
export * from './derive/desk.ts';
export * from './derive/models.ts';
export * from './derive/controls.ts';
export * from './derive/vram.ts';
