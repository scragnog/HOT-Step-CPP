import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'
import { ENGINES } from './src/definitions/engines.ts'

// The existing frontend's rules (ui/eslint.config.js), for the React layer in src/app, and the React layer's own two
// rules (docs/09-architecture.md, "The rules the lint keeps"). The core (src/derive, src/definitions) and the tools are
// checked by `npm run check` (the TypeScript compiler, the validator and the tests).

/* 1. A part has no route to the store. A file in src/app/parts may import react and react-dom, the part kit
   (src/app/kit, which also gives a bespoke part responsible for a server conversation the boundary's `service`), its
   sibling parts, and nothing else: not the store, not the core, not the definitions, not the shell, not a service's
   implementation. It is given its values, its words and setters for exactly what its node
   binds (kit/types.ts PartView), and the store, the core and the binding do the rest. The rule catches an import of
   anything else; with no import there is no name to reach the store by (the test build's compatibility globals are
   test-only, and tools/test-parts-contract.mjs plants a part that tries both and must fail this lint). */
const PART_ONLY = [
  { group: ['../*', '../**', '!../kit', '!../kit/**', '../../*', '../../**'], message: "A part imports only ../kit (its props and helpers, and the server boundary's service), react and its sibling parts. It is given its values and setters: docs/09, 'How a part is written'." },
  { group: ['zustand', 'zustand/**', 'i18next', 'react-i18next'], message: 'A part has no store and no translation of its own: it says words with useSay from ../kit.' },
]

/* 2. The view decides nothing about a value. Outside the files that are about values by their job (the store, the
   binding, the test layer, the model index, the core), the React layer may not:
   - name an engine (a string that is an engine id): what an engine has or says is the definitions' and the core's;
   - read a definition's fields that decide a value or what is shown (engines, labUnder, builtIn, source, gated,
     when, range, default, on, per, on a variable named as a definition or a parameter is: d, def, p, param, x; or a
     parameter's face): ask the core (usedHere, tileAtTier, partModel ...);
   - import the core's value functions (valueOf, has, applies, changed, isAuto, resolve, wordsOf, setValue,
     cleanValue, formatValue, boardNodes, idleOf, stripOf): the core's models already hold their answers;
   - format or parse a number (toLocaleString, toPrecision, Intl, parseFloat, parseInt, Number()) or change the case
     of words (toUpperCase and the rest): the core writes values, useSay prints words in capitals;
   - write English: a JSX text with a word in it, or a literal title, aria-label, placeholder or alt; every word is a
     message (useSay, ui()).
   What it cannot catch: a decision written with none of these (comparing two values the core gave, arithmetic on a
   number to place a hand, which is the part's job), English inside a template literal or a constant, a definition
   field read through a variable name, or the same rule stated twice inside the core. Those are the review's to find;
   the allowed drawing files below may use toFixed for coordinates. */
const ENGINE_ID = `/^(${ENGINES.map(e => e.id.replace(/[-]/g, '\\-')).join('|')})$/`
const VIEW_RULES = [
  { selector: `Literal[value=${ENGINE_ID}]`, message: 'The view names no engine: what an engine has or says is the core\'s (docs/09, "The one rule").' },
  { selector: 'MemberExpression[object.name=/^(d|def|definition|p|param|x)$/][property.name=/^(engines|labUnder|builtIn|source|gated|when|range|default|on|per)$/], MemberExpression[object.property.name="face"][property.name=/^(range|default|unit|options)$/]', message: 'This definition field decides a value or what is shown: ask the core (src/derive/models.ts) instead of reading it here.' },
  { selector: 'CallExpression[callee.property.name=/^(toLocaleString|toPrecision|toUpperCase|toLowerCase|toLocaleUpperCase|toLocaleLowerCase)$/]', message: 'The core writes values and useSay prints words in capitals (say(slot, { upper: true })): no formatting or casing in the view.' },
  { selector: 'CallExpression[callee.name=/^(parseFloat|parseInt|Number)$/], NewExpression[callee.object.name="Intl"]', message: 'Reading a number is the core\'s (parseTyped): hand the text to the setter (p.typed).' },
  { selector: 'JSXText[value=/[A-Za-z]{2,}/]', message: 'Every word is a message: say(ui(...)) or the core\'s words, never English in the markup.' },
  { selector: 'JSXAttribute[name.name=/^(title|aria-label|placeholder|alt)$/] > Literal', message: 'Every word is a message: say(ui(...)) or the core\'s words.' },
]
/* 3. One server boundary (docs/09, "The service boundary"): the page asks for a service by name (service('models')),
   never imports an implementation; main.tsx registers them all (the simulated server, services/mock, until package 10's
   real adapters take its place) and the test layer drives the simulated server's own knobs. */
const NO_MOCKS = { group: ['**/services/mock', '**/services/mock/**'], message: "Ask for the service by name (service('name'), register/services.ts): main.tsx chooses the implementation, so the real adapters can take the simulated server's place in one line." }
const VALUE_FUNCTIONS = ['valueOf', 'has', 'applies', 'changed', 'isAuto', 'resolve', 'wordsOf', 'setValue', 'cleanValue', 'formatValue', 'boardNodes', 'idleOf', 'stripOf']
const NO_VALUE_FUNCTIONS = { name: '../../derive.ts', importNames: VALUE_FUNCTIONS, message: 'The view asks the core\'s models (src/derive/models.ts), which already hold these answers.' }
/* the files that are about values by their job, and the drawing files whose numbers are coordinates */
const VALUE_FILES = ['src/app/store/**', 'src/app/render/bind.ts', 'src/app/testing/**', 'src/app/model/**', 'src/app/i18n/**', 'src/app/services/**']
/* drawing: coordinates and sizes read from the page (parseFloat of a computed style) and key names, not values; the
   engines' marks and colours and the definitions' drawings, which are per engine by what they are */
const DRAWING = ['src/app/board/cables.ts', 'src/app/parts/xyMotion.ts', 'src/app/parts/motion.ts', 'src/app/desk/layout.ts', 'src/app/desk/ring.ts', 'src/app/register/keys.ts']
const IDENTITY = ['src/app/shell/engines.ts', 'src/app/thumbs/**']

export default defineConfig([
  globalIgnores(['dist', 'dist-file', 'dist-test', 'dist-switch', 'tools', 'src/derive', 'src/definitions', 'src/derive.ts']),
  {
    files: ['src/app/**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    rules: {
      // the view talks to the engine only through src/app/services (the existing frontend's rule)
      'no-restricted-globals': ['error', { name: 'fetch', message: 'Use a service (src/app/services, register/services.ts service(name)) for HTTP transport.' }],
      'no-restricted-syntax': ['error', ...VIEW_RULES],
      'no-restricted-imports': ['error', { paths: [NO_VALUE_FUNCTIONS], patterns: [NO_MOCKS] }],
    },
  },
  {
    files: ['src/app/parts/**/*.{ts,tsx}'],
    /* a part cannot reach a definition at all (rule 1), so the rule on definition fields is not asked there: a part's
       own ParamView has a range and a default the core worked out */
    rules: {
      'no-restricted-imports': ['error', { paths: [NO_VALUE_FUNCTIONS], patterns: [...PART_ONLY, NO_MOCKS] }],
      'no-restricted-syntax': ['error', ...VIEW_RULES.filter(r => !/MemberExpression/.test(r.selector)),
        { selector: 'Identifier[name=/^(window|globalThis|self)$/]', message: 'A part reaches nothing through the page globals: it is given its values and setters.' }],
    },
  },
  { files: VALUE_FILES, rules: { 'no-restricted-syntax': 'off', 'no-restricted-imports': ['error', { patterns: [NO_MOCKS] }] } },
  { files: ['src/app/main.tsx', 'src/app/testing/**', 'src/app/services/mock/**'], rules: { 'no-restricted-imports': 'off' } },
  { files: DRAWING, rules: { 'no-restricted-syntax': ['error', ...VIEW_RULES.filter(r => !/toLocaleString|parseFloat/.test(r.selector))] } },
  { files: IDENTITY, rules: { 'no-restricted-syntax': 'off' } },
])
