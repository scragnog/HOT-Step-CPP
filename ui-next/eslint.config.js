import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

// The existing frontend's rules (ui/eslint.config.js), for the React layer in src/app. The core (src/derive,
// src/definitions) and the tools are checked by `npm run check` (the TypeScript compiler and the validator).
export default defineConfig([
  globalIgnores(['dist', 'dist-file', 'dist-test', 'tools', 'src/derive', 'src/definitions', 'src/derive.ts']),
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
      'no-restricted-globals': ['error', { name: 'fetch', message: 'Use the engine (src/app/services) for HTTP transport.' }],
    },
  },
])
