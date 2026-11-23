import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

// fetchQuery is deprecated in TanStack Query v5 (removed next major);
// queryClient.query() has identical semantics. Keep the P1 sweep permanent.
const noFetchQuery = {
  selector: "CallExpression[callee.property.name='fetchQuery']",
  message: 'queryClient.fetchQuery is deprecated — use queryClient.query() (docs/dev/tanstack-query.md).',
}

// P8.4: a component observes server state through a factory; it does not read
// the cache imperatively. Every file below is audited — a command path (read
// what the server just created, or the revision a write must match), a one-shot
// form seed (convention 9 in docs/dev/tanstack-query.md) or the pipeline stage
// adoption effect. Shrink this list as sites move to useQuery; add a file only
// with an entry in the P8 appendix.
const componentQueryExemptions = [
  'src/components/insta-gen/InstaGenPanel.tsx',
  'src/components/song-builder/SongBuilder.tsx',
  'src/components/lyric-studio/RecordingsTab.tsx',
  'src/components/lyric-studio/ProviderSelector.tsx',
  'src/components/stem-studio/StemStudio.tsx',
  'src/components/training-studio/Yue2OptimisePanel.tsx',
  'src/components/training-studio/Yue2LadderReview.tsx',
  'src/components/training-studio/sendAuditionToCustomGen.ts',
  'src/components/training-studio/SendToLyricStudio.tsx',
  'src/components/training-studio/BatchImportWizard.tsx',
  'src/components/training-studio/ReviewPanel.tsx',
  'src/components/training-studio/Yue2AitkTrainCard.tsx',
  'src/components/training-studio/RefinePanel.tsx',
  'src/components/training-studio/PipelineStageProgress.tsx',
]

const noComponentQuery = {
  selector: "CallExpression[callee.property.name='query']",
  message: 'Imperative cache read in a component — observe the factory with useQuery, or keep the read in the owning service/store (docs/dev/tanstack-query.md#appendix-p8-reactive-read-migration).',
}

export default defineConfig([
  globalIgnores(['dist', 'scripts/skin-fixtures-shells/**']),
  {
    files: ['**/*.{ts,tsx}'],
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
      'no-restricted-syntax': ['error', noFetchQuery],
    },
  },
  // App code shares contracts only; tests may exercise server implementations.
  {
    files: ['**/*.{ts,tsx}'],
    ignores: ['**/*.test.{ts,tsx}'],
    rules: {
      'no-restricted-imports': ['error', {
        patterns: [{
          regex: '(^|/)server/src/(?!contracts/)',
          message: 'Import shared contracts from server/src/contracts/ instead of server implementation code.',
        }],
      }],
    },
  },
  // Shell skins (docs/dev/skins.md#shell-skins) import their own folder, skin-api, react,
  // lucide-react and the kit, and nothing else from src/. This block replaces the one above
  // for these files. It catches the common reaches;
  // scripts/check-skins.mjs resolves every import path exactly.
  {
    files: ['src/shells/**/*.{ts,tsx}'],
    ignores: ['**/*.test.{ts,tsx}'],
    rules: {
      'no-restricted-imports': ['error', {
        patterns: [
          {
            regex: '(^|/)server/src/',
            message: 'A shell may not import from server/src. Types it needs come through skin-api.',
          },
          {
            regex: '^(?![./])(?!react$)(?!react/jsx-runtime$)(?!lucide-react$)',
            message: 'A shell may import only react, lucide-react, its own folder, skin-api and the kit.',
          },
          {
            regex: '(^|/)components/(?!shared/kit/)',
            message: 'A shell may import only components/shared/kit/. Classic components stay in classic.',
          },
          {
            regex: '^(\\.\\./)+(app|audio|constants|context|data|hooks|i18n|queries|services|stores|themes|types|utils|assets)(/|$)',
            message: 'A shell reaches the app only through skin-api (docs/dev/skins.md).',
          },
        ],
      }],
    },
  },
  // Components read through an options factory. See P8.4 in the TanStack doc.
  {
    files: ['src/components/**/*.{ts,tsx}'],
    ignores: [...componentQueryExemptions, 'src/components/**/*.test.*'],
    rules: {
      'no-restricted-syntax': ['error', noFetchQuery, noComponentQuery],
    },
  },
  {
    files: ['src/**/*.{ts,tsx}'],
    ignores: ['src/services/**', 'src/queries/**', '**/*.test.{ts,tsx}', 'src/stores/audioQueueStoreTestFake.ts'],
    rules: {
      'no-restricted-globals': ['error', { name: 'fetch', message: 'Use a service in src/services/ for HTTP transport.' }],
      'no-restricted-properties': ['error', ...['window', 'globalThis', 'self'].map(object => ({
        object, property: 'fetch', message: 'Use a service in src/services/ for HTTP transport.',
      }))],
    },
  },
])
