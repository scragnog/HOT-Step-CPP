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
  globalIgnores(['dist']),
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
  // Components read through an options factory. See P8.4 in the TanStack doc.
  {
    files: ['src/components/**/*.{ts,tsx}'],
    ignores: [...componentQueryExemptions, 'src/components/**/*.test.*'],
    rules: {
      'no-restricted-syntax': ['error', noFetchQuery, noComponentQuery],
    },
  },
])
