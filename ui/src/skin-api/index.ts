// skin-api version 0 (unstable). The only module a shell skin may import from the app
// (besides its own folder, react, lucide-react and the kit). Explicit named exports only,
// never `export *`: adding a name is a minor change, removing or renaming one is a major bump.
// Reference and rules: docs/dev/skins.md#shell-skins.

export const SKIN_API_VERSION = 0;

export { useTheme, useThemeSkin, openInStudio, openClassic, useSkinStorage, SKIN_STORAGE_LIMIT } from './misc';
export { useParam, useParamMeta, params, useParamHistory } from './params';
export { useSongs, useRecentSongs, useQueue, generate, cancel } from './library';
export { useBackends, useActiveBackend, backend, useCapabilities, useBackendModels, useVram, useEngineLog } from './engine';
export { ClassicPage } from '../app/ClassicPage';

export type { ClassicPageView } from '../app/ClassicPage';
export type {
  ThemeMode, ParamId, ParamOption, ParamMeta, ParamHistoryStep, SkinSong, QueueState, SkinQueueItem,
  GenerateRequest, SkinBackend, SkinCapabilities, SkinModelCatalogue, SkinVram, SkinLogLine, StudioKind,
} from './types';
