// config.ts — Environment-based configuration for HOT-Step CPP server
import { config as dotenvConfig, parse as dotenvParse } from 'dotenv';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Project root is two levels up from server/src/
const PROJECT_ROOT = path.resolve(__dirname, '../..');

// Load .env from project root (optional — smart defaults work without it)
dotenvConfig({ path: path.join(PROJECT_ROOT, '.env') });

// Smart defaults: resolve paths relative to project root so users can
// build the engine and drop models in place without editing any config.
//
// Binary location depends on the CMake generator used:
//   - Visual Studio (multi-config): engine/build/Release/ace-server.exe
//   - Ninja / Makefiles (single-config): engine/build/ace-server.exe
// We check both and use whichever exists.
const BUILD_DIR = path.join(PROJECT_ROOT, 'engine', 'build');
const EXE_CANDIDATES = [
  path.join(BUILD_DIR, 'Release', 'ace-server.exe'),  // Visual Studio
  path.join(BUILD_DIR, 'ace-server.exe'),              // Ninja / Makefiles
  path.join(BUILD_DIR, 'Debug', 'ace-server.exe'),     // VS Debug build
];
const DEFAULT_EXE = EXE_CANDIDATES.find(p => fs.existsSync(p)) || EXE_CANDIDATES[0];
const DEFAULT_MODELS = path.join(PROJECT_ROOT, 'models');
const DEFAULT_ADAPTERS = path.join(PROJECT_ROOT, 'adapters');
const DEFAULT_NOISE_SAMPLES = path.join(PROJECT_ROOT, 'noise_samples');

export const config = {
  // ace-server configuration
  aceServer: {
    exe: process.env.ACESTEPCPP_EXE || DEFAULT_EXE,
    models: process.env.ACESTEPCPP_MODELS || DEFAULT_MODELS,
    adapters: process.env.ACESTEPCPP_ADAPTERS || DEFAULT_ADAPTERS,
    port: parseInt(process.env.ACESTEPCPP_PORT || '8085', 10),
    host: process.env.ACESTEPCPP_HOST || '127.0.0.1',
    noiseProfile: process.env.ACESTEPCPP_NOISE_PROFILE || (() => {
      // Auto-detect: find first .wav in noise_samples/
      const dir = DEFAULT_NOISE_SAMPLES;
      if (fs.existsSync(dir)) {
        const wavs = fs.readdirSync(dir).filter(f => f.endsWith('.wav'));
        if (wavs.length > 0) return path.join(dir, wavs[0]);
      }
      return '';
    })(),
    get url() {
      return `http://${this.host}:${this.port}`;
    },
  },

  // Essentia audio analysis
  essentia: {
    bin: process.env.ESSENTIA_BIN || path.join(PROJECT_ROOT, 'Essentia', 'essentia_streaming_extractor_music.exe'),
  },

  // Node.js server
  server: {
    port: parseInt(process.env.SERVER_PORT || '3001', 10),
    host: process.env.SERVER_HOST || '0.0.0.0',
  },

  // Data paths
  data: {
    dir: path.resolve(__dirname, '..', process.env.DATA_DIR || './data'),
    get dbPath() {
      return path.join(this.dir, 'hotstep.db');
    },
    get audioDir() {
      return path.join(this.dir, 'audio');
    },
  },

  // Lyric Studio / Lireek
  lireek: {
    geniusAccessToken: process.env.GENIUS_ACCESS_TOKEN || '',
    geminiApiKey: process.env.GEMINI_API_KEY || '',
    openaiApiKey: process.env.OPENAI_API_KEY || '',
    anthropicApiKey: process.env.ANTHROPIC_API_KEY || '',
    ollamaBaseUrl: process.env.OLLAMA_BASE_URL || 'http://localhost:11434',
    lmstudioBaseUrl: process.env.LMSTUDIO_BASE_URL || 'http://localhost:1234/v1',
    unslothBaseUrl: process.env.UNSLOTH_BASE_URL || 'http://127.0.0.1:8888',
    unslothUsername: process.env.UNSLOTH_USERNAME || '',
    unslothPassword: process.env.UNSLOTH_PASSWORD || '',
    defaultProvider: process.env.DEFAULT_LLM_PROVIDER || 'gemini',
    geminiModel: process.env.GEMINI_MODEL || 'gemini-2.5-flash',
    openaiModel: process.env.OPENAI_MODEL || 'gpt-4o-mini',
    anthropicModel: process.env.ANTHROPIC_MODEL || 'claude-3-5-haiku-20241022',
    ollamaModel: process.env.OLLAMA_MODEL || 'llama3',
    lmstudioModel: process.env.LMSTUDIO_MODEL || '',
    unslothModel: process.env.UNSLOTH_MODEL || '',
    get dbPath() {
      return path.join(config.data.dir, 'lireek.db');
    },
    exportDir: process.env.LYRICS_EXPORT_DIR || path.join(
      path.resolve(__dirname, '..', process.env.DATA_DIR || './data'), 'lyrics'
    ),
  },

  // VST3 Post-Processing
  vst: {
    /** Path to vst-host.exe — lives in same dir as ace-server.exe */
    get exe() {
      const aceExe = config.aceServer.exe;
      return aceExe
        ? path.join(path.dirname(aceExe), 'vst-host.exe')
        : path.join(BUILD_DIR, 'Release', 'vst-host.exe');
    },
    /** Directory for .vststate binary blobs */
    get statesDir() {
      return path.join(config.data.dir, 'vst', 'states');
    },
    /** Persistent chain config file */
    get chainFile() {
      return path.join(config.data.dir, 'vst', 'chain.json');
    },
  },
};

// ── .env hot-reload infrastructure ──────────────────────────────────

/** Absolute path to the project .env file */
export const ENV_FILE_PATH = path.join(PROJECT_ROOT, '.env');

/** Keys exposed to the Settings UI (whitelist — nothing else leaks) */
export const EXPOSED_ENV_KEYS = [
  // Engine
  'ACESTEPCPP_MODELS', 'ACESTEPCPP_ADAPTERS', 'ACESTEPCPP_PORT', 'ACESTEPCPP_HOST',
  // Server
  'SERVER_PORT', 'DATA_DIR',
  // API keys
  'GENIUS_ACCESS_TOKEN', 'GEMINI_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY',
  // LLM config
  'DEFAULT_LLM_PROVIDER',
  'GEMINI_MODEL', 'OPENAI_MODEL', 'ANTHROPIC_MODEL',
  'OLLAMA_MODEL', 'LMSTUDIO_MODEL', 'UNSLOTH_MODEL',
  // LLM endpoints
  'OLLAMA_BASE_URL', 'LMSTUDIO_BASE_URL',
  'UNSLOTH_BASE_URL', 'UNSLOTH_USERNAME', 'UNSLOTH_PASSWORD',
  // Paths
  'LYRICS_EXPORT_DIR',
] as const;

/** Keys that require an app restart to take effect */
export const RESTART_REQUIRED_KEYS = new Set([
  'ACESTEPCPP_MODELS', 'ACESTEPCPP_ADAPTERS', 'ACESTEPCPP_PORT', 'ACESTEPCPP_HOST',
  'SERVER_PORT', 'DATA_DIR',
]);

/**
 * Re-read the .env file and hot-patch the live config object.
 * Returns a list of keys that actually changed.
 */
export function reloadEnvConfig(): string[] {
  const envContent = fs.existsSync(ENV_FILE_PATH)
    ? fs.readFileSync(ENV_FILE_PATH, 'utf-8')
    : '';
  const parsed = dotenvParse(envContent);
  const changed: string[] = [];

  // Helper: update a config property if the env value changed
  const apply = (envKey: string, setter: (val: string) => void, getter: () => string) => {
    const newVal = parsed[envKey] ?? '';
    if (newVal !== getter()) {
      setter(newVal);
      changed.push(envKey);
    }
  };

  // ── Engine (values stored but won't affect running child process) ──
  apply('ACESTEPCPP_MODELS', v => { config.aceServer.models = v || DEFAULT_MODELS; },
    () => config.aceServer.models);
  apply('ACESTEPCPP_ADAPTERS', v => { config.aceServer.adapters = v || DEFAULT_ADAPTERS; },
    () => config.aceServer.adapters);
  apply('ACESTEPCPP_PORT', v => { config.aceServer.port = parseInt(v || '8085', 10); },
    () => String(config.aceServer.port));
  apply('ACESTEPCPP_HOST', v => { config.aceServer.host = v || '127.0.0.1'; },
    () => config.aceServer.host);

  // ── Server ──
  apply('SERVER_PORT', v => { config.server.port = parseInt(v || '3001', 10); },
    () => String(config.server.port));
  apply('DATA_DIR', v => {
    config.data.dir = path.resolve(__dirname, '..', v || './data');
  }, () => config.data.dir);

  // ── Lireek / LLM (hot-reloaded — takes effect immediately) ──
  apply('GENIUS_ACCESS_TOKEN', v => { config.lireek.geniusAccessToken = v; },
    () => config.lireek.geniusAccessToken);
  apply('GEMINI_API_KEY', v => { config.lireek.geminiApiKey = v; },
    () => config.lireek.geminiApiKey);
  apply('OPENAI_API_KEY', v => { config.lireek.openaiApiKey = v; },
    () => config.lireek.openaiApiKey);
  apply('ANTHROPIC_API_KEY', v => { config.lireek.anthropicApiKey = v; },
    () => config.lireek.anthropicApiKey);
  apply('OLLAMA_BASE_URL', v => { config.lireek.ollamaBaseUrl = v || 'http://localhost:11434'; },
    () => config.lireek.ollamaBaseUrl);
  apply('LMSTUDIO_BASE_URL', v => { config.lireek.lmstudioBaseUrl = v || 'http://localhost:1234/v1'; },
    () => config.lireek.lmstudioBaseUrl);
  apply('UNSLOTH_BASE_URL', v => { config.lireek.unslothBaseUrl = v || 'http://127.0.0.1:8888'; },
    () => config.lireek.unslothBaseUrl);
  apply('UNSLOTH_USERNAME', v => { config.lireek.unslothUsername = v; },
    () => config.lireek.unslothUsername);
  apply('UNSLOTH_PASSWORD', v => { config.lireek.unslothPassword = v; },
    () => config.lireek.unslothPassword);
  apply('DEFAULT_LLM_PROVIDER', v => { config.lireek.defaultProvider = v || 'gemini'; },
    () => config.lireek.defaultProvider);
  apply('GEMINI_MODEL', v => { config.lireek.geminiModel = v || 'gemini-2.5-flash'; },
    () => config.lireek.geminiModel);
  apply('OPENAI_MODEL', v => { config.lireek.openaiModel = v || 'gpt-4o-mini'; },
    () => config.lireek.openaiModel);
  apply('ANTHROPIC_MODEL', v => { config.lireek.anthropicModel = v || 'claude-3-5-haiku-20241022'; },
    () => config.lireek.anthropicModel);
  apply('OLLAMA_MODEL', v => { config.lireek.ollamaModel = v || 'llama3'; },
    () => config.lireek.ollamaModel);
  apply('LMSTUDIO_MODEL', v => { config.lireek.lmstudioModel = v; },
    () => config.lireek.lmstudioModel);
  apply('UNSLOTH_MODEL', v => { config.lireek.unslothModel = v; },
    () => config.lireek.unslothModel);
  apply('LYRICS_EXPORT_DIR', v => {
    config.lireek.exportDir = v || path.join(config.data.dir, 'lyrics');
  }, () => config.lireek.exportDir);

  if (changed.length > 0) {
    console.log(`[Config] Hot-reloaded ${changed.length} setting(s): ${changed.join(', ')}`);
  }

  return changed;
}
