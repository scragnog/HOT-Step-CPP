// config.ts — Environment-based configuration for HOT-Step CPP server
import { config as dotenvConfig, parse as dotenvParse } from 'dotenv';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ── Portable mode detection ─────────────────────────────────────────
// When HOT_STEP_ROOT is set (by the release launcher), all paths resolve
// from the distribution root. Otherwise, fall back to __dirname-based
// resolution for development mode.
export const PORTABLE_MODE = !!process.env.HOT_STEP_ROOT;
export const PROJECT_ROOT = process.env.HOT_STEP_ROOT
  ? path.resolve(process.env.HOT_STEP_ROOT)
  : path.resolve(__dirname, '../..');  // two levels up from server/src/

// Load .env from project root (optional — smart defaults work without it)
// On first launch, bootstrap .env from .env.example so settings are writable.
const ENV_PATH = path.join(PROJECT_ROOT, '.env');
const ENV_EXAMPLE_PATH = path.join(PROJECT_ROOT, '.env.example');
if (!fs.existsSync(ENV_PATH) && fs.existsSync(ENV_EXAMPLE_PATH)) {
  try {
    fs.copyFileSync(ENV_EXAMPLE_PATH, ENV_PATH);
    console.log('[Config] Created .env from .env.example (first launch)');
  } catch (e: any) {
    console.warn('[Config] Could not create .env:', e.message);
  }
}
dotenvConfig({ path: ENV_PATH });

// Smart defaults: resolve paths relative to project root so users can
// build the engine and drop models in place without editing any config.
//
// Binary location depends on the layout:
//   - Portable release: engine/ace-server.exe (flat)
//   - Visual Studio (multi-config): engine/build/Release/ace-server.exe
//   - Ninja / Makefiles (single-config): engine/build/ace-server.exe
// We check all and use whichever exists.
const ENGINE_DIR = path.join(PROJECT_ROOT, 'engine');
const BUILD_DIR = path.join(ENGINE_DIR, 'build');

/** Platform-aware binary extension: .exe on Windows, empty on macOS/Linux */
const BIN_EXT = process.platform === 'win32' ? '.exe' : '';

const EXE_CANDIDATES = [
  path.join(ENGINE_DIR, `ace-server${BIN_EXT}`),             // Portable release (flat)
  path.join(BUILD_DIR, 'Release', `ace-server${BIN_EXT}`),   // Visual Studio
  path.join(BUILD_DIR, `ace-server${BIN_EXT}`),               // Ninja / Makefiles
  path.join(BUILD_DIR, 'Debug', `ace-server${BIN_EXT}`),      // VS Debug build
];
const DEFAULT_EXE = EXE_CANDIDATES.find(p => fs.existsSync(p)) || EXE_CANDIDATES[0];
const DEFAULT_MODELS = path.join(PROJECT_ROOT, 'models');
const DEFAULT_ADAPTERS = path.join(PROJECT_ROOT, 'adapters');
const DEFAULT_NOISE_SAMPLES = path.join(PROJECT_ROOT, 'noise_samples');
const DEFAULT_ONNX_DIR = path.join(PROJECT_ROOT, 'models', 'onnx');

// ── FFmpeg path resolution ──────────────────────────────────────────
// Portable: bundled ffmpeg.exe alongside the server.
// Dev mode: ffmpeg-static npm package provides the binary.
// Uses lazy init — resolved on first call, cached thereafter.

import { createRequire } from 'module';

let _ffmpegPath: string | null | undefined; // undefined = not yet resolved

/** Get the resolved path to ffmpeg. Checks portable location first, then ffmpeg-static. */
export function getFFmpegPath(): string | null {
  if (_ffmpegPath !== undefined) return _ffmpegPath;

  // 1. Portable: ffmpeg binary next to the bundled server
  const portablePath = path.join(PROJECT_ROOT, 'server', `ffmpeg${BIN_EXT}`);
  if (fs.existsSync(portablePath)) {
    _ffmpegPath = portablePath;
    return _ffmpegPath;
  }

  // 2. Dev: ffmpeg-static npm package
  try {
    const require = createRequire(import.meta.url);
    const ffmpegStatic = require('ffmpeg-static') as string | null;
    if (ffmpegStatic && fs.existsSync(ffmpegStatic)) {
      _ffmpegPath = ffmpegStatic;
      return _ffmpegPath;
    }
  } catch {
    // ffmpeg-static not installed — expected in portable mode
  }

  // 3. Not found
  _ffmpegPath = null;
  return null;
}


export const config = {
  // ace-server configuration
  aceServer: {
    exe: process.env.ACESTEPCPP_EXE || DEFAULT_EXE,
    models: process.env.ACESTEPCPP_MODELS || DEFAULT_MODELS,
    adapters: process.env.ACESTEPCPP_ADAPTERS || DEFAULT_ADAPTERS,
    port: parseInt(process.env.ACESTEPCPP_PORT || '8085', 10),
    host: process.env.ACESTEPCPP_HOST || '127.0.0.1',
    vaeChunk: parseInt(process.env.ACESTEPCPP_VAE_CHUNK || '1024', 10),
    vaeOverlap: parseInt(process.env.ACESTEPCPP_VAE_OVERLAP || '64', 10),
    noiseProfile: process.env.ACESTEPCPP_NOISE_PROFILE || (() => {
      // Auto-detect: find first .wav in noise_samples/
      const dir = DEFAULT_NOISE_SAMPLES;
      if (fs.existsSync(dir)) {
        const wavs = fs.readdirSync(dir).filter(f => f.endsWith('.wav'));
        if (wavs.length > 0) return path.join(dir, wavs[0]);
      }
      return '';
    })(),
    // Draft LM for speculative decoding — DISABLED
    // GGML per-call overhead (~10ms) makes sequential 0.6B forwards nearly as
    // expensive as the 4B target, negating the speedup. Left for future use if
    // persistent graphs or CUDA graphs reduce per-call overhead.
    // To re-enable: set ACESTEPCPP_DRAFT_LM env var or uncomment auto-detect.
    draftLm: process.env.ACESTEPCPP_DRAFT_LM || '',
    onnxDir: process.env.ACESTEPCPP_ONNX_DIR || DEFAULT_ONNX_DIR,
    /** Pass --keep-loaded to the spawned ace-server, flipping the engine's
     *  ModelStore to EVICT_NEVER at startup so DiT + adapter + LoKr
     *  precompute stay resident across requests. Default true: the cold-
     *  start LoKr precompute is ~17 s, hot-start ~50 ms — keeping models
     *  loaded is the right default for an interactive product. Override
     *  with ACESTEPCPP_KEEP_LOADED=0 if VRAM is tight and you'd rather
     *  pay the cold-start cost than carry several DiT+adapter combos. */
    keepLoaded: (process.env.ACESTEPCPP_KEEP_LOADED ?? '1') !== '0',
    /** Post /warm to the engine after it boots, pre-loading the configured
     *  DiT + VAE + adapter so the first user-facing /synth skips the ~7 min
     *  cold-start VRAM-copy phase. Requires keepLoaded — under EVICT_STRICT
     *  the engine drops the modules instantly, making the warm a waste.
     *  Default '1' when keepLoaded is on; set ACESTEPCPP_WARM_ON_STARTUP=0
     *  to disable. */
    warmOnStartup: (process.env.ACESTEPCPP_WARM_ON_STARTUP ?? '1') !== '0',
    /** Filename of the DiT to warm (resolved against models dir).
     *  Falls back to ACESTEP_DIT_MODEL (the worker hotstep.conf knob) so
     *  the warm naturally matches what the worker submits. */
    warmDit: process.env.ACESTEPCPP_WARM_DIT || process.env.ACESTEP_DIT_MODEL || '',
    /** Filename of the VAE to warm. */
    warmVae: process.env.ACESTEPCPP_WARM_VAE || process.env.ACESTEP_VAE_MODEL || '',
    /** Filename of the adapter to warm (resolved against adapters dir).
     *  Empty disables adapter pre-load — DiT/VAE alone still cuts most of
     *  the cold-start. */
    warmAdapter: process.env.ACESTEPCPP_WARM_ADAPTER || '',
    /** Adapter scale to use for the warm. Matches ACESTEP_LORA_SCALE default
     *  so the warm and the worker hit the same LoKr-delta cache key. */
    warmAdapterScale: parseFloat(process.env.ACESTEPCPP_WARM_ADAPTER_SCALE || process.env.ACESTEP_LORA_SCALE || '0.97'),
    /** TensorRT runtime DLL directory — auto-detected or TENSORRT_LIBS env override */
    trtLibs: process.env.TENSORRT_LIBS || (() => {
      // Auto-detect from engine/deps/tensorrt_libs/ (downloaded by Model Manager)
      const depsDir = path.join(ENGINE_DIR, 'deps', 'tensorrt_libs');
      if (fs.existsSync(path.join(depsDir, 'nvinfer_10.dll')) ||
          fs.existsSync(path.join(depsDir, 'libnvinfer.so.10'))) {
        return depsDir;
      }
      return '';
    })(),
    // draftLm: process.env.ACESTEPCPP_DRAFT_LM || (() => {
    //   const dir = process.env.ACESTEPCPP_MODELS || DEFAULT_MODELS;
    //   if (fs.existsSync(dir)) {
    //     const drafts = fs.readdirSync(dir)
    //       .filter(f => f.endsWith('.gguf') && (f.includes('-0.6B-') || f.includes('_0.6B_')))
    //       .sort((a, b) => {
    //         const aBF = a.includes('BF16') ? 1 : 0;
    //         const bBF = b.includes('BF16') ? 1 : 0;
    //         return bBF - aBF;
    //       });
    //     if (drafts.length > 0) return path.join(dir, drafts[0]);
    //   }
    //   return '';
    // })(),
    get url() {
      return `http://${this.host}:${this.port}`;
    },
  },

  // Essentia audio analysis
  essentia: {
    bin: process.env.ESSENTIA_BIN || path.join(PROJECT_ROOT, 'Essentia', `essentia_streaming_extractor_music${BIN_EXT}`),
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
    openaiCompatBaseUrl: process.env.OPENAI_COMPAT_BASE_URL || '',
    openaiCompatApiKey: process.env.OPENAI_COMPAT_API_KEY || '',
    openaiCompatModel: process.env.OPENAI_COMPAT_MODEL || '',
    openaiCompatName: process.env.OPENAI_COMPAT_NAME || 'OpenAI Compatible',
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
        ? path.join(path.dirname(aceExe), `vst-host${BIN_EXT}`)
        : path.join(BUILD_DIR, 'Release', `vst-host${BIN_EXT}`);
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

  // Whisper speech-to-text (lyrics transcription)
  whisper: {
    exe: process.env.WHISPER_EXE || path.join(PROJECT_ROOT, 'tools', 'whisper', `whisper-cli${BIN_EXT}`),
    modelsDir: process.env.WHISPER_MODELS_DIR || path.join(process.env.ACESTEPCPP_MODELS || DEFAULT_MODELS, 'whisper'),
  },
};

// ── .env hot-reload infrastructure ──────────────────────────────────

/** Absolute path to the project .env file */
export const ENV_FILE_PATH = path.join(PROJECT_ROOT, '.env');

/** Keys exposed to the Settings UI (whitelist — nothing else leaks) */
export const EXPOSED_ENV_KEYS = [
  // Engine
  'ACESTEPCPP_MODELS', 'ACESTEPCPP_ADAPTERS', 'ACESTEPCPP_PORT', 'ACESTEPCPP_HOST',
  'ACESTEPCPP_VAE_CHUNK', 'ACESTEPCPP_VAE_OVERLAP',
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
  'OPENAI_COMPAT_BASE_URL', 'OPENAI_COMPAT_API_KEY', 'OPENAI_COMPAT_MODEL', 'OPENAI_COMPAT_NAME',
  // Paths
  'LYRICS_EXPORT_DIR',
] as const;

/** Keys that require an app restart to take effect */
export const RESTART_REQUIRED_KEYS = new Set([
  'ACESTEPCPP_MODELS', 'ACESTEPCPP_ADAPTERS', 'ACESTEPCPP_PORT', 'ACESTEPCPP_HOST',
  'ACESTEPCPP_VAE_CHUNK', 'ACESTEPCPP_VAE_OVERLAP',
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
  apply('ACESTEPCPP_VAE_CHUNK', v => { config.aceServer.vaeChunk = parseInt(v || '1024', 10); },
    () => String(config.aceServer.vaeChunk));
  apply('ACESTEPCPP_VAE_OVERLAP', v => { config.aceServer.vaeOverlap = parseInt(v || '64', 10); },
    () => String(config.aceServer.vaeOverlap));

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
  apply('OPENAI_COMPAT_BASE_URL', v => { config.lireek.openaiCompatBaseUrl = v; },
    () => config.lireek.openaiCompatBaseUrl);
  apply('OPENAI_COMPAT_API_KEY', v => { config.lireek.openaiCompatApiKey = v; },
    () => config.lireek.openaiCompatApiKey);
  apply('OPENAI_COMPAT_MODEL', v => { config.lireek.openaiCompatModel = v; },
    () => config.lireek.openaiCompatModel);
  apply('OPENAI_COMPAT_NAME', v => { config.lireek.openaiCompatName = v || 'OpenAI Compatible'; },
    () => config.lireek.openaiCompatName);
  apply('LYRICS_EXPORT_DIR', v => {
    config.lireek.exportDir = v || path.join(config.data.dir, 'lyrics');
  }, () => config.lireek.exportDir);

  if (changed.length > 0) {
    console.log(`[Config] Hot-reloaded ${changed.length} setting(s): ${changed.join(', ')}`);
  }

  return changed;
}
