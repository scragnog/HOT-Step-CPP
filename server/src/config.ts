// config.ts — Environment-based configuration for HOT-Step CPP server
import { config as dotenvConfig, parse as dotenvParse } from 'dotenv';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { AsyncLocalStorage } from 'node:async_hooks';
// Must stay the first local import: it picks (and if needed moves) the release
// data folder before anything below reads a path.
import { RUNTIME } from './dataroot/bootstrap.js';
import { defaultServerHost } from './services/security/originPolicy.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ── Portable mode detection ─────────────────────────────────────────
// When HOT_STEP_ROOT is set (by the release launcher), app files resolve
// from the distribution root and user data from the data folder RUNTIME
// chose (dataroot/bootstrap.ts). Otherwise app files resolve from the
// checkout, and user data from the checkout too unless Settings gave this
// checkout a data folder (its dev pointer).
export const PORTABLE_MODE = RUNTIME.mode === 'release';
export const PROJECT_ROOT = RUNTIME.appRoot ?? path.resolve(__dirname, '../..');  // two levels up from server/src/
/** Data folder (null in a dev checkout that has none). */
export const DATA_ROOT = RUNTIME.dataRoot;
export const LOGS_DIR = RUNTIME.logsDir ?? path.join(PROJECT_ROOT, 'logs');
export const RESTART_MARKER_PATH = RUNTIME.restartMarkerPath ?? path.join(PROJECT_ROOT, '.restart-requested');

/** A DATA_DIR-style value: resolved against the data folder when there is
 *  one, otherwise against server/ as dev always has. */
function resolveDataDir(v: string | undefined): string {
  if (RUNTIME.dataRoot) return v ? path.resolve(RUNTIME.dataRoot, v) : RUNTIME.libraryDir!;
  return path.resolve(__dirname, '..', v || './data');
}
/** Other path values: relative ones resolve against the data folder when there is one. */
function resolveDataPath(v: string): string {
  return RUNTIME.dataRoot && v ? path.resolve(RUNTIME.dataRoot, v) : v;
}

// Load .env (optional — smart defaults work without it). Dev: project root.
// With a data folder: <data>/settings/.env. On first launch, bootstrap it from
// .env.example so settings are writable; a data folder drops the example's
// DATA_DIR line, which belonged to the old in-app layout.
const ENV_PATH = RUNTIME.envPath ?? path.join(PROJECT_ROOT, '.env');

/** The release tag, from the VERSION file the release workflow writes at the
 *  package root; a git checkout has none and reports "dev" (#185). */
export const APP_VERSION: string = (() => {
  try { return fs.readFileSync(path.join(PROJECT_ROOT, 'VERSION'), 'utf-8').trim() || 'dev'; }
  catch { return 'dev'; }
})();

/** build-info.json, written beside VERSION by release/build-info.mjs. */
export interface BuildInfo {
  format: 1;
  tag: string | null;
  sha: string;
  channel: 'release' | 'master';
  os: 'windows' | 'linux' | 'macos';
  arch: string;
  /** Exact asset variant (cuda13.1, cuda12-volta, metal...), unlike engine/.variant. */
  variant: string;
  engineSha: string;
  updaterProtocol: number;
}
// Same table and rules as release/build-info.mjs (buildInfoErrors).
const BUILD_VARIANTS: Record<string, string[]> = {
  'windows/x64': ['cuda13.1', 'cuda12.8', 'cuda12-volta', 'vulkan', 'cpu'],
  'linux/x64': ['cuda13.1', 'cuda12.8', 'cuda12-volta', 'vulkan', 'rocm', 'cpu'],
  'macos/arm64': ['metal'],
};
/** Highest updater protocol this server speaks (release/build-info.mjs MAX_UPDATER_PROTOCOL). */
export const MAX_UPDATER_PROTOCOL = 1;
const FULL_SHA = /^[0-9a-f]{40}$/;
/** A build-info.json that meets the whole contract, else null (unknown identity). */
export function parseBuildInfo(text: string): BuildInfo | null {
  let i: any;
  try { i = JSON.parse(text); } catch { return null; }
  const ok = i && typeof i === 'object'
    && i.format === 1
    && (i.tag === null || (typeof i.tag === 'string' && /^v\S+$/.test(i.tag)))
    && typeof i.sha === 'string' && FULL_SHA.test(i.sha)
    && typeof i.engineSha === 'string' && FULL_SHA.test(i.engineSha)
    && i.channel === (i.tag ? 'release' : 'master')
    && BUILD_VARIANTS[`${i.os}/${i.arch}`]?.includes(i.variant) === true
    && (i.updaterProtocol === 0 || i.updaterProtocol === MAX_UPDATER_PROTOCOL);
  return ok ? i : null;
}
/** Protocol of the stable launcher that started this process (release/updater/launcher.mjs),
 *  0 when it was not started by one. Apply needs it to equal build-info's updaterProtocol. */
export const LAUNCHER_PROTOCOL = typeof process.send === 'function' && process.env.HOT_STEP_INSTALL_ROOT
  ? Number.parseInt(process.env.HOT_STEP_LAUNCHER_PROTOCOL ?? '', 10) || 0 : 0;
/** The package's exact identity; null (unknown) for older releases, git checkouts and invalid files. */
export const BUILD_INFO: BuildInfo | null = (() => {
  let text: string;
  try { text = fs.readFileSync(path.join(PROJECT_ROOT, 'build-info.json'), 'utf-8'); }
  catch (e: any) {
    if (e.code !== 'ENOENT') console.warn('[Config] Could not read build-info.json:', e.message);
    return null;
  }
  const info = parseBuildInfo(text);
  if (!info) console.warn('[Config] Ignoring build-info.json that breaks its contract; identity unknown');
  return info;
})();
const ENV_EXAMPLE_PATH = path.join(PROJECT_ROOT, '.env.example');
if (!fs.existsSync(ENV_PATH) && fs.existsSync(ENV_EXAMPLE_PATH)) {
  try {
    if (DATA_ROOT) {
      fs.mkdirSync(path.dirname(ENV_PATH), { recursive: true });
      fs.writeFileSync(ENV_PATH, fs.readFileSync(ENV_EXAMPLE_PATH, 'utf-8').replace(/^DATA_DIR=.*$/m, '# DATA_DIR='), 'utf-8');
    } else {
      fs.copyFileSync(ENV_EXAMPLE_PATH, ENV_PATH);
    }
    console.log('[Config] Created .env from .env.example (first launch)');
  } catch (e: any) {
    console.warn('[Config] Could not create .env:', e.message);
  }
}
dotenvConfig({ path: ENV_PATH });

// An EMPTY CUDA_VISIBLE_DEVICES is not "no preference" — it is "no GPUs at
// all", and every child we spawn (ace-server, ace-train) inherits it and falls
// back to CPU without saying why. The Settings GPU picker writes exactly that
// line when the user selects Auto, so drop it here rather than propagate it.
// "all" (keep every GPU visible) is a Settings value, not a CUDA one: CUDA reads
// it as no GPUs, so it must not reach inherited child envs either.
const GPU_SETTING_ALL = process.env.CUDA_VISIBLE_DEVICES?.trim().toLowerCase() === 'all' ? 'all' : undefined;
if (process.env.CUDA_VISIBLE_DEVICES !== undefined && (process.env.CUDA_VISIBLE_DEVICES.trim() === '' || GPU_SETTING_ALL)) {
  delete process.env.CUDA_VISIBLE_DEVICES;
}

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
const DEFAULT_MODELS = RUNTIME.modelsDir ?? path.join(PROJECT_ROOT, 'models');
const DEFAULT_ADAPTERS = RUNTIME.adaptersDir ?? path.join(PROJECT_ROOT, 'adapters');
/** Model Manager `role: runtime` DLLs: the data folder in a release, beside ace-server in dev. */
export const RUNTIME_LIB_DIR = RUNTIME.runtimeLibDir;
/** User plugin overlay (release only); the engine scans it after the shipped plugins. */
export const USER_PLUGINS_DIR = RUNTIME.userPluginsDir;

// Downloaded runtime libraries (CUDA, cuDNN, TRT) used to sit beside the
// engine binaries; in a release they are in the data folder, so put that on
// the library search path every child process inherits.
if (RUNTIME_LIB_DIR) {
  const key = process.platform === 'win32' ? (Object.keys(process.env).find(k => k.toUpperCase() === 'PATH') ?? 'PATH')
    : 'LD_LIBRARY_PATH';
  process.env[key] = process.env[key] ? `${RUNTIME_LIB_DIR}${path.delimiter}${process.env[key]}` : RUNTIME_LIB_DIR;
}
const DEFAULT_NOISE_SAMPLES = path.join(PROJECT_ROOT, 'noise_samples');

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

/** `env` with the app's ffmpeg folder first on PATH. Some ace-train stages
 *  (yue2-sheet) run plain `ffmpeg`, which a machine without a system ffmpeg
 *  cannot find. */
export function withFfmpegOnPath(env: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const ffmpeg = getFFmpegPath();
  if (!ffmpeg) return env;
  const key = Object.keys(env).find(k => k.toUpperCase() === 'PATH') ?? 'PATH';
  return { ...env, [key]: `${path.dirname(ffmpeg)}${path.delimiter}${env[key] ?? ''}` };
}


/** UNSLOTH_KEYLESS is on only for an explicit '1'; unset, empty and anything else are off
 *  (Settings writes '1'/'0'), so startup and hot reload agree. */
export function parseUnslothKeyless(v: string | undefined): boolean {
  return v === '1';
}

/** Worker scope behind config.aceServer.url. Declared here so the getter has
 *  no import cycle; services/engineWorkers.ts owns what goes in it. A
 *  secondary worker sets scopedUrl; the primary leaves it undefined. */
export const engineWorkerScope = new AsyncLocalStorage<{ readonly scopedUrl?: string }>();

export const config = {
  // ace-server configuration
  aceServer: {
    exe: process.env.ACESTEPCPP_EXE || DEFAULT_EXE,
    models: resolveDataPath(process.env.ACESTEPCPP_MODELS || '') || DEFAULT_MODELS,
    adapters: resolveDataPath(process.env.ACESTEPCPP_ADAPTERS || '') || DEFAULT_ADAPTERS,
    port: parseInt(process.env.ACESTEPCPP_PORT || '8085', 10),
    host: process.env.ACESTEPCPP_HOST || '127.0.0.1',
    vaeChunk: parseInt(process.env.ACESTEPCPP_VAE_CHUNK || '1024', 10),
    vaeOverlap: parseInt(process.env.ACESTEPCPP_VAE_OVERLAP || '64', 10),
    /** GPU device selection — maps to CUDA_VISIBLE_DEVICES env var for the engine process */
    cudaVisibleDevices: GPU_SETTING_ALL ?? (process.env.CUDA_VISIBLE_DEVICES || ''),
    /** Multi-GPU opt-in: comma-separated NVIDIA GPU UUIDs, one ace-server per
     *  card (services/engineWorkers.ts). Empty (the default) is the single
     *  engine above, launched exactly as before. Read once at boot. */
    workerGpus: process.env.ACESTEPCPP_WORKER_GPUS || '',
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
    /** Pass --keep-loaded to the spawned ace-server, flipping the engine's
     *  ModelStore to EVICT_NEVER at startup so DiT + adapter + LoKr precompute
     *  stay resident across requests (cold-start LoKr precompute is ~17 s,
     *  hot-start ~50 ms). Default OFF: keeping ~13 GB resident is a VRAM
     *  trade-off that's wrong for VRAM-tight desktops. Toggle from the Settings
     *  UI (Environment tab → "Keep models in VRAM") or set ACESTEPCPP_KEEP_LOADED=1.
     *  Restart-required (it's a spawn-time flag). Note: the engine also honors a
     *  per-request `?keep_loaded=1`, which the UI "co-resident" toggle uses. */
    keepLoaded: (process.env.ACESTEPCPP_KEEP_LOADED ?? '0') !== '0',
    /** Post /warm to the engine after it boots, pre-loading the configured
     *  DiT + VAE + adapter so the first user-facing /synth skips the cold-start.
     *  Requires keepLoaded — under EVICT_STRICT the engine drops the modules
     *  instantly, making the warm a waste. Gated on keepLoaded && warmDit, so
     *  with keep-loaded off (the default) this never fires. */
    warmOnStartup: (process.env.ACESTEPCPP_WARM_ON_STARTUP ?? '1') !== '0',
    /** Filename of the DiT to warm (resolved against the models dir). Empty
     *  disables warm-on-startup entirely. */
    warmDit: process.env.ACESTEPCPP_WARM_DIT || '',
    /** Filename of the VAE to warm. */
    warmVae: process.env.ACESTEPCPP_WARM_VAE || '',
    /** Filename of the adapter to warm (resolved against the adapters dir).
     *  Empty disables adapter pre-load — DiT/VAE alone still cuts most cold-start. */
    warmAdapter: process.env.ACESTEPCPP_WARM_ADAPTER || '',
    /** Adapter scale to use for the warm. Should match what the UI submits so
     *  the warm and the render hit the same LoKr-delta cache key. */
    warmAdapterScale: parseFloat(process.env.ACESTEPCPP_WARM_ADAPTER_SCALE || '1.0'),
    /** TensorRT runtime DLL directory — auto-detected or TENSORRT_LIBS env override */
    trtLibs: resolveDataPath(process.env.TENSORRT_LIBS || '') || RUNTIME.runtimeLibDir || (() => {
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
    /** The engine the current async chain is bound to: a secondary worker
     *  inside engineWorkers.runWithWorker(), the primary everywhere else. */
    get url() {
      return engineWorkerScope.getStore()?.scopedUrl ?? `http://${this.host}:${this.port}`;
    },
  },

  // Essentia audio analysis
  essentia: {
    bin: process.env.ESSENTIA_BIN || path.join(PROJECT_ROOT, 'Essentia', `essentia_streaming_extractor_music${BIN_EXT}`),
  },

  // Dataset Studio labeling throughput (training/rateLimit.ts).
  // concurrency = calls in flight; minInterval = spacing between call STARTS.
  // Defaults reproduce the old serial-with-courtesy-sleep behaviour for the two
  // network services so raising the queue's parallelism cannot surprise anyone
  // with a quota bill or a Genius block.
  labeling: {
    essentiaConcurrency: parseInt(process.env.LABEL_ESSENTIA_CONCURRENCY || '2', 10),
    geniusConcurrency: parseInt(process.env.LABEL_GENIUS_CONCURRENCY || '1', 10),
    geniusMinIntervalMs: parseInt(process.env.LABEL_GENIUS_MIN_INTERVAL_MS || '400', 10),
    captionConcurrency: parseInt(process.env.LABEL_CAPTION_CONCURRENCY || '1', 10),
    captionMinIntervalMs: parseInt(process.env.LABEL_CAPTION_MIN_INTERVAL_MS || '250', 10),
  },

  // Node.js server
  server: {
    port: parseInt(process.env.SERVER_PORT || '3001', 10),
    // Loopback by default; set SERVER_HOST=0.0.0.0 (or a LAN/Tailscale IP) to expose it.
    host: defaultServerHost(process.env.SERVER_HOST),
  },

  // Data paths
  data: {
    dir: resolveDataDir(process.env.DATA_DIR),
    /** Where /api/download/:id/save writes; '' = browser download only. */
    downloadDir: resolveDataPath(process.env.DOWNLOAD_DIR || ''),
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
    // One completion call's abort limit, every provider. A local model on
    // unified memory can need more than the old fixed 300 s while SuperSep
    // or the DiT holds the GPU (#136).
    llmTimeoutMs: Math.max(10_000, parseInt(process.env.LLM_TIMEOUT_MS || '300000', 10) || 300_000),
    // LM Studio can require a bearer token once its server has Authentication
    // switched on, which is what you need for its MCP features (issue #121).
    // Empty means anonymous, which is how a default LM Studio install answers.
    lmstudioApiKey: process.env.LMSTUDIO_API_KEY || '',
    unslothBaseUrl: process.env.UNSLOTH_BASE_URL || 'http://127.0.0.1:8888',
    unslothUsername: process.env.UNSLOTH_USERNAME || '',
    unslothPassword: process.env.UNSLOTH_PASSWORD || '',
    // Opt-in for a local Studio running with its "keyless API access" setting
    // enabled (off by default in Studio): /v1 then needs no username/password.
    // Must be set explicitly — the loopback default URL alone must never make
    // the provider look installed on a fresh checkout.
    unslothKeyless: parseUnslothKeyless(process.env.UNSLOTH_KEYLESS),
    defaultProvider: process.env.DEFAULT_LLM_PROVIDER || 'gemini',
    geminiModel: process.env.GEMINI_MODEL || 'gemini-2.5-flash',
    openaiModel: process.env.OPENAI_MODEL || 'gpt-4o-mini',
    anthropicModel: process.env.ANTHROPIC_MODEL || 'claude-haiku-4-5-20251001',
    ollamaModel: process.env.OLLAMA_MODEL || 'llama3',
    lmstudioModel: process.env.LMSTUDIO_MODEL || '',
    unslothModel: process.env.UNSLOTH_MODEL || '',
    llamacppBaseUrl: process.env.LLAMACPP_BASE_URL || 'http://127.0.0.1:8080/v1',
    llamacppModel: process.env.LLAMACPP_MODEL || '',
    openaiCompatBaseUrl: process.env.OPENAI_COMPAT_BASE_URL || '',
    openaiCompatApiKey: process.env.OPENAI_COMPAT_API_KEY || '',
    openaiCompatModel: process.env.OPENAI_COMPAT_MODEL || '',
    openaiCompatName: process.env.OPENAI_COMPAT_NAME || 'OpenAI Compatible',
    // Default reasoning budget for runtimes that expose graded effort rather
    // than a boolean (NInfer: none|low|medium|xhigh). Empty means "send nothing
    // and let the server pick", which is the safe default for endpoints that
    // reject the field. Per-request reasoning_effort and no_think both win.
    openaiCompatReasoningEffort: process.env.OPENAI_COMPAT_REASONING_EFFORT || '',
    get dbPath() {
      return path.join(config.data.dir, 'lireek.db');
    },
    exportDir: resolveDataPath(process.env.LYRICS_EXPORT_DIR || '') || path.join(resolveDataDir(process.env.DATA_DIR), 'lyrics'),
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

  // Training Studio / Dataset Studio
  // NOTE: deliberately NOT in EXPOSED_ENV_KEYS — none of these are user-facing
  // settings, and exposing a key without a matching apply() line in
  // reloadEnvConfig() is the classic silent-no-op bug.
  training: {
    dir: resolveDataPath(process.env.TRAINING_DIR || '') || path.join(resolveDataDir(process.env.DATA_DIR), 'training'),
    understandTimeoutMs: parseInt(process.env.TRAINING_UNDERSTAND_TIMEOUT_MS || '1200000', 10), // 20 min/file
    maxScanFiles: parseInt(process.env.TRAINING_MAX_SCAN_FILES || '5000', 10),
  },

  // Training on another PC (services/training/trainingWorkers.ts).
  workers: {
    /** This machine drives these workers: `Name=http://host:3001`, comma-separated. */
    list: process.env.TRAINING_WORKERS || '',
    /** Token this machine sends to its workers. */
    token: process.env.TRAINING_WORKER_TOKEN || '',
    /** Set = this machine IS a worker: API calls from other machines must carry it. */
    acceptToken: process.env.WORKER_TOKEN || '',
  },

  // Whisper speech-to-text (lyrics transcription)
  whisper: {
    exe: resolveDataPath(process.env.WHISPER_EXE || '') || path.join(RUNTIME.whisperDir ?? path.join(PROJECT_ROOT, 'tools', 'whisper'), `whisper-cli${BIN_EXT}`),
    modelsDir: resolveDataPath(process.env.WHISPER_MODELS_DIR || '') || path.join(resolveDataPath(process.env.ACESTEPCPP_MODELS || '') || DEFAULT_MODELS, 'whisper'),
  },
};

// ── .env hot-reload infrastructure ──────────────────────────────────

/** Absolute path to the project .env file */
export const ENV_FILE_PATH = ENV_PATH;

/** Write one .env value so dotenv reads back exactly `v` (a bare `#` would start a comment). */
export function formatEnvValue(v: string): string {
  if (!/[#'"`]|^\s|\s$/.test(v)) return v;
  if (v.includes('`')) throw new Error('Value cannot contain a backtick');
  return `\`${v}\``; // backticks are the one dotenv quote with no escape processing
}

/** Inverse of formatEnvValue for the raw text after `KEY=`. */
export function parseEnvValue(raw: string): string {
  return dotenvParse(`K=${raw}`).K ?? '';
}

/** Keys exposed to the Settings UI (whitelist — nothing else leaks) */
export const EXPOSED_ENV_KEYS = [
  // Engine
  'ACESTEPCPP_MODELS', 'ACESTEPCPP_ADAPTERS', 'ACESTEPCPP_PORT', 'ACESTEPCPP_HOST',
  'ACESTEPCPP_VAE_CHUNK', 'ACESTEPCPP_VAE_OVERLAP', 'ACESTEPCPP_KEEP_LOADED',
  // GPU
  'CUDA_VISIBLE_DEVICES', 'ACESTEPCPP_WORKER_GPUS',
  // Server
  'SERVER_PORT', 'DATA_DIR',
  // API keys
  'GENIUS_ACCESS_TOKEN', 'GEMINI_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY',
  // LLM config
  'DEFAULT_LLM_PROVIDER', 'LLM_TIMEOUT_MS',
  'GEMINI_MODEL', 'OPENAI_MODEL', 'ANTHROPIC_MODEL',
  'OLLAMA_MODEL', 'LMSTUDIO_MODEL', 'UNSLOTH_MODEL',
  // LLM endpoints
  'OLLAMA_BASE_URL', 'LMSTUDIO_BASE_URL', 'LMSTUDIO_API_KEY',
  'UNSLOTH_BASE_URL', 'UNSLOTH_USERNAME', 'UNSLOTH_PASSWORD', 'UNSLOTH_KEYLESS',
  'LLAMACPP_BASE_URL', 'LLAMACPP_MODEL',
  'OPENAI_COMPAT_BASE_URL', 'OPENAI_COMPAT_API_KEY', 'OPENAI_COMPAT_MODEL', 'OPENAI_COMPAT_NAME',
  'OPENAI_COMPAT_REASONING_EFFORT',
  // Dataset Studio labeling throughput
  'LABEL_ESSENTIA_CONCURRENCY',
  'LABEL_GENIUS_CONCURRENCY', 'LABEL_GENIUS_MIN_INTERVAL_MS',
  'LABEL_CAPTION_CONCURRENCY', 'LABEL_CAPTION_MIN_INTERVAL_MS',
  // Paths
  'LYRICS_EXPORT_DIR', 'MUSCRIPTOR_MODELS_DIR', 'DOWNLOAD_DIR',
  // Training workers
  'TRAINING_WORKERS', 'TRAINING_WORKER_TOKEN', 'WORKER_TOKEN',
] as const;

/**
 * Run after every successful reloadEnvConfig().
 *
 * Modules that hold state DERIVED from config (rather than reading it on each
 * use) register here so a Settings save reaches them without a restart —
 * rateLimit.ts's limiters being the first. A direct call from reloadEnvConfig()
 * would be an import cycle, since those modules import config themselves.
 */
export const configReloadListeners: Array<() => void> = [];

/** Keys that require an app restart to take effect */
export const RESTART_REQUIRED_KEYS = new Set([
  'ACESTEPCPP_MODELS', 'ACESTEPCPP_ADAPTERS', 'ACESTEPCPP_PORT', 'ACESTEPCPP_HOST',
  'ACESTEPCPP_VAE_CHUNK', 'ACESTEPCPP_VAE_OVERLAP', 'ACESTEPCPP_KEEP_LOADED',
  'CUDA_VISIBLE_DEVICES', 'ACESTEPCPP_WORKER_GPUS',
  'SERVER_PORT', 'DATA_DIR', 'MUSCRIPTOR_MODELS_DIR',
]);

/**
 * Re-read the .env file and hot-patch the live config object.
 * Returns a list of keys that actually changed.
 */
export function reloadEnvConfig(envText?: string): string[] {
  const envContent = envText ?? (fs.existsSync(ENV_FILE_PATH)
    ? fs.readFileSync(ENV_FILE_PATH, 'utf-8')
    : '');
  const parsed = dotenvParse(envContent);
  const changed: string[] = [];

  // Helper: update a config property if the env value changed. Compare the
  // RESOLVED values before/after applying, not raw .env text against the live
  // config — setters transform (resolveDataPath, parseInt, defaults), so a
  // raw-vs-resolved compare reports every empty/default key as "changed" on
  // every reload and makes the UI demand a restart that isn't needed.
  const apply = (envKey: string, setter: (val: string) => void, getter: () => string) => {
    const before = getter();
    setter(parsed[envKey] ?? '');
    if (getter() !== before) changed.push(envKey);
  };

  // ── Engine (values stored but won't affect running child process) ──
  apply('ACESTEPCPP_MODELS', v => { config.aceServer.models = resolveDataPath(v) || DEFAULT_MODELS; },
    () => config.aceServer.models);
  apply('ACESTEPCPP_ADAPTERS', v => { config.aceServer.adapters = resolveDataPath(v) || DEFAULT_ADAPTERS; },
    () => config.aceServer.adapters);
  apply('ACESTEPCPP_PORT', v => { config.aceServer.port = parseInt(v || '8085', 10); },
    () => String(config.aceServer.port));
  apply('ACESTEPCPP_HOST', v => { config.aceServer.host = v || '127.0.0.1'; },
    () => config.aceServer.host);
  apply('ACESTEPCPP_VAE_CHUNK', v => { config.aceServer.vaeChunk = parseInt(v || '1024', 10); },
    () => String(config.aceServer.vaeChunk));
  apply('ACESTEPCPP_VAE_OVERLAP', v => { config.aceServer.vaeOverlap = parseInt(v || '64', 10); },
    () => String(config.aceServer.vaeOverlap));
  apply('CUDA_VISIBLE_DEVICES', v => { config.aceServer.cudaVisibleDevices = v; },
    () => config.aceServer.cudaVisibleDevices);
  apply('ACESTEPCPP_WORKER_GPUS', v => { config.aceServer.workerGpus = v; },
    () => config.aceServer.workerGpus);
  apply('ACESTEPCPP_KEEP_LOADED', v => { config.aceServer.keepLoaded = (v || '0') !== '0'; },
    () => (config.aceServer.keepLoaded ? '1' : '0'));

  // ── Server ──
  apply('SERVER_PORT', v => { config.server.port = parseInt(v || '3001', 10); },
    () => String(config.server.port));
  apply('DATA_DIR', v => {
    config.data.dir = resolveDataDir(v);
  }, () => config.data.dir);

  // ── Lireek / LLM (hot-reloaded — takes effect immediately) ──
  apply('LLM_TIMEOUT_MS', v => { config.lireek.llmTimeoutMs = Math.max(10_000, parseInt(v || '300000', 10) || 300_000); },
    () => String(config.lireek.llmTimeoutMs));
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
  apply('LMSTUDIO_API_KEY', v => { config.lireek.lmstudioApiKey = v; },
    () => config.lireek.lmstudioApiKey);
  apply('UNSLOTH_BASE_URL', v => { config.lireek.unslothBaseUrl = v || 'http://127.0.0.1:8888'; },
    () => config.lireek.unslothBaseUrl);
  apply('UNSLOTH_USERNAME', v => { config.lireek.unslothUsername = v; },
    () => config.lireek.unslothUsername);
  apply('UNSLOTH_PASSWORD', v => { config.lireek.unslothPassword = v; },
    () => config.lireek.unslothPassword);
  apply('UNSLOTH_KEYLESS', v => { config.lireek.unslothKeyless = parseUnslothKeyless(v); },
    () => (config.lireek.unslothKeyless ? '1' : '0'));
  apply('DEFAULT_LLM_PROVIDER', v => { config.lireek.defaultProvider = v || 'gemini'; },
    () => config.lireek.defaultProvider);
  apply('GEMINI_MODEL', v => { config.lireek.geminiModel = v || 'gemini-2.5-flash'; },
    () => config.lireek.geminiModel);
  apply('OPENAI_MODEL', v => { config.lireek.openaiModel = v || 'gpt-4o-mini'; },
    () => config.lireek.openaiModel);
  apply('ANTHROPIC_MODEL', v => { config.lireek.anthropicModel = v || 'claude-haiku-4-5-20251001'; },
    () => config.lireek.anthropicModel);
  apply('OLLAMA_MODEL', v => { config.lireek.ollamaModel = v || 'llama3'; },
    () => config.lireek.ollamaModel);
  apply('LMSTUDIO_MODEL', v => { config.lireek.lmstudioModel = v; },
    () => config.lireek.lmstudioModel);
  apply('UNSLOTH_MODEL', v => { config.lireek.unslothModel = v; },
    () => config.lireek.unslothModel);
  apply('LLAMACPP_BASE_URL', v => { config.lireek.llamacppBaseUrl = v || 'http://127.0.0.1:8080/v1'; },
    () => config.lireek.llamacppBaseUrl);
  apply('LLAMACPP_MODEL', v => { config.lireek.llamacppModel = v; },
    () => config.lireek.llamacppModel);
  apply('OPENAI_COMPAT_BASE_URL', v => { config.lireek.openaiCompatBaseUrl = v; },
    () => config.lireek.openaiCompatBaseUrl);
  apply('OPENAI_COMPAT_API_KEY', v => { config.lireek.openaiCompatApiKey = v; },
    () => config.lireek.openaiCompatApiKey);
  apply('OPENAI_COMPAT_MODEL', v => { config.lireek.openaiCompatModel = v; },
    () => config.lireek.openaiCompatModel);
  apply('OPENAI_COMPAT_NAME', v => { config.lireek.openaiCompatName = v || 'OpenAI Compatible'; },
    () => config.lireek.openaiCompatName);
  apply('OPENAI_COMPAT_REASONING_EFFORT', v => { config.lireek.openaiCompatReasoningEffort = v; },
    () => config.lireek.openaiCompatReasoningEffort);
  apply('TRAINING_WORKERS', v => { config.workers.list = v; }, () => config.workers.list);
  apply('TRAINING_WORKER_TOKEN', v => { config.workers.token = v; }, () => config.workers.token);
  apply('WORKER_TOKEN', v => { config.workers.acceptToken = v; }, () => config.workers.acceptToken);
  apply('DOWNLOAD_DIR', v => { config.data.downloadDir = resolveDataPath(v); },
    () => config.data.downloadDir);
  apply('LYRICS_EXPORT_DIR', v => {
    config.lireek.exportDir = resolveDataPath(v) || path.join(config.data.dir, 'lyrics');
  }, () => config.lireek.exportDir);

  // ── Labeling throughput ──
  // Numeric, so apply()'s string compare needs the current value stringified.
  // An empty/garbage entry falls back to the shipped default rather than 0,
  // which would deadlock a limiter.
  const applyInt = (envKey: string, fallback: number, setter: (n: number) => void, getter: () => number) => {
    const raw = parsed[envKey] ?? '';
    const next = raw === '' ? fallback : parseInt(raw, 10);
    const safe = Number.isFinite(next) && next >= 0 ? next : fallback;
    if (safe !== getter()) {
      setter(safe);
      changed.push(envKey);
    }
  };
  applyInt('LABEL_ESSENTIA_CONCURRENCY', 2, n => { config.labeling.essentiaConcurrency = n; },
    () => config.labeling.essentiaConcurrency);
  applyInt('LABEL_GENIUS_CONCURRENCY', 1, n => { config.labeling.geniusConcurrency = n; },
    () => config.labeling.geniusConcurrency);
  applyInt('LABEL_GENIUS_MIN_INTERVAL_MS', 400, n => { config.labeling.geniusMinIntervalMs = n; },
    () => config.labeling.geniusMinIntervalMs);
  applyInt('LABEL_CAPTION_CONCURRENCY', 1, n => { config.labeling.captionConcurrency = n; },
    () => config.labeling.captionConcurrency);
  applyInt('LABEL_CAPTION_MIN_INTERVAL_MS', 250, n => { config.labeling.captionMinIntervalMs = n; },
    () => config.labeling.captionMinIntervalMs);

  if (changed.length > 0) {
    console.log(`[Config] Hot-reloaded ${changed.length} setting(s): ${changed.join(', ')}`);
  }

  for (const listener of configReloadListeners) {
    try {
      listener();
    } catch (err) {
      console.error('[Config] Reload listener failed:', err);
    }
  }

  return changed;
}
