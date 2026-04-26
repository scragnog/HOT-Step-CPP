// config.ts — Environment-based configuration for HOT-Step CPP server
import { config as dotenvConfig } from 'dotenv';
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
    get exportDir() {
      return process.env.LYRICS_EXPORT_DIR || path.join(config.data.dir, 'lyrics');
    },
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
