// appSettings.ts — the ace-settings shape and its defaults, headless so any shell
// can read them. The Settings panel re-exports both for its older importers.

export interface AppSettings {
  coResident: boolean;
  cacheLmCodes: boolean;
  speculativeDecoding: boolean;
  // Download defaults
  downloadFormat: 'wav' | 'flac' | 'opus' | 'mp3';
  downloadMp3Bitrate: number;
  downloadOpusBitrate: number;
  downloadVersion: 'original' | 'mastered' | 'both';
  downloadIncludeLatent: boolean;
  downloadEnhancedLrc: boolean;
  // Adapter trigger word
  triggerUseFilename: boolean;
  triggerPlacement: 'prepend' | 'append' | 'replace';
  discoKickExtract: boolean;
  // Pipeline parallelism (opt-in)
  parallelWhisper: boolean;
  parallelQualityEval: boolean;
  parallelCoverArt: boolean;
  // Generation timeout (minutes) — how long to wait before declaring a generation timed out
  generationTimeoutMinutes: number;
}

export const DEFAULT_SETTINGS: AppSettings = {
  coResident: false,
  cacheLmCodes: true,
  speculativeDecoding: true,
  downloadFormat: 'flac',
  downloadMp3Bitrate: 192,
  downloadOpusBitrate: 192,
  downloadVersion: 'mastered',
  downloadIncludeLatent: false,
  downloadEnhancedLrc: false,
  triggerUseFilename: false,
  triggerPlacement: 'prepend',
  discoKickExtract: false,
  parallelWhisper: false,
  parallelQualityEval: false,
  parallelCoverArt: false,
  generationTimeoutMinutes: 30,
};
