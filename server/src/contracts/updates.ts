// contracts/updates.ts — wire shapes for /api/updates (routes/updates.ts).
// Preparation stages a verified payload; POST /apply hands it to the stable
// Windows launcher (release/updater/launcher.mjs), which installs it.

export type UpdateChannel = 'release' | 'master';

export const BUILD_VARIANTS = ['cuda13.1', 'cuda12.8', 'cuda12-volta', 'vulkan', 'rocm', 'cpu', 'metal'] as const;
export type BuildVariant = (typeof BUILD_VARIANTS)[number];

/** build-info.json beside VERSION (format 1). */
export interface BuildInfo {
  format: 1;
  tag: string | null;
  sha: string;
  channel: UpdateChannel;
  os: 'windows' | 'linux' | 'macos';
  arch: string;
  variant: BuildVariant;
  engineSha: string;
  updaterProtocol: number;
}

export type GitRelation = 'up-to-date' | 'behind' | 'ahead' | 'diverged' | 'unknown';

export interface GitInstall {
  head: string;
  /** null = detached HEAD. */
  branch: string | null;
  /** Tracked, untracked or submodule changes. */
  dirty: boolean;
  /** origin is scragnog/HOT-Step-CPP. */
  originOk: boolean;
}

export interface InstalledIdentity {
  /** 'portable' = release package, 'git' = source checkout, 'unknown' = neither could be proven. */
  kind: 'portable' | 'git' | 'unknown';
  /** VERSION file content ('dev' when absent). */
  version: string;
  /** Exact release tag from build-info.json, else null. */
  tag: string | null;
  /** Full source commit: build-info.json first, else the checkout's HEAD. */
  sha: string | null;
  /** Exact asset variant from build-info.json only; never guessed. */
  variant: BuildVariant | null;
  /** null = no usable build-info.json (older release or a git clone). */
  buildInfo: BuildInfo | null;
  /** Why build-info.json was ignored, when it exists but is unusable. */
  buildInfoProblem: string | null;
  git: GitInstall | null;
}

export interface LatestRelease {
  tag: string;
  name: string;
  url: string;
  publishedAt: string | null;
}

export interface LatestMaster {
  sha: string;
  url: string;
}

export type UpdateStatusKind =
  | 'never-checked'
  | 'up-to-date'
  | 'update-available'
  /** Installed is newer than the feed (dev build, unreleased commit). */
  | 'ahead'
  /** Installed identity or relation can't be compared; the latest is shown with its link. */
  | 'unknown'
  /** Last attempt failed. Never read this as up to date; `latest` is the stale result. */
  | 'could-not-check';

export interface UpdatesStatusResponse {
  /** Present on the HTTP endpoints; absent on the notice checker's internal result. */
  prepare?: UpdatePrepareStatus;
  /** Present on the HTTP endpoints: whether a prepared update can be installed, and the last outcome. */
  install?: UpdateInstallStatus;
  channel: UpdateChannel;
  installed: InstalledIdentity;
  status: UpdateStatusKind;
  /** Latest known feed result for the selected channel (may be stale). */
  latest: { release: LatestRelease | null; master: LatestMaster | null };
  /** Installed commit versus the master tip (Master channel on a checkout only). */
  gitRelation: GitRelation;
  /** Where to read what changed; null until a feed answered. */
  notesUrl: string | null;
  lastSuccessAt: string | null;
  lastAttemptAt: string | null;
  /** Cached result is older than 24 h or the last attempt failed. */
  stale: boolean;
  /** Message from the last failed attempt, else null. */
  error: string | null;
  /** No GitHub request before this time (403/429 back-off), else null. */
  retryAfter: string | null;
  /** A refresh is running now. */
  checking: boolean;
  apply: {
    /** Always false in the notices batch. */
    offered: false;
    reason: 'notices-only';
    /** What would also stop an automatic update on this install (empty = nothing else). */
    blockers: string[];
  };
}

export interface UpdatesCheckRequest {
  /** Switch the check feed first; never changes the installation. */
  channel?: UpdateChannel;
}

export interface UpdatesCheckResponse extends UpdatesStatusResponse {
  /** True when a manual check was inside the one-minute limit and the cached result was returned. */
  throttled: boolean;
}

export interface UpdateCandidate {
  /** Opaque, server-issued, expires after one hour or a server restart. */
  id: string;
  tag: string;
  sha: string;
  variant: BuildVariant;
  assetName: string;
  bytes: number;
}

export interface UpdatePrepareStatus {
  offered: boolean;
  /** Plain reason when prepare is unavailable, otherwise null. */
  reason: string | null;
  candidate: UpdateCandidate | null;
  phase: 'idle' | 'checking' | 'downloading' | 'verifying' | 'extracting' | 'ready' | 'failed' | 'cancelled';
  downloadedBytes: number;
  totalBytes: number;
  error: string | null;
  /** Opaque directory ID for the launcher, never a client-supplied path. */
  preparedId: string | null;
}

export interface UpdatesPrepareRequest { candidateId: string }
/** POST /api/updates/apply: the prepared candidate's ID and nothing else. */
export interface UpdatesApplyRequest { candidateId: string }

/** What the launcher recorded about the last update (updater/last-result.json). */
export interface UpdateApplyResult {
  /** installed: committed and running. rolled-back: the new version failed before commit;
   *  the old one runs on its restored data. recovery-required: nothing was started; the
   *  launcher window says what to fix. failed: stopped before anything changed. */
  outcome: 'installed' | 'rolled-back' | 'recovery-required' | 'failed';
  /** Payload folder names under app/ (`<tag>-<sha12>-<variant>`). */
  from: string | null;
  to: string | null;
  at: string;
  message: string | null;
}

export interface UpdateInstallStatus {
  /** True when this install can apply a prepared update: updater protocol 1 or
   *  higher, started by the stable launcher, Windows portable. */
  supported: boolean;
  /** Why not, when unsupported; otherwise null. */
  reason: string | null;
  /** 'stopping': apply was accepted and the server is about to stop for the launcher. */
  phase: 'idle' | 'stopping';
  lastResult: UpdateApplyResult | null;
}

/** 202 from POST /api/updates/apply. The server stops about 300 ms later; poll
 *  GET /api/health until it answers again, then read install.lastResult. */
export interface UpdatesApplyResponse {
  phase: 'stopping';
  from: string;
  to: string;
}
export interface UpdatesCancelPrepareRequest { candidateId: string }
