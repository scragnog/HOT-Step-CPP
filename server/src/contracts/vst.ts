// contracts/vst.ts — wire shapes for /api/vst (server/src/routes/vst.ts).
//
// VstPlugin and ChainEntry used to be declared in the route file itself and
// duplicated again as api.ts's own copies (VstPlugin, VstChainEntry) — same
// shape, three places. This is now the one definition.

export interface VstPlugin {
  name: string;
  vendor: string;
  version: string;
  path: string;
  uid: string;
  subcategories: string;
}

export interface ChainEntry {
  uid: string;
  name: string;
  vendor: string;
  /** .vst3 module path. */
  path: string;
  enabled: boolean;
  /** .vststate file path (may not exist yet). */
  statePath: string;
}

export interface ChainConfig { plugins: ChainEntry[] }

export interface ScanPluginsResponse { plugins: VstPlugin[] }
export type GetChainResponse = ChainConfig;
export type UpdateChainResponse = ChainConfig;

export interface GuiResponse { ok: true; pid: number | undefined }
export interface ProcessResponse { ok: true; skipped?: true; elapsed?: number }

export interface MonitorStartResponse { ok: true; pid: number | undefined; plugins: number }
export interface MonitorStopResponse { ok: true; wasRunning: boolean }
export interface MonitorSwitchResponse { ok: true }
export interface MonitorStatusResponse { running: boolean; paused: boolean; pid: number | null; position: number; duration: number }
export interface MonitorSeekResponse { ok: true; position: number }
export interface MonitorPauseResponse { ok: true }
export interface MonitorResumeResponse { ok: true }
export type MonitorRestartResponse = MonitorStartResponse;
