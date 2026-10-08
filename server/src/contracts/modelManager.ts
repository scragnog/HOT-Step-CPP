// contracts/modelManager.ts — wire shapes for /api/model-manager
// (server/src/routes/modelManager.ts, server/src/services/modelDownloadService.ts).
//
// RegistryFile and StarterPack used to be two independent interfaces — one
// in modelDownloadService.ts, one in ui/src/types.ts — kept in sync by hand.
// They had already drifted: the UI's copy was missing repoPath, sha256 and
// companions, and the top-level registry response was missing variant and
// cudaMajor. One definition here, imported by both ends, so that can't
// happen silently again.

export type FileRole =
  | 'dit' | 'lm' | 'embedding' | 'vae' | 'pp-vae' | 'supersep' | 'whisper'
  | 'stablestep' | 'runtime' | 'mm3' | 'moss' | 'yue2';

/** Which generation backend (or "shared" across several) a file belongs to.
 *  Absent on entries added before this field existed; the UI falls back to
 *  a role -> family mapping in that case. */
export type FileFamily = 'as1.5' | 'mm3' | 'yue2' | 'shared';

export interface RegistryCompanion {
  filename: string;
  /** Path within the HuggingFace repo; defaults to filename. */
  repoPath?: string;
}

/** One catalogue entry, before installed-status enrichment. */
export interface RegistryFileEntry {
  id: string;
  filename: string;
  role: FileRole;
  subdir?: string;
  /** Path within the HuggingFace repo (e.g. "runtime/cublas64_13.dll"). */
  repoPath?: string;
  displayName: string;
  scale?: 'standard' | 'xl' | null;
  variant?: string | null;
  quant: string;
  sizeBytes: number;
  /** sha256 of the published file, lowercase hex. Declared on any entry whose
   *  content can change under a reused filename, so a stale local copy can
   *  be told apart from a fresh one. */
  sha256?: string;
  repo: string;
  description: string;
  tags: string[];
  companions?: RegistryCompanion[];
  /** TensorRT builder-resource entries only: CUDA compute capability
   *  (MAJOR*10+MINOR, e.g. 120 for Blackwell consumer). Absent otherwise. */
  sm?: number;
  family?: FileFamily;
}

/** What `GET /api/model-manager/registry` sends per file: the catalogue
 *  entry plus this installation's status. */
export interface RegistryFile extends RegistryFileEntry {
  installed: boolean;
  outdated: boolean;
}

export interface StarterPack {
  id: string;
  name: string;
  description: string;
  fileIds: string[];
  family?: FileFamily;
}

export interface ModelRegistryResponse {
  packs: StarterPack[];
  files: RegistryFile[];
  modelsDir: string;
  variant: string;
  cudaMajor: number;
}

export interface DownloadStartResponse { jobId: string }

export interface DownloadJob {
  jobId: string;
  fileId: string;
  filename: string;
  status: 'queued' | 'downloading' | 'paused' | 'completed' | 'failed' | 'cancelled';
  bytesDownloaded: number;
  totalBytes: number;
  speed: number;
  error?: string;
}
