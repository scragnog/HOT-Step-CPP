// contracts/mastering.ts — wire shapes for /api/mastering (server/src/routes/mastering.ts).

export interface UploadReferenceResponse {
  name: string;
  /** Absolute filesystem path — server-internal, but already sent today. */
  path: string;
  url: string;
}

export interface ReferenceEntry {
  name: string;
  path: string;
  size: number;
  url: string;
}

export interface ListReferencesResponse { references: ReferenceEntry[] }
export interface DeleteReferenceResponse { ok: true }

export interface RunMasteringResponse {
  ok: true;
  masteredUrl: string;
  songId: string;
}
