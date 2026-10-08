// contracts/profiles.ts — wire shapes for /api/profiles (server/src/routes/profiles.ts).
//
// A profile is a named snapshot of every generation parameter, stored as
// plain preset JSON wrapped with { name, saved_at, data } — the same shape
// as the UI's exported preset file, so one can be dropped into the profiles
// folder by hand.

export interface ProfileFile {
  name: string;
  saved_at: string;
  data: Record<string, unknown>;
}

export interface ListProfilesResponse { profiles: ProfileFile[]; count: number }
export type GetProfileResponse = ProfileFile;
export interface SaveProfileResponse { ok: true; name: string; saved_at: string }
export interface RenameProfileResponse { ok: true; name: string }
export interface DeleteProfileResponse { ok: true; deleted: string }
