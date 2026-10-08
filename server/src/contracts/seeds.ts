// contracts/seeds.ts — wire shapes for /api/seeds (server/src/routes/seeds.ts).
//
// File format is intentionally identical to MD_Nodes/SeedSaver (ComfyUI):
// { seed, saved_at, metadata }. A ComfyUI seeds/ directory drops in and
// loads immediately.

export interface SeedListEntry {
  name: string;
  seed: number | null;
  saved_at: string | null;
  description: string;
  tags: string[];
  favorite: boolean;
}

export interface ListSeedsResponse { seeds: SeedListEntry[]; count: number }

export interface FavoriteSeedEntry { name: string; seed: number; saved_at: string; favorite: true }
export interface ListFavoritesResponse { seeds: FavoriteSeedEntry[] }

export interface RandomSeedResponse { name: string; seed: number; saved_at: string }

export interface GetSeedResponse {
  name: string;
  seed: number;
  saved_at: string;
  description: string;
  tags: string[];
  favorite: boolean;
}

export interface SaveSeedResponse { ok: true; name: string; seed: number }
export interface DeleteSeedResponse { ok: true; deleted: string }
export interface ToggleFavoriteResponse { ok: true; name: string; favorite: boolean }
