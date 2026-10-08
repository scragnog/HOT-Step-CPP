// contracts/auth.ts — wire shapes for /api/auth (server/src/routes/auth.ts).
//
// Local single-user auto-auth: no passwords, UUID bearer tokens, in-memory
// token map. Shared here so the UI's User/AuthState (ui/src/types.ts) can't
// drift from what the route actually returns — before this they were two
// independently hand-kept copies.

export interface AuthUser {
  id: string;
  username: string;
  bio: string;
  avatar_url: string;
  banner_url: string;
  created_at: string;
}

export interface AutoLoginResponse { user: AuthUser; token: string }
export interface MeResponse { user: AuthUser }
export interface UsernameUpdateResponse { user: AuthUser; token: string }
