// auth.ts — Simple local auto-auth for single-user mode
//
// No passwords, no tokens. Just auto-create a user on first launch
// and return it on every request. Tokens are simple UUIDs for API compat.

import { Router } from 'express';
import { v4 as uuidv4 } from 'uuid';
import { getDb } from '../db/database.js';
import type { AuthUser, AutoLoginResponse, MeResponse, UsernameUpdateResponse } from '../contracts/auth.js';

const router = Router();

// In-memory token → userId map (resets on restart, which is fine for local single-user)
const tokens = new Map<string, string>();

/** Get or create the default local user */
function getOrCreateUser(): AuthUser {
  const db = getDb();

  let user = db.prepare('SELECT * FROM users LIMIT 1').get() as AuthUser | undefined;
  if (!user) {
    const id = uuidv4();
    db.prepare('INSERT INTO users (id, username) VALUES (?, ?)').run(id, 'Producer');
    user = db.prepare('SELECT * FROM users WHERE id = ?').get(id) as AuthUser;
  }
  return user;
}

/** Create a token for the user */
function createToken(userId: string): string {
  const token = uuidv4();
  tokens.set(token, userId);
  return token;
}

// GET /api/auth/auto — auto-login: get or create local user
router.get('/auto', (_req, res) => {
  const user = getOrCreateUser();
  const token = createToken(user.id);
  res.json({ user, token } satisfies AutoLoginResponse);
});

// POST /api/auth/setup — set username on first launch
router.post('/setup', (req, res) => {
  const { username } = req.body;
  if (!username || typeof username !== 'string') {
    res.status(400).json({ error: 'Username is required' });
    return;
  }

  const user = getOrCreateUser();
  getDb().prepare('UPDATE users SET username = ? WHERE id = ?').run(username.trim(), user.id);
  const updated = getDb().prepare('SELECT * FROM users WHERE id = ?').get(user.id) as AuthUser;
  const token = createToken(user.id);
  res.json({ user: updated, token } satisfies UsernameUpdateResponse);
});

// GET /api/auth/me — get current user
router.get('/me', (req, res) => {
  const token = req.headers.authorization?.replace('Bearer ', '');
  if (!token || !tokens.has(token)) {
    res.status(401).json({ error: 'Unauthorized' });
    return;
  }
  const userId = tokens.get(token)!;
  const user = getDb().prepare('SELECT * FROM users WHERE id = ?').get(userId) as AuthUser | undefined;
  if (!user) {
    res.status(404).json({ error: 'User not found' });
    return;
  }
  res.json({ user } satisfies MeResponse);
});

// PATCH /api/auth/username — update username
router.patch('/username', (req, res) => {
  const token = req.headers.authorization?.replace('Bearer ', '');
  if (!token || !tokens.has(token)) {
    res.status(401).json({ error: 'Unauthorized' });
    return;
  }
  const { username } = req.body;
  if (!username || typeof username !== 'string') {
    res.status(400).json({ error: 'Username is required' });
    return;
  }
  const userId = tokens.get(token)!;
  getDb().prepare('UPDATE users SET username = ? WHERE id = ?').run(username.trim(), userId);
  const user = getDb().prepare('SELECT * FROM users WHERE id = ?').get(userId) as AuthUser;
  const newToken = createToken(userId);
  res.json({ user, token: newToken } satisfies UsernameUpdateResponse);
});

// POST /api/auth/logout
router.post('/logout', (_req, res) => {
  res.json({ success: true });
});

// Helper: extract userId from token (used by other routes)
export function getUserId(req: any): string | null {
  const token = req.headers.authorization?.replace('Bearer ', '');
  if (!token) return null;
  return tokens.get(token) || null;
}

export default router;
