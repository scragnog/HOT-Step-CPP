// trackScope.ts — a stable, key-safe id for one track in browser storage.
//
// Per-song settings cannot key on the library row's id: the lyric editor is
// handed an audio URL, not a song, and the server keys alignment by that URL
// too (see POST /songs/lyrics/align). The URL is therefore the identity every
// surface already agrees on.
//
// The scope keeps a readable piece of the filename so a key can be recognised
// in devtools, and appends a 32-bit hash of the whole path so two tracks with
// the same basename in different folders do not share settings.

/** A readable, key-safe scope for one audio URL. */
export function scopeFromAudioUrl(audioUrl: string): string {
  const path = audioUrl.split(/[?#]/, 1)[0] || audioUrl;
  const base = path.split('/').pop() || 'track';
  const readable = base
    .replace(/\.[a-z0-9]{1,8}$/i, '')
    .replace(/[^a-z0-9._-]+/gi, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48) || 'track';
  return `${readable}-${hash32(path).toString(36)}`;
}

/** FNV-1a, 32-bit: short, stable across sessions, and not a security boundary. */
function hash32(value: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < value.length; i++) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}
