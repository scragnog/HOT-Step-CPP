// masterToAlbumPref.ts — the Lyric Studio "Master to Album Track" toggle.
//
// When on, every Lyric Studio render turns Mastering on, skips the VST chain,
// and masters to a random track of the album's training dataset (the server
// draws the track at render time). The key and reader live here so the queue
// store can read it without importing the sidebar component.

export const MASTER_TO_ALBUM_KEY = 'lireek-masterToAlbumTrack';

/** True only when the user has ticked the box — default is OFF. */
export function masterToAlbumEnabled(): boolean {
  try {
    const raw = localStorage.getItem(MASTER_TO_ALBUM_KEY);
    return raw !== null && JSON.parse(raw) === true;
  } catch {
    return false;
  }
}
