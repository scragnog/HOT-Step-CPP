// songDisplay.ts — how a song reads in a list, as opposed to how it is stored.
//
// Nothing here is ever written back. Rename edits the stored title; these
// helpers only decide what the card, row and table show.

import type { Song } from '../types';

/**
 * Trailing model tags people leave on a title when comparing writers —
 * "Ninety Minute Tape - Claude Opus", "Sunny Delaney - Opus 5",
 * "This Is My Flatmate (Opus 5)". Useful while auditioning, noise afterwards.
 *
 * Deliberately anchored to a known list of model families: a bare
 * "Title - Something" is far more likely to be part of the name than a tag.
 */
const MODEL_TAG = new RegExp(
  '\\s*[-–—(\\[]\\s*' +                                   // the separator
  '(?:claude\\s+|anthropic\\s+|openai\\s+|google\\s+)?' + // optional vendor
  '(?:opus|sonnet|haiku|fable|gpt|chatgpt|gemini|grok|llama|mistral|mixtral|' +
  'qwen|deepseek|kimi|glm|command-?r|phi|nova|o[13-9])' + // the family
  '[\\d.\\s-]*' +                                         // version, e.g. " 5", "-4.1"
  '(?:mini|pro|turbo|preview|flash|thinking|max|air)?' +  // and its variant
  '[\\d.\\s-]*' +
  '[)\\]]?\\s*$',                                         // optional closer
  'i',
);

/** Drop a trailing model tag from a title. Leaves anything else alone. */
export function stripModelTag(title: string): string {
  const stripped = title.replace(MODEL_TAG, '').trim();
  // Never strip a title down to nothing — a song called "Opus 5" keeps its name.
  return stripped || title.trim();
}

function params(song: Song): any {
  return (song.generationParams || song.generation_params) as any;
}

/** Who the song is by, from Song Info or the row itself. Empty when unset. */
export function songArtist(song: Song): string {
  const gp = params(song);
  const fromParams = typeof gp?.artist === 'string' ? gp.artist.trim() : '';
  return fromParams || (song.artistName || '').trim();
}

/** What the song is about, from Song Info. Empty when unset. */
export function songSubject(song: Song): string {
  const gp = params(song);
  return typeof gp?.subject === 'string' ? gp.subject.trim() : '';
}

/**
 * Title as DISPLAYED: the artist in front, the model tag gone.
 * Never what rename edits, which stays the stored title.
 */
export function displayTitle(song: Song): string {
  const title = stripModelTag(song.title || 'Untitled') || 'Untitled';
  const artist = songArtist(song);
  // Don't double up when the stored title already carries the artist.
  if (!artist || title.toLowerCase().startsWith(`${artist.toLowerCase()} - `)) return title;
  return `${artist} - ${title}`;
}

/**
 * The description under the title. The subject wins when there is one: it says
 * what this track is, where the caption is the same handful of genre words on
 * every track of an album.
 */
export function displaySubtext(song: Song): string {
  return songSubject(song) || song.style || song.caption || '';
}
