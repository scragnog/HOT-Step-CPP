// textPairing.ts — match two word sequences that are the same words, retimed.
//
// Two callers look different and are not: scoring a fixture against a run, and
// auditing CTC timings against Whisper's anchors. Both sides of both comparisons
// are the same lyric text, so the only difference to absorb is a word that one
// side dropped or invented.
//
// That is why this is not Needleman-Wunsch. A general sequence alignment would
// happily stretch one block over a longer stretch of the other and report a
// small average error, which is precisely the failure both callers exist to
// catch. Looking a few words ahead on each side and resyncing keeps a dropped
// word local: everything after it is compared to the right partner.

export interface TextWord {
  word: string;
}

/** How far either side a mismatch may look before that pair is given up on. */
const RESYNC_WORDS = 4;
/** A mismatch wider than the near window is only bridged on a shared anchor,
 *  so a whole dropped line still resyncs instead of shifting the rest. */
const BLOCK_RESYNC_WORDS = 16;

/** Comparison form: case and punctuation are not what a timing measures. */
export function normaliseWord(word: string): string {
  return word.toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');
}

/** Whether `run` consecutive words match at `a[i]` and `b[j]`. */
function matchesAt<A extends TextWord, B extends TextWord>(
  a: A[], i: number, b: B[], j: number, run: number,
): boolean {
  for (let k = 0; k < run; k++) {
    if (i + k >= a.length || j + k >= b.length) return false;
    const left = normaliseWord(a[i + k].word);
    if (left === '' || left !== normaliseWord(b[j + k].word)) return false;
  }
  return true;
}

/**
 * Whether a resync anchor starts here: two words when both sides have two, one
 * when the shorter side is down to its last. One word is not enough evidence —
 * "the" or "i" would jump the comparison to a coincidence — except where there
 * is nothing else left to compare.
 */
function anchorAt<A extends TextWord, B extends TextWord>(
  a: A[], ai: number, b: B[], bj: number,
): boolean {
  const left = Math.min(a.length - ai, b.length - bj);
  if (left <= 0) return false;
  return matchesAt(a, ai, b, bj, Math.min(2, left));
}

/**
 * Pair two sequences in reading order.
 *
 * Whichever side can resync soonest wins. When neither side can resync within
 * the near window, a shared two-word anchor further out is looked for: that is
 * what carries the comparison past a whole dropped line, rather than both sides
 * stepping once and every word after it being compared to the wrong partner.
 * With no anchor at all the pairing stops, because a tail nothing in common can
 * only be paired by guesswork — and the callers exist to detect exactly that.
 */
export function pairByText<A extends TextWord, B extends TextWord>(
  a: A[], b: B[], resync = RESYNC_WORDS,
): Array<[A, B]> {
  const pairs: Array<[A, B]> = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (normaliseWord(a[i].word) === normaliseWord(b[j].word)) {
      pairs.push([a[i], b[j]]);
      i++;
      j++;
      continue;
    }
    let skipA = -1;
    let skipB = -1;
    for (let k = 1; k <= resync; k++) {
      if (skipA < 0 && i + k < a.length && normaliseWord(a[i + k].word) === normaliseWord(b[j].word)) {
        skipA = k;
      }
      if (skipB < 0 && j + k < b.length && normaliseWord(b[j + k].word) === normaliseWord(a[i].word)) {
        skipB = k;
      }
      if (skipA >= 0 || skipB >= 0) break;
    }
    if (skipA >= 0 && (skipB < 0 || skipA <= skipB)) { i += skipA; continue; }
    if (skipB >= 0) { j += skipB; continue; }

    // Nothing within the near window: look for a shared anchor past it, on
    // either side, and jump both sides to it. This is the dropped-block case.
    let blockA = -1;
    let blockB = -1;
    for (let k = 1; k <= BLOCK_RESYNC_WORDS; k++) {
      if (anchorAt(a, i + k, b, j)) { blockA = k; break; }
      if (anchorAt(a, i, b, j + k)) { blockB = k; break; }
    }
    if (blockA >= 0) { i += blockA; continue; }
    if (blockB >= 0) { j += blockB; continue; }
    break;
  }
  return pairs;
}
