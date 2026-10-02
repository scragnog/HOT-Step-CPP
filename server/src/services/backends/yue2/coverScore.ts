// SheetSage writes quoted chord symbols before notes in the tune body.
// Keep ABC headers, voice declarations, lyrics and comments untouched.
const INLINE_CHORD = /"[A-G](?:#{1,2}|b{1,2})?(?:m\(maj7\)|maj7|m7b5|dim7|7sus4|sus[24]|m[67]?|dim|aug|[67])?(?:\/[A-G](?:#{1,2}|b{1,2})?)?"/g;

export function stripYue2CoverChords(abc: string): string {
  return abc.split(/(\r?\n)/).map((line, index) => {
    if (index % 2 || /^\s*(?:[A-Za-z+]:|%)/.test(line)) return line;
    const commentAt = line.indexOf('%');
    const music = commentAt < 0 ? line : line.slice(0, commentAt);
    return music.replace(INLINE_CHORD, '') + (commentAt < 0 ? '' : line.slice(commentAt));
  }).join('');
}

export function coverScoreSnapshot(fullScore: string, keepChords: boolean) {
  return { fullScore, renderedAbc: keepChords ? fullScore : stripYue2CoverChords(fullScore) };
}
