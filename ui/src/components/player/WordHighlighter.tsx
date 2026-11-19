import React from 'react';
import type { LyricsWord } from '@/utils/lyrics/wordLrcUtils';

interface WordHighlighterProps {
  words: LyricsWord[];
  activeWordIndex: number;
}

export const WordHighlighter: React.FC<WordHighlighterProps> = ({ words, activeWordIndex }) => {
  return (
    <span className="lyrics-word-container">
      {words.map((w, i) => {
        let cls = 'lyrics-word';
        if (i < activeWordIndex) cls += ' lyrics-word--past';
        else if (i === activeWordIndex) cls += ' lyrics-word--active';
        else cls += ' lyrics-word--future';
        return (
          <span key={`${i}-${w.start}`} className={cls}>
            {w.word}{i < words.length - 1 ? ' ' : ''}
          </span>
        );
      })}
    </span>
  );
};
