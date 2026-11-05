export const MULACOVER_PACK = { name: 'MuLaCover', license: 'CC BY-NC 4.0' } as const;
export type HeartmulaCoverTags = Record<'topic' | 'genre' | 'instrument' | 'mood', string>;
export const EMPTY_HEARTMULA_TAGS: HeartmulaCoverTags = { topic: '', genre: '', instrument: '', mood: '' };
