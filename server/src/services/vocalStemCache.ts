// vocalStemCache.ts — the one vocal-stem cache both features share.
//
// A stem is keyed on the audio's content hash, the separation level and the
// cache version, so the file Cover Studio produces for a track is exactly the
// file lyric alignment wants for it. Two constructors with two roots
// (`tmp/cover-vocals` and `tmp/lyric-vocals`) is how alignment ended up never
// finding one, while still paying for the lookup. This module is the single
// owner of the root, the level and the separator, so the two callers cannot
// drift apart again; `coverStemCache` stays the implementation.
//
// Nothing here runs at import time beyond building the object: the directory is
// created on the first separation, and `find()` is a read.

import path from 'node:path';
import { config } from '../config.js';
import { createCoverStemCache } from './coverStemCache.js';
import { separateOne, YUE2_STEMS_DEFAULT_LEVEL } from './training/yue2Stems.js';

export const vocalStems = createCoverStemCache({
  root: path.join(config.data.dir, 'tmp', 'cover-vocals'),
  level: YUE2_STEMS_DEFAULT_LEVEL,
  separate: (src, out, level) => separateOne(src, out, level, () => {}),
});
