import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import type { TrainingDatasetRow, TrainingSample } from './types.js';

function chunk(id: string, data: Buffer): Buffer {
  const bytes = Buffer.alloc(8 + data.length + (data.length % 2));
  bytes.write(id, 0, 'ascii');
  bytes.writeUInt32LE(data.length, 4);
  data.copy(bytes, 8);
  return bytes;
}

function writeTaggedWave(file: string, artist: string, album: string, title: string) {
  const fmt = Buffer.alloc(16);
  fmt.writeUInt16LE(1, 0); // PCM
  fmt.writeUInt16LE(1, 2); // mono
  fmt.writeUInt32LE(8000, 4); // sample rate
  fmt.writeUInt32LE(16000, 8); // byte rate
  fmt.writeUInt16LE(2, 12); // block align
  fmt.writeUInt16LE(16, 14); // bits per sample
  const info = Buffer.concat([
    Buffer.from('INFO'),
    chunk('IART', Buffer.from(artist + '\0')),
    chunk('IPRD', Buffer.from(album + '\0')),
    chunk('INAM', Buffer.from(title + '\0')),
  ]);
  fs.writeFileSync(file, chunk('RIFF', Buffer.concat([
    Buffer.from('WAVE'), chunk('fmt ', fmt), chunk('data', Buffer.alloc(160)), chunk('LIST', info),
  ])));
}

test('UTF-8 WAV tags survive Lyric Studio preview and export with fresh and legacy labels', async t => {
  const root = fs.mkdtempSync(path.resolve('.test-unicode-export-'));
  process.env.DATA_DIR = root;
  process.env.TRAINING_DIR = path.join(root, 'training');
  const { config } = await import('../../config.js');
  config.lireek.geniusAccessToken = '';
  const database = await import('../../db/database.js');
  const rows = await import('../../db/lireekDb.js');
  const { previewLyricStudioExport, commitLyricStudioExport } = await import('./lyricStudioExport.js');
  const { sanitizeTag } = await import('./audioMeta.js');
  database.initDb();
  try {
    const artist = 'Исполнитель';
    const album = 'Альбом';
    const title = 'Песня';
    const file = path.join(root, '01.wav');
    writeTaggedWave(file, artist, album, title);
    const ds = { id: 'unicode-fixture', slug: 'unicode-fixture', name: 'Fixture', sourceDir: root,
      defaultArtist: '', defaultAlbum: '' } as TrainingDatasetRow;
    const legacyAscii = (tag: string) => sanitizeTag(Buffer.from(tag).toString('ascii'));
    const cases = [
      { name: 'fresh source', tags: ['', '', ''], expectedTitle: title },
      { name: 'legacy label', tags: [artist, album, title].map(legacyAscii), expectedTitle: title },
      { name: 'edited title', tags: [legacyAscii(artist), legacyAscii(album), 'Edited title'], expectedTitle: 'Edited title' },
    ];
    for (const { name, tags, expectedTitle } of cases) {
      await t.test(name, async () => {
        const [tagArtist, tagAlbum, tagTitle] = tags;
        const sample = {
          sampleId: 'fixture', filename: '01.wav', relPath: '01.wav', audioPath: file,
          tagArtist, tagAlbum, tagTitle, lyrics: 'Test lyrics',
          isInstrumental: false, caption: '', genre: '', key: '', signature: '',
          language: '', excluded: false, fileMissing: false,
        } as TrainingSample;
        const preview = await previewLyricStudioExport(ds, [{ ...sample }]);
        assert.deepEqual([preview.artist, preview.album, preview.songs[0].title], [artist, album, expectedTitle]);

        // The POST request rebuilds samples from disk and receives the artist
        // and album that the user confirmed in the preview.
        const result = await commitLyricStudioExport(ds, [sample], {
          artist: preview.artist, album: preview.album, linkAdapters: false,
        });
        const set = rows.getLyricsSet(result.lyricsSetId)!;
        assert.equal(rows.findArtistByName(artist)?.id, set.artist_id);
        assert.equal(set.album, album);
        assert.deepEqual([set.songs[0].title, set.songs[0].album], [expectedTitle, album]);
      });
    }
  } finally {
    database.closeDb();
    fs.rmSync(root, { recursive: true, force: true });
  }
});
