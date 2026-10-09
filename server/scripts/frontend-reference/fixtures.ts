// fixtures.ts — tiny, disposable inputs the reference client sends: a
// minimal valid WAV buffer, and the one real asset-acquisition call
// (POST /api/upload/audio) every source-audio workflow needs before it can
// reference a file by URL. No network dependency beyond the fake server
// itself — no ffmpeg, no device capture.

/** A structurally valid, silent, 16-bit PCM WAV. 16-bit PCM is required: it
 *  is the one format services/audioConvert.ts's ensureEngineFormat() treats
 *  as already engine-compatible, so SuperSep's `POST /api/supersep/separate`
 *  reads and forwards it as-is instead of needing a real conversion tool. */
export function makeFixtureWav(durationSec = 0.1, sampleRate = 44100, channels = 1): Buffer {
  const frames = Math.round(durationSec * sampleRate);
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;
  const dataSize = frames * channels * bytesPerSample;
  const buf = Buffer.alloc(44 + dataSize);
  buf.write('RIFF', 0, 'ascii');
  buf.writeUInt32LE(36 + dataSize, 4);
  buf.write('WAVE', 8, 'ascii');
  buf.write('fmt ', 12, 'ascii');
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(channels, 22);
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * channels * bytesPerSample, 28);
  buf.writeUInt16LE(channels * bytesPerSample, 32);
  buf.writeUInt16LE(bitsPerSample, 34);
  buf.write('data', 36, 'ascii');
  buf.writeUInt32LE(dataSize, 40);
  return buf;
}

export interface UploadedAsset {
  audio_url: string;
  filename: string;
  asset_id: string;
}

/** POST /api/upload/audio with a fixture WAV — the real way a client gets a
 *  `sourceAudioUrl`/`assetId` onto the server, rather than writing into the
 *  data dir directly. */
export async function uploadFixtureAudio(origin: string, token: string, filename = 'fixture.wav'): Promise<UploadedAsset> {
  const form = new FormData();
  form.append('audio', new Blob([makeFixtureWav()], { type: 'audio/wav' }), filename);
  const res = await fetch(`${origin}/api/upload/audio`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  });
  if (!res.ok) throw new Error(`upload fixture audio failed (${res.status}): ${await res.text()}`);
  return res.json() as Promise<UploadedAsset>;
}
