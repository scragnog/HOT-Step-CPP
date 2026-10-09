// trainingFixtures.ts — the files the training start routes look for, made
// in the harness's isolated dirs so each route's own checks pass. Nothing
// here is executed or loaded as a model: the trainer binary is a placeholder
// whose existence aceTrainExe() checks, and the runners that would have used
// these files are replaced (fakeTrainingRunner.ts).
//
// Paths come from the env fakeServer.ts set: HOT_STEP_ROOT (the engine dir,
// where ace-train lives), ACESTEPCPP_MODELS and TRAINING_DIR.

import fs from 'node:fs';
import path from 'node:path';

const put = (file: string, content: string) => {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, content);
};

/** Shared by every family: the trainer binary next to ace-server, and the
 *  base model files the MM3 and YuE2 routes check for. Once per process. */
export function installTrainerFixtures(): void {
  const root = process.env.HOT_STEP_ROOT!;
  const models = process.env.ACESTEPCPP_MODELS!;
  put(path.join(root, 'engine', process.platform === 'win32' ? 'ace-train.exe' : 'ace-train'), 'fixture');
  put(path.join(models, 'fake-dit.gguf'), 'fixture');
  put(path.join(models, 'mm3', 'mm3-lm-q8_0.gguf'), 'fixture');
  put(path.join(models, 'mm3', 'mm3-depth-f16.gguf'), 'fixture');
  put(path.join(models, 'yue2', 'yue2-lm-bf16.gguf'), 'fixture');
  put(path.join(models, 'yue2', 'yue2-vae-standard-f32.gguf'), 'fixture');
  put(path.join(models, 'yue2', 'minted_manifest.json'), '{}');
  put(path.join(models, 'yue2', 'minted_codes.i32'), 'fixture');
}

/** The dataset's prepared state: a build (dataset.json), an ACE preprocess
 *  variant, MM3 codes and caption, and a YuE2 latent cache manifest. */
export function installDatasetFixtures(slug: string, sourceDir: string): void {
  const training = process.env.TRAINING_DIR!;
  put(path.join(sourceDir, 'dataset.json'), JSON.stringify({ metadata: {}, samples: [{ filename: 'track-01.wav' }] }));
  put(path.join(sourceDir, 'track-01.mm3.txt'), 'fixture caption');
  put(path.join(training, 'tensors', slug, 'fake-dit', 'preprocess_meta.json'),
    JSON.stringify({ created_at: '2026-01-01T00:00:00Z', model_variant: 'fake-dit.gguf', custom_tag: 'trig', tag_position: 'prepend' }));
  put(path.join(training, 'datasets', slug, 'mm3-codes', 'codes', 'track-01.codes'), 'fixture');
  put(path.join(training, 'datasets', slug, 'yue2-latents', 'yue2_preprocess.json'), JSON.stringify({
    n_sources: 1, n_clips: 1, codec_ids_present: true,
    sources: [{ name: 'track-01', cursor_words: 'w', codec_ids: 'c' }], clips: [{ codec_ids: 'c' }],
  }));
}
