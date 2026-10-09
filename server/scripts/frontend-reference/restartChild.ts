// restartChild.ts — one life of the harness for client.restart.test.ts.
//
//   node --import tsx restartChild.ts start <dataDir> <sourceDir>
//   node --import tsx restartChild.ts check <dataDir> <state.json>
//
// `start` brings the harness up on <dataDir>, starts work that never
// finishes on its own (a training job and a workflow job, their runners held
// by fixtures), prints what it started as one JSON line, and waits to be
// killed. `check` brings the harness up again on the same <dataDir> and
// reports, over HTTP, what became of that work. Both install the harness's
// safety guards before any production import (startFakeServer does) and
// report their violations.

import fs from 'node:fs';
import { startFakeServer } from './fakeServer.js';
import { ReferenceClient } from './client.js';
import { makeFixtureWav } from './fixtures.js';
import { heldRunner } from './fakeTrainingRunner.js';
import { installDatasetFixtures, installTrainerFixtures } from './trainingFixtures.js';

const [phase, dataDir, extra] = process.argv.slice(2);
const server = await startFakeServer({ dataDir });
const client = new ReferenceClient(server.origin);
await client.login();
const api = async (method: string, p: string, body?: unknown) => {
  const res = await fetch(`${server.origin}${p}`, { method, headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${client.token}` },
    body: body === undefined ? undefined : JSON.stringify(body) });
  return { status: res.status, body: await res.json() as any };
};
const until = async (check: () => Promise<boolean>, what: string) => {
  for (let i = 0; i < 100; i++) { if (await check()) return; await new Promise(r => setTimeout(r, 100)); }
  throw new Error(`timed out waiting for ${what}`);
};

if (phase === 'start') {
  // Runners that hold until the process dies: the work stays "running".
  const queue = await import('../../src/services/training/labelingQueue.js');
  queue.overrideTrainingRunner('build', heldRunner(queue));
  queue.overrideTrainingRunner('yue2-preprocess', heldRunner(queue));
  installTrainerFixtures();
  const { overrideCoverWorkflowDeps } = await import('../../src/services/workflows/coverWorkflow.js');
  overrideCoverWorkflowDeps({ caption: () => new Promise(() => { /* never settles */ }) });

  fs.writeFileSync(`${extra}/track-01.wav`, makeFixtureWav(1));
  const dataset = await api('POST', '/api/training/datasets', { name: 'Restart Fixture', sourceDir: extra });
  if (dataset.status !== 201) throw new Error(`dataset: ${JSON.stringify(dataset.body)}`);
  const build = await api('POST', `/api/training/datasets/${dataset.body.dataset.id}/build`, {});
  if (build.status !== 202) throw new Error(`build: ${JSON.stringify(build.body)}`);
  const trainingJobId = build.body.jobId as string;
  await until(async () => (await api('GET', `/api/training/jobs/${trainingJobId}`)).body.status === 'running', 'the training job to run');

  // A preparation pipeline caught mid-stage, on a second dataset (one job per dataset).
  const second = `${extra}-prep`;
  fs.mkdirSync(second);
  fs.writeFileSync(`${second}/track-01.wav`, makeFixtureWav(1));
  const prepSet = await api('POST', '/api/training/datasets', { name: 'Restart Prep Fixture', sourceDir: second });
  installDatasetFixtures(prepSet.body.dataset.slug, second);
  const ctx = (await api('GET', `/api/training/ops/preparation/context/${prepSet.body.dataset.id}`)).body;
  const prep = await api('POST', '/api/training/ops/preparation', { version: 1, operation: { kind: 'yue2-preparation', idempotencyKey: 'restart-prep' },
    worker: { kind: 'local' }, dataset: ctx.dataset, sources: [ctx.source],
    payload: { recipes: {}, mode: 'prepare-only', stages: ['latents'], trigger: '', lyricTiming: false } });
  if (prep.status !== 200) throw new Error(`preparation: ${JSON.stringify(prep.body)}`);
  const pipelineId = prep.body.pipeline.id as string;
  await until(async () => (await api('GET', `/api/training/ops/preparation/${pipelineId}`)).body.pipeline.stages
    .some((st: { status: string }) => st.status === 'running'), 'the preparation stage to run');

  const asset = await client.uploadAudio(makeFixtureWav(1));
  const open = await client.waitForWorkflowJob((await client.submitWorkflowJob('cover-open', { assetId: asset.asset_id })).id);
  const { documentId, revision } = open.result as { documentId: string; revision: number };
  const caption = await client.submitWorkflowJob('cover-caption', { documentId, revision, artistId: 1, provider: 'fixture', model: '', force: true });
  await until(async () => (await client.getWorkflowJob(caption.id)).status === 'running', 'the workflow job to run');

  process.stdout.write(`${JSON.stringify({ trainingJobId, workflowJobId: caption.id, datasetId: dataset.body.dataset.id, pipelineId, violations: server.violations })}\n`);
  await new Promise(() => { /* killed by the parent */ });
}

if (phase === 'check') {
  const state = JSON.parse(fs.readFileSync(extra, 'utf8')) as { trainingJobId: string; workflowJobId: string; datasetId: string; pipelineId: string };
  const training = await api('GET', `/api/training/jobs/${state.trainingJobId}`);
  const list = await api('GET', `/api/training/jobs?datasetId=${state.datasetId}`);
  const workflow = await client.getWorkflowJob(state.workflowJobId);
  const pipeline = (await api('GET', `/api/training/ops/preparation/${state.pipelineId}`)).body.pipeline;
  process.stdout.write(`${JSON.stringify({ training: { status: training.status, body: training.body }, listed: list.body.jobs, workflow, pipeline, violations: server.violations })}\n`);
  await server.close();
  process.exit(0);
}
