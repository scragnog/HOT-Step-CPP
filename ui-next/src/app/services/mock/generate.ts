// The simulated generation server (the interface is services/generate.ts; services/mock/index.ts registers this): the
// concept's simulation (35-gobar.js runBatch): one run renders at a time, the head of the queue,
// for 3.6 seconds of its own; each take starts a little after the one before; it plans, steps, decodes and runs the
// pedals; MiniMax-Music3 refuses a run that will not fit its card before loading anything; YuE2 re-plans a runaway
// score a few times. Its songs are kept in this browser, as a server keeps them, so they outlive a reload; its queue
// is not (a reload drops what was rendering, as the concept's page does). What the
// songs of a run are (draft) is the concept's too: a title from the first words, a length of 150 s and up to 90 more
// where the model ends the song (Stable Audio 3's 30 s without one), a colour at random, a repaint's from its song's.

import { attemptsAt, failWords, readyOf, refusedWords, stageOf, takeProgress, vramNeed, type RunSpec } from '../../../derive.ts';
import { ix } from '../../model/index.ts';
import { service } from '../../register/services.ts';
import type { GenerateService, Job, Song } from '../generate.ts';
import { draftRun } from './songs.ts';

/* ---------------------------------------------------------------- the simulated server */

const SONGS_KEY = 'hotstep.mock.songs';
const RUN_MS = 3600;
const DEMO: Song[] = [
  { id: '1', title: 'Studio sketch 01', style: 'Lo-fi hip hop, mellow electric piano', text: 'Lo-fi hip hop, mellow electric piano', dur: 171, hue: 215, engine: 'ace', createdAt: 6 },
  { id: '2', title: 'Evening texture', style: 'Ambient, warm pads, slow evolving textures', text: 'Ambient, warm pads, slow evolving textures', dur: 236, hue: 330, engine: 'sa3', createdAt: 5 },
  { id: '3', title: 'Untitled take', style: 'Folk rock, acoustic guitar, brushed drums', text: 'Folk rock, acoustic guitar, brushed drums', dur: 183, hue: 150, engine: 'yue2', createdAt: 4 },
  { id: '4', title: 'Late bus', style: 'Synth-pop, driving bass, bright lead', text: 'Synth-pop, driving bass, bright lead', dur: 221, hue: 265, engine: 'ace', createdAt: 3 },
  { id: '5', title: 'Field recording', style: 'Post-rock, swelling guitars, big drums', text: 'Post-rock, swelling guitars, big drums', dur: 300, hue: 30, engine: 'minimax-m3', createdAt: 2 },
  { id: '6', title: 'Rhodes at 2am', style: 'Soul, warm Rhodes, tight groove', text: 'Soul, warm Rhodes, tight groove', dur: 198, hue: 5, engine: 'ace', createdAt: 1 },
];
const isSong = (x: unknown): x is Song => !!x && typeof x === 'object' && typeof (x as Song).id === 'string' && typeof (x as Song).title === 'string' && typeof (x as Song).dur === 'number';
function readSongs(): Song[] {
  try { const a = JSON.parse(localStorage.getItem(SONGS_KEY) ?? 'null'); if (Array.isArray(a)) return a.filter(isSong); } catch { /* unreadable: the demo songs */ }
  return DEMO.map(s => ({ ...s }));
}

interface MockJob { id: string; spec: RunSpec; ids: string[]; alive: Set<string>; prog: number; status: Job['status']; att: number; attempts: number; refused?: Job['refused']; failed?: Job['failed']; fails?: boolean; result?: Song[] }

/** The card the simulated server has, for a scenario that makes it refuse what the page thought would fit (the
 *  page's figures are an estimate; the server checks its own). Undefined: the models service's card. */
export const mockCard = { gb: undefined as number | undefined };
/** The next run the simulated server takes stops half-way, as a run whose engine fails does. */
export const mockFail = { next: false };

export function mockGenerate(): GenerateService {
  let songs = readSongs();
  const jobs: MockJob[] = [], watchers = new Set<(jobs: Job[]) => void>();
  let raf = 0, last = 0, next = 1, card = 31.8, figures: Awaited<ReturnType<ReturnType<typeof service<'models'>>['vram']>> | undefined;
  void service('models').vram().then(f => { figures = f; card = f.card; });
  const keep = () => { try { localStorage.setItem(SONGS_KEY, JSON.stringify(songs.slice(0, 200))); } catch { /* the server's disk is full */ } };
  const live = () => jobs.filter(j => j.status === 'queued' || j.status === 'running');
  const view = (): Job[] => {
    const l = live();
    return jobs.map(j => {
      const f = Math.min(1, j.prog / RUN_MS), sp = j.spec;
      return {
        id: j.id, engine: sp.engine, status: j.status, ahead: Math.max(0, l.indexOf(j)), f, attempts: j.attempts,
        songs: j.ids.map((id, i) => ({ id, f: takeProgress(f, i), stage: stageOf(takeProgress(f, i), sp.engine, sp.steps, sp.lit), ready: sp.stream ? readyOf(takeProgress(f, i)) : 0 })).filter(s => j.alive.has(s.id)),
        ...(j.result ? { result: j.result } : {}), ...(j.refused ? { refused: j.refused } : {}), ...(j.failed ? { failed: j.failed } : {}),
      };
    });
  };
  const emit = () => { const v = view(); for (const w of watchers) w(v); for (const j of [...jobs]) if (j.status !== 'queued' && j.status !== 'running') jobs.splice(jobs.indexOf(j), 1); };
  const frame = (now: number) => {
    raf = 0;
    const dt = Math.min(100, Math.max(0, now - last)); last = now;
    const head = live()[0];
    if (head) {
      head.status = 'running'; head.prog += dt;
      const sp = head.spec, n = head.ids.length, f = Math.min(1, head.prog / RUN_MS);
      /* MiniMax-Music3 checks before it loads anything: a run its card cannot hold is refused at once */
      const own = mockCard.gb ?? card, v = vramNeed(ix, { ...sp.desk, ...(figures ? { vram: { ...figures, card: own } } : {}) }, n);
      if (v.refuses) { head.status = 'refused'; head.refused = refusedWords(sp.engine, n, v); }
      else {
        if (head.fails && f >= 0.5) { head.status = 'failed'; head.failed = { why: failWords(sp.engine) }; }
        else if (sp.tries) { head.att ||= 1 + Math.floor(Math.random() * sp.tries); head.attempts = attemptsAt(f, head.att); }
        if (head.status === 'running' && f >= 1) {
          head.status = 'done';
          head.result = head.ids.filter(id => head.alive.has(id)).map(id => { const i = head.ids.indexOf(id), s = sp.songs[i]!; return { id, title: s.title, ...(s.take ? { take: s.take, of: s.of! } : {}), style: s.style, text: s.text, dur: s.dur, hue: s.hue, engine: sp.engine, createdAt: Date.now(), ...(s.seed !== undefined ? { seed: s.seed } : {}) }; });
          songs = [...head.result, ...songs]; keep();
        }
      }
    }
    emit();
    if (live().length) raf = requestAnimationFrame(frame);
  };
  const run = () => { if (!raf && live().length) { last = performance.now(); raf = requestAnimationFrame(frame); } };
  return {
    songs: async () => songs.map(s => ({ ...s })),
    draft: (ask, rnd = Math.random) => draftRun(ask, rnd),
    submit: async (spec, ids) => {
      const id = `job-${next++}`;
      jobs.push({ id, spec, ids, alive: new Set(ids), prog: 0, status: 'queued', att: 0, attempts: 0, fails: mockFail.next }); mockFail.next = false;
      run(); return { jobId: id };
    },
    cancel: async (jobId, ids) => {
      const j = jobs.find(x => x.id === jobId); if (!j) return;
      for (const id of ids) j.alive.delete(id);
      if (!j.alive.size) j.status = 'cancelled';
      emit(); run();
    },
    live: async () => live().map(j => j.id),
    watch: fn => { watchers.add(fn); return () => { watchers.delete(fn); }; },
  };
}

