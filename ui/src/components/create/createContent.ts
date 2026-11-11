// createContent.ts — Create's per-song content as request fields. Shared by
// CreatePanel's Generate and by resuming a saved draft, so a resumed draft
// produces the same request the form would.
import type { GenerationParams } from '../../types';
import type { Yue2Continuation } from '../../../../server/src/contracts/yue2Continuation';

export interface CreateContent {
  caption: string; lyrics: string; negativePrompt: string; instrumental: boolean;
  loraTrigger: string; beatIntro: boolean; introBars: number;
  title: string; artist: string; subject: string;
  bpm: number; keyScale: string; timeSignature: string; duration: number; vocalLanguage: string;
  sourceLatentUrl: string;
  yue2Continuation?: Yue2Continuation;
}

/** Edit restores per-song source state separately from shared YuE2 controls. */
export function createYue2Replay(saved: Record<string, unknown>): {
  continuation: Yue2Continuation | undefined; shared: Record<string, unknown>;
} {
  return {
    continuation: saved.yue2Continuation as Yue2Continuation | undefined,
    shared: Object.fromEntries(Object.entries(saved).filter(([key]) => key.startsWith('yue2')
      && !['yue2', 'yue2Request', 'yue2Adapters', 'yue2Continuation'].includes(key))),
  };
}

/** LoRA trigger word prepended (unless already typed), beat intro/outro request appended. */
export function composeCaption(base: string, loraTrigger: string, beatIntro: boolean, introBars: number): string {
  const trigger = loraTrigger.trim();
  const start = base.trimStart();
  const hasTrigger = trigger.length > 0
    && start.slice(0, trigger.length).toLowerCase() === trigger.toLowerCase()
    && (start.length === trigger.length || /[,\s]/.test(start[trigger.length]));
  const loraText = trigger && !hasTrigger ? `${trigger}, ` : '';
  const beatText = beatIntro ? `, with a clean ${introBars}-bar percussive intro and outro for DJ mixing` : '';
  return `${loraText}${base}${beatText}`;
}

/** The content part of a Create request. `resolvedCaption`/`resolvedLyrics`
 *  are the wildcard-expanded texts when auto-expand ran. */
export function createContentParams(c: CreateContent, opts: { mm3Mode: boolean; yue2Active?: boolean; durationMax?: number; resolvedCaption?: string; resolvedLyrics?: string }): Partial<GenerationParams> {
  const params: Partial<GenerationParams> = {
    caption: composeCaption(opts.resolvedCaption ?? c.caption, opts.yue2Active ? '' : c.loraTrigger, c.beatIntro, c.introBars),
    lyrics: c.instrumental ? '[Instrumental]' : (opts.resolvedLyrics ?? c.lyrics),
    ...(c.negativePrompt.trim() ? { negative_prompt: c.negativePrompt.trim() } : {}),
    instrumental: c.instrumental,
    bpm: c.bpm, keyScale: c.keyScale, timeSignature: c.timeSignature, vocalLanguage: c.vocalLanguage,
    // MM3 has no length input — a duration there is a frame cap that can only
    // truncate the planner's own ending, so every MM3 render is auto. The
    // control is hidden in MM3 mode (MetadataSection), and this stops the
    // persisted ACE value riding along behind it. The backend enforces the
    // same thing, so a stale row or a direct API call cannot reinstate a cap.
    // YuE2 hides the control too (duration.editable: false) and its backend
    // trims to any duration it is sent, so it gets -1 for the same reason.
    duration: opts.mm3Mode || opts.yue2Active ? -1 : c.duration > 0
      ? Math.min(c.duration, opts.durationMax ?? Infinity) : c.duration,
    // vocalGender is deliberately NOT sent: neither backend has a wire field
    // for it. It reaches the model only by being written into the caption's
    // Vocal Details section by Mm3ComposeButton, and the caption is what
    // travels. Adding it to the request would create another dead knob.
    taskType: 'text2music',
  };
  // Optional song info fields — only include if populated
  if (c.title.trim()) params.title = c.title.trim();
  if (c.artist.trim()) params.artist = c.artist.trim();
  if (c.subject.trim()) params.subject = c.subject.trim();
  if (c.sourceLatentUrl) params.sourceLatentUrl = c.sourceLatentUrl;
  if (opts.yue2Active && c.yue2Continuation) Object.assign(params, { yue2Continuation: c.yue2Continuation });
  return params;
}

/** A saved Create draft's fields, read with the form's own defaults. */
export function createContentFromDraft(fields: Record<string, unknown>): CreateContent {
  const get = <T>(key: string, fallback: T): T => (Object.hasOwn(fields, key) ? fields[key] : fallback) as T;
  return {
    caption: get('hs-caption', ''), lyrics: get('hs-lyrics', ''), negativePrompt: get('hs-negative-prompt', ''),
    instrumental: get('hs-instrumental', false), loraTrigger: get('hs-lora-trigger', ''),
    beatIntro: get('hs-beat-intro', false), introBars: get('hs-intro-bars', 2),
    title: get('hs-title', ''), artist: get('hs-artist', ''), subject: get('hs-subject', ''),
    bpm: get('hs-bpm', 0), keyScale: get('hs-keyScale', ''), timeSignature: get('hs-timeSignature', ''),
    duration: get('hs-duration', -1), vocalLanguage: get('hs-vocalLanguage', 'en'),
    sourceLatentUrl: get('hs-sourceLatentUrl', ''),
  };
}

/** A Create draft resumes only on the backend it was saved with: its duration,
 *  caption and request rules depend on it, and switching the active backend is
 *  an app-wide setting a draft load must not change. '' accepts the draft. */
export function createDraftBackendRefusal(body: { backendId?: string }, activeBackendId: string | null | undefined,
  nameOf: (backendId: string) => string): string {
  if (!body.backendId || body.backendId === activeBackendId) return '';
  const name = nameOf(body.backendId);
  return `This draft was saved for ${name}. Switch the backend to ${name} to load it.`;
}

/** Apply an accepted Create draft through the same persisted keys the form reads. */
export function applyCreateDraft(body: { fields: Record<string, unknown> }, write: (key: string, value: unknown) => void): void {
  for (const [key, value] of Object.entries(body.fields)) if (key.startsWith('hs-')) write(key, value);
}

/** A refusal check that reads the active backend when it runs, which is after
 *  a draft's fetch finishes, so a backend switched mid-load is the one checked. */
export function liveBackendRefusal(activeBackendId: () => string | null | undefined, nameOf: (backendId: string) => string) {
  return (body: { backendId?: string }) => createDraftBackendRefusal(body, activeBackendId(), nameOf);
}
