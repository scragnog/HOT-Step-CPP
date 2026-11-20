// recordingLink.ts — list a finished render under its written song (Lyric
// Studio's Recordings tab). A Lyric Studio render carries `meta.lyricLink` on
// the audio queue, and the queue calls linkRenderedRecording each time an item
// succeeds, so a render that failed and was retried from the queue is still
// listed. The lyric workflow links too; both go through linkRecording, which
// writes one row per job.
import * as db from '../../db/lireekDb.js';
import type { AudioIntentItem } from '../../contracts/audioQueue.js';

export interface LyricLink { generationId: number; lyricsSetId?: number }

export function linkRecording(link: LyricLink, jobId: string, result: Record<string, unknown> | null | undefined): void {
  const listed = db.getAudioGenerations(link.generationId, link.lyricsSetId).some((row: any) => row.hotstep_job_id === jobId);
  if (!listed) db.linkAudioGeneration(link.generationId, jobId, link.lyricsSetId);
  const url = result?.audioUrl || (result?.audioUrls as string[] | undefined)?.[0];
  if (typeof url === 'string' && url) db.resolveAudioGeneration(jobId, url);
}

/** The audio queue's onSucceeded hook. */
export function linkRenderedRecording(item: AudioIntentItem): void {
  const link = item.meta?.lyricLink as LyricLink | undefined;
  if (!link?.generationId || !item.jobId) return;
  linkRecording(link, item.jobId, item.result as Record<string, unknown> | null);
}
