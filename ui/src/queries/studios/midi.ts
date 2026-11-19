import { queryOptions, type QueryClient } from '@tanstack/react-query';
import { serverKey, resourceKey } from '@/queries/keys';
import { pollWhile } from '@/queries/contract';
import { queryManager } from '@/queries/invalidation';
import { getMidiStatus, listMidiJobs, getMidiProgress, getMidiNotes, deleteMidiJob, type MidiJobSummary, type MidiStudioStatus } from '@/services/midiStudioApi';

export const midiActive = (job: { status: string }) => job.status === 'queued' || job.status === 'transcribing';
export const midiStatusOptions = () => queryOptions({
  queryKey: serverKey('studios/midi-status'),
  queryFn: ({ signal }) => getMidiStatus({ signal }),
  refetchInterval: pollWhile<MidiStudioStatus>(data => !!data && Object.values(data.models).some(model => model.downloading)),
});

// One cancellable list request owns each polling cycle. Progress reads share
// its signal, so deleting a job aborts the whole old snapshot before it lands.
export const midiJobsOptions = () => queryOptions({
  queryKey: serverKey('studios/midi-jobs'),
  queryFn: async ({ signal }) => {
    const jobs = await listMidiJobs({ signal });
    return Promise.all(jobs.map(async job => midiActive(job)
      ? { ...job, ...await getMidiProgress(job.id, { signal }) }
      : job));
  },
  refetchInterval: pollWhile<MidiJobSummary[]>(jobs => !!jobs?.some(midiActive)),
});

export const midiNotesOptions = (id: string) => queryOptions({
  queryKey: serverKey('studios/midi-notes', { id }),
  queryFn: ({ signal }) => getMidiNotes(id, { signal }),
});

export async function removeMidiJob(client: QueryClient, id: string) {
  const list = { queryKey: resourceKey('studios/midi-jobs') };
  const notes = { queryKey: resourceKey('studios/midi-notes', undefined, id) };
  await Promise.all([client.cancelQueries(list), client.cancelQueries(notes)]);
  try {
    await deleteMidiJob(id);
    // A timer may have started another read while DELETE was pending.
    await Promise.all([client.cancelQueries(list), client.cancelQueries(notes)]);
    client.setQueriesData<MidiJobSummary[]>(list, jobs => jobs?.filter(job => job.id !== id));
    client.removeQueries(notes);
  } finally {
    await queryManager.filter(client, list);
  }
}
