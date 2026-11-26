// The Concept notes and the engine log's own state (package 11): whether each stands open, what the server said about
// itself and what the queue holds, for the log. View state and what the server owns: nothing here is about a value of
// the desk, nothing is kept between visits, and Undo covers none of it.
//
// The queue is the generation package's to tell (`tellQueue`): until it does, the log says the queue is idle.

import { create } from 'zustand';
import type { EngineStatus, QueueSummary } from '../../derive.ts';

export interface NotesLogsState {
  notes: boolean;
  logs: boolean;
  status: EngineStatus | null;
  queue: QueueSummary | null;
}
export const useNotesLogs = create<NotesLogsState>()(() => ({ notes: false, logs: false, status: null, queue: null }));

/** What the queue holds now, for the engine log's queue line (the generation package calls it as jobs come and go). */
export const tellQueue = (queue: QueueSummary | null): void => useNotesLogs.setState({ queue });
