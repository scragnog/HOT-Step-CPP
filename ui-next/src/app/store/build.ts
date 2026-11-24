// What the server sessions of package 3c tell the desk (desk.sessions): the build session's takes and the section
// looked at (services/build.ts), and the lengths of the recordings given to Continue from recording
// (services/recording.ts). The core draws from them (src/derive/build.ts, recording.ts). Server-owned: told as the
// engine's lists are told (store/desk.ts told), never a step, never saved with the desk, never taken back by Undo.
//
// One thing follows the desk the other way: a song the build rendered is in the Library while the render that made it
// stands. Undoing a render takes its song out of the record's songs, and the session withdraws it; redoing it puts it
// back (B-06: Render song is one undo step).

import type { Json } from '../../definitions/types.ts';
import { useDesk } from './desk.ts';
import { buildSession } from '../services/build.ts';
import { recordings } from '../services/recording.ts';

/** The state whose record is the build, and the file whose length the recording span needs. */
const BUILD = 'build', RECORDING = 'contFile';

let build: Json = null, recording: { [file: string]: number } = {};
const told = () => {
  const b = buildSession.state();
  build = { planning: b.planning, sessions: b.sessions } as unknown as Json;
};
const tell = () => {
  const s = useDesk.getState(), file = s.desk.state[RECORDING];
  if (typeof file === 'string' && file && recording[file] === undefined) recording = { ...recording, [file]: recordings.length(file) };
  if (s.desk.sessions?.build === build && s.desk.sessions?.recording === recording) return;
  useDesk.setState({ desk: { ...s.desk, sessions: { ...s.desk.sessions, build, recording } } });
};
told(); tell();
buildSession.subscribe(() => { told(); tell(); });
recordings.subscribe(() => { recording = recordings.known(); tell(); });

const songsOf = (rec: Json | undefined): string[] => { const s = (rec as { songs?: unknown } | null | undefined)?.songs; return Array.isArray(s) ? s.filter((x): x is string => typeof x === 'string') : []; };
useDesk.subscribe((s, prev) => {
  tell();
  if (s.desk.state[BUILD] === prev.desk.state[BUILD]) return;
  const now = songsOf(s.desk.state[BUILD]), was = songsOf(prev.desk.state[BUILD]);
  for (const id of was) if (!now.includes(id)) buildSession.withdraw(id, true);
  for (const id of now) if (!was.includes(id)) buildSession.withdraw(id, false);
});
