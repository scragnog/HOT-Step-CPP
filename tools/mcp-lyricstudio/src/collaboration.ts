// Shared discussion storage for independent stdio MCP processes.
// No imports from the app and no connection to hotstep.db.
import Database from 'better-sqlite3';
import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';
import { setTimeout as delay } from 'node:timers/promises';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { COORDINATION_SCHEMA, DiscussionCoordination, mentionHandle } from './discussion-coordination.js';
import { CONSENSUS_SCHEMA, DiscussionConsensus } from './discussion-consensus.js';
import { PRESENCE_SCHEMA, DiscussionPresence } from './discussion-presence.js';
import { MAX_EVIDENCE_CHARS, MAX_EVIDENCE_ITEMS, MAX_POSITION_CHARS, POSITIONS_SCHEMA, DiscussionPositions, normaliseEvidence, renderPosition } from './discussion-positions.js';
import { detectBaseCommit } from './discussion-base-commit.js';
import { CHANNEL_NOTIFICATION, WakeTracker, channelEnabled } from './discussion-wake.js';
import { registerWorkTools } from './work-tools.js';

export const DEFAULT_COLLAB_DB = fileURLToPath(new URL('../../../data/collaboration.db', import.meta.url));
export const DEFAULT_MAX_ROUNDS = 2;

export const DISCUSSION_PROTOCOL = `You are participating as the current chat agent, not launching another model.
Read the brief and transcript before replying. Post concrete proposals and critiques with code references where useful.
Sealed positions: while discussion.phase is "positions", read the brief and any user direction, research if needed, then submit one independent position with collab_submit_position (up to 24000 characters): your proposed plan, what would change your mind, and an evidence list of concrete sources (file paths with lines, commit SHAs, log paths, listening results, measurements; at least one, and the critique should target the evidence, not the taste). Other agents' positions stay hidden until every present agent has submitted or the user reveals them; replies, decisions and agreement are refused until then. Do not ask another agent what it thinks first. After the reveal, positions appear in the transcript in submission order without consuming a turn, and the turn rules below apply. Critique the other position from your assigned role before converging.
Base commit: the room records the commit it was created against (discussion.base_commit). Plan against that source. Say so when you rely on something newer.
Reconciler: a participant joined with reconciler=true sits outside the pair. It submits no position and casts no agreement vote. While a reconciler is present, only the reconciler records plans: read both positions and both critiques, draft the merged plan with open_items for what the evidence does not settle, and record it. Other agents critique and agree. If you are the reconciler, do not take a side before both critiques are in.
Outcome: after a room closes, whoever ships or abandons the plan records the result with collab_record_outcome (shipped, partial, abandoned or superseded, with a note and the commit). A closed room with a plan and no outcome is an unpaid debt; collab_list_discussions marks it needs_outcome.
Open items: collab_record_decision takes open_items, each an issue with an owner (You or a joined agent name). Free-text disagreements count as one item owned by You. Consensus is refused while the current revision has open items. When a revision at or past max_rounds still has open items, the room pauses itself and the user rules; do not keep negotiating past that. After the ruling, record a revision that removes each item and cites the direction message.
Aim for 150 words per reply; agent messages are limited to 2400 characters. State only new evidence, disagreements, or the next decision. Do not repeat a peer's proposal or announce that you will reply later.
One agent contribution per turn, including a decision. After posting, wait for a different speaker (another agent or the human) before posting again. Do not send an acknowledgement followed by a proposal or decision. Combine them into one contribution. Rejoining or relaying user_direction does not bypass this rule.
Once a solid plan is set, one agent records it with collab_record_decision. Each agent, including its author, must then read the latest plan and all messages and call collab_agree_plan for that revision. Recording a plan does not count as agreement. Agreement is a control action, allowed directly after recording or replying, but does not unlock another discussion turn. Do not post a separate agreement announcement. All present agents must agree, with at least two distinct agent names, before the room closes automatically with Consensus reached. New discussion, a revised plan, or reopening clears agreements. Keep waiting until the room closes or the user stops you. Consensus is agreement on a plan, not permission to implement.
An @mention requests a reply from that participant. Read coordination.requests and wait if another agent was asked. Requests are queued, not proof that an idle VSCode chat was woken.
Before investigating, use collab_set_activity(researching, reason). It reserves the room for 120 seconds, renewable up to 300 seconds per call, without consuming your reply. Other agents may read but must hold proposals and decisions. Human steering stays open. Do not post a separate "please hold" message.
Before answering from research, read all new messages and pass next_after_id as read_after_id with your reply or decision. The answer releases the hold and resolves your pending mentions. Use activity=idle to release without answering; the human can also clear a hold or unanswered mention. Renew before expiry if more time is needed.
If a ping needs no substantive answer, use collab_decline_request with a short reason after reading it. Activity changes and coordination events are not invitations to reply.
Check new user_direction messages before continuing the plan; the user can post directly from the group chat as You. Address their questions and constraints in the room so every participant can follow.
Relay user instructions that affect the shared plan as kind=user_direction, clearly identifying them as the user's words or a paraphrase. Never invent user approval.
Each join returns a participant_id for this chat; retain it and identify yourself honestly. These IDs prevent accidental mixups, not malicious impersonation by trusted local clients.
Rejoining under the same name reuses its room identity. Use distinct names for distinct agents. Pass participant_id on reads and waits to renew your presence. Monitoring expires after 90 seconds without a room call; a research hold keeps you present until its lease ends, so renew it during longer research. Before ending your chat turn or stopping participation, call collab_leave_discussion. Leaving removes your presence immediately without deleting your messages or changing the discussion turn.
After reading a page, retain next_after_id. If has_more is true, read the next page before replying. Never use your posted message ID as the read cursor: other messages may have arrived before it.
Reads are compact by default: the brief arrives on the initial read only; the live participant list arrives on every read; decision text arrives initially and with a new decision event. Retain earlier values. Older decision bodies are revision references. Use compact=false for full historical text or refreshed participant metadata. Write results acknowledge IDs without echoing your text.
Use collab_wait_for_message with that cursor between responses. A timeout is not a message: do not post filler or respond repeatedly to your own messages.
An empty wait ends only that tool call, not your participation. Keep calling collab_wait_for_message while the discussion is active, including while another participant researches. There is no automatic idle-time or reply-count cutoff. Stop when the requested discussion is complete, the room is paused or closed, or the user asks you to stop or sets a deadline that has arrived. Do not end your chat turn merely because repeated waits return no messages. Keep individual waits short so user steering stays responsive.
Pause or close the room when asked; use collab_agree_plan for consensus completion rather than closing the room unilaterally. All participants must stop discussion work when its status is paused or closed. Resume only on user direction.
record_decision saves an agent proposal and unresolved disagreements; it does not confer user approval or permission to implement.
MCP does not automatically wake a chat after its turn ends. The user must start or resume participation in each chat.
Keep training, generation, and source edits outside this discussion unless separately authorized.`;

type Phase = 'positions' | 'discussion';
type Room = { id: string; brief: string; status: 'active' | 'paused' | 'closed'; revision: number; created_at: string; phase: Phase; base_commit: string | null; max_rounds: number };
type Message = { id: number; room: string; participant_id: string; author: string; kind: string; body: string; reply_to: number | null; request_id: string; created_at: string };
export type OpenItem = { issue: string; owner: string };
export type OutcomeStatus = 'shipped' | 'partial' | 'abandoned' | 'superseded';
type Outcome = { room: string; status: OutcomeStatus; note: string; commit_ref: string | null; recorded_by: string; message_id: number; created_at: string };
type Decision = { room: string; revision: number; message_id: number; plan: string; disagreements: string; open_items: OpenItem[] };
type CreateOptions = { blind_positions?: boolean; max_rounds?: number; base_commit?: string | null };
export const MAX_AGENT_REPLY_CHARS = 2400;
// A none-clause at the start, ended by punctuation, counts as no disagreement;
// "None of the crop questions are settled" does not.
const NO_DISAGREEMENT = /^(|none( recorded| outstanding| open| remaining)?|n\/a|no (open )?disagreements)\s*([.!;:]|$)/i;

function compactPage(page: ReturnType<DiscussionStore['read']>, after: number) {
  const decision = page.decision;
  const includeDecision = after === 0 || page.messages.some(m => m.id === decision?.message_id);
  const includeOutcome = after === 0 || page.messages.some(m => m.kind === 'outcome');
  return {
    discussion: { id: page.discussion.id, status: page.discussion.status, revision: page.discussion.revision, phase: page.discussion.phase,
      ...(after === 0 ? { brief: page.discussion.brief, base_commit: page.discussion.base_commit, max_rounds: page.discussion.max_rounds } : {}) },
    ...(page.discussion.phase === 'positions' || after === 0 ? { positions: page.positions } : {}),
    coordination: page.coordination,
    consensus: page.consensus,
    messages: page.messages.map(({ id, author, kind, body, reply_to, mentions }) => {
      if (kind === 'decision') {
        try { body = JSON.stringify({ revision: JSON.parse(body).expected_revision + 1, superseded: id !== decision?.message_id }); }
        catch { /* Preserve unrecognised historical events. */ }
      }
      return { id, author, kind, body, ...(reply_to !== null ? { reply_to } : {}), ...(mentions.length ? { mentions } : {}) };
    }),
    has_more: page.has_more, next_after_id: page.next_after_id,
    participants: page.participants,
    ...(includeDecision ? { decision } : {}),
    ...(includeOutcome ? { outcome: page.outcome } : {}),
  };
}

export class DiscussionStore {
  private db: Database.Database;
  private coordination: DiscussionCoordination;
  private consensus: DiscussionConsensus;
  private presence: DiscussionPresence;
  private positions: DiscussionPositions;
  private readonly: boolean;

  constructor(dbPath: string, options: { readonly?: boolean } = {}) {
    if (!options.readonly) mkdirSync(dirname(resolve(dbPath)), { recursive: true });
    this.db = new Database(dbPath, { timeout: 5000, readonly: options.readonly ?? false, fileMustExist: options.readonly ?? false });
    this.coordination = new DiscussionCoordination(this.db);
    this.consensus = new DiscussionConsensus(this.db);
    this.presence = new DiscussionPresence(this.db, randomUUID());
    this.positions = new DiscussionPositions(this.db);
    this.readonly = options.readonly ?? false;
    if (options.readonly) return;
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('foreign_keys = ON');
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS discussions (
        id TEXT PRIMARY KEY, brief TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','paused','closed')),
        revision INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS participants (
        id TEXT PRIMARY KEY, room TEXT NOT NULL REFERENCES discussions(id),
        name TEXT NOT NULL, joined_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, room TEXT NOT NULL REFERENCES discussions(id),
        participant_id TEXT NOT NULL REFERENCES participants(id), author TEXT NOT NULL,
        kind TEXT NOT NULL, body TEXT NOT NULL, reply_to INTEGER REFERENCES messages(id),
        request_id TEXT NOT NULL, created_at TEXT NOT NULL,
        UNIQUE(participant_id, request_id)
      );
      CREATE INDEX IF NOT EXISTS messages_room_id ON messages(room, id);
      CREATE TABLE IF NOT EXISTS decisions (
        room TEXT NOT NULL REFERENCES discussions(id), revision INTEGER NOT NULL,
        message_id INTEGER NOT NULL REFERENCES messages(id), plan TEXT NOT NULL,
        disagreements TEXT NOT NULL, PRIMARY KEY(room, revision)
      );
    `);
    this.db.exec(COORDINATION_SCHEMA);
    this.db.exec(CONSENSUS_SCHEMA);
    this.db.exec(PRESENCE_SCHEMA);
    this.db.exec(POSITIONS_SCHEMA);
    // Side tables, not new columns: an MCP process still running the previous
    // code inserts into participants and decisions positionally and must keep working.
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS discussion_roles (
        participant_id TEXT PRIMARY KEY REFERENCES participants(id), role TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS discussion_reconcilers (
        participant_id TEXT PRIMARY KEY REFERENCES participants(id)
      );
      CREATE TABLE IF NOT EXISTS discussion_open_items (
        room TEXT NOT NULL REFERENCES discussions(id), revision INTEGER NOT NULL,
        items TEXT NOT NULL DEFAULT '[]', PRIMARY KEY(room, revision)
      );
      CREATE TABLE IF NOT EXISTS discussion_outcomes (
        room TEXT PRIMARY KEY REFERENCES discussions(id),
        status TEXT NOT NULL CHECK(status IN ('shipped','partial','abandoned','superseded')),
        note TEXT NOT NULL, commit_ref TEXT, recorded_by TEXT NOT NULL,
        message_id INTEGER NOT NULL REFERENCES messages(id), created_at TEXT NOT NULL
      );
    `);
    this.migrate();
  }

  // Additive columns for databases created before sealed positions. Serialised
  // across processes; a concurrent writer that got there first is not an error.
  private migrate() {
    const columns = [
      ['discussions', 'phase', "TEXT NOT NULL DEFAULT 'discussion'"],
      ['discussions', 'base_commit', 'TEXT'],
      ['discussions', 'max_rounds', `INTEGER NOT NULL DEFAULT ${DEFAULT_MAX_ROUNDS}`],
      ['discussion_positions', 'evidence', "TEXT NOT NULL DEFAULT '[]'"],
    ] as const;
    this.db.transaction(() => {
      for (const [table, column, ddl] of columns) {
        const existing = this.db.prepare(`PRAGMA table_info(${table})`).all() as { name: string }[];
        if (existing.some(c => c.name === column)) continue;
        try { this.db.exec(`ALTER TABLE ${table} ADD COLUMN ${column} ${ddl}`); }
        catch (error) { if (!/duplicate column/i.test(String(error))) throw error; }
      }
    }).immediate();
  }

  close() {
    if (!this.readonly) this.db.transaction(() => this.presence.disconnect()).immediate();
    this.db.close();
  }
  private normaliseRoom(row: Partial<Room> & { id: string }): Room {
    return { ...row, phase: row.phase ?? 'discussion', base_commit: row.base_commit ?? null, max_rounds: row.max_rounds ?? DEFAULT_MAX_ROUNDS } as Room;
  }
  room(id: string): Room {
    const room = this.db.prepare('SELECT * FROM discussions WHERE id = ?').get(id) as Room | undefined;
    if (!room) throw new Error(`Unknown discussion: ${id}. Join it first.`);
    return this.normaliseRoom(room);
  }
  // Reconcilers are outside the pair: no position, no vote, and the pen on plans.
  private reconcilerIds(): Set<string> {
    if (!this.db.prepare("SELECT 1 FROM sqlite_master WHERE name = 'discussion_reconcilers'").get()) return new Set();
    return new Set((this.db.prepare('SELECT participant_id FROM discussion_reconcilers').all() as { participant_id: string }[]).map(r => r.participant_id));
  }
  isReconciler(participant: string) {
    return this.reconcilerIds().has(participant);
  }
  participantName(room: string, id: string): string | null {
    return (this.db.prepare('SELECT name FROM participants WHERE room = ? AND id = ?').get(room, id) as { name: string } | undefined)?.name ?? null;
  }
  private presentReconciler(room: string) {
    const ids = this.reconcilerIds();
    return this.presence.list(room).find(p => ids.has(p.id)) ?? null;
  }
  // The agents expected to submit positions and vote: present, not the human, not a reconciler.
  private agentRoster(room: string) {
    const ids = this.reconcilerIds();
    return this.presence.list(room).filter(p => p.name !== 'You' && !ids.has(p.id));
  }
  private roleOf(participant: string): string | null {
    if (!this.db.prepare("SELECT 1 FROM sqlite_master WHERE name = 'discussion_roles'").get()) return null;
    return (this.db.prepare('SELECT role FROM discussion_roles WHERE participant_id = ?').get(participant) as { role: string } | undefined)?.role ?? null;
  }
  latestOutcome(room: string): Outcome | null {
    if (!this.db.prepare("SELECT 1 FROM sqlite_master WHERE name = 'discussion_outcomes'").get()) return null;
    return (this.db.prepare('SELECT * FROM discussion_outcomes WHERE room = ?').get(room) as Outcome | undefined) ?? null;
  }
  // Auto-reveal: every present agent has submitted and at least two positions
  // exist. Re-checked when the roster shrinks, since a departed agent must not
  // keep the room sealed. Runs inside the caller's write transaction.
  private autoReveal(room: string, participant: string) {
    if (this.room(room).phase !== 'positions') return false;
    const roster = this.agentRoster(room);
    const ids = this.reconcilerIds();
    const submitted = new Set(this.positions.list(room).filter(p => !ids.has(p.participant_id)).map(p => p.agent));
    const revealed = submitted.size >= 2 && roster.every(p => submitted.has(p.name.trim().toLowerCase()));
    if (revealed) this.reveal(room, participant, randomUUID(), 'every present agent has submitted');
    return revealed;
  }
  private createRoom(room: string, brief: string, options: CreateOptions) {
    const rounds = options.max_rounds ?? DEFAULT_MAX_ROUNDS;
    if (!Number.isInteger(rounds) || rounds < 1 || rounds > 6) throw new Error('max_rounds must be a whole number from 1 to 6.');
    this.db.prepare('INSERT INTO discussions (id, brief, created_at, phase, base_commit, max_rounds) VALUES (?, ?, ?, ?, ?, ?)')
      .run(room, brief, new Date().toISOString(), options.blind_positions ? 'positions' : 'discussion', options.base_commit ?? null, rounds);
  }
  private participant(room: string, id: string) {
    const participant = this.db.prepare('SELECT name FROM participants WHERE room = ? AND id = ?').get(room, id) as { name: string } | undefined;
    if (!participant) throw new Error('Unknown participant for this discussion. Use the participant_id returned by join.');
    return participant;
  }
  private active(room: string) {
    const current = this.room(room);
    if (current.status !== 'active') throw new Error(`Discussion is ${current.status}. Resume only on user direction.`);
    return current;
  }
  list(limit: number) {
    return (this.db.prepare('SELECT * FROM discussions ORDER BY created_at DESC, id LIMIT ?').all(limit) as Room[]).map(r => {
      const room = this.normaliseRoom(r);
      const outcome = this.latestOutcome(room.id);
      return { ...room, outcome: outcome?.status ?? null, needs_outcome: room.status === 'closed' && room.revision > 0 && !outcome };
    });
  }
  join(room: string, name: string, brief?: string, options: CreateOptions & { role?: string; reconciler?: boolean } = {}) {
    if (name.trim().toLowerCase() === 'you') throw new Error('You is reserved for the human discussion viewer. Use your honest agent name.');
    // git runs before the write lock is taken; only a creating join needs it.
    const create = brief && options.base_commit === undefined ? { ...options, base_commit: detectBaseCommit() } : options;
    return this.db.transaction(() => {
      const existing = this.db.prepare('SELECT id FROM discussions WHERE id = ?').get(room);
      if (!existing) {
        if (!brief) throw new Error('A brief is required to create a discussion.');
        this.createRoom(room, brief, create);
      }
      const previous = this.db.prepare('SELECT id FROM participants WHERE room = ? AND lower(trim(name)) = ? ORDER BY joined_at DESC, rowid DESC LIMIT 1')
        .get(room, name.trim().toLowerCase()) as { id: string } | undefined;
      const id = previous?.id ?? randomUUID();
      if (!previous) this.db.prepare('INSERT INTO participants (id, room, name, joined_at) VALUES (?, ?, ?, ?)').run(id, room, name.trim(), new Date().toISOString());
      const role = options.role?.trim() ?? '';
      if (role) this.db.prepare('INSERT OR REPLACE INTO discussion_roles (participant_id, role) VALUES (?, ?)').run(id, role);
      if (options.reconciler === true) {
        if (this.positions.list(room).some(p => p.participant_id === id)) {
          throw new Error('You already submitted a position in this room, so you cannot become its reconciler. Rejoin as a pair agent, or reconcile a fresh room.');
        }
        const other = this.presentReconciler(room);
        if (other && other.id !== id) throw new Error(`${other.name} is already the reconciler in this room. Join without reconciler=true.`);
        this.db.prepare('INSERT OR IGNORE INTO discussion_reconcilers VALUES (?)').run(id);
      } else if (options.reconciler === false) {
        this.db.prepare('DELETE FROM discussion_reconcilers WHERE participant_id = ?').run(id);
      }
      this.presence.touch(room, id);
      const discussion = this.room(room);
      const reconciler = this.isReconciler(id);
      const sealed = discussion.phase === 'positions';
      return {
        discussion, participant_id: id, role: this.roleOf(id), reconciler, protocol: DISCUSSION_PROTOCOL,
        ...(sealed ? { positions: this.positions.snapshot(room, 'positions', this.agentRoster(room)) } : {}),
        outcome: this.latestOutcome(room),
        next_step: reconciler
          ? (sealed ? 'You are the reconciler: wait for the reveal, then read both positions and both critiques before drafting the merged plan. You submit no position and cast no vote.' : 'You are the reconciler: read every position and critique, then record the merged plan with open_items for whatever the evidence does not settle.')
          : sealed
            ? 'Sealed positions phase: read from after_id=0 for the brief and user direction, then submit your independent position with collab_submit_position. Do not post replies until positions are revealed.'
            : 'Read from after_id=0 before replying. Joining never overwrites an existing brief or resumes a room.',
      };
    }).immediate();
  }
  createViewerDiscussion(room: string, brief: string, participantId: string, requestId: string, options: CreateOptions = {}) {
    const create = options.base_commit === undefined ? { ...options, base_commit: detectBaseCommit() } : options;
    return this.db.transaction(() => {
      const existing = this.db.prepare('SELECT id FROM discussions WHERE id = ?').get(room);
      if (existing) {
        const previous = this.previous(participantId, requestId);
        if (!previous || previous.room !== room || previous.kind !== 'user_direction' || previous.body !== brief || this.room(room).brief !== brief) {
          throw new Error(`Discussion ${room} already exists. Select it above or choose another name.`);
        }
        this.joinViewer(room, participantId);
        return { discussion: this.room(room), participant_id: participantId, message: previous };
      }
      this.createRoom(room, brief, create);
      this.joinViewer(room, participantId);
      const message = this.insert(room, participantId, requestId, 'user_direction', brief);
      return { discussion: this.room(room), participant_id: participantId, message };
    }).immediate();
  }
  joinViewer(room: string, participantId: string) {
    return this.db.transaction(() => {
      this.room(room);
      const existing = this.db.prepare('SELECT room, name FROM participants WHERE id = ?').get(participantId) as { room: string; name: string } | undefined;
      if (existing) {
        if (existing.room !== room || existing.name !== 'You') throw new Error('Viewer identity belongs to another participant. Reload the page with a fresh viewer identity.');
      } else {
        this.db.prepare('INSERT INTO participants (id, room, name, joined_at) VALUES (?, ?, ?, ?)').run(participantId, room, 'You', new Date().toISOString());
      }
      return participantId;
    }).immediate();
  }
  monitor(room: string, participant: string) {
    this.db.transaction(() => { this.presence.touch(room, participant); this.autoReveal(room, participant); }).immediate();
  }
  leave(room: string, participant: string) {
    return this.db.transaction(() => {
      this.participant(room, participant);
      this.presence.leave(room, participant);
      this.coordination.answered(room, participant);
      this.autoReveal(room, participant);
      return { participant_id: participant, present: false };
    }).immediate();
  }
  read(room: string, after: number, limit: number) {
    return this.db.transaction(() => {
      const discussion = this.room(room);
      const rows = this.db.prepare('SELECT * FROM messages WHERE room = ? AND id > ? ORDER BY id LIMIT ?').all(room, after, limit + 1) as Message[];
      const messages = rows.slice(0, limit).map(m => ({ ...m, mentions: this.coordination.mentions(m.id) }));
      const reconcilers = this.reconcilerIds();
      const participants = this.presence.list(room).map(p => ({ ...p, role: this.roleOf(p.id), reconciler: reconcilers.has(p.id), handle: mentionHandle(p.name) }));
      return {
        discussion, messages, has_more: rows.length > limit,
        next_after_id: messages.at(-1)?.id ?? after,
        participants,
        positions: this.positions.snapshot(room, discussion.phase, participants.filter(p => p.name !== 'You' && !p.reconciler)),
        decision: this.latestDecision(room),
        outcome: this.latestOutcome(room),
        coordination: this.coordination.snapshot(room),
        consensus: this.consensus.snapshot(room, discussion.revision, reconcilers),
      };
    })();
  }
  latestDecision(room: string): Decision | null {
    const row = this.db.prepare('SELECT * FROM decisions WHERE room = ? ORDER BY revision DESC LIMIT 1').get(room) as Omit<Decision, 'open_items'> | undefined;
    if (!row) return null;
    let items: OpenItem[] = [];
    if (this.db.prepare("SELECT 1 FROM sqlite_master WHERE name = 'discussion_open_items'").get()) {
      const stored = this.db.prepare('SELECT items FROM discussion_open_items WHERE room = ? AND revision = ?').get(room, row.revision) as { items: string } | undefined;
      try { items = JSON.parse(stored?.items ?? '[]'); } catch { /* Unreadable value counts as no structured items. */ }
    }
    return { ...row, open_items: Array.isArray(items) ? items : [] };
  }
  exportTranscript(room: string) {
    // One read transaction freezes the whole export while other processes post.
    // Read messages, not discussion_positions: unrevealed positions stay sealed.
    return this.db.transaction(() => ({
      discussion: this.room(room),
      exported_at: new Date().toISOString(),
      participants: this.db.prepare(`SELECT p.name, r.role FROM participants p
        LEFT JOIN discussion_roles r ON r.participant_id = p.id
        WHERE p.room = ? ORDER BY p.joined_at, p.rowid`).all(room) as { name: string; role: string | null }[],
      messages: this.db.prepare('SELECT * FROM messages WHERE room = ? ORDER BY id').all(room) as Message[],
    }))();
  }
  exportPlan(room: string, options: { transcript?: boolean } = {}) {
    return this.db.transaction(() => {
      const discussion = this.room(room);
      const decision = this.latestDecision(room);
      if (!decision) throw new Error('No proposed plan has been recorded for this discussion.');
      // Older clients sometimes sent literal backslash-n separators throughout.
      const prose = (value: string) => value.includes('\n') ? value : value.replaceAll('\\n', '\n');
      const items = decision.open_items.length
        ? decision.open_items.map((item, i) => `${i + 1}. ${item.issue.replaceAll('\n', ' ')} (owner: ${item.owner})`).join('\n')
        : 'None. Consensus was possible for this revision.';
      const outcome = this.latestOutcome(room);
      const outcomeText = outcome
        ? `${outcome.status}${outcome.commit_ref ? ` (${outcome.commit_ref})` : ''}, recorded by ${outcome.recorded_by} ${outcome.created_at}\n\n${prose(outcome.note)}`
        : discussion.status === 'closed' ? 'Not recorded yet. Record it with collab_record_outcome or from the viewer when the plan ships or is dropped.' : 'Not applicable until the room closes.';
      let markdown = `# ${room}: proposed plan\n\nRevision: ${decision.revision} of at most ${discussion.max_rounds} before the user rules\n\nBase commit: ${discussion.base_commit ?? 'not recorded'}\n\nRoom status: ${discussion.status}\n\nThis is an agent proposal, not user approval or permission to implement.\n\n## Plan\n\n${prose(decision.plan)}\n\n## Open items\n\n${items}\n\n## Open disagreements (free text)\n\n${prose(decision.disagreements) || 'None recorded.'}\n\n## Outcome\n\n${outcomeText}\n`;
      if (options.transcript) {
        const rows = this.db.prepare('SELECT * FROM messages WHERE room = ? ORDER BY id').all(room) as Message[];
        const render = (m: Message) => {
          if (m.kind === 'status' || m.kind === 'decision' || m.kind === 'agreement' || m.kind === 'outcome') {
            try {
              const value = JSON.parse(m.body);
              if (m.kind === 'outcome') return `Outcome ${value.status}${value.commit_ref ? ` (${value.commit_ref})` : ''}: ${value.note}`;
              if (m.kind === 'agreement') return `Agreed to plan revision ${value.revision}.`;
              if (m.kind === 'status') return `${value.status}: ${value.reason}`;
              return `Recorded plan revision ${value.expected_revision + 1} with ${Array.isArray(value.open_items) ? value.open_items.length : 0} open item(s).`;
            } catch { /* Keep unrecognised historical events verbatim. */ }
          }
          return prose(m.body);
        };
        markdown += `\n## Transcript\n\nBrief: ${prose(discussion.brief)}\n\n${rows.map(m => `### #${m.id} ${m.author} (${m.kind.replaceAll('_', ' ')}) ${m.created_at}\n\n${render(m)}\n`).join('\n')}`;
      }
      return { filename: `${room}-r${decision.revision}.md`, markdown, revision: decision.revision };
    })();
  }
  // Nothing but the human's steering may enter the transcript while positions are sealed.
  private requireDiscussionPhase(room: string, participant: string) {
    if (this.participant(room, participant).name === 'You') return;
    if (this.room(room).phase === 'positions') {
      throw new Error('Sealed positions phase: submit your independent position with collab_submit_position. Replies, decisions and agreement open when every present agent has submitted or the user reveals the positions.');
    }
  }
  private checkTurn(room: string, participant: string, body?: string, readAfter?: number) {
    const { name } = this.participant(room, participant);
    if (name === 'You') return; // Human messages can always steer an active room.
    this.coordination.beforeContribution(room, participant, readAfter);
    if (body !== undefined && body.length > MAX_AGENT_REPLY_CHARS) {
      throw new Error(`Keep agent replies within ${MAX_AGENT_REPLY_CHARS} characters. Combine only new evidence and your proposed next step.`);
    }
    // Agent control events and revealed positions must neither consume nor unlock a discussion turn.
    const last = this.db.prepare("SELECT author FROM messages WHERE room = ? AND (kind NOT IN ('status', 'coordination', 'agreement', 'position', 'outcome') OR author = 'You') ORDER BY id DESC LIMIT 1").get(room) as { author: string } | undefined;
    if (last?.author.trim().toLowerCase() === name.trim().toLowerCase()) {
      throw new Error('Wait for another agent or the human to reply before posting again. Do not retry, rejoin, or post user_direction to bypass the turn limit.');
    }
  }
  private previous(participant: string, request: string) {
    return this.db.prepare('SELECT * FROM messages WHERE participant_id = ? AND request_id = ?').get(participant, request) as Message | undefined;
  }
  private insert(room: string, participant: string, request: string, kind: string, body: string, replyTo?: number) {
    const { name } = this.participant(room, participant);
    if (replyTo !== undefined && !this.db.prepare('SELECT id FROM messages WHERE room = ? AND id = ?').get(room, replyTo)) {
      throw new Error('reply_to must refer to a message in this discussion.');
    }
    const result = this.db.prepare('INSERT INTO messages (room, participant_id, author, kind, body, reply_to, request_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)')
      .run(room, participant, name, kind, body, replyTo ?? null, request, new Date().toISOString());
    return this.db.prepare('SELECT * FROM messages WHERE id = ?').get(result.lastInsertRowid) as Message;
  }
  post(room: string, participant: string, request: string, kind: string, body: string, replyTo?: number, readAfter?: number) {
    return this.db.transaction(() => {
      this.participant(room, participant);
      const previous = this.previous(participant, request);
      if (previous) {
        if (previous.kind !== kind || previous.body !== body || previous.reply_to !== (replyTo ?? null)) throw new Error('request_id was already used for different content.');
        return previous;
      }
      this.active(room);
      if (replyTo !== undefined && !this.db.prepare('SELECT id FROM messages WHERE room = ? AND id = ?').get(room, replyTo)) {
        throw new Error('reply_to must refer to a message in this discussion.');
      }
      this.requireDiscussionPhase(room, participant);
      this.checkTurn(room, participant, body, readAfter);
      const message = this.insert(room, participant, request, kind, body, replyTo);
      this.consensus.clear(room);
      this.coordination.answered(room, participant);
      this.coordination.enqueue(room, participant, message.id, body);
      this.presence.touch(room, participant);
      return message;
    }).immediate();
  }
  status(room: string, participant: string, request: string, status: Room['status'], reason: string) {
    return this.db.transaction(() => {
      this.participant(room, participant);
      this.requireDiscussionPhase(room, participant); // a status reason is free text in the transcript
      const body = JSON.stringify({ status, reason });
      const previous = this.previous(participant, request);
      if (previous) {
        if (previous.kind !== 'status' || previous.body !== body) throw new Error('request_id was already used for different content.');
        return { event: previous, discussion: this.room(room) };
      }
      const previousStatus = this.room(room).status;
      this.db.prepare('UPDATE discussions SET status = ? WHERE id = ?').run(status, room);
      if (status === 'active' && previousStatus !== 'active') this.consensus.clear(room);
      if (status !== 'active') this.coordination.release(room, participant, true);
      if (status !== 'active') this.presence.clear(room);
      if (status === 'closed') this.coordination.clearRequests(room);
      return { event: this.insert(room, participant, request, 'status', body), discussion: this.room(room) };
    }).immediate();
  }
  // Structured open items, each owned by You or a joined agent. Legacy free-text
  // disagreements become one item owned by You so they cannot slip past the gate.
  private openItems(room: string, disagreements: string, items: OpenItem[]) {
    const names = new Map((this.db.prepare('SELECT name FROM participants WHERE room = ?').all(room) as { name: string }[])
      .map(p => [p.name.trim().toLowerCase(), p.name.trim()]));
    const resolved = items.map(item => {
      const issue = item.issue.trim();
      const owner = item.owner.trim();
      if (!issue || issue.length > 2000) throw new Error('Each open item needs an issue of 1 to 2000 characters.');
      if (owner.toLowerCase() === 'you') return { issue, owner: 'You' };
      const known = names.get(owner.toLowerCase());
      if (!known || known === 'You') throw new Error(`Open item owner "${owner}" is not You or a joined agent name (${[...names.values()].filter(n => n !== 'You').join(', ') || 'none yet'}).`);
      return { issue, owner: known };
    });
    if (!resolved.length && !NO_DISAGREEMENT.test(disagreements.trim())) resolved.push({ issue: disagreements.trim(), owner: 'You' });
    return resolved;
  }
  decide(room: string, participant: string, request: string, expected: number, plan: string, disagreements: string, readAfter?: number, openItems: OpenItem[] = []) {
    return this.db.transaction(() => {
      this.participant(room, participant);
      const items = this.openItems(room, disagreements, openItems);
      const body = JSON.stringify({ plan, disagreements, open_items: items, expected_revision: expected });
      const previous = this.previous(participant, request);
      if (previous) {
        if (previous.kind !== 'decision' || previous.body !== body) throw new Error('request_id was already used for different content.');
        const stored = this.db.prepare('SELECT * FROM decisions WHERE message_id = ?').get(previous.id) as Omit<Decision, 'open_items'>;
        return { ...stored, open_items: items, paused: this.room(room).status === 'paused' };
      }
      const current = this.active(room);
      if (current.revision !== expected) throw new Error(`Decision changed: expected revision ${expected}, current ${current.revision}. Read it before revising.`);
      this.requireDiscussionPhase(room, participant);
      const reconciler = this.presentReconciler(room);
      if (reconciler && !this.isReconciler(participant)) {
        throw new Error(`${reconciler.name} is the reconciler and records plans in this room. Post your critique or agreement instead.`);
      }
      this.checkTurn(room, participant, undefined, readAfter);
      const message = this.insert(room, participant, request, 'decision', body);
      const revision = expected + 1;
      this.db.prepare('INSERT INTO decisions (room, revision, message_id, plan, disagreements) VALUES (?, ?, ?, ?, ?)')
        .run(room, revision, message.id, plan, disagreements);
      this.db.prepare('INSERT OR REPLACE INTO discussion_open_items VALUES (?, ?, ?)').run(room, revision, JSON.stringify(items));
      this.db.prepare('UPDATE discussions SET revision = ? WHERE id = ?').run(revision, room);
      this.consensus.clear(room);
      this.coordination.answered(room, participant);
      this.presence.touch(room, participant);
      // Bounded closure: past the round cap, open items go to the user, not another pass.
      let paused = false;
      if (items.length && revision >= current.max_rounds) {
        this.db.prepare("UPDATE discussions SET status = 'paused' WHERE id = ?").run(room);
        this.coordination.release(room, participant, true);
        this.presence.clear(room);
        const reason = `Round cap: revision ${revision} of ${current.max_rounds} still lists ${items.length} open item(s): ${items.map(i => `${i.issue} (${i.owner})`).join('; ')}. The user decides: resume the room, then post the ruling as a message.`;
        this.insert(room, participant, randomUUID(), 'status', JSON.stringify({ status: 'paused', reason }));
        paused = true;
      }
      return { room, revision, message_id: message.id, plan, disagreements, open_items: items, paused };
    }).immediate();
  }
  submitPosition(room: string, participant: string, request: string, body: string, evidence: string[]) {
    const cited = normaliseEvidence(evidence);
    return this.db.transaction(() => {
      const { name } = this.participant(room, participant);
      if (name === 'You') throw new Error('Positions belong to agents. The human steers with messages.');
      if (this.isReconciler(participant)) throw new Error('Reconcilers submit no position. Wait for the reveal, then draft the merged plan from both positions and critiques.');
      const previous = this.positions.previous(participant, request);
      if (previous) {
        if (previous.body !== body || JSON.stringify(this.positions.evidenceOf(previous)) !== JSON.stringify(cited)) throw new Error('request_id was already used for different content.');
        const current = this.room(room);
        return { ...this.positions.snapshot(room, current.phase, this.agentRoster(room)), submitted_now: true };
      }
      const current = this.active(room);
      if (current.phase !== 'positions') throw new Error('This discussion has no sealed positions phase open. Use collab_post_message.');
      if (body.length > MAX_POSITION_CHARS) throw new Error(`Keep positions within ${MAX_POSITION_CHARS} characters.`);
      const agent = name.trim().toLowerCase();
      if (this.positions.get(room, agent)) throw new Error('You already submitted a position. Wait for the reveal; do not resubmit or post elsewhere.');
      this.positions.submit(room, agent, participant, name, body, request, cited);
      this.coordination.answered(room, participant); // the position is the answer from any research hold
      this.presence.touch(room, participant);
      const revealed = this.autoReveal(room, participant);
      return { ...this.positions.snapshot(room, revealed ? 'discussion' : 'positions', this.agentRoster(room)), submitted_now: true };
    }).immediate();
  }
  // Copies sealed positions into the transcript, oldest first, after one event
  // that marks the phase change. Runs inside the caller's write transaction.
  private reveal(room: string, participant: string, request: string, reason: string) {
    const ids = this.reconcilerIds();
    const rows = this.positions.list(room).filter(p => !ids.has(p.participant_id));
    this.db.prepare("UPDATE discussions SET phase = 'discussion' WHERE id = ?").run(room);
    this.coordination.release(room, participant, true); // no hold survives the phase change
    const event = this.insert(room, participant, request, 'coordination',
      `Positions revealed (${rows.length}: ${rows.map(r => r.name).join(', ')}) because ${reason}. They follow in submission order. Discussion is open; critique before converging.`);
    for (const row of rows) {
      const message = this.insert(room, row.participant_id, `position:${row.request_id}`, 'position', renderPosition(row.body, this.positions.evidenceOf(row)));
      this.positions.markRevealed(room, row.agent, message.id);
    }
    return event;
  }
  agreePlan(room: string, participant: string, request: string, revision: number, readAfter: number) {
    return this.db.transaction(() => {
      const { name } = this.participant(room, participant);
      if (name === 'You') throw new Error('Plan agreements belong to agents. The human can end the discussion.');
      if (this.isReconciler(participant)) throw new Error('Reconcilers do not vote. The pair agrees to the plan you recorded.');
      const body = JSON.stringify({ revision });
      const previous = this.previous(participant, request);
      if (previous) {
        if (previous.room !== room || previous.kind !== 'agreement' || previous.body !== body) throw new Error('request_id was already used for different content.');
        return { id: previous.id, discussion: this.room(room), consensus: this.consensus.snapshot(room, this.room(room).revision, this.reconcilerIds()) };
      }
      const current = this.active(room);
      if (revision < 1 || revision !== current.revision) throw new Error('Read the current recorded plan before agreeing; its revision must match.');
      const decision = this.latestDecision(room);
      if (decision?.open_items.length) {
        throw new Error(`Consensus requires no open items. Revision ${revision} lists ${decision.open_items.length}: ${decision.open_items.map(i => `${i.issue} (${i.owner})`).join('; ')}. Record a revision that resolves each one or carries the user's ruling.`);
      }
      const latest = this.db.prepare('SELECT MAX(id) AS id FROM messages WHERE room = ?').get(room) as { id: number };
      if (readAfter !== latest.id) throw new Error('Read all new room messages before agreeing; pass next_after_id as read_after_id.');
      this.coordination.beforeContribution(room, participant, readAfter);
      const message = this.insert(room, participant, request, 'agreement', body);
      this.consensus.agree(room, revision, name, participant, message.id);
      this.coordination.answered(room, participant);
      this.presence.touch(room, participant);
      const consensus = this.consensus.snapshot(room, revision, this.reconcilerIds());
      if (consensus.agents.length >= 2 && consensus.agents.every(agent => agent.agreed)) {
        this.consensus.complete(room, revision, consensus.agents);
        this.db.prepare("UPDATE discussions SET status = 'closed' WHERE id = ?").run(room);
        this.coordination.release(room, participant, true);
        this.coordination.clearRequests(room);
        this.presence.clear(room);
        this.insert(room, participant, randomUUID(), 'status', JSON.stringify({ status: 'closed', reason: `Consensus reached on plan revision ${revision}.` }));
      }
      return { id: message.id, discussion: this.room(room), consensus: this.consensus.snapshot(room, revision, this.reconcilerIds()) };
    }).immediate();
  }
  // Follow-through: what happened to the plan. Allowed on closed rooms, by the
  // human or any joined agent; a control event that consumes no turn.
  recordOutcome(room: string, participant: string, request: string, status: OutcomeStatus, note: string, commit?: string) {
    const trimmed = note.trim();
    if (!trimmed || trimmed.length > 4000) throw new Error('Give an outcome note of 1 to 4000 characters: what shipped, where, or why not.');
    const ref = commit?.trim() || null;
    if (ref && (ref.length > 100 || /[\r\n<>]/.test(ref))) throw new Error('commit must be a git SHA or a short reference without line breaks or angle brackets.');
    return this.db.transaction(() => {
      const { name } = this.participant(room, participant);
      const body = JSON.stringify({ status, note: trimmed, commit_ref: ref });
      const previous = this.previous(participant, request);
      if (previous) {
        if (previous.kind !== 'outcome' || previous.body !== body) throw new Error('request_id was already used for different content.');
        return { id: previous.id, outcome: this.latestOutcome(room) };
      }
      const current = this.room(room);
      if (current.revision < 1) throw new Error('No plan was recorded in this discussion, so there is nothing to report an outcome for.');
      if (current.status !== 'closed') throw new Error(`Discussion is ${current.status}. Record the outcome after the room closes; until then the plan is still being decided.`);
      const message = this.insert(room, participant, request, 'outcome', body);
      this.db.prepare('INSERT OR REPLACE INTO discussion_outcomes (room, status, note, commit_ref, recorded_by, message_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)')
        .run(room, status, trimmed, ref, name, message.id, new Date().toISOString());
      if (name !== 'You') this.presence.touch(room, participant);
      return { id: message.id, outcome: this.latestOutcome(room) };
    }).immediate();
  }
  activity(room: string, participant: string, activity: 'researching' | 'idle', reason: string, seconds: number) {
    return this.db.transaction(() => {
      if (this.participant(room, participant).name === 'You') throw new Error('Research holds belong to agents.');
      if (activity === 'idle') this.coordination.release(room, participant);
      else {
        this.active(room);
        const current = this.coordination.snapshot(room).research;
        if (!current || current.participant_id !== participant) {
          this.checkTurn(room, participant);
        }
        if (!reason.trim() || reason.length > 240) throw new Error('Give a research reason of 1 to 240 characters.');
        if (!Number.isInteger(seconds) || seconds < 30 || seconds > 300) throw new Error('Research holds must last 30 to 300 seconds.');
        this.coordination.research(room, participant, reason, seconds);
      }
      this.presence.touch(room, participant);
      return this.coordination.snapshot(room);
    }).immediate();
  }
  clearCoordination(room: string, participant: string, request: string, action: 'release_research' | 'clear_requests' | 'reveal_positions') {
    return this.db.transaction(() => {
      if (this.participant(room, participant).name !== 'You') throw new Error('Only the human can clear room coordination.');
      const previous = this.previous(participant, request);
      if (previous) {
        const matches = action === 'reveal_positions' ? previous.body.startsWith('Positions revealed') : previous.body === action;
        if (previous.kind !== 'coordination' || !matches) throw new Error('request_id was already used for different content.');
        return previous;
      }
      if (action === 'reveal_positions') {
        if (this.active(room).phase !== 'positions') throw new Error('Positions are already revealed.');
        if (!this.positions.list(room).length) throw new Error('No positions have been submitted yet.');
        // The human's reveal event carries the request id so a retried click cannot reveal twice.
        return this.reveal(room, participant, request, 'the user opened them');
      }
      if (action === 'release_research') this.coordination.release(room, participant, true);
      else this.coordination.clearRequests(room);
      return this.insert(room, participant, request, 'coordination', action);
    }).immediate();
  }
  declineRequest(room: string, participant: string, request: string, reason: string, readAfter: number) {
    return this.db.transaction(() => {
      this.participant(room, participant);
      const body = `Declined reply request: ${reason}`;
      const previous = this.previous(participant, request);
      if (previous) {
        if (previous.kind !== 'coordination' || previous.body !== body) throw new Error('request_id was already used for different content.');
        return { id: previous.id };
      }
      this.requireDiscussionPhase(room, participant); // the reason is free text in the transcript
      this.coordination.decline(room, participant, readAfter);
      return { id: this.insert(room, participant, request, 'coordination', body).id };
    }).immediate();
  }
  async wait(room: string, after: number, timeoutMs: number, limit: number, signal?: AbortSignal, participant?: string) {
    if (participant) this.monitor(room, participant);
    const deadline = Date.now() + timeoutMs;
    const initialCoordination = JSON.stringify(this.coordination.snapshot(room));
    const initialPresence = JSON.stringify(this.presence.list(room));
    try {
      while (true) {
        signal?.throwIfAborted();
        const page = this.read(room, after, limit);
        if (page.messages.length || page.discussion.status !== 'active' || JSON.stringify(page.coordination) !== initialCoordination || JSON.stringify(this.presence.list(room)) !== initialPresence) return { ...page, timed_out: false };
        const remaining = deadline - Date.now();
        if (remaining <= 0) return { ...page, timed_out: true };
        await delay(Math.min(250, remaining), undefined, { signal });
      }
    } finally {
      if (participant) {
        if (signal?.aborted) this.leave(room, participant);
        else this.monitor(room, participant);
      }
    }
  }
}

export function registerCollaborationTools(server: McpServer, dbPath = process.env.HOTSTEP_COLLAB_DB ?? DEFAULT_COLLAB_DB, options: { wake?: boolean } = {}) {
  const work = registerWorkTools(server, dbPath, options);
  // Lazy opening keeps existing lyric-only clients independent of collaboration storage.
  let store: DiscussionStore | undefined;
  const joined = new Map<string, string>();
  const get = () => store ??= new DiscussionStore(dbPath);
  // Automatic wake (opt-in): poll for events this connection's participant has
  // not seen and push them as channel notifications. See discussion-wake.ts.
  const wake = new WakeTracker(get);
  let wakeTimer: NodeJS.Timeout | undefined;
  const wakePollMs = Math.max(250, Number(process.env.HOTSTEP_COLLAB_WAKE_POLL_MS) || 3000);
  const startWake = () => {
    if (wakeTimer || !(options.wake ?? channelEnabled())) return;
    wakeTimer = setInterval(() => {
      for (const event of wake.poll()) {
        void server.server.notification({
          method: CHANNEL_NOTIFICATION,
          params: { content: event.content, meta: { room: event.room, event: event.event, message_id: String(event.message_id), participant_id: event.participant_id } },
        }).catch(() => { /* A closed transport ends the wake with the connection. */ });
      }
    }, wakePollMs);
    wakeTimer.unref();
  };
  const result = async (action: () => unknown | Promise<unknown>) => {
    try { return { content: [{ type: 'text' as const, text: JSON.stringify(await action()) }] }; }
    catch (error) { return { isError: true, content: [{ type: 'text' as const, text: error instanceof Error ? error.message : String(error) }] }; }
  };
  const room = z.string().min(1).max(100).regex(/^[a-zA-Z0-9][a-zA-Z0-9._-]*$/).describe('Shared discussion name, for example mm3-cache-design');
  const identity = { room, participant_id: z.string().uuid().describe('ID returned by your join call'), request_id: z.string().min(1).max(100).describe('Unique ID for this write. Reuse it only when retrying identical content.') };
  const cursor = { room, participant_id: identity.participant_id.optional().describe('Your joined identity; renews your monitoring presence. Defaults to this connection\'s last join in this room.'), after_id: z.number().int().min(0).default(0).describe('Last message ID actually read, initially 0'), limit: z.number().int().min(1).max(100).default(50), compact: z.boolean().default(true).describe('Omit repeated metadata and old decision bodies; false returns the full transcript format.') };
  const body = z.string().trim().min(1).max(24000);
  const readAfter = z.number().int().min(0).optional().describe('Last next_after_id read; required to answer while holding the research turn.');

  server.tool('collab_list_discussions', 'List shared project discussions. Does not start another agent.', { limit: z.number().int().min(1).max(100).default(30) },
    async ({ limit }) => result(() => get().list(limit)));
  server.tool('collab_join_discussion', 'Join as this chat agent. Creates a room only if missing and a brief is supplied. Returns participation instructions; read and follow them. If the room is in the sealed positions phase, submit your independent position before anything else.',
    { room, name: z.string().trim().min(1).max(100).describe('Honest chat identity, e.g. Codex or Claude'), brief: body.optional(),
      role: z.string().trim().max(200).optional().describe('Your assigned role for this discussion, e.g. "engine/logic lead" or "app/integration lead". Shown to every participant.'),
      blind_positions: z.boolean().default(false).describe('Creation only: start in the sealed positions phase so each agent commits an independent position before seeing the other.'),
      max_rounds: z.number().int().min(1).max(6).default(DEFAULT_MAX_ROUNDS).describe('Creation only: plan revisions allowed with open items before the room pauses for the user.'),
      reconciler: z.boolean().optional().describe('Join as the reconciler outside the pair: no position, no vote, and while present the only participant who records plans.') },
    async ({ room, name, brief, role, blind_positions, max_rounds, reconciler }) => result(() => { const value = get().join(room, name, brief, { role, blind_positions, max_rounds, reconciler }); joined.set(room, value.participant_id); wake.joined(room, value.participant_id); startWake(); return value; }));
  server.tool('collab_submit_position', 'Sealed positions phase only: submit your one independent position (plan and what would change your mind; max 24000 characters) with an evidence list. Hidden from other agents until every present agent has submitted or the user reveals. Does not consume a turn.',
    { ...identity, body: z.string().trim().min(1).max(MAX_POSITION_CHARS),
      evidence: z.array(z.string().trim().min(1).max(MAX_EVIDENCE_CHARS)).min(1).max(MAX_EVIDENCE_ITEMS).describe('Concrete sources this position rests on: file paths with lines, commit SHAs, log paths, listening results, measurements. At least one.') },
    async ({ room, participant_id, request_id, body, evidence }) => result(() => get().submitPosition(room, participant_id, request_id, body, evidence)));
  server.tool('collab_record_outcome', 'After the room closes: record what happened to the plan (shipped, partial, abandoned, superseded) with a note and the commit. Replaces an earlier outcome; consumes no turn. Not user approval.',
    { ...identity, status: z.enum(['shipped', 'partial', 'abandoned', 'superseded']), note: z.string().trim().min(1).max(4000), commit: z.string().trim().max(100).optional().describe('Git SHA or short reference of what shipped') },
    async ({ room, participant_id, request_id, status, note, commit }) => result(() => get().recordOutcome(room, participant_id, request_id, status, note, commit)));
  server.tool('collab_leave_discussion', 'Leave when you stop monitoring or before ending your chat turn. Removes your live presence and research hold, preserves transcript and identity, and does not unlock a discussion turn.',
    { room, participant_id: identity.participant_id },
    async ({ room, participant_id }) => result(() => { const value = get().leave(room, participant_id); if (joined.get(room) === participant_id) joined.delete(room); return value; }));
  server.tool('collab_read_discussion', 'Read ordered messages, status and latest proposed decision. Page until has_more=false before replying. Retain next_after_id.', cursor,
    async ({ room, participant_id, after_id, limit, compact }) => result(() => { const who = participant_id ?? joined.get(room); if (who) get().monitor(room, who); const page = get().read(room, after_id, limit); wake.read(room, page.next_after_id); return compact ? compactPage(page, after_id) : page; }));
  server.tool('collab_post_message', 'Post one reply (max 2400 characters), then wait for another speaker. Relay user steering accurately. No implementation permission.',
    { ...identity, kind: z.enum(['proposal', 'critique', 'question', 'reply', 'user_direction', 'summary']).default('reply'), body, reply_to: z.number().int().positive().optional(), read_after_id: readAfter },
    async ({ room, participant_id, request_id, kind, body, reply_to, read_after_id }) => result(() => { const message = get().post(room, participant_id, request_id, kind, body, reply_to, read_after_id); return { id: message.id, kind: message.kind }; }));
  server.tool('collab_set_activity', 'Reserve the room while researching (no reply consumed), renew the hold, or release it with idle. Human steering remains open. Read new messages before answering.',
    { room, participant_id: identity.participant_id, activity: z.enum(['researching', 'idle']), reason: z.string().trim().max(240).default(''), lease_seconds: z.number().int().min(30).max(300).default(120) },
    async ({ room, participant_id, activity, reason, lease_seconds }) => result(() => get().activity(room, participant_id, activity, reason, lease_seconds)));
  server.tool('collab_decline_request', 'Resolve your pending ping when no substantive reply is appropriate. Read it first. Does not consume or unlock a discussion turn.',
    { ...identity, reason: z.string().trim().min(1).max(240), read_after_id: z.number().int().min(0) },
    async ({ room, participant_id, request_id, reason, read_after_id }) => result(() => get().declineRequest(room, participant_id, request_id, reason, read_after_id)));
  server.tool('collab_wait_for_message', 'Wait for new messages in this active turn; does not wake idle chats. Retain next_after_id and repeat empty waits while active, including during peer research. No automatic idle or reply-count cutoff. Stop on completion, pause/close, or user stop/deadline. Never post filler on timeout.',
    { ...cursor, timeout_ms: z.number().int().min(0).max(25000).default(20000) },
    async ({ room, participant_id, after_id, limit, timeout_ms, compact }, extra) => result(async () => {
      wake.waiting(room, 1);
      try {
        const page = await get().wait(room, after_id, timeout_ms, limit, extra.signal, participant_id ?? joined.get(room));
        wake.read(room, page.next_after_id);
        return compact ? { ...compactPage(page, after_id), timed_out: page.timed_out } : page;
      } finally { wake.waiting(room, -1); }
    }));
  server.tool('collab_set_status', 'Pause/close on user request; use collab_agree_plan for consensus completion. Resume (active) only on user direction. Status changes are visible to all waiting participants.',
    { ...identity, status: z.enum(['active', 'paused', 'closed']), reason: body },
    async ({ room, participant_id, request_id, status, reason }) => result(() => { const value = get().status(room, participant_id, request_id, status, reason); return { id: value.event.id, status: value.discussion.status }; }));
  server.tool('collab_agree_plan', 'Agree to the current recorded plan after reading all messages. Each agent, including the author, must agree. All present agents (at least two) agreeing closes the room and releases waiters. Does not consume or unlock a discussion turn. Not implementation approval.',
    { ...identity, revision: z.number().int().positive(), read_after_id: z.number().int().min(0) },
    async ({ room, participant_id, request_id, revision, read_after_id }) => result(() => get().agreePlan(room, participant_id, request_id, revision, read_after_id)));
  server.tool('collab_record_decision', 'Save the plan as your one contribution this turn; do not post an announcement first. List every unresolved contradiction in open_items with an owner; consensus is refused while any remain, and a revision at the round cap with open items pauses the room for the user. After reading it, each agent including the author must use collab_agree_plan to reach consensus. Not user approval. expected_revision prevents overwrites.',
    { ...identity, expected_revision: z.number().int().min(0), plan: body, disagreements: z.string().max(24000).default('').describe('Legacy free text; non-empty text becomes one open item owned by You.'),
      open_items: z.array(z.object({ issue: z.string().trim().min(1).max(2000), owner: z.string().trim().min(1).max(100).describe('You or a joined agent name') })).max(20).default([]),
      read_after_id: readAfter },
    async ({ room, participant_id, request_id, expected_revision, plan, disagreements, open_items, read_after_id }) => result(() => { const value = get().decide(room, participant_id, request_id, expected_revision, plan, disagreements, read_after_id, open_items); return { revision: value.revision, message_id: value.message_id, open_items: value.open_items, paused: value.paused }; }));
  return { close: () => { work.close(); if (wakeTimer) clearInterval(wakeTimer); wakeTimer = undefined; store?.close(); store = undefined; } };
}
