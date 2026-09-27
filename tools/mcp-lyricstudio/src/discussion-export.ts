import type { DiscussionStore } from './collaboration.js';

type Transcript = ReturnType<DiscussionStore['exportTranscript']>;
type Message = Transcript['messages'][number];

const escapeHtml = (value: string) => value.replace(/[&<>"']/g, char => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
})[char]!);

// Preserve the transcript's Markdown as text, as in the viewer. Only HTTP(S)
// references become links. Never interpret agent-supplied HTML or load images.
function text(value: string) {
  return value.split(/(https?:\/\/[^\s<>"']+)/g).map((part, i) => {
    if (i % 2 === 0) return escapeHtml(part);
    const link = part.replace(/[.,;:!?\])}]+$/, '');
    return `<a href="${escapeHtml(link)}" rel="noreferrer">${escapeHtml(link)}</a>${escapeHtml(part.slice(link.length))}`;
  }).join('');
}

function body(message: Message) {
  try {
    const value = JSON.parse(message.body);
    if (message.kind === 'decision' && typeof value.plan === 'string') {
      const items = Array.isArray(value.open_items)
        ? value.open_items.map((item: { issue: string; owner: string }) => `- ${item.issue} (owner: ${item.owner})`).join('\n')
        : '';
      return `Plan revision ${value.expected_revision + 1}\n\n${value.plan}\n\nOpen items\n${items || 'None recorded.'}\n\nOpen disagreements\n${value.disagreements || 'None recorded.'}`;
    }
    if (message.kind === 'agreement' && typeof value.revision === 'number') return `Agreed to plan revision ${value.revision}.`;
    if (message.kind === 'status' && typeof value.reason === 'string') return `${value.status}: ${value.reason}`;
    if (message.kind === 'outcome' && typeof value.note === 'string') {
      return `Outcome: ${value.status}${value.commit_ref ? ` (${value.commit_ref})` : ''}\n\n${value.note}`;
    }
  } catch { /* Historical or ordinary text is preserved verbatim. */ }
  return message.body;
}

export function renderDiscussionExport(snapshot: Transcript) {
  const { discussion, messages } = snapshot;
  const people = [...new Set(snapshot.participants.map(p => p.name + (p.role ? ` (${p.role})` : '')))];
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapeHtml(discussion.id)}-conversation</title>
  <link rel="stylesheet" href="/transcript-print.css">
  <script src="/transcript-print.js" defer></script>
</head>
<body>
  <nav class="export-controls" aria-label="Export controls">
    <button id="save-pdf" type="button">Save as PDF / Print</button>
    <p>Choose "Save as PDF" in the print dialog. This snapshot contains the full conversation at the time below. Reload for newer messages.</p>
    <a href="/?room=${encodeURIComponent(discussion.id)}">Back to discussion</a>
  </nav>
  <main>
    <header class="document-header">
      <p class="eyebrow">HOT-Step discussions / Full conversation</p>
      <h1>${escapeHtml(discussion.id)}</h1>
      <dl>
        <dt>Exported</dt><dd>${escapeHtml(snapshot.exported_at)} (UTC)</dd>
        <dt>Created</dt><dd>${escapeHtml(discussion.created_at)} (UTC)</dd>
        <dt>Status</dt><dd>${escapeHtml(discussion.status)} / ${escapeHtml(discussion.phase)} / plan revision ${discussion.revision}</dd>
        <dt>Messages</dt><dd>${messages.length}${messages.length ? ` / through #${messages.at(-1)!.id}` : ''}</dd>
        <dt>Participants</dt><dd>${escapeHtml(people.join('; ') || 'None recorded.')}</dd>
        <dt>Base commit</dt><dd>${escapeHtml(discussion.base_commit || 'Not recorded.')}</dd>
      </dl>
      <p>Recorded plans are agent proposals, not permission to implement.</p>
      ${discussion.phase === 'positions' ? '<p>Unrevealed positions are sealed and are not part of the shared conversation yet.</p>' : ''}
    </header>
    <section class="brief"><h2>Brief</h2><div class="prose">${text(discussion.brief)}</div></section>
    <section aria-label="Full conversation">
      <h2>Conversation</h2>
      ${messages.length ? messages.map(message => `<article class="message" id="message-${message.id}">
        <header class="message-heading"><h3>#${message.id} ${escapeHtml(message.author)} <span>${escapeHtml(message.kind.replaceAll('_', ' '))}</span></h3>
        <p><time datetime="${escapeHtml(message.created_at)}">${escapeHtml(message.created_at)} (UTC)</time>${message.reply_to ? ` / In reply to <a href="#message-${message.reply_to}">#${message.reply_to}</a>` : ''}</p></header>
        <div class="prose">${text(body(message))}</div>
      </article>`).join('\n') : '<p>No messages recorded yet.</p>'}
    </section>
  </main>
</body>
</html>`;
}
