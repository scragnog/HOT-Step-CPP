<script lang="ts">
	import { app } from '../lib/state.svelte.js';
	import { SSE_RECONNECT_MS, LOG_MAX_LINES } from '../lib/config.js';
	import { ChevronDown, ChevronRight } from '@lucide/svelte';

	let lines = $state<string[]>([]);

	$effect(() => {
		let es: EventSource | null = null;
		let timer = 0;

		function connect() {
			es = new EventSource('logs');
			es.onmessage = (e: MessageEvent) => {
				lines.push(e.data);
				if (lines.length > LOG_MAX_LINES) lines.splice(0, lines.length - LOG_MAX_LINES);
			};
			es.onerror = () => {
				es?.close();
				es = null;
				lines.push('[Client] Server unavailable');
				if (lines.length > LOG_MAX_LINES) lines.splice(0, lines.length - LOG_MAX_LINES);
				timer = setTimeout(connect, SSE_RECONNECT_MS) as unknown as number;
			};
		}

		connect();
		return () => {
			clearTimeout(timer);
			es?.close();
		};
	});
</script>

<div class="card">
	<button class="card-header" onclick={() => (app.logsOpen = !app.logsOpen)}>
		{#if app.logsOpen}
			<ChevronDown size={14} />
		{:else}
			<ChevronRight size={14} />
		{/if}
		<span class="card-label">Server logs</span>
	</button>
	{#if app.logsOpen}
		<pre class="log-body">{lines.join('\n')}</pre>
	{/if}
</div>

<style>
	.card {
		display: flex;
		flex-direction: column;
		border: none;
		border-radius: 4px;
		background: var(--bg-card);
		overflow: hidden;
	}
	.card-header {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		padding: 0.3rem 0.5rem;
		background: var(--bg-input);
		border: none;
		cursor: pointer;
		color: var(--fg);
		font-size: 0.8rem;
		text-align: left;
	}
	.card-header:hover {
		background: var(--bg-btn-hover);
	}
	.card-label {
		font-weight: 600;
	}
	.log-body {
		margin: 0;
		padding: 0.4rem 0.5rem;
		font-family: monospace;
		font-size: 0.7rem;
		line-height: 1.4;
		color: var(--fg-dim);
		background: var(--bg-card);
		white-space: pre;
		overflow-x: auto;
	}
</style>
