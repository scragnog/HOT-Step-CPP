<script lang="ts">
	import { Play, Square, Pencil, Ear, Download, Trash2 } from '@lucide/svelte';
	import { app, setRequest, toast } from '../lib/state.svelte.js';
	import { deleteSong } from '../lib/db.js';
	import { understandSubmit, pollJob, jobResultJson } from '../lib/api.js';
	import { saveJob, clearJob } from '../lib/db.js';
	import type { Song } from '../lib/types.js';
	import Waveform from './Waveform.svelte';

	let { song }: { song: Song } = $props();

	let playing = $state(false);
	let time = $state(0);
	let dur = $state(0);
	let rangeStart = $state(0);
	let rangeEnd = $state(0);

	let isRef = $derived(app.refSongId === song.id);
	let isSrc = $derived(app.srcSongId === song.id);

	function toggleRef() {
		if (isRef) {
			app.refSongId = null;
		} else {
			app.refSongId = song.id ?? null;
		}
	}

	function toggleSrc() {
		if (isSrc) {
			app.srcSongId = null;
			app.srcRangeStart = null;
			app.srcRangeEnd = null;
			rangeStart = 0;
			rangeEnd = 0;
		} else {
			app.srcSongId = song.id ?? null;
		}
	}

	// waveform drag to global state
	$effect(() => {
		if (isSrc && rangeEnd > rangeStart) {
			app.srcRangeStart = rangeStart;
			app.srcRangeEnd = rangeEnd;
		}
	});

	// global state to waveform visual (field input)
	$effect(() => {
		if (isSrc) {
			const rs = app.srcRangeStart;
			const re = app.srcRangeEnd;
			if (rs != null && re != null && re > rs) {
				rangeStart = rs;
				rangeEnd = re;
			} else {
				rangeStart = 0;
				rangeEnd = 0;
			}
		}
	});

	function toggle() {
		playing = !playing;
	}

	function load() {
		app.name = song.name;
		setRequest({ ...song.request });
		app.pendingRequests = [];
		app.pendingIndex = 0;
	}

	let scanning = $state(false);

	// analyze audio: send to /understand, fill form with detected metadata.
	// persists the job under 'lm' key so page reload resumes polling.
	async function scan() {
		scanning = true;
		try {
			const jobId = await understandSubmit(
				song.audio,
				app.request.lm_model as string,
				app.request.synth_model as string
			);
			saveJob('lm', jobId);
			await pollJob(jobId);
			const results = await jobResultJson(jobId);
			clearJob('lm');
			app.name = song.name;
			if (results.length > 0) {
				setRequest(results[0]);
			}
			app.pendingRequests = results;
			app.pendingIndex = 0;
		} catch (e: unknown) {
			toast(e instanceof Error ? e.message : String(e));
		} finally {
			scanning = false;
		}
	}

	function downloadAudio() {
		const url = URL.createObjectURL(song.audio);
		const a = document.createElement('a');
		a.href = url;
		const safe = song.name.replace(/[\\/:*?"<>|\x00-\x1f]/g, '') || 'song';
		const ext = song.format.startsWith('wav') ? '.wav' : '.mp3';
		a.download = `${safe}${ext}`;
		a.click();
		URL.revokeObjectURL(url);
	}

	async function remove() {
		if (song.id == null) return;
		if (app.refSongId === song.id) app.refSongId = null;
		if (app.srcSongId === song.id) app.srcSongId = null;
		await deleteSong(song.id);
		const idx = app.songs.findIndex((s) => s.id === song.id);
		if (idx >= 0) app.songs.splice(idx, 1);
	}

	// MM:SS:XX (hundredths) for current position
	function fmtPos(s: number): string {
		const m = Math.floor(s / 60);
		const sec = Math.floor(s % 60);
		const cs = Math.floor((s * 100) % 100);
		return (
			String(m).padStart(2, '0') +
			':' +
			String(sec).padStart(2, '0') +
			':' +
			String(cs).padStart(2, '0')
		);
	}

	// MM:SS for total duration
	function fmtDur(s: number): string {
		const m = Math.floor(s / 60);
		const sec = Math.floor(s % 60);
		return String(m).padStart(2, '0') + ':' + String(sec).padStart(2, '0');
	}
</script>

<div class="card">
	<div class="card-header">
		<button class="icon-btn" onclick={toggle} title={playing ? 'Stop' : 'Play'}>
			{#if playing}
				<Square size={14} />
			{:else}
				<Play size={14} />
			{/if}
		</button>
		<span class="card-name">{song.name}</span>
		<div class="card-actions">
			<button class="icon-btn" onclick={load} title="Edit prompt"><Pencil size={14} /> Edit</button>
			<button class="icon-btn" onclick={downloadAudio} title="Download track"
				><Download size={14} /> Down</button
			>
			<button class="icon-btn" onclick={remove} title="Delete track"
				><Trash2 size={14} /> Del</button
			>
		</div>
	</div>
	<Waveform
		audio={song.audio}
		bind:playing
		bind:time
		bind:dur
		selectable={isSrc}
		bind:rangeStart
		bind:rangeEnd
	/>
	<div class="card-footer">
		<span class="format-badge">{song.format.toUpperCase()}</span>
		<span class="timecode">{fmtPos(time)} / {fmtDur(dur)}</span>
		<div class="card-actions">
			<button class="icon-btn" onclick={scan} disabled={scanning} title="Analyze audio"
				><Ear size={14} /> Scan</button
			>
			<label class="icon-btn"
				><input
					type="checkbox"
					class="ref-check"
					checked={isSrc}
					onchange={toggleSrc}
					title="Source audio"
				/> Src</label
			>
			<label class="icon-btn"
				><input
					type="checkbox"
					class="ref-check"
					checked={isRef}
					onchange={toggleRef}
					title="Timbre reference"
				/> Ref</label
			>
		</div>
	</div>
</div>

<style>
	.card {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		padding: 0.5rem;
		border: none;
		border-radius: 4px;
		background: var(--bg-card);
	}
	.card-header {
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}
	.card-footer {
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}
	.card-name {
		font-size: 0.8rem;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
		flex: 1;
	}
	.format-badge {
		font-size: 0.6rem;
		font-family: monospace;
		padding: 0.05rem 0.3rem;
		border-radius: 2px;
		background: var(--fg);
		color: var(--bg);
		flex-shrink: 0;
	}
	.timecode {
		font-size: 0.7rem;
		font-family: monospace;
		color: var(--fg);
		white-space: nowrap;
		flex: 1;
	}
	.card-actions {
		display: flex;
		align-items: center;
		gap: 0.2rem;
		flex-shrink: 0;
		font-size: 0.8rem;
	}
	.icon-btn {
		background: none;
		border: none;
		cursor: pointer;
		padding: 0.15rem;
		color: var(--fg);
		display: flex;
		align-items: center;
		gap: 0.2rem;
		font-size: 0.8rem;
	}
	.icon-btn:hover {
		color: var(--focus);
	}
	.ref-check {
		cursor: pointer;
		accent-color: var(--focus);
	}
</style>
