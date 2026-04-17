// all ACE-Step examples bundled at build time (Vite eager glob).
// toRequest maps any ACE-Step JSON format to our AceRequest fields.

import type { AceRequest } from './types.js';

const modules = import.meta.glob('../../dice/**/*.json', { eager: true });

const examples: Record<string, any>[] = Object.values(modules).map((m: any) => m.default ?? m);

// map ACE-Step JSON keys to AceRequest fields
function toRequest(ex: Record<string, any>): Partial<AceRequest> {
	const r: Partial<AceRequest> = {};

	// caption (text2music) or description (simple_mode)
	r.caption = String(ex.caption ?? ex.description ?? '');

	// instrumental flag: simple_mode uses a boolean, our backend uses lyrics convention
	if (ex.instrumental === true) {
		r.lyrics = '[Instrumental]';
	} else if (ex.lyrics) {
		r.lyrics = String(ex.lyrics);
	}

	// vocal_language (simple_mode) or language (text2music), skip "unknown"
	const lang = ex.vocal_language ?? ex.language;
	if (lang && lang !== 'unknown') {
		r.vocal_language = String(lang);
	}

	// direct mappings (text2music only, undefined for simple_mode)
	if (ex.bpm != null) r.bpm = Number(ex.bpm);
	if (ex.duration != null) r.duration = Number(ex.duration);
	if (ex.keyscale) r.keyscale = String(ex.keyscale);
	if (ex.timesignature) r.timesignature = String(ex.timesignature);

	return r;
}

// pick a random example and return an AceRequest (caption is always set)
export function rollDice(): AceRequest {
	return toRequest(examples[Math.floor(Math.random() * examples.length)]) as AceRequest;
}
