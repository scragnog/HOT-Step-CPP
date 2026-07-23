// index.ts — MCP server for HOT-Step Lyric Studio
//
// Lets the Antigravity agent act as the LLM for lyric generation,
// refinement, and profile building. Connects to hotstep.db directly.

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import * as db from './db.js';
import * as prompts from './prompts.js';

const server = new McpServer({
  name: 'lyricstudio',
  version: '1.0.0',
});

// ── Model-name title suffix ─────────────────────────────────────────────────
// Titles are tagged with the LLM that wrote them: "Song Name - Fable 5".
// The appending happens HERE (not in the LLM prompt) so it is deterministic.

/** Turn a model id like "claude-fable-5" / "claude-opus-4-8" into "Fable 5" / "Opus 4.8".
 *  Friendly names ("Fable 5", "Gemini 3 Pro") pass through unchanged. */
function prettifyModel(raw: string): string {
  let s = raw.trim();
  if (!s) return s;
  if (/\s/.test(s) && !/[-_/]/.test(s)) return s; // already a friendly name
  s = s.replace(/^(us\.)?(anthropic[./])?(claude-)?/i, '');
  s = s.replace(/[-.]?\d{8}$/, '');               // date suffix e.g. -20251001
  s = s.replace(/[-.]?v\d+(:\d+)?$/i, '');        // bedrock-style :0 / v1 suffix
  const parts = s.split(/[-_]/).filter(Boolean);
  const out: string[] = [];
  for (const part of parts) {
    // Join consecutive single-digit segments as a version: opus 4 8 → opus 4.8
    if (/^\d$/.test(part) && out.length && /^\d+(\.\d+)*$/.test(out[out.length - 1])) {
      out[out.length - 1] += `.${part}`;
    } else {
      out.push(/^\d/.test(part) ? part : part.charAt(0).toUpperCase() + part.slice(1));
    }
  }
  return out.join(' ') || raw.trim();
}

/** Append " - <Model>" to a title unless it already carries that suffix. */
function withModelSuffix(title: string, model?: string): string {
  const t = title.trim();
  if (!model) return t;
  const pretty = prettifyModel(model);
  if (!pretty || t.toLowerCase().endsWith(`- ${pretty.toLowerCase()}`)) return t;
  return `${t} - ${pretty}`;
}

const MODEL_PARAM_DESC =
  "Name of the model YOU are running as (e.g. 'Fable 5', 'claude-opus-4-8', 'Gemini 3 Pro'). " +
  'Always pass this — it is appended to the song title ("Song Name - Fable 5") and stored with the generation.';

// ── list_artists ────────────────────────────────────────────────────────────

server.tool(
  'list_artists',
  'List all artists in the Lyric Studio database with their IDs and lyrics set counts',
  {},
  async () => {
    const artists = db.listArtists();
    if (!artists.length) {
      return { content: [{ type: 'text', text: 'No artists found in the database.' }] };
    }
    const lines = ['# Artists\n'];
    for (const a of artists) {
      lines.push(`- **${a.name}** (ID: ${a.id}) — ${a.lyrics_set_count} lyrics set(s)`);
    }
    return { content: [{ type: 'text', text: lines.join('\n') }] };
  }
);

// ── list_profiles ───────────────────────────────────────────────────────────

server.tool(
  'list_profiles',
  'List artist profiles with album info. Use artist_id to filter by artist.',
  { artist_id: z.number().optional().describe('Filter by artist ID') },
  async ({ artist_id }) => {
    const profiles = db.listProfiles(artist_id);
    if (!profiles.length) {
      return { content: [{ type: 'text', text: 'No profiles found.' }] };
    }
    const lines = ['# Profiles\n'];
    for (const p of profiles) {
      const album = p.album ? ` — Album: "${p.album}"` : ' — (all songs)';
      lines.push(`- **Profile ${p.id}**: ${p.artist_name}${album} (built with ${p.provider}/${p.model}, ${p.created_at})`);
    }
    return { content: [{ type: 'text', text: lines.join('\n') }] };
  }
);

// ── get_profile ─────────────────────────────────────────────────────────────

server.tool(
  'get_profile',
  'Get the full profile data for a specific profile ID. Returns the artist style analysis.',
  { profile_id: z.number().describe('Profile ID to retrieve') },
  async ({ profile_id }) => {
    const profile = db.getProfile(profile_id);
    if (!profile) {
      return { content: [{ type: 'text', text: `Profile ${profile_id} not found.` }] };
    }
    const pd = profile.profile_data;
    const lines = [
      `# Profile: ${profile.artist_name}`,
      profile.album ? `**Album:** ${profile.album}` : '**Album:** (all songs)',
      `**Built with:** ${profile.provider} / ${profile.model}`,
      `**Created:** ${profile.created_at}`,
      '',
      '## Style Summary',
      pd.raw_summary || '(no summary)',
      '',
      '## Key Stats',
      `- Themes: ${(pd.themes || []).join(', ')}`,
      `- Avg verse: ${pd.avg_verse_lines} lines, Avg chorus: ${pd.avg_chorus_lines} lines`,
      `- Perspective: ${pd.perspective || 'unknown'}`,
      `- Blueprints: ${(pd.structure_blueprints || []).join(', ')}`,
    ];
    if (pd.tone_and_mood) lines.push(`- Tone: ${pd.tone_and_mood}`);
    if (pd.vocabulary_notes) lines.push(`- Vocabulary: ${pd.vocabulary_notes}`);
    return { content: [{ type: 'text', text: lines.join('\n') }] };
  }
);

// ── prepare_generation ──────────────────────────────────────────────────────

server.tool(
  'prepare_generation',
  'Prepare prompts for generating new lyrics. Returns the metadata planning prompt that the agent should respond to with a JSON object containing subject, bpm, key, caption, and duration.',
  {
    profile_id: z.number().describe('Profile ID to generate from'),
    extra_instructions: z.string().optional().describe('Extra instructions for generation'),
    user_subject: z.string().optional().describe('User-specified subject for the song'),
  },
  async ({ profile_id, extra_instructions, user_subject }) => {
    const profile = db.getProfile(profile_id);
    if (!profile) {
      return { content: [{ type: 'text', text: `Profile ${profile_id} not found.` }] };
    }
    const pd = profile.profile_data;
    const history = db.getGenerationHistory(profile.artist_id);

    const metadataUserPrompt = prompts.buildMetadataPrompt(
      pd, history.usedSubjects, history.usedBpms, history.usedKeys, history.usedDurations, user_subject
    );

    const text = [
      `# Generation Prep: ${profile.artist_name}`,
      profile.album ? `**Album:** ${profile.album}` : '',
      `**Profile ID:** ${profile_id}`,
      `**Previous generations:** ${history.usedTitles.length} songs`,
      '',
      '---',
      '',
      '## Step 1: Plan Song Metadata',
      '',
      'Respond to the following prompt with a JSON object containing: subject, bpm, key, caption, duration, structure.',
      '',
      '### System Prompt',
      '```',
      prompts.SONG_METADATA_SYSTEM_PROMPT,
      '```',
      '',
      '### User Prompt',
      '```',
      metadataUserPrompt,
      '```',
      '',
      '---',
      '',
      '## Step 2',
      'After generating the metadata JSON, call `build_lyrics_prompt` with the profile_id and your generated metadata values (subject, bpm, duration, structure, and optionally extra_instructions).',
    ].filter(Boolean).join('\n');

    return { content: [{ type: 'text', text }] };
  }
);

// ── build_lyrics_prompt ─────────────────────────────────────────────────────

server.tool(
  'build_lyrics_prompt',
  'Build the full lyrics generation prompt using profile data and metadata. Call this after generating metadata. Returns system + user prompts for lyric writing.',
  {
    profile_id: z.number().describe('Profile ID'),
    subject: z.string().describe('Song subject from metadata generation'),
    bpm: z.number().describe('BPM from metadata generation'),
    duration: z.number().describe('Duration in seconds from metadata generation'),
    structure: z.string().optional().describe('Planned song structure from metadata generation, e.g. "I-V-C-V-C-B-C-O"'),
    extra_instructions: z.string().optional().describe('Extra instructions'),
  },
  async ({ profile_id, subject, bpm, duration, structure, extra_instructions }) => {
    const profile = db.getProfile(profile_id);
    if (!profile) {
      return { content: [{ type: 'text', text: `Profile ${profile_id} not found.` }] };
    }
    const pd = profile.profile_data;

    // Add subject to extra instructions
    let fullInstructions = `The song must be about: ${subject}`;
    if (extra_instructions) fullInstructions += `\n\n${extra_instructions}`;

    const userPrompt = prompts.buildGenerationPrompt(pd, fullInstructions, duration, bpm, structure);

    const text = [
      `# Lyrics Generation: ${profile.artist_name}`,
      `**Subject:** ${subject}`,
      `**BPM:** ${bpm} | **Duration:** ${duration}s${structure ? ` | **Structure:** ${structure}` : ''}`,
      '',
      '---',
      '',
      '## System Prompt',
      '```',
      prompts.GENERATION_SYSTEM_PROMPT,
      '```',
      '',
      '## User Prompt',
      '```',
      userPrompt,
      '```',
      '',
      '---',
      '',
      'Write the lyrics following the system prompt rules. Then call `save_generation` to save the result — ' +
      "include the `model` param with your own model name (e.g. 'Fable 5') so it is appended to the title.",
    ].join('\n');

    return { content: [{ type: 'text', text }] };
  }
);

// ── save_generation ─────────────────────────────────────────────────────────

server.tool(
  'save_generation',
  'Save a completed lyric generation to the database. The result appears in the Lyric Studio UI immediately.',
  {
    profile_id: z.number().describe('Profile ID'),
    lyrics: z.string().describe('Generated lyrics'),
    title: z.string().describe('Song title (WITHOUT model name — the server appends it)'),
    model: z.string().optional().describe(MODEL_PARAM_DESC),
    subject: z.string().optional().describe('Song subject'),
    bpm: z.number().optional().describe('BPM'),
    key: z.string().optional().describe('Musical key (e.g. "C Major")'),
    caption: z.string().optional().describe('Audio style caption'),
    duration: z.number().optional().describe('Duration in seconds'),
  },
  async ({ profile_id, lyrics, title, model, subject, bpm, key, caption, duration }) => {
    const profile = db.getProfile(profile_id);
    if (!profile) {
      return { content: [{ type: 'text', text: `Profile ${profile_id} not found.` }] };
    }

    const saved = db.saveGeneration({
      profileId: profile_id,
      provider: 'mcp',
      model: model || 'unknown',
      lyrics,
      title: withModelSuffix(title, model),
      subject,
      bpm,
      key,
      caption,
      duration,
    });

    return {
      content: [{
        type: 'text',
        text: [
          `✅ Generation saved!`,
          '',
          `**ID:** ${saved.id}`,
          `**Title:** ${saved.title}`,
          `**Artist:** ${profile.artist_name}${profile.album ? ` — ${profile.album}` : ''}`,
          `**Subject:** ${saved.subject}`,
          `**BPM:** ${saved.bpm} | **Key:** ${saved.key} | **Duration:** ${saved.duration}s`,
          '',
          'The generation is now visible in the Lyric Studio UI.',
        ].join('\n'),
      }],
    };
  }
);

// ── prepare_refinement ──────────────────────────────────────────────────────

server.tool(
  'prepare_refinement',
  'Prepare prompts for refining an existing generation. Returns the refinement system + user prompts.',
  { generation_id: z.number().describe('Generation ID to refine') },
  async ({ generation_id }) => {
    const gen = db.getGeneration(generation_id);
    if (!gen) {
      return { content: [{ type: 'text', text: `Generation ${generation_id} not found.` }] };
    }
    const profile = db.getProfile(gen.profile_id);
    const lyricsSet = profile ? db.getLyricsSet(profile.lyrics_set_id) : null;
    const artistName = lyricsSet?.artist_name || profile?.artist_name || 'Unknown';
    const pd = profile?.profile_data;

    const userPrompt = prompts.buildRefinementPrompt(gen.lyrics, artistName, gen.title, pd);

    const text = [
      `# Refinement: "${gen.title}" by ${artistName}`,
      `**Generation ID:** ${generation_id}`,
      `**BPM:** ${gen.bpm} | **Key:** ${gen.key} | **Duration:** ${gen.duration}s`,
      '',
      '### Original Lyrics',
      '```',
      gen.lyrics,
      '```',
      '',
      '---',
      '',
      '## System Prompt',
      '```',
      prompts.REFINEMENT_SYSTEM_PROMPT,
      '```',
      '',
      '## User Prompt',
      '```',
      userPrompt,
      '```',
      '',
      '---',
      '',
      'Write the refined lyrics (starting with "Title: ..."). Then call `save_refinement` to save — ' +
      "include the `model` param with your own model name (e.g. 'Fable 5') so it is appended to the title.",
    ].join('\n');

    return { content: [{ type: 'text', text }] };
  }
);

// ── save_refinement ─────────────────────────────────────────────────────────

server.tool(
  'save_refinement',
  'Save refined lyrics as a new generation linked to the parent.',
  {
    generation_id: z.number().describe('Parent generation ID'),
    lyrics: z.string().describe('Refined lyrics'),
    title: z.string().describe('Refined title (WITHOUT model name — the server appends it)'),
    model: z.string().optional().describe(MODEL_PARAM_DESC),
  },
  async ({ generation_id, lyrics, title, model }) => {
    const parent = db.getGeneration(generation_id);
    if (!parent) {
      return { content: [{ type: 'text', text: `Generation ${generation_id} not found.` }] };
    }

    const saved = db.saveGeneration({
      profileId: parent.profile_id,
      provider: 'mcp',
      model: model || 'unknown',
      lyrics,
      title: withModelSuffix(title, model),
      subject: parent.subject,
      bpm: parent.bpm,
      key: parent.key,
      caption: parent.caption,
      duration: parent.duration,
      parentGenerationId: generation_id,
    });

    return {
      content: [{
        type: 'text',
        text: [
          `✅ Refinement saved!`,
          '',
          `**ID:** ${saved.id} (parent: ${generation_id})`,
          `**Title:** ${saved.title}`,
          '',
          'The refined generation is now visible in the Lyric Studio UI.',
        ].join('\n'),
      }],
    };
  }
);

// ── list_generations ────────────────────────────────────────────────────────

server.tool(
  'list_generations',
  'List recent generations. Filter by profile_id or artist_id.',
  {
    profile_id: z.number().optional().describe('Filter by profile ID'),
    artist_id: z.number().optional().describe('Filter by artist ID'),
    limit: z.number().optional().default(20).describe('Max results (default 20)'),
  },
  async ({ profile_id, artist_id, limit }) => {
    const gens = db.listGenerations(profile_id, artist_id, limit);
    if (!gens.length) {
      return { content: [{ type: 'text', text: 'No generations found.' }] };
    }
    const lines = ['# Generations\n'];
    for (const g of gens) {
      const parent = g.parent_generation_id ? ` (refined from #${g.parent_generation_id})` : '';
      lines.push(
        `- **#${g.id}** "${g.title || '(untitled)'}" — ${g.artist_name}${g.album ? ` / ${g.album}` : ''}` +
        ` | ${g.provider}/${g.model} | BPM ${g.bpm} | ${g.key} | ${g.duration}s${parent}`
      );
    }
    return { content: [{ type: 'text', text: lines.join('\n') }] };
  }
);

// ── prepare_title ───────────────────────────────────────────────────────────

server.tool(
  'prepare_title',
  'Prepare a title derivation prompt for lyrics.',
  {
    lyrics: z.string().describe('The lyrics to title'),
    artist_name: z.string().describe('Artist name'),
    album: z.string().optional().describe('Album name'),
    used_titles: z.array(z.string()).optional().describe('Titles already used (for diversity)'),
  },
  async ({ lyrics, artist_name, album, used_titles }) => {
    const userPrompt = prompts.buildTitlePrompt(lyrics, artist_name, album, used_titles);
    const text = [
      '## Title Derivation',
      '',
      '### System Prompt',
      '```',
      prompts.TITLE_DERIVATION_PROMPT,
      '```',
      '',
      '### User Prompt',
      '```',
      userPrompt,
      '```',
      '',
      'Respond with ONLY the title. Do NOT include your model name in it — ' +
      'when saving, pass your model name via the `model` param and the server appends it automatically.',
    ].join('\n');

    return { content: [{ type: 'text', text }] };
  }
);

// ── prepare_profile_build ───────────────────────────────────────────────────

server.tool(
  'prepare_profile_build',
  'Prepare prompts for building an artist profile from a lyrics set. Returns 4 sequential prompts that need to be answered in order.',
  {
    lyrics_set_id: z.number().describe('Lyrics set ID to build profile from'),
  },
  async ({ lyrics_set_id }) => {
    const lyricsSet = db.getLyricsSet(lyrics_set_id);
    if (!lyricsSet) {
      return { content: [{ type: 'text', text: `Lyrics set ${lyrics_set_id} not found.` }] };
    }

    const songs = lyricsSet.songs;
    const artistName = lyricsSet.artist_name;
    const album = lyricsSet.album;

    // Note: The rule-based analysis (rhyme, meter, structure, etc.) would normally
    // be computed by the profilerService. Since we don't have that code here,
    // we provide a simplified profile build that focuses on the LLM analysis parts.
    // The agent should note that rule-based stats will need to be calculated separately
    // or filled in from the lyrics set data.

    const songList = songs.map((s: any) => `--- ${s.title} ---\n${s.lyrics}`).join('\n\n');
    const header = [
      `Artist: ${artistName}`,
      album ? `Album: ${album}` : '',
      `Songs analysed: ${songs.length}`,
      '',
      '=== COMPLETE LYRICS ===',
      '',
      songList,
    ].filter(Boolean).join('\n');

    const text = [
      `# Profile Build: ${artistName}${album ? ` — "${album}"` : ''}`,
      `**Lyrics Set ID:** ${lyrics_set_id}`,
      `**Songs:** ${songs.length}`,
      '',
      '---',
      '',
      '## Process',
      'Building a profile requires 4 sequential LLM calls. Answer each prompt in order with a JSON response.',
      '',
      '### Call 1/4: Themes & Vocabulary',
      '**System Prompt:**',
      '```',
      prompts.PROFILE_PROMPT_1,
      '```',
      '**User Prompt:**',
      '```',
      header,
      '```',
      '',
      '### Call 2/4: Tone & Structure',
      '**System Prompt:**',
      '```',
      prompts.PROFILE_PROMPT_2,
      '```',
      '**User Prompt:** (same as above)',
      '',
      '### Call 3/4: Imagery & Signature',
      '**System Prompt:**',
      '```',
      prompts.PROFILE_PROMPT_3,
      '```',
      '**User Prompt:** (same as above)',
      '',
      '### Call 4/4: Song Subjects',
      '**System Prompt:**',
      '```',
      prompts.SUBJECT_ANALYSIS_PROMPT,
      '```',
      '**User Prompt:**',
      '```',
      prompts.buildSubjectAnalysisPrompt(songs),
      '```',
      '',
      '---',
      '',
      'After generating all 4 JSON responses, call `save_profile` with the merged data.',
      '',
      '> **Note:** Rule-based stats (rhyme analysis, meter, syllable counts, structure blueprints, etc.) are computed by the app\'s profiler service, not by the LLM. When saving the profile, include the LLM-generated fields. The app will have partial data but the most important parts (themes, vocabulary, tone, imagery, subjects) will be present.',
    ].join('\n');

    return { content: [{ type: 'text', text }] };
  }
);

// ── save_profile ────────────────────────────────────────────────────────────

server.tool(
  'save_profile',
  'Save a built profile to the database.',
  {
    lyrics_set_id: z.number().describe('Lyrics set ID the profile was built from'),
    profile_data: z.string().describe('JSON string of the merged profile data object'),
  },
  async ({ lyrics_set_id, profile_data }) => {
    const lyricsSet = db.getLyricsSet(lyrics_set_id);
    if (!lyricsSet) {
      return { content: [{ type: 'text', text: `Lyrics set ${lyrics_set_id} not found.` }] };
    }

    let parsed: any;
    try {
      parsed = JSON.parse(profile_data);
    } catch (e) {
      return { content: [{ type: 'text', text: 'Failed to parse profile_data JSON.' }] };
    }

    // Add artist name to the profile data
    parsed.artist = lyricsSet.artist_name;
    if (lyricsSet.album) parsed.album = lyricsSet.album;

    const saved = db.saveProfile(lyrics_set_id, 'antigravity', 'claude-opus-4', parsed);

    return {
      content: [{
        type: 'text',
        text: [
          `✅ Profile saved!`,
          '',
          `**ID:** ${saved.id}`,
          `**Artist:** ${lyricsSet.artist_name}${lyricsSet.album ? ` — ${lyricsSet.album}` : ''}`,
          `**Lyrics Set:** ${lyrics_set_id}`,
          '',
          'The profile is now available for generation in the Lyric Studio UI.',
        ].join('\n'),
      }],
    };
  }
);

// ── Start server ────────────────────────────────────────────────────────────

async function main() {
  db.initDb();
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('[mcp-lyricstudio] Server started');
}

main().catch((err) => {
  console.error('[mcp-lyricstudio] Fatal error:', err);
  process.exit(1);
});
