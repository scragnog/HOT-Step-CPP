// index.ts — MCP server for HOT-Step Lyric Studio
//
// Lets the Antigravity agent act as the LLM for lyric generation,
// refinement, and profile building. Connects to hotstep.db directly.

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import * as db from './db.js';
import * as prompts from './prompts.js';
// Model-name title suffix ("Song Name - Fable 5") — canonical implementation
// shared with the in-app pipeline; appended HERE (not in the LLM prompt) so it
// is deterministic.
import { withModelSuffix } from '../../../server/src/services/lireek/modelName.js';
import { recalculateProfileStats } from '../../../server/src/services/lireek/profilerService.js';
import { registerCollaborationTools } from './collaboration.js';
import { channelServerOptions } from './discussion-wake.js';

const server = new McpServer({
  name: 'lyricstudio',
  version: '1.0.0',
}, channelServerOptions());

const collaboration = registerCollaborationTools(server);
server.server.onclose = () => collaboration.close();

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

// ── list_lyrics_sets ────────────────────────────────────────────────────────

server.tool(
  'list_lyrics_sets',
  'List lyrics sets with their IDs, song counts, and whether a profile has been built from them. ' +
    'Use this to find the lyrics_set_id needed by prepare_profile_build/save_profile, and to find ' +
    'which artists still need a profile. Set unprofiled_only to list just the sets with no profile yet.',
  {
    artist_id: z.number().optional().describe('Filter by artist ID'),
    unprofiled_only: z
      .boolean()
      .optional()
      .describe('Only return lyrics sets that do not have a profile yet'),
  },
  async ({ artist_id, unprofiled_only }) => {
    let sets = db.getLyricsSets(artist_id);
    if (unprofiled_only) sets = sets.filter((s: any) => s.profile_id == null);
    if (!sets.length) {
      return {
        content: [
          {
            type: 'text',
            text: unprofiled_only
              ? 'No unprofiled lyrics sets found — every lyrics set already has a profile.'
              : 'No lyrics sets found.',
          },
        ],
      };
    }
    const lines = [unprofiled_only ? '# Lyrics Sets (unprofiled)\n' : '# Lyrics Sets\n'];
    for (const s of sets) {
      const album = s.album ? ` — Album: "${s.album}"` : ' — (all songs)';
      const profile =
        s.profile_id == null ? '**no profile yet**' : `profile ${s.profile_id}`;
      lines.push(
        `- **Lyrics Set ${s.id}**: ${s.artist_name}${album} — ${s.total_songs} song(s), ${profile}`
      );
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
      lines.push(`- **Profile ${p.id}** (lyrics set ${p.lyrics_set_id}): ${p.artist_name}${album} (built with ${p.provider}/${p.model}, ${p.created_at})`);
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

    // Enrichment is PURE derived data — recompute from the lyrics set every
    // time rather than trusting the copy frozen into profile_data at build
    // time, which goes stale when the set gains captions or the distillation
    // changes (see the same note in server/src/routes/lireek/llmRoutes.ts).
    {
      const set = db.getLyricsSet(profile.lyrics_set_id);
      if (set) pd.audio_enrichment = prompts.computeAlbumEnrichment(set.songs);
    }

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
      '',
      'The full chain is: `prepare_generation` → `build_lyrics_prompt` → write lyrics → ' +
      '`prepare_mm3_caption` → write the MM3 caption → `save_generation`. The `caption` in this ' +
      'metadata JSON is the ACE-Step one; the MiniMax-Music3 caption is a separate, differently ' +
      'formatted field written later, once the lyrics exist.',
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

    // Same live re-derivation as prepare_generation.
    {
      const set = db.getLyricsSet(profile.lyrics_set_id);
      if (set) pd.audio_enrichment = prompts.computeAlbumEnrichment(set.songs);
    }

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
      'Write the lyrics following the system prompt rules. Then call `prepare_mm3_caption` with the ' +
      'finished lyrics to write the MiniMax-Music3 caption, and finally `save_generation` to save ' +
      "everything — include the `model` param with your own model name (e.g. 'Fable 5') so it is " +
      'appended to the title.',
    ].join('\n');

    return { content: [{ type: 'text', text }] };
  }
);

// ── prepare_mm3_caption ─────────────────────────────────────────────────────
//
// MiniMax-Music3 is the second generation backend and takes a completely
// different caption from ACE-Step's: a three-heading Structured Caption with
// twelve fixed labels. The two are not interchangeable — an ACE caption fed to
// MM3 lands off-genre (measured A/B, see prompts.ts). So a generation carries
// BOTH, and this is the MCP counterpart of the dedicated MM3 caption call the
// in-app pipeline makes in llm/orchestration.ts.
//
// Called AFTER the lyrics are written, on purpose: the Arrangement section is a
// section-by-section timeline of this song, so it needs the real section tags.

server.tool(
  'prepare_mm3_caption',
  'Prepare the prompts for writing this song\'s MiniMax-Music3 Structured Caption. Call this AFTER writing the lyrics and BEFORE save_generation — the caption describes the song section by section, so it needs the finished lyrics. Pass the result to save_generation as caption_mm3.',
  {
    profile_id: z.number().describe('Profile ID'),
    lyrics: z.string().describe('The FINISHED lyrics, including their section tags'),
    bpm: z.number().optional().describe('BPM from metadata generation'),
    key: z.string().optional().describe('Musical key from metadata generation, e.g. "E minor"'),
    caption: z.string().optional().describe('The ACE-Step caption planned for this song — used as evidence of the intended sound'),
    subject: z.string().optional().describe('Song subject'),
  },
  async ({ profile_id, lyrics, bpm, key, caption, subject }) => {
    const profile = db.getProfile(profile_id);
    if (!profile) {
      return { content: [{ type: 'text', text: `Profile ${profile_id} not found.` }] };
    }
    const pd = profile.profile_data;

    // Same live re-derivation as prepare_generation — the enrichment supplies
    // the album's measured genres and time signature.
    const set = db.getLyricsSet(profile.lyrics_set_id);
    if (set) pd.audio_enrichment = prompts.computeAlbumEnrichment(set.songs);

    const userPrompt = prompts.buildMm3CaptionPrompt(pd, {
      lyrics,
      aceCaption: caption,
      subject,
      bpm,
      key,
      signature: pd.audio_enrichment?.signatures?.[0],
      instrumental: !lyrics.trim(),
    });

    const text = [
      `# MM3 Structured Caption: ${profile.artist_name}`,
      `**Profile ID:** ${profile_id}`,
      '',
      '---',
      '',
      '## System Prompt',
      '```',
      prompts.MM3_CAPTION_SYSTEM_PROMPT,
      '```',
      '',
      '## User Prompt',
      '```',
      userPrompt,
      '```',
      '',
      '---',
      '',
      'Write the Structured Caption, then call `save_generation` with it as the `caption_mm3` param ' +
      '(alongside the normal `caption`). The `Basic Attributes:` line is rebuilt from the stored ' +
      'bpm/key on save, so do not worry about getting those digits exactly right — but DO name a ' +
      'specific genre, because that part is kept.',
    ].join('\n');

    return { content: [{ type: 'text', text }] };
  }
);

// ── prepare_yue2_caption ────────────────────────────────────────────────────
//
// The YuE2 planner is prompted with ONE sentence in a fixed order (language →
// genre → vocal → instruments → mood → production → BPM) — the order its
// training captions take. A third caption, next to the ACE and MM3 ones.
// Called after the lyrics exist, like MM3's, because the section tags are
// evidence of the arrangement.

server.tool(
  'prepare_yue2_caption',
  'Prepare the prompts for writing this song\'s YuE2 planner caption: ONE sentence in the fixed order language → genre → vocal → instruments → mood → production → BPM. Call this AFTER writing the lyrics and BEFORE save_generation; pass the result to save_generation as caption_yue2.',
  {
    profile_id: z.number().describe('Profile ID'),
    lyrics: z.string().describe('The FINISHED lyrics, including their section tags'),
    bpm: z.number().optional().describe('BPM from metadata generation — the sentence must end with it'),
    key: z.string().optional().describe('Musical key (not stated in the caption; context only)'),
    caption: z.string().optional().describe('The ACE-Step caption planned for this song — evidence of the intended sound'),
    subject: z.string().optional().describe('Song subject'),
  },
  async ({ profile_id, lyrics, bpm, key, caption, subject }) => {
    const profile = db.getProfile(profile_id);
    if (!profile) {
      return { content: [{ type: 'text', text: `Profile ${profile_id} not found.` }] };
    }
    const pd = profile.profile_data;
    const set = db.getLyricsSet(profile.lyrics_set_id);
    if (set) pd.audio_enrichment = prompts.computeAlbumEnrichment(set.songs);
    const language = set?.songs?.find((s: any) => s?.language)?.language;

    const userPrompt = prompts.buildYue2CaptionPrompt(pd, {
      lyrics, aceCaption: caption, subject, bpm, key, language,
      instrumental: !lyrics.trim(),
    });

    const text = [
      `# YuE2 Caption: ${profile.artist_name}`,
      `**Profile ID:** ${profile_id}`,
      '',
      '---',
      '',
      '## System Prompt',
      '```',
      prompts.YUE2_CAPTION_SYSTEM_PROMPT,
      '```',
      '',
      '## User Prompt',
      '```',
      userPrompt,
      '```',
      '',
      '---',
      '',
      'Write the one sentence, then call `save_generation` with it as the `caption_yue2` param ' +
      '(alongside `caption` and `caption_mm3`). The trailing "<N> BPM" is rebuilt from the stored bpm on save.',
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
    caption: z.string().optional().describe('Audio style caption for the ACE-Step backend'),
    caption_mm3: z.string().optional().describe('MiniMax-Music3 Structured Caption, from prepare_mm3_caption. A SEPARATE caption in MM3\'s own three-heading format — never a copy of `caption`.'),
    caption_yue2: z.string().optional().describe('YuE2 planner caption, from prepare_yue2_caption: ONE sentence, fixed order (language → genre → vocal → instruments → mood → production → BPM). Never a copy of `caption`.'),
    duration: z.number().optional().describe('Duration in seconds'),
  },
  async ({ profile_id, lyrics, title, model, subject, bpm, key, caption, caption_mm3, caption_yue2, duration }) => {
    const profile = db.getProfile(profile_id);
    if (!profile) {
      return { content: [{ type: 'text', text: `Profile ${profile_id} not found.` }] };
    }

    // Same arrangement + duration policy as the in-app path (orchestration.ts).
    // Intro FIRST — a declared instrumental intro is time the duration
    // derivation counts, so applying it afterwards would under-time the song.
    let introNote = '';
    const withIntro = prompts.ensureInstrumentalIntro(lyrics, `${profile.artist_name}|${title}`);
    if (withIntro !== null) {
      lyrics = withIntro;
      introNote = `\n🎸 Instrumental intro applied (${prompts.INSTRUMENTAL_INTRO_TAG})`;
    }

    // The final duration derives from the written lyrics at the artist's
    // measured pacing, so duration==content holds no matter what was planned.
    let durationNote = '';
    {
      const set = db.getLyricsSet(profile.lyrics_set_id);
      const rate = set ? (prompts.computeAlbumEnrichment(set.songs)?.wordsPerSec || 0) : 0;
      const reconciled = prompts.reconcileDurationToLyrics(lyrics, bpm ?? 0, duration ?? 0, rate);
      if (reconciled !== (duration ?? 0)) {
        durationNote = `\n⏱ Duration reconciled to lyrics: ${duration ?? 0}s → ${reconciled}s (${rate > 0 ? `artist rate ${rate.toFixed(2)} w/s` : 'global rate'})`;
        duration = reconciled;
      }
    }

    // The MM3 caption gets the SAME deterministic treatment as the in-app
    // path: markdown stripped, and `Basic Attributes:` rebuilt from the
    // bpm/key/signature we hold exactly rather than the digits the model typed.
    let mm3Note = '';
    let captionMm3 = '';
    if (caption_mm3 && caption_mm3.trim()) {
      const enrich = (() => {
        const set = db.getLyricsSet(profile.lyrics_set_id);
        return set ? prompts.computeAlbumEnrichment(set.songs) : null;
      })();
      captionMm3 = prompts.normalizeMm3Caption(caption_mm3, {
        bpm, key, signature: enrich?.signatures?.[0], fallbackGenre: enrich?.genres?.[0],
      });
      const issues = prompts.validateMm3Caption(captionMm3);
      mm3Note = issues.length
        ? `\n⚠ MM3 caption saved WITH FORMAT ISSUES: ${issues.join('; ')} — fix it in the Lyric Studio UI or re-run prepare_mm3_caption`
        : `\n🎛 MM3 caption saved (${captionMm3.split(/\s+/).length} words, format OK)`;
    } else {
      mm3Note = '\n⚠ No MM3 caption supplied — this song cannot be generated well on the MiniMax-Music3 backend. Call prepare_mm3_caption and re-save.';
    }

    // The YuE2 sentence: same deterministic tidy-up as the in-app path, with
    // the BPM tail rebuilt from the number we hold.
    let yue2Note = '';
    let captionYue2 = '';
    if (caption_yue2 && caption_yue2.trim()) {
      captionYue2 = prompts.normalizeYue2Caption(caption_yue2, { bpm });
      const issues = prompts.validateYue2Caption(captionYue2);
      yue2Note = issues.length
        ? `\n⚠ YuE2 caption saved WITH FORMAT ISSUES: ${issues.join('; ')}`
        : `\n🎼 YuE2 caption saved (${captionYue2.split(/\s+/).length} words, format OK)`;
    } else {
      yue2Note = '\nℹ No YuE2 caption supplied — a YuE2 render will fall back to a dataset-track caption. Call prepare_yue2_caption and re-save to give it one of its own.';
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
      captionMm3,
      captionYue2,
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
          introNote,
          durationNote,
          mm3Note,
          yue2Note,
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
      captionMm3: parent.caption_mm3,
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
    // Measured audio facts from a Training Studio export — ground truth for
    // genre/tempo. Plain Genius-fetched sets yield null and the block is absent.
    const enrichment = prompts.computeAlbumEnrichment(songs);
    const header = [
      `Artist: ${artistName}`,
      album ? `Album: ${album}` : '',
      `Songs analysed: ${songs.length}`,
      ...(enrichment ? [
        '',
        ...prompts.formatAlbumEnrichment(enrichment),
        'Treat the detected genre and tempo as ground truth — fold them into tone_and_mood and additional_notes rather than inferring a genre from the lyrics alone.',
      ] : []),
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
      '> **Note:** Rule-based stats (rhyme analysis, meter, syllable counts, structure blueprints, representative excerpts, etc.) are computed deterministically from the lyrics — do NOT invent them. Just include the LLM-generated prose fields; `save_profile` computes and attaches all rule-based stats automatically on save.',
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
    model: z.string().describe(MODEL_PARAM_DESC),
    provider: z.string().optional().describe(
      "Where the model ran (e.g. 'anthropic', 'mcp'). Defaults to 'mcp'.",
    ),
  },
  async ({ lyrics_set_id, profile_data, model, provider }) => {
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
    // Rule-based stats (meter, rhyme, repetition, vocabulary), structure
    // blueprints, representative excerpts and audio enrichment are deterministic
    // and computed from the lyrics — never agent-derived. Without this, MCP-built
    // profiles render "Average verse length: undefined lines" and carry no
    // excerpts or structural vocabulary into generation prompts.
    parsed = recalculateProfileStats(lyricsSet.songs, parsed);

    // The provider/model were hardcoded to 'antigravity'/'claude-opus-4' here, so
    // EVERY profile built through MCP was stamped claude-opus-4 whatever actually
    // wrote it — 36 of them, unattributable after the fact. save_generation has
    // taken a real `model` since it was written; this now matches it (2026-08-06).
    const saved = db.saveProfile(lyrics_set_id, provider || 'mcp', model, parsed);

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
