// Module "Prompt". Concept: src/js/33-prompt.js and src/css/22-prompt.css.

import type { ModuleDef } from '../types.ts';

export const prompt = {
  kind: 'module',
  id: 'prompt',
  name: 'Prompt',
  desc: 'The main text box and the song info: what the engine reads first.',
  at: 0,
  category: 'Words',
  builtIn: true,
  engines: ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula'],
  owner: 'song',
  stored: 'content',
  size: { w: 4, quickW: 4 },
  per: [
    { engines: ['minimax-m3'], name: 'Caption' },
    { engines: ['sa3'], name: 'Prompt' },
    { engines: ['heartmula'], name: 'Tags' },
  ],
  params: [
    {
      key: 'prompt.text',
      type: 'text',
      label: 'Describe the song',
      undoName: 'the description',
      default: '',
      engines: ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula'],
      wire: 'content:caption',
      /* what the box's tools do to it, each one undo step (concept 33-prompt.js and 25-module-tempo.js) */
      steps: {
        write: { says: 'Write it for me', changes: ['lyrics.text', 'song.title'] },
        compose: 'Compose caption',
        copy: { says: 'Edit a copy of {name}', changes: ['cap.src'] },
        addTag: { says: 'Add "{tag}"', changes: ['cap.src'] },
        removeTag: { says: 'Take out "{tag}"', changes: ['cap.src'] },
        expand: { says: 'Expand wildcards', changes: ['lyrics.text'] },
        toLyrics: { says: 'Move to lyrics', changes: ['lyrics.text'] },
        tempo: { says: 'Set the tempo to {bpm} BPM', changes: ['tempo.bpm'] },
        plain: 'Plain {words}',
        preset: 'Style preset',
        addChip: 'Add {chip} to {name}',
        removeChip: 'Take {chip} out of {name}',
      },
      per: [
        { engines: ['minimax-m3'], label: 'Caption', undoName: 'the caption', wire: ['content:caption', { to: 'intent:captionSource', via: 'mm3CaptionSource' }] },
        { engines: ['sa3'], label: 'Prompt', undoName: 'the prompt' },
        { engines: ['heartmula'], label: 'Tags', undoName: 'the tags' },
      ],
    },
    {
      key: 'wild.auto',
      type: 'boolean',
      label: 'Expand {a|b|c} wildcards at Generate',
      undoName: 'Expand on Generate',
      tip: 'Expand wildcards in the caption and lyrics when you press Generate, seeded from the song\'s seed',
      default: false,
      engines: ['ace', 'minimax-m3', 'yue2', 'sa3'],
      stored: 'settings',
      wire: 'intent:compose.autoExpand',
    },
    { key: 'song.artist', type: 'text', label: 'Artist', placeholder: 'optional', maxLength: 60, default: '', wire: 'content:artist' },
    { key: 'song.title', type: 'text', label: 'Title', placeholder: 'optional', maxLength: 80, default: '', wire: 'content:title' },
    { key: 'song.subject', type: 'text', label: 'Subject', placeholder: 'what the song is about', maxLength: 120, default: '', wire: 'content:subject' },
    {
      key: 'ad.trigger',
      type: 'text',
      label: 'Trigger',
      undoName: 'Trigger word',
      tip: 'The word your adapter was trained to answer to, added to the prompt. Up to 24 characters.',
      default: '',
      placeholder: 'add a word',
      maxLength: 24,
      when: { any: [{ state: 'adapters', atLeast: 1 }, { param: 'ad.planner', not: '' }] },
      engines: ['ace', 'minimax-m3', 'yue2', 'heartmula'],
      wire: { to: 'intent:compose.loraTrigger', note: 'The server puts it at the start of the caption (composeCreateCaption).' },
      per: [
        {
          engines: ['yue2'],
          wire: {
            none: 'YuE2 composes its adapter\'s own trigger (yue2LmAdapterTrigger); the server drops a typed trigger word there.',
          },
        },
        {
          engines: ['heartmula'],
          wire: {
            none: 'HeartMuLa adds its adapter\'s own trigger (withHeartmulaTrigger); the existing Create page sends no typed trigger word there.',
          },
        },
      ],
    },
    {
      key: 'cap.src',
      type: 'choice',
      label: 'Caption source',
      default: 'custom',
      options: {
        from: 'caption.sources',
        include: [{ value: 'custom', label: 'Custom' }, { value: 'auto', label: 'Automatic' }],
      },
      engines: ['minimax-m3', 'yue2'],
      wire: { to: 'intent:captionSource', via: 'mm3CaptionSource' },
      per: [{ engines: ['yue2'], label: 'Description source', wire: { to: 'intent:captionSource', via: 'yue2CaptionSource' } }],
    },
    {
      key: 'cap.ds',
      type: 'choice',
      label: 'Dataset',
      default: '',
      options: { from: 'caption.datasets', include: [{ value: '', label: 'The adapter\'s dataset' }] },
      engines: ['minimax-m3', 'yue2'],
      wire: {
        none: 'MiniMax-Music3 takes its caption tracks from a Lyric Studio handoff, not from a dataset dropdown.',
      },
      per: [{ engines: ['yue2'], wire: { to: 'intent:captionSource', via: 'yue2CaptionSource' } }],
    },
  ],
  state: [
    {
      key: 'latent',
      kind: 'file',
      noun: 'the latent',
      /* importing a latent brings its tempo and key, and fills an empty lyric sheet and box (concept importLatent) */
      steps: { load: { says: 'Import latent', changes: ['tempo.bpm', 'tempo.key', 'lyrics.text', 'prompt.text'] }, clear: 'Stop using the latent' },
      owner: 'song',
      stored: 'content',
      wire: [{ engines: ['ace'], wire: 'content:sourceLatentUrl' }],
    },
    {
      key: 'capTracks',
      kind: 'list',
      noun: 'the caption tracks',
      owner: 'song',
      stored: 'none',
      /* the tracks a caption source picks from: a Lyric Studio handoff on MiniMax-Music3, the dataset's on YuE2 */
      wire: [
        { engines: ['minimax-m3'], wire: { to: 'intent:captionSource', via: 'mm3CaptionSource' } },
        { engines: ['yue2'], wire: { to: 'intent:captionSource', via: 'yue2CaptionSource' } },
      ],
    },
  ],
  face: [
    {
      part: 'promptBox',
      param: 'prompt.text',
      /* the trigger word is a chip on the card at every tier; the cover art subject is song info while the Cover art
         unit is on the board and on (concept html/create/20-centre.html, 33-prompt.js) */
      params: ['song.artist', 'song.title', 'song.subject', 'cap.src', 'cap.ds', 'wild.auto', 'ad.trigger', 'pp.art.subject'],
      state: ['capTracks', 'latent'],
      /* the card's tools, each a named action the prompt feature runs (src/app/features/prompt-tools.ts) */
      actions: [
        { do: 'bespoke', name: 'stylePresets', label: 'Presets & genres' },
        { do: 'bespoke', name: 'promptWrite', label: 'Write it for me' },
        { do: 'bespoke', name: 'composeCaption', label: 'Compose caption', tip: "Turn your words into MiniMax-Music3's Structured Caption, using your BPM, key and vocals" },
        { do: 'bespoke', name: 'expandWildcards', label: 'Expand {a|b} choices', tip: 'Replace each {a|b|c} with one of its choices' },
        { do: 'bespoke', name: 'moveToLyrics', label: 'Move them to the lyrics sheet' },
        { do: 'bespoke', name: 'tempoFromPrompt', label: 'Set the tempo field' },
        { do: 'bespoke', name: 'plainCaption', label: 'Use it as plain words' },
        { do: 'bespoke', name: 'editCaptionCopy', label: 'Edit a copy' },
        { do: 'bespoke', name: 'datasetTag', label: 'Tags' },
        { do: 'bespoke', name: 'chooseDataset', label: 'From' },
        {
          do: 'bespoke',
          name: 'importLatent',
          label: 'Import latent',
          tip: 'Continue from a saved .latent (or .hslat) file. Fills BPM, key, lyrics and caption from its metadata.',
        },
        { do: 'bespoke', name: 'clearLatent', label: 'Stop using this latent' },
        { do: 'bespoke', name: 'promptLocked', label: 'Why the caption is locked' },
      ],
    },
  ],
} satisfies ModuleDef;
