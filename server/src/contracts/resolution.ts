// contracts/resolution.ts — wire schemas for server-side caption and content
// resolution (POST /api/resolve/preview).
//
// zod 4 schemas; the TypeScript types are inferred from them, so the shape the
// route validates and the shape clients compile against cannot drift. A
// client that only needs the types imports them with `import type`.
//
// Everything the browser used to read from its own storage to make these
// decisions (caption source choice, "Use LLM duration", filename trigger
// settings, the "Use LM adapter" toggle) arrives as an explicit field; Node
// never reads or writes a stored selection on the client's behalf.

import { z } from 'zod/v4';

const captionSourceMode = z.enum(['auto', 'track', 'custom']);
export type CaptionSourceMode = z.infer<typeof captionSourceMode>;

/** A MiniMax-Music3 caption choice for one written song. Absent = auto. */
export const mm3CaptionSelectionSchema = z.object({
  mode: captionSourceMode,
  /** Only read for mode 'track'. */
  selectedTitle: z.string().optional(),
});
export type Mm3CaptionSelection = z.infer<typeof mm3CaptionSelectionSchema>;

/** A YuE2 caption choice. Absent = auto when a dataset is linked and an
 *  adapter is in force, otherwise custom. */
export const yue2CaptionSelectionSchema = z.object({
  mode: captionSourceMode,
  /** Only read for mode 'track'. */
  selectedName: z.string().optional(),
});
export type Yue2CaptionSelection = z.infer<typeof yue2CaptionSelectionSchema>;

/** One album source track with an MM3 caption, in album order. */
export const mm3SourceTrackSchema = z.object({
  title: z.string(),
  bpm: z.number().optional(),
  caption: z.string(),
});
export type Mm3SourceTrack = z.infer<typeof mm3SourceTrackSchema>;

/** One YuE2 dataset track, as GET /api/training/yue2-dataset-captions lists it. */
export interface Yue2SourceTrack {
  name: string;
  caption: string;
  genre?: string;
  bpm?: string | number;
  key?: string;
  styled?: string;
}

const engineId = z.string().min(1);
const params = z.record(z.string(), z.unknown());

/** Create panel: the body Create would submit, with its caption steps still to
 *  do. `params.caption` and `params.lyrics` are the boxes as typed (for a
 *  locked caption source, the user's own caption, used if the source has
 *  nothing to offer). */
export const createIntentSchema = z.object({
  kind: z.literal('create'),
  /** Engine the request is for. Default: the active engine. */
  engine: engineId.optional(),
  params,
  compose: z.object({
    /** Expand {a|b} wildcards in caption and lyrics before submit. */
    autoExpand: z.boolean().optional(),
    /** Manual LoRA trigger prepended to the caption unless already there. */
    loraTrigger: z.string().optional(),
    /** Append the clean percussive intro/outro request. */
    beatIntro: z.boolean().optional(),
    introBars: z.number().optional(),
  }).optional(),
  /** Caption source for the engine in use. Omitted = the caption box. */
  captionSource: z.discriminatedUnion('engine', [
    z.object({
      engine: z.literal('yue2'),
      datasetId: z.string(),
      selection: yue2CaptionSelectionSchema.optional(),
      adapterInForce: z.boolean(),
    }),
    z.object({
      engine: z.literal('minimax-m3'),
      /** The song's own MM3 caption ("Custom"). */
      customCaption: z.string(),
      selection: mm3CaptionSelectionSchema.optional(),
      /** Album to read source tracks from. */
      lyricsSetId: z.number().int().positive().optional(),
      /** Source tracks as handed over, when there is no lyrics set. */
      tracks: z.array(mm3SourceTrackSchema).optional(),
    }),
  ]).optional(),
});
export type CreateIntent = z.infer<typeof createIntentSchema>;

/** Lyric Studio: render one written song. */
export const writtenSongIntentSchema = z.object({
  kind: z.literal('written-song'),
  engine: engineId.optional(),
  generationId: z.number().int().positive(),
  /** Album the song renders as (the Render-as target, else its own). */
  lyricsSetId: z.number().int().nonnegative(),
  /** The song's own album, when rendered as another one. */
  sourceLyricsSetId: z.number().int().nonnegative().optional(),
  artistName: z.string().optional(),
  /** The global parameter snapshot the queue item carries. */
  params,
  /** Browser settings the queue reads today. */
  settings: z.object({
    useLlmDuration: z.boolean().optional(),
    useLmAdapter: z.boolean().optional(),
    triggerUseFilename: z.boolean().optional(),
    triggerPlacement: z.enum(['prepend', 'append', 'replace']).optional(),
    randomizeTimbreRef: z.boolean().optional(),
    /** Create's stored time signature and language, used when params lack them. */
    timeSignature: z.string().optional(),
    vocalLanguage: z.string().optional(),
    /** App settings copied onto the request (coResident, ...). */
    app: params.optional(),
  }).optional(),
  /** Per-song caption choices. */
  mm3Selection: mm3CaptionSelectionSchema.optional(),
  yue2Selection: yue2CaptionSelectionSchema.optional(),
  /** YuE2 model pick captured when the song was queued; omitted = derived
   *  from `yue2Defaults` and the album preset, as the queue does. */
  yue2Pick: z.record(z.string(), z.union([z.string(), z.number()])).optional(),
  /** The YuE2 picker's current defaults (paths and dials). */
  yue2Defaults: params.optional(),
});
export type WrittenSongIntent = z.infer<typeof writtenSongIntentSchema>;

export const resolveIntentSchema = z.discriminatedUnion('kind', [createIntentSchema, writtenSongIntentSchema]);
export type ResolveIntent = z.infer<typeof resolveIntentSchema>;

/** Where each resolved field came from. */
export interface ResolveProvenance {
  engine: string;
  caption: {
    /** 'box' | 'song' | 'song-mm3' | 'song-yue2' | 'dataset-track' | 'album-track'. */
    source: string;
    mode?: CaptionSourceMode;
    fromTrack?: string;
    datasetId?: string;
    renderingAs?: boolean;
  };
  wildcards?: { seed: number; seedFrom: 'dit-seed' | 'random'; caption: boolean; lyrics: boolean };
  duration?: { source: 'request' | 'llm' | 'estimate' | 'fallback' | 'auto'; value: number };
  trigger?: { source: 'compose' | 'preset-filename' | 'cleared' | 'none'; words?: string[] };
  preset?: { lyricsSetId: number; adapterPath?: string; lmAdapterPath?: string; mm3AdapterPath?: string; referenceTrack?: string };
}

export interface ResolvePreviewResponse {
  /** The exact body to POST to /api/generate. */
  request: Record<string, unknown>;
  provenance: ResolveProvenance;
  warnings: string[];
  /** Selection writes the browser makes on this path today; Node does not. */
  uiEffects: Array<{ key: string; value: unknown }>;
  /** sha256 of the canonical request; identifies this resolved version. */
  version: string;
}
