export interface CaptionReplaceRequest {
  sampleIds: string[];
  find: string;
  replacement: string;
  /** Omitted for preview. Apply requires the exact preview fingerprint. */
  previewToken?: string;
}

export interface CaptionReplacePreview {
  previewToken: string;
  tracks: number;
  files: Array<{ sampleId: string; file: string }>;
}
