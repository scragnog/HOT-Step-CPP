import type { CoverLeadSheetSource, CoverLeadSheetResult } from '../../../../server/src/contracts/studioWorkflows';
import type { CoverDraftResult } from '../../services/coverWorkflowApi';

export const LEAD_SHEET_SOURCES: CoverLeadSheetSource[] = ['sheetsage2', 'sheetsage2-refined', 'muscriptor', 'yourmt3-chordnet'];

export function readLeadSheetSource(value: unknown): CoverLeadSheetSource | undefined {
  return LEAD_SHEET_SOURCES.includes(value as CoverLeadSheetSource) ? value as CoverLeadSheetSource : undefined;
}

/** An unset new draft follows Models until cover-open returns its snapshot. */
export function selectedLeadSheetSource(stored: unknown, readinessDefault?: CoverLeadSheetSource) {
  return readLeadSheetSource(stored) ?? readinessDefault;
}

export function leadSheetReadinessPath(source?: CoverLeadSheetSource) {
  return `/yue2-cover/readiness${source ? `?source=${encodeURIComponent(source)}` : ''}`;
}

/** Source mutations clear the score; incomplete transcriptions still supply editable ABC. */
export function leadSheetScoreUpdate(result: Pick<CoverDraftResult, 'abc' | 'leadSheet' | 'scoreSource'>) {
  return { abc: result.abc ?? '', approved: null,
    scoreSource: result.leadSheet?.scoreSource ?? result.scoreSource ?? null,
    chordQuality: result.leadSheet?.chordQuality ?? null } satisfies {
      abc: string; approved: null; scoreSource: string | null; chordQuality: CoverLeadSheetResult['chordQuality'] | null;
    };
}
