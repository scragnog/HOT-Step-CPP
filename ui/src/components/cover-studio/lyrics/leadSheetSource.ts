import type { CoverLeadSheetSource, CoverLeadSheetResult } from '../../../../../server/src/contracts/studioWorkflows';
import type { CoverDraftResult } from '@/services/coverWorkflowApi';

export const LEAD_SHEET_SOURCES: CoverLeadSheetSource[] = ['sheetsage2', 'sheetsage2-refined', 'muscriptor', 'dettato', 'yourmt3-chordnet'];

export function readLeadSheetSource(value: unknown): CoverLeadSheetSource | undefined {
  return LEAD_SHEET_SOURCES.includes(value as CoverLeadSheetSource) ? value as CoverLeadSheetSource : undefined;
}

/** An unset new draft follows Models until cover-open returns its snapshot. */
export function selectedLeadSheetSource(stored: unknown, readinessDefault?: CoverLeadSheetSource) {
  return readLeadSheetSource(stored) ?? readinessDefault;
}

/** `repair=0` is dettato's repairer switched off, which drops its folder from the readiness check. */
export function leadSheetReadinessPath(source?: CoverLeadSheetSource, dettatoRepair = true) {
  const query = [source ? `source=${encodeURIComponent(source)}` : '', dettatoRepair ? '' : 'repair=0'].filter(Boolean).join('&');
  return `/yue2-cover/readiness${query ? `?${query}` : ''}`;
}

/** Source mutations clear the score; incomplete transcriptions still supply editable ABC. */
export function leadSheetScoreUpdate(result: Pick<CoverDraftResult, 'abc' | 'leadSheet' | 'scoreSource'>) {
  return { abc: result.abc ?? '', approved: null,
    scoreSource: result.leadSheet?.scoreSource ?? result.scoreSource ?? null,
    chordQuality: result.leadSheet?.chordQuality ?? null } satisfies {
      abc: string; approved: null; scoreSource: string | null; chordQuality: CoverLeadSheetResult['chordQuality'] | null;
    };
}
