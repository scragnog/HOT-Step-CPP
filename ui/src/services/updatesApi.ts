import { apiClient, type ReadOptions } from './httpClient';
import type {
  UpdatePrepareStatus, UpdateApplyResult, UpdatesApplyResponse, UpdateChannel,
  UpdatesStatusResponse, UpdatesCheckResponse,
} from '../../../server/src/contracts/updates';
export type {
  UpdateApplyResult, UpdateChannel, UpdateInstallStatus, UpdatePrepareStatus,
  UpdateStatusKind, UpdatesApplyResponse,
} from '../../../server/src/contracts/updates';

export type UpdatesStatus = UpdatesStatusResponse;
export type UpdatesCheckResult = UpdatesCheckResponse;

export interface UpdateRestartAttempt {
  from: string | null;
  to: string | null;
  previousResult: UpdateApplyResult | null;
  startedAt: number;
}

export const updatesApi = {
  status: (opts?: ReadOptions) => apiClient.get<UpdatesStatus>('/updates/status', { ...opts, silent: true }),
  check: (channel?: UpdateChannel) => apiClient.post<UpdatesCheckResult>('/updates/check', channel ? { channel } : {}, { silent: true }),
  prepare: (candidateId: string) => apiClient.post<UpdatePrepareStatus>('/updates/prepare', { candidateId }, { silent: true }),
  cancel: (candidateId: string) => apiClient.post<UpdatePrepareStatus>('/updates/prepare/cancel', { candidateId }, { silent: true }),
  apply: (candidateId: string) => apiClient.post<UpdatesApplyResponse>('/updates/apply', { candidateId }, { silent: true }),
  health: (opts?: ReadOptions) => apiClient.get('/health', { ...opts, silent: true, timeout: 4000 }),
};
