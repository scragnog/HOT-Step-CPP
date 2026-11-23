import fs from 'node:fs/promises';
import { listYue2TrainLogs } from './datasetProfile.js';

export interface GateHistory { readings: Array<{ ep: number; loss: number }>; steps: Array<{ step: number; ep: number; narMse: number; frozen?: boolean }>; frozenAt?: number }

/** Read only gate records; tolerate an incomplete final line of a live log. */
export function parseGateHistory(logs: string[]): GateHistory {
  const readings = new Map<number, number>();
  const steps = new Map<number, { step: number; ep: number; narMse: number; frozen?: boolean }>();
  let frozenAt: number | undefined;
  let lastJoint: number | undefined;
  let lastFrozen: boolean | undefined;
  for (const log of logs) for (const line of log.split(/\r?\n/)) {
    try {
      const e = JSON.parse(line);
      if (!Number.isFinite(e.step)) continue;
      if (e.stage === 'planner_gate' && Number.isFinite(e.ce)) readings.set(e.step, e.ce);
      if (e.stage === 'joint' && Number.isFinite(e.nar_mse)) steps.set(e.step, { step: e.step, ep: e.step, narMse: e.nar_mse, ...(e.planner_frozen === true ? { frozen: true } : {}) });
      if (e.stage === 'planner_frozen' && e.reason === 'planner_ce') frozenAt = e.step;
      // The trainer's freeze event is stdout-only; joint records carry the
      // frozen flag from the first decoder update onward.
      if (e.stage === 'joint') {
        if (e.planner_frozen === true && lastFrozen === false && frozenAt === undefined && lastJoint === e.step - 1) frozenAt = lastJoint;
        lastJoint = e.step;
        lastFrozen = e.planner_frozen === true;
      }
    } catch { /* malformed or incomplete line */ }
  }
  return { readings: [...readings].sort((a, b) => a[0] - b[0]).map(([ep, loss]) => ({ ep, loss })), steps: [...steps.values()].sort((a, b) => a.step - b.step), frozenAt };
}

export async function readGateHistory(output: string): Promise<GateHistory> {
  const files = listYue2TrainLogs(output);
  const logs = await Promise.all(files.map(({ file }) => fs.readFile(file, 'utf8')));
  return parseGateHistory(logs);
}
