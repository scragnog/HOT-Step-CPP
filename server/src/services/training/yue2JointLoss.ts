// The one place a joint step's reported loss is composed.
//
// Two readers need the same number from the same train.jsonl line: the live
// relay that feeds the chart, the log and the ladder milestones
// (yue2JointTrainRunner.parseYue2JointEvent), and the on-disk catalogue that
// labels checkpoints for the resume list (yue2AitkRuns.checkpointRecords).
// They used to duplicate the formula, and both required all three terms —
// which is why a frozen planner stopped every label dead at its last value.

/** The KL weight the trainer itself uses for its `--target-loss` composite
 *  (`yue2-aitk-runtime.cpp`, the `composite` there is ar_ce + kl_weight *
 *  ar_kl + nar_mse + cursor). Not a display constant: the number shown has to
 *  be the number the stop compares against. */
export const YUE2_JOINT_KL_WEIGHT = 0.2;

/** The raw numbers of one `stage:"joint"` JSONL line. */
export interface Yue2JointStepNumbers {
  ar_ce?: unknown;
  ar_kl?: unknown;
  nar_mse?: unknown;
  cursor_ce?: unknown;
  cursor_weight?: unknown;
}

export interface Yue2JointStepLoss {
  loss: number;
  /** True when the planner is frozen: `loss` is the decoder's flow loss
   *  alone, not the planner+decoder composite the earlier steps report. */
  decoderOnly: boolean;
}

/** A joint step's loss, or undefined when the step reported nothing usable.
 *
 *  A frozen planner has no objective left — the trainer skips its forward, its
 *  CE/KL and its backward, and says so by omitting `ar_ce`, `ar_kl` and the
 *  cursor terms (`yue2-aitk-runtime.cpp`, the `planner_frozen` branch). The
 *  decoder keeps training and still emits `nar_mse`. Treating the missing
 *  planner terms as "no measurement" is what froze every label built from
 *  these steps at the last planner value; the decoder's own loss is the honest
 *  number for a decoder step, and it is flagged so the labels can say which
 *  objective they are showing. */
export function yue2JointStepLoss(event: Yue2JointStepNumbers): Yue2JointStepLoss | undefined {
  const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
  if (!finite(event.nar_mse)) return undefined;
  if (finite(event.ar_ce) && finite(event.ar_kl)) {
    const cursor = finite(event.cursor_ce) && finite(event.cursor_weight)
      ? event.cursor_ce * event.cursor_weight : 0;
    return { loss: event.ar_ce + YUE2_JOINT_KL_WEIGHT * event.ar_kl + event.nar_mse + cursor, decoderOnly: false };
  }
  return { loss: event.nar_mse, decoderOnly: true };
}
