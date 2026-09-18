# Stopping rules and resuming runs

> Reference for the `mm3-lm-adapter-training` skill. Read only when the task needs it.

## Target loss as a stopping rule (2026-08-26) — available, and a trap here

`ace-train mm3-lm-train` now takes `--target-loss <f>`, with
`--target-loss-metric train|eval` and `--target-loss-epochs N` (default 5), and
the Training Studio exposes it as **Train until: Target loss**. `--steps` stays
the hard cap in that mode, so a target that never arrives still ends the run.

**Defaults as of 2026-09-05: target loss 1.0 on the training metric, cap 1000
steps.** It was 0.1 from 2026-08-27 until a blind depth ladder on albumA
(3 songs, base + checkpoints 100/250/500) showed the step-250 checkpoint
(5-epoch mean ~1.06) beating step 500 (0.25 saved, 0.08 best) on every song,
with the step-500 checkpoint emitting an empty plan on one of them. Below ~1
the planner memorises and its plans degenerate (early EOS, droning intros),
exactly as the AS1.5 planner does. Measured trajectories at the shipped recipe
(5-epoch mean):

| step | 25 | 100 | 250 | 500 |
|---|---|---|---|---|
| albumD | 3.14 | 2.50 | 1.43 | 0.59 |
| albumE | 3.28 | 2.23 | 0.92 | 0.31 |
| limbizkit_starfish | 3.62 | 2.72 | 2.03 | - |

So 0.1 binds past step 500 on albums that converge fast and never arrives on the
slow ones. The cap is doing real work, not decoration.

Read the section above before reaching for it. **For an album clone, loss is not
the quantity you want to minimise**: held-out CE bottoms out 1-8x earlier than
the checkpoint that sounds right, and a training loss under ~0.05 was pure
sequence memorisation on the runs that got there. A target-loss run is therefore
a way to say "stop wasting GPU once it plateaus", not a way to pick a
checkpoint - the ladder still decides that.

Where it earns its place: an unfamiliar album where you do not yet know the step
count, run it with a generous cap and a target read off a previous album's
curve, then audition the ladder as usual.

**The window is whole PASSES, not a step count, and the difference is not
cosmetic.** One step here is one crop of one song and swings further than a
whole run's improvement, so the target has to be checked against an average -
that much is obvious. What is less obvious: a 25-step window on an 11-song album
covers two passes plus three songs, so three tracks weigh triple and eight weigh
double, and the window then rises and falls with WHICH songs it caught, a
sawtooth tied to the phase of the pass. A whole number of passes counts every
song identically. Same reasoning as the DiT trainer's ma5, which is the same
decision on the same kind of curve.

The consequence is that the window's LENGTH IN STEPS follows the dataset: 5
epochs is ~55 steps on an 11-song album and 225 on a 45-song one. That is
correct - it is 5 passes either way - but it means a small dataset can stop
sooner in wall-clock than a large one at the same target.

The epoch history rides in the resume state (format v3). It has to: previews
pause the trainer every 50 steps by default, which closes only ~4 epochs per
segment on a 10-song album, so a window that restarted empty each segment would
never fill and the target would silently never fire. Pre-v3 state files load
fine and simply refill the window over the next few epochs.

The `eval` metric only fires on evaluation steps, so at the default
`--eval-every 250` it can fire once in a 250-step run - lower it first or the
target is decorative.

**The LR schedule does not shorten with the run.** The cosine is laid out over
`--steps`, so a run that stops at 300 of a 1000-step cap stops with the learning
rate still high. Two runs that both ended at step 300 - one capped there, one
stopped there by a target - are not the same run.

## Continuing a finished or halted run (2026-08-26)

Every run directory holds `resume-state.bin` (weights + optimizer momentum + RNG
+ the shuffled pass) and, as of this change, `resume-state.json` describing it
and `hotstep-run.json` recording the recipe. Training Studio -> the dataset ->
Train now lists previous runs with a **Continue** control: pick one, say how many
more steps, and it carries on in the same directory under the same adapter name.

Two things to know before using it:

* **The engine now saves state on a clean exit, not only at preview pauses.**
  Runs finished BEFORE this change hold state from their last preview pause, so
  continuing one retrains the steps between that pause and where it stopped -
  50 of them at the usual `--save-every 50` cadence. The UI says how many.
* **Raising the cap restarts the tail of the cosine.** Continuing a 250-step run
  to 500 does not extend the old schedule; it lays a 500-step schedule over a
  run that is already 250 steps in, so the LR jumps back up. That is a real
  difference from having asked for 500 in the first place, and it is why a
  continued run is not interchangeable with a longer one.

The engine refuses a resume whose rank, alpha, tensor count, optimizer or
train/held-out split differs from the state file, so a mismatched continuation
fails loudly rather than training on a half-restored adapter. Prodigy resumes
too (state format v2), and its `x0` lives once in the run directory.
