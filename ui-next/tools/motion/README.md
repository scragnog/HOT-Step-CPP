# Motion

Checks how the page moves, where `tools/visual-baseline` checks how it looks at rest and `tools/behaviour` what it
does (`scenarios/38-motion.mjs` reads the page at chosen moments of each motion). Needs Playwright with Chromium
(`PLAYWRIGHT_MODULE`, as the other tools) and the test build (built first when it is missing or older than the
sources).

```
npm run motion                                        # every motion against its approved samples; exit 1 on a difference
node tools/motion/frames.mjs --only "^tier-"          # some motions, by id
node tools/motion/frames.mjs --list                   # the motions, and which have approved samples
node tools/motion/frames.mjs --shots                  # also a screenshot of every step, in out/<motion>/
node tools/motion/frames.mjs --workers 3              # motions at once (default: half the machine's threads, 6 on 12)
npm run motion:approve -- --only "^tile-leaves$"      # record the current build's samples as the approved ones
node tools/motion/cost.mjs                            # what each motion costs the main thread (a measure, not a check)
```

`motions.mjs` lists the motions: the tier change in each direction and in the dark theme, a tile arriving, leaving,
coming back by Undo and moved by a key, a pedal joining and leaving the board, the same under reduced motion, the
prompt box growing and shrinking, a suggestion and Song info opening, Generate starting, Quick's fold opening and
folding. Each sets a scene through the test build's compatibility layer, sets the motion off as a user does (a click,
a key, typing), and names the boxes that move: a selector (`|attr` names each match by an attribute) or `vt:<name>`, a
View Transition part.

`frames.mjs` holds the page's animation timeline still (DevTools `Animation.setPlaybackRate(0)`), sets the motion off,
then moves every running animation by hand to 0, 16, 32 ... ms of its run and reads, at each step, each box's place
and size on screen (a running transform included) and, where the motion fades, its opacity; for a View Transition part,
its group's box and its old and new images' opacity. Each step is compared with the motion's approved samples
(`approved/<motion>.json`, committed: one line per step, 336 KB for all twenty) within **0.5 px** and **0.01 of
opacity** (`TOL`, `OTOL`; the boxes are read to a tenth of a pixel, and two runs of one build agree to that, also six
motions at once). A key the approved samples have and the page lacks, or the other way round, is a difference. A
motion in which nothing moved fails too (the motion was not set off), but for one marked `still`.

**After a deliberate change to a motion** (a duration, an easing, what moves): run `npm run motion`, read the keys it
names, look at the steps (`--shots`), then approve that motion alone (`npm run motion:approve -- --only "^<id>$"`) and
commit its `approved/<id>.json` with the change. A new motion in `motions.mjs` fails until it is approved.

The motions were last compared with the concept page's at the tag `rebuild-parity`; since then the approved samples are
the reference.
