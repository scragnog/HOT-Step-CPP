# Batch 1 generation intent replay

`batch1-intent-golden.json` contains all 31 representative request bodies from
`server/data/dev-captures/generate/INDEX_BATCH1_2026_10_07.json` at base
`2c364862`. The original capture directory is ignored by Git. Free text,
model and adapter paths, reference paths, and filename trigger words were
replaced with stable fixture values; numeric and boolean wire values were kept.

Each case has the captured legacy body (`expected`), a compact intent rebuilt
from the request's control values, and the captured backend extension keys.
The original browser control state was not recorded, so these reconstructed
intents establish Node wire parity for those values; they do not establish
that a future UI migration sends the same state.

The comparison includes caption and content fields, with their sanitized
captured values supplied through `input`. No body fields are excluded. The
written-song queue applies album preset overrides after assembling global
parameters. Its `masteringReference`, `timbreReference`, `triggerWord`,
`triggerWords`, and `triggerPlacement` enter `input` for that reason. It can
also retain earlier `triggerSpecs`, `adapterSectionAlignAt`, and
`adapterSectionIsolation` after replacing the global adapter stack with a
single preset adapter. Those observed values enter `input` and remain in the
full body comparison. See `ui/src/stores/audioGenQueueStore.ts` near the
adapter and reference override blocks.
