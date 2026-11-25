"""Run dettato 2.0.0 on one WAV with the repairer bypassed and capture every stage's inputs and outputs.

Usage: python capture.py --upstream <clone> --weights <dir> --audio <wav> --out <dir> --device cpu|cuda
The real pipeline (Dettato.transcribe) runs unchanged, with two differences:
  * the evidence repairer is replaced by a stub that returns the transcriber's own decoded notes (its weights are never used);
    with --repairer the real repairer runs instead and its inputs and outputs are recorded (repair.* tensors, result.repair);
  * observation wrappers record what the stages exchange. They return their results untouched.
cpu = reference (fp32, deterministic algorithms, 4 intra-op threads, 1 inter-op thread); cuda = upstream as shipped (bf16 autocast)."""
import argparse
import time

import numpy as np

import common

ap = argparse.ArgumentParser()
ap.add_argument('--upstream', required=True)
ap.add_argument('--weights', required=True)
ap.add_argument('--audio', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--device', choices=('cpu', 'cuda'), required=True)
ap.add_argument('--repairer', action='store_true', help='run the real evidence repairer and capture its inputs and outputs')
a = ap.parse_args()
torch = common.load_upstream(a.upstream, a.weights, a.device)

from dettato import Dettato                                  # noqa: E402
from dettato.drums import DrumTranscriber                     # noqa: E402
from dettato.grouper import core as gcore, retune             # noqa: E402
from dettato.grouper.core import Grouper                      # noqa: E402
from dettato.transcriber import Transcriber                   # noqa: E402
from dettato.transcriber.core import HEADS                    # noqa: E402

rec = {}
dump = common.Dumps(a.out)


def notes_json(n):
    return {k: np.asarray(v).tolist() for k, v in n.items()}


def spy(obj, name, after):
    """Replace obj.name by a wrapper that calls the original, hands (args, kwargs, result) to `after`, returns the result."""
    real = getattr(obj, name)

    def run(*args, **kw):
        out = real(*args, **kw)
        after(args, kw, out)
        return out
    setattr(obj, name, run)


# ---- transcriber: all-window maps, and the every-other-window sub-run the grouper reads
def after_run(args, kw, out):
    maps, _bufs, (smaps, sbufs) = out
    rec['x_retuned'] = args[0]
    for h in HEADS:
        dump.put(f'maps.{h}', maps[h])
        dump.put(f'sub.maps.{h}', smaps[h])
    for k in ('hidden', 'stem'):
        dump.put(f'sub.tap.{k}', sbufs[k])                  # float16, (T, 88, 368 | 96)
    rec['notes'] = notes_json(tr.notes(maps))
    rec['sub_notes'] = notes_json(tr.notes(smaps))


# ---- grouper
def after_features(args, kw, out):
    meta, feat, fsc = out
    dump.put('grouper.meta', meta)                          # (N, 12) float32
    dump.put('grouper.feat', feat)                          # (N, 3, 368) int8
    dump.put('grouper.fsc', fsc)                            # (N, 3) float16 per-row scales


def after_predict(args, kw, out):
    dump.put('grouper.probs', out['probs'])
    dump.put('grouper.emb', out['emb'])
    dump.put('grouper.h', out['h'])
    wins = out['wins']
    dump.put('grouper.win.t0', np.array([w[0] for w in wins], np.float64))
    dump.put('grouper.win.count', np.array([len(w[1]) for w in wins], np.int64))
    if wins:
        dump.put('grouper.win.sel', np.concatenate([w[1] for w in wins]).astype(np.int64))
        dump.put('grouper.win.emb', np.concatenate([w[2] for w in wins]))     # float16
        dump.put('grouper.win.probs', np.concatenate([w[3] for w in wins]))   # float16
    rec['grouper_window_s'] = float(out['W'])


real_decode = gcore.decode


def decode_spy(*args, **kw):
    out = real_decode(*args, **kw)
    rec['groups'] = [[int(k), np.asarray(ids).astype(int).tolist()] for k, ids in out]
    rec['decode_params'] = dict(kw)
    return out


# ---- drums
def after_probs(args, kw, out):
    dump.put('drums.features', args[0])                     # (T, 280), upstream's float16 rounding
    for k, v in out.items():
        dump.put(f'drums.{k}', np.atleast_1d(np.asarray(v)))
    rec['drum_probs'] = out


def after_drum_decode(args, kw, out):
    on, keys, vel, which = out
    rec['drum_decode'] = dict(onset=on.tolist(), key=keys.tolist(), velocity=np.asarray(vel).tolist(), model=which)


def after_drum_transcribe(args, kw, out):
    dump.put('x44', args[0])
    rec['drum_hits'] = out


def after_ratio(args, kw, out):
    rec['tuning'] = dict(estimate_cents=out[0], strength=out[1], shift_cents=out[2], ratio=out[3])
    dump.put('x16', args[0])


class Bypass:
    """Stands in for the evidence repairer: the transcriber's own notes, nothing dropped or inserted."""

    def from_run(self, transcriber, maps, hidden):
        n = transcriber.notes(maps)
        notes = {'onset': n['onset'], 'offset': n['offset'], 'pitch': n['pitch']}
        return notes, dict(notes), {'transcribed': int(len(n['onset'])), 'dropped': 0, 'inserted': 0}


def repair_spies(rp):
    """Record the evidence repairer: the all-window hidden tap, the transcriber notes and their evidence rows, the sorted
    input notes with the model's edits, and the repaired notes."""
    from dettato.evidence import repair as rmod
    real_efr = rmod.evidence_from_run

    def efr(transcriber, maps, H):
        res, ev = real_efr(transcriber, maps, H)
        dump.put('repair.hidden', H)                        # float16 (T, 88, 368), averaged over every window
        dump.put('repair.ev', ev)                           # (N, 769) float32, row-aligned with the transcriber's notes
        rec['repair_input_notes'] = notes_json(res)
        return res, ev

    rmod.evidence_from_run = efr

    def after_edits(args, kw, out):
        k = args[0]
        pk, on_r, off_r, ins = out
        for f in k.dtype.names:
            dump.put(f'repair.k.{f}', k[f])                 # the sorted input notes (inst, pitch, vel, on, off)
        keep = ~np.isnan(pk)                                # NaN = the note's window passed through unchanged
        dump.put('repair.window_ran', keep)
        dump.put('repair.pk', np.where(keep, pk, 0.0))
        dump.put('repair.on_r', np.where(keep, on_r, 0.0))
        dump.put('repair.off_r', np.where(keep, off_r, 0.0))
        rec['repair_insertions'] = [list(x) for x in ins]   # (onset, end, pitch, score), all proposals

    def after_apply(args, kw, out):
        rec['repair_output'] = {f: np.asarray(v).tolist() for f, v in out.items()}

    spy(rp, 'edits', after_edits)
    spy(rp, 'apply', after_apply)
    return rp


tr = Transcriber.from_pretrained(a.device)
gr = Grouper.from_pretrained(a.device)
dr = DrumTranscriber.from_pretrained(a.device)
if a.repairer:
    from dettato.evidence import EvidenceRepairer
    rep = repair_spies(EvidenceRepairer.from_pretrained(a.device))
else:
    rep = Bypass()
model = Dettato(tr, rep, gr, dr, a.device)

spy(tr, 'run', after_run)
spy(gr, 'features_from_run', after_features)
spy(gr, 'predict', after_predict)
gcore.decode = decode_spy
spy(dr, 'probabilities', after_probs)
spy(dr, 'decode', after_drum_decode)
spy(dr, 'transcribe', after_drum_transcribe)
retune.ratio = (lambda real: lambda x: (lambda out: (after_ratio((x,), {}, out), out)[1])(real(x)))(retune.ratio)

if a.device == 'cuda':
    torch.cuda.synchronize()
t0 = time.time()
res = model.transcribe(a.audio)
if a.device == 'cuda':
    torch.cuda.synchronize()
wall = time.time() - t0

if rec['tuning']['shift_cents']:
    dump.put('x16_retuned', rec['x_retuned'])

# Decode both drum branches: no fixture reaches the orchestral model, so the port needs its expected hits too.
branches = {}
drum_decode = rec.get('drum_decode')          # the pipeline's own decode, before the extra calls below overwrite it
if 'drum_probs' in rec:
    for name, router in (('drum-kit', 0.0), ('orchestral', 1.0)):
        on, keys, vel, _ = dr.decode({**rec['drum_probs'], 'router': router})
        branches[name] = dict(onset=on.tolist(), key=keys.tolist(), velocity=np.asarray(vel).tolist())

result = dict(
    audio=a.audio, device=a.device, upstream_commit=common.UPSTREAM_COMMIT, torch=torch.__version__,
    repairer='on' if a.repairer else 'bypassed', wall_s=round(wall, 2), seconds=res.seconds,
    repair=dict(input_notes=rec.get('repair_input_notes'), insertions=rec.get('repair_insertions'), output=rec.get('repair_output'),
                keep_th=getattr(rep, 'keep_th', None), insert_th=getattr(rep, 'insert_th', None)) if a.repairer else None,
    tuning=rec['tuning'], transcriber_notes=rec['notes'], sub_notes=rec['sub_notes'],
    groups=rec.get('groups', []), decode_params=rec.get('decode_params'), grouper_window_s=rec.get('grouper_window_s'),
    drum_router=rec['drum_probs']['router'] if 'drum_probs' in rec else None,
    drum_decode=drum_decode, drum_branches=branches, drum_hits=rec.get('drum_hits'),
    drum_info=res.info.get('drums'),
    tracks=[dict(name=t.name, program=int(t.program), is_drum=t.is_drum, notes=t.notes.tolist()) for t in res.tracks],
    info={k: v for k, v in res.info.items() if k != 'drums'})
if a.device == 'cuda':
    result['peak_alloc_gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
common.write_json(f'{a.out}/result.json', result)
print(f'{a.out}: {wall:.1f}s, tracks', [(t.name, len(t.notes)) for t in res.tracks], 'transcribed', len(rec['notes']['onset']))
