"""Conditioning ablation for the MM3 latent refiner (HOTSTEP-1007).  Upstream package untouched; everything is patched from here.
Clips: the ten native renders + the two damaged HOTSTEP-1001 controls (control/degraded.wav = mild, degraded_heavy.wav).
The frozen constants (style C, MERT E) are means over the ten NATIVE clips only, so the two controls test them out of sample.

Conditions (steps 32, 30 s windows, 2 s overlap, sequential):
  A1/A2  real conditioning, run twice from the cache (refiner-only numerical floor)
  Afull  stock pipeline, nothing cached or patched (floor incl. DAV/MERT/CLAP recompute; also proves the cache path == stock)
  B  style <- refiner.style_null            C  style <- mean of real CLAP vectors over all windows of all clips
  D  MERT layers <- refiner.mert_null       E  MERT layers <- per-layer/per-dim mean over all frames of all clips (post-norm)
  F  best of B/C with best of D/E (picked by lowest max ratio over the clips)
  Cloo  extra, not in the contract: C with the clip's own windows left out of the mean (in-sample check; native clips only)

Resumable: every unit of work lands in state/ and is skipped on rerun.  Exits 2 when the time budget ran out
(rerun), 0 when everything is done.   usage: ablate.py [--budget 500]
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch, pyloudnorm as pyln

HERE = Path(__file__).parent
REF = HERE.parent
os.environ.setdefault("HF_HOME", str(REF / "hf-cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(REF))
from minimax_music3_latent_refiner import MiniMaxMusic3RefinerPipeline
from minimax_music3_latent_refiner.model import bridge_sample

NATIVE = REF.parent / "native"
NATIVE_CLIPS = sorted(p.stem for p in NATIVE.glob("*.wav"))
SRC = {c: NATIVE / f"{c}.wav" for c in NATIVE_CLIPS}
SRC["ctl_mild"] = REF / "control/degraded.wav"
SRC["ctl_heavy"] = REF / "control/degraded_heavy.wav"
CLIPS = list(SRC)
assert len(NATIVE_CLIPS) == 10 and all(p.exists() for p in SRC.values())
STATE = HERE / "state"; (STATE / "cache").mkdir(parents=True, exist_ok=True); (STATE / "lat").mkdir(exist_ok=True)
STEPS, WIN, OVL = 32, 30.0, 2.0
T0 = time.time()
BUDGET = float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 500.0
def over(): return time.time() - T0 > BUDGET

def log(*a): print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)

def load_wav(clip):
    data, sr = sf.read(SRC[clip], dtype="float32", always_2d=True)
    assert sr == 44100
    n = len(data); pad = (-n) % 512      # runner's zero-pad (upstream window_starts bug)
    padded = np.concatenate([data, np.zeros((pad, data.shape[1]), np.float32)]) if pad else data
    return data, padded, n

pipe = MiniMaxMusic3RefinerPipeline.from_pretrained(str(REF / "repo"), device="cuda", local_files_only=True)
dev = pipe.device
orig_chunk = pipe._refine_chunk
ref = pipe.refiner
cfg = ref.config

class Ctl:           # patched _refine_chunk state
    mode = "stock"; idx = 0; cache = None; cond = None
ctl = Ctl()

def patched(waveform, steps, audio_encoder):
    if ctl.mode == "stock":
        return orig_chunk(waveform, steps, audio_encoder)
    i = ctl.idx; ctl.idx += 1
    if ctl.mode == "cache":       # real conditioning + identity output (so the blend yields the input latent for free)
        deg_raw = audio_encoder.encode(waveform.to(device=dev, dtype=next(pipe.audio_vae.parameters()).dtype))
        deg_raw = deg_raw.transpose(1, 2).to(dev, pipe.dtype)                    # [1,L,128]
        layers = pipe._mert_layers(waveform, deg_raw.shape[1])
        style = pipe._clap_style(waveform)
        lm, ls, ym, ys = pipe._normalization_tensors()
        ctl.cache.append(dict(deg_raw=deg_raw.cpu(), deg=((deg_raw - lm) / ls).cpu(),
                              layers=((layers - ym) / ys).to(pipe.dtype).cpu(), style=style.cpu()))
        return deg_raw.transpose(1, 2)
    c = ctl.cache[i]
    deg, layers, style = c["deg"], c["layers"], c["style"]
    layers, style = ctl.cond(layers, style)
    lm, ls, _, _ = pipe._normalization_tensors()
    gen = bridge_sample(ref, layers[:, cfg.mert_input_layer], layers, deg, style, steps)
    return (gen * ls + lm).transpose(1, 2)
pipe._refine_chunk = patched

def run_refine(padded):
    ctl.idx = 0
    wav = torch.from_numpy(padded.T.copy())
    return pipe.refine_latents(wav, 44100, steps=STEPS, window_seconds=WIN, overlap_seconds=OVL)

def gpu_cache(c):
    return [{k: v.to(dev) for k, v in ch.items()} for ch in c]

# ---------------------------------------------------------------- stage 1: cache real conditioning
def stage_cache():
    for clip in CLIPS:
        f = STATE / "cache" / f"{clip}.pt"
        if f.exists(): continue
        if over(): return False
        _, padded, n = load_wav(clip)
        ctl.mode, ctl.cache = "cache", []
        t = time.time(); lat, _ = run_refine(padded)
        torch.save(dict(chunks=ctl.cache, input_lat=lat.cpu(), n=n, npad=len(padded)), f)
        log("cached", clip, len(ctl.cache), "windows", f"{time.time() - t:.1f}s")
    return True

def stage_means():
    f = STATE / "means.pt"
    if f.exists(): return torch.load(f)
    styles, lsum, lcnt = [], torch.zeros(13, 768, dtype=torch.float64), 0
    for clip in NATIVE_CLIPS:
        for ch in torch.load(STATE / "cache" / f"{clip}.pt")["chunks"]:
            styles.append(ch["style"][0].double())
            l = ch["layers"][0].double()                 # [13,L,768]
            lsum += l.sum(1); lcnt += l.shape[1]
    m = dict(style=torch.stack(styles).mean(0).float(), layer=(lsum / lcnt).float(), n_windows=len(styles), n_frames=lcnt)
    # how much of the real signal the constant leaves: rms(real-mean)/rms(real)
    sr = torch.stack(styles); m["style_resid_rel"] = float(((sr - sr.mean(0)).pow(2).mean().sqrt()) / sr.pow(2).mean().sqrt())
    rs, rt = 0.0, 0.0
    for clip in NATIVE_CLIPS:
        for ch in torch.load(STATE / "cache" / f"{clip}.pt")["chunks"]:
            l = ch["layers"][0].double(); rs += (l - m["layer"].double()[:, None]).pow(2).sum().item(); rt += l.pow(2).sum().item()
    m["mert_resid_rel"] = math.sqrt(rs / rt)
    torch.save(m, f); return m

# ---------------------------------------------------------------- stage 2/3: conditions
def loo_style(clip):
    sty = []
    for o in NATIVE_CLIPS:
        if o != clip: sty += [ch["style"][0].double() for ch in torch.load(STATE / "cache" / f"{o}.pt")["chunks"]]
    return torch.stack(sty).mean(0).float()

def cond_fn(name, means, fchoice=None):
    sn, mn = ref.style_null.detach(), ref.mert_null.detach()
    def style_of(k, s):
        return {"B": lambda: sn[None].to(s).expand_as(s), "C": lambda: means["style"][None].to(s).expand_as(s), "Cloo": lambda: means["style_loo"][None].to(s).expand_as(s), "real": lambda: s}[k]()
    def layers_of(k, l):
        return {"D": lambda: mn[None, :, None, :].to(l).expand_as(l), "E": lambda: means["layer"][None, :, None, :].to(l).expand_as(l), "real": lambda: l}[k]()
    if name in ("B", "C", "Cloo"): sk, lk = name, "real"
    elif name in ("D", "E"): sk, lk = "real", name
    elif name == "F": sk, lk = fchoice
    else: sk = lk = "real"
    return lambda l, s: (layers_of(lk, l), style_of(sk, s))

def lat_file(cond, clip): return STATE / "lat" / f"{cond}__{clip}.pt"

def run_cond(cond, clip, cache, padded, means, fchoice=None):
    f = lat_file(cond, clip)
    if f.exists(): return
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated() / 2**30
    ctl.cache = cache
    if cond == "Cloo": means = dict(means, style_loo=loo_style(clip))
    if cond == "Afull": ctl.mode = "stock"
    else: ctl.mode, ctl.cond = "run", cond_fn("A" if cond in ("A1", "A2") else cond, means, fchoice)
    t = time.time(); lat, npad = run_refine(padded); torch.cuda.synchronize(); dt = time.time() - t
    torch.save(dict(lat=lat.cpu()[0], npad=npad, time_s=dt, peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30, resident_gib=base), f)
    log(f"{cond:5s} {clip:20s} {dt:5.1f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

def stage_conds(conds, means, fchoice=None):
    for clip in CLIPS:
        pend = [c for c in conds if not lat_file(c, clip).exists() and not (c == "Cloo" and clip not in NATIVE_CLIPS)]
        if not pend: continue
        if over(): return False
        c = torch.load(STATE / "cache" / f"{clip}.pt"); _, padded, _ = load_wav(clip)
        cache = gpu_cache(c["chunks"])
        for cond in pend: run_cond(cond, clip, cache, padded, means, fchoice)
        del cache; torch.cuda.empty_cache()
    return True

# ---------------------------------------------------------------- metrics
def rms(x): return float(x.double().pow(2).mean().sqrt())
def metrics(cond, clip):
    A = torch.load(lat_file("A1", clip))["lat"]; X = torch.load(lat_file(cond, clip))["lat"]
    I = torch.load(STATE / "cache" / f"{clip}.pt")["input_lat"][0]
    d = (X - A).double()
    return dict(cos=float((X.double().flatten() @ A.double().flatten()) / (rms(X) * rms(A) * X.numel())),
                rel_rms=rms(d) / rms(A), max_abs=float(d.abs().max()), refiner_effect=rms(A - I) / rms(I),
                ratio=rms(d) / rms(A - I), A_rms=rms(A))

def pick_f():
    def worst(c): return max(metrics(c, k)["ratio"] for k in CLIPS)
    return (min(("B", "C"), key=worst), min(("D", "E"), key=worst))

# ---------------------------------------------------------------- decode + proxy
def decode_all(conds):
    meter = pyln.Meter(44100)
    for cond in conds:
        d = HERE / "wav" / cond; d.mkdir(parents=True, exist_ok=True)
        for clip in CLIPS:
            out_f = d / f"{clip}.wav"
            if out_f.exists(): continue
            if over(): return False
            data, _, n = load_wav(clip)
            r = torch.load(lat_file(cond, clip))
            audio = pipe.decode_latents(r["lat"][None].to(dev), r["npad"])
            out = audio.squeeze(0).T.numpy()[:n]
            gain = meter.integrated_loudness(data) - meter.integrated_loudness(out)
            out = np.clip(out * 10 ** (gain / 20), -1, 1)
            sf.write(out_f, out, 44100, subtype="PCM_24")
    log("decoded", conds); return True

def proxy_all(conds):
    sys.path.insert(0, str(REF))
    import mert20
    pf = STATE / "proxy.json"; P = json.loads(pf.read_text()) if pf.exists() else {}
    for clip in CLIPS:
        if f"in/{clip}" not in P: P[f"in/{clip}"] = mert20.embed20(str(SRC[clip])).tolist()
        for cond in conds:
            if f"{cond}/{clip}" in P: continue
            if over(): pf.write_text(json.dumps(P)); return False
            P[f"{cond}/{clip}"] = mert20.embed20(str(HERE / "wav" / cond / f"{clip}.wav")).tolist()
    pf.write_text(json.dumps(P)); return True

def main():
    if not stage_cache(): sys.exit(2)
    means = stage_means()
    if not stage_conds(["A1", "A2", "Afull", "B", "C", "D", "E", "Cloo"], means): sys.exit(2)
    fch = pick_f(); (STATE / "fchoice.json").write_text(json.dumps(fch)); log("F =", fch)
    if not stage_conds(["F"], means, fch): sys.exit(2)
    ctl.mode = "stock"
    if not decode_all(["A1", "B", "C", "D", "E", "F"]): sys.exit(2)
    if not proxy_all(["A1", "B", "C", "D", "E", "F"]): sys.exit(2)
    log("ALL DONE")

if __name__ == "__main__":
    main()
