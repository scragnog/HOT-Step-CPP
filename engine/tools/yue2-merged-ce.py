"""Teacher-forced CE of a training song with our AR adapter merged the way
upstream's ar_generate.py merges it (W += scale * B @ A), on the CPU here.

Why: Rob hears our step-800 adapter as WEAK through upstream's Python pipeline
(arms 42, 48) and STRONG through our engine (40, 51). The trainer says artist
CE 0.313 at step 800 (mean over the first six songs). If the Python merge is
complete, CE at scale 1 lands near that; if it lands near the base (~2-5) the
Python path under-applies the adapter; a scale sweep says which multiplier
our engine corresponds to.

usage: merged_ce_cpu.py <adapter.pt> <song index 0..11> [scales...]
"""
import sys, json, os, time, numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(30)
from yue2.protocol import SongRequest, token_prefixes, CODEC_OFFSET, MUSIC_END
from yue2.pipeline import YuE2TextTokenizer
from yue2.modeling_yue2 import YuE2ForCausalLM

CK, IDX = sys.argv[1], int(sys.argv[2]); SCALES = [float(s) for s in sys.argv[3:]] or [0.0, 1.0, 2.0]
snap = "K:/yue2/models/YuE2-3B"
tok = YuE2TextTokenizer(snap + "/qwen.tiktoken")
man = json.load(open("D:/Ace-Step-Latest/hot-step-cpp/_experiments/yue2-adapters/greenday_warning/cache/yue2_preprocess.json"))
src = man["sources"][IDX]
style = "albumb, " + src["caption"]                      # OUR trainer's template (yue2_at_style)
prefix = token_prefixes(SongRequest(style=style, lyrics=src["lyrics"], cot="off", seed=1, id="x"), tok)
codes = np.fromfile("D:/Ace-Step-Latest/hot-step-cpp/_experiments/yue2-adapters/greenday_warning/cache/" + src["codec_ids"], dtype=np.int32)
ids = list(prefix) + [int(c) + CODEC_OFFSET for c in codes] + [MUSIC_END]
P = len(prefix); S = len(ids)
print(f"song {src['name']}: prefix {P} + {len(codes)} codec + END = {S} tokens (trainer's S for this song should match)", flush=True)

t0 = time.time()
model = YuE2ForCausalLM.from_pretrained(snap, local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
bb = model.model
print(f"model loaded on cpu in {time.time()-t0:.0f}s", flush=True)
ck = torch.load(CK, map_location="cpu", weights_only=False)

def deltas():
    it = iter(ck["lora"]); out = []
    for layer in bb.layers:
        for mod, names in ((layer.self_attn, ("q_proj","k_proj","v_proj","o_proj")), (layer.mlp, ("gate_proj","up_proj","down_proj"))):
            for n in names:
                A = next(it).float(); B = next(it).float(); out.append((getattr(mod, n), B @ A))
    return out
D = deltas()
base_w = [lin.weight.detach().clone() for lin, _ in D]

@torch.no_grad()
def ce():
    x = torch.tensor([ids]); out = model(input_ids=x).logits[0].float()
    h = out[P-1:-1]; tgt = x[0, P:]
    return F.cross_entropy(h, tgt, reduction="mean").item()

for s in SCALES:
    with torch.no_grad():
        for (lin, d), w0 in zip(D, base_w):
            lin.weight.copy_(w0 + s * d)
    t0 = time.time(); v = ce()
    print(f"scale {s:4.2f}: artist CE {v:.4f}   ({time.time()-t0:.0f}s)", flush=True)
print("done")
