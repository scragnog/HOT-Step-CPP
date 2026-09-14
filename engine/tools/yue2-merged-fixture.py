"""Fabricate a yue2-probe --ar-parity fixture (02_semantic/) from the HF model on
the CPU, with an AR LoRA merged the way our ENGINE merges it (A, B rounded to
bf16; delta in f32; delta rounded to bf16; base + delta rounded to bf16), so
the probe's logits comparison isolates the engine's forward with merged
weights from everything else.

Two fixture roots are written: <out>/base (no adapter, the control that sets
the noise floor) and <out>/merged (adapter at scale 1).

usage: make_merged_fixture.py <adapter.pt> <song index> <out dir>
"""
import sys, os, json, numpy as np, torch
torch.set_num_threads(30)
from yue2.protocol import SongRequest, token_prefixes, CODEC_OFFSET, MUSIC_END
from yue2.pipeline import YuE2TextTokenizer
from yue2.modeling_yue2 import YuE2ForCausalLM

CK, IDX, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
snap = "K:/yue2/models/YuE2-3B"
tok = YuE2TextTokenizer(snap + "/qwen.tiktoken")
man = json.load(open("D:/Ace-Step-Latest/hot-step-cpp/_experiments/yue2-adapters/greenday_warning/cache/yue2_preprocess.json"))
src = man["sources"][IDX]
style = "albumb, " + src["caption"]
prefix = token_prefixes(SongRequest(style=style, lyrics=src["lyrics"], cot="off", seed=1, id="x"), tok)
codes = np.fromfile("D:/Ace-Step-Latest/hot-step-cpp/_experiments/yue2-adapters/greenday_warning/cache/" + src["codec_ids"], dtype=np.int32)
forced = [int(c) + CODEC_OFFSET for c in codes]
ids = list(prefix) + forced + [MUSIC_END]
P, S = len(prefix), len(ids)
pins = [p for p in [0, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000] if p <= len(codes)] + [len(codes)]
abs_pos = [P - 1 + p for p in pins]
print(f"song {src['name']}: P {P} codec {len(codes)} S {S}; pins {pins} -> abs {abs_pos}", flush=True)

model = YuE2ForCausalLM.from_pretrained(snap, local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
bb = model.model
ck = torch.load(CK, map_location="cpu", weights_only=False)

def bf(x): return x.to(torch.bfloat16).to(torch.float32)
def merge(scale):
    it = iter(ck["lora"])
    with torch.no_grad():
        for layer in bb.layers:
            for mod, names in ((layer.self_attn, ("q_proj","k_proj","v_proj","o_proj")), (layer.mlp, ("gate_proj","up_proj","down_proj"))):
                for n in names:
                    A = bf(next(it).float()); B = bf(next(it).float()); lin = getattr(mod, n)
                    delta = bf(B @ A) * scale
                    lin.weight.copy_(bf(lin.weight + delta))

@torch.no_grad()
def logits_at(rows):
    out = model(input_ids=torch.tensor([ids])).logits[0].float()
    return out[rows].contiguous().numpy().astype(np.float32)

def write(root, L):
    d = os.path.join(root, "02_semantic"); os.makedirs(d, exist_ok=True)
    np.asarray(prefix, dtype=np.int32).tofile(os.path.join(d, "prefix_pos_ids.bin"))
    np.asarray(forced, dtype=np.int32).tofile(os.path.join(d, "forced_semantic_ids.bin"))
    L.tofile(os.path.join(d, "logits_pinned_cond.bin"))
    json.dump({"truncated": False, "cfg_active": False, "activation_pinned_positions": pins,
               "note": "fabricated by make_merged_fixture.py from the HF model on CPU; hidden_pinned_cond.bin absent on purpose"},
              open(os.path.join(d, "manifest.json"), "w"), indent=1)
    print("wrote", d, L.shape, flush=True)

Lb = logits_at(abs_pos); write(os.path.join(OUT, "base"), Lb)
merge(1.0)
Lm = logits_at(abs_pos); write(os.path.join(OUT, "merged"), Lm)
# a quick CE sanity print for both, on the supervised rows at the pins
tg = np.asarray(ids)[np.asarray(abs_pos) + 1]
for name, L in (("base", Lb), ("merged", Lm)):
    z = L - L.max(1, keepdims=True); lp = z - np.log(np.exp(z).sum(1, keepdims=True))
    print(name, "CE at pins:", np.round(-lp[np.arange(len(tg)), tg], 3).tolist(), flush=True)
print("done")
