"""HOTSTEP-1007: metrics table from state/ (run after ablate.py).  Writes results.json and results.md; prints the gate verdict.
cosine / rel_rms / max_abs of condition X vs A1 on the raw refined latent [128, T]; ratio = rms(X-A1) / rms(A1-input);
refiner effect = rms(A1-input) / rms(input).  Gate: every clip cosine >= 0.999 and ratio <= 0.1.  Floor = A2 vs A1, Afull vs A1.
Proxy (advisory only): 1 - cos(MERT-v2 L20 embedding of decoded X, of decoded A1)."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.argv = [sys.argv[0]]
HERE = Path(__file__).parent; STATE = HERE / "state"
CLIPS = sorted(p.stem.split("__")[1] for p in (STATE / "lat").glob("A1__*.pt"))
NATIVE = [c for c in CLIPS if not c.startswith("ctl_")]
CONDS = ["A2", "Afull", "B", "C", "D", "E", "F", "Cloo"]
rms = lambda x: float(x.double().pow(2).mean().sqrt())
lat = lambda c, k: torch.load(STATE / "lat" / f"{c}__{k}.pt")["lat"]
R = {}
for k in CLIPS:
    A = lat("A1", k); I = torch.load(STATE / "cache" / f"{k}.pt")["input_lat"][0]
    for c in CONDS:
        if c == "Cloo" and k not in NATIVE: continue
        X = lat(c, k); d = (X - A).double()
        R.setdefault(c, {})[k] = dict(cos=float((X.double().flatten() @ A.double().flatten()) / (X.double().norm() * A.double().norm())),
            rel_rms=rms(d) / rms(A), max_abs=float(d.abs().max()), ratio=rms(d) / rms(A - I))
    R.setdefault("_effect", {})[k] = dict(effect_rel=rms(A - I) / rms(I), effect_rms=rms(A - I), A_rms=rms(A), A_max=float(A.abs().max()))
P = json.loads((STATE / "proxy.json").read_text())
cs = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
for c in ["B", "C", "D", "E", "F"]:
    for k in CLIPS: R[c][k]["proxy_delta"] = 1 - cs(np.array(P[f"{c}/{k}"]), np.array(P[f"A1/{k}"]))
for k in CLIPS: R["_effect"][k]["proxy_A1_vs_input"] = 1 - cs(np.array(P[f"A1/{k}"]), np.array(P[f"in/{k}"]))
fch = json.loads((STATE / "fchoice.json").read_text())
means = torch.load(STATE / "means.pt")
R["_meta"] = dict(F=fch, n_windows=means["n_windows"], n_frames=means["n_frames"], style_resid_rel=means["style_resid_rel"], mert_resid_rel=means["mert_resid_rel"])
(HERE / "results.json").write_text(json.dumps(R, indent=1))
L = []
def tab(c, keys):
    L.append(f"\n### {c}\n\n| clip | cosine | rel RMS | max-abs | ratio | MERT-20 proxy |\n|---|---|---|---|---|---|")
    for k in keys:
        r = R[c][k]; L.append(f"| {k} | {r['cos']:.6f} | {r['rel_rms']:.4f} | {r['max_abs']:.4f} | {r['ratio']:.4f} | {r.get('proxy_delta', float('nan')):.5f} |")
for c in CONDS: tab(c, CLIPS if c != "Cloo" else NATIVE)
L.append("\n### Refiner effect (A1 vs input)\n\n| clip | rms(A1-input)/rms(input) | rms(A1-input) | max abs latent | MERT-20 1-cos(A1,input) |\n|---|---|---|---|---|")
for k in CLIPS: e = R["_effect"][k]; L.append(f"| {k} | {e['effect_rel']:.4f} | {e['effect_rms']:.4f} | {e['A_max']:.2f} | {e['proxy_A1_vs_input']:.5f} |")
L.append("\n### Gate (cosine >= 0.999 and ratio <= 0.1 on every clip)\n\n| cond | min cosine | max ratio | worst clip | pass all 12 | pass 10 native |\n|---|---|---|---|---|---|")
for c in ["B", "C", "D", "E", "F", "Cloo", "A2", "Afull"]:
    ks = list(R[c]); w = max(ks, key=lambda k: R[c][k]["ratio"])
    ok = lambda kk: all(R[c][k]["cos"] >= 0.999 and R[c][k]["ratio"] <= 0.1 for k in kk)
    L.append(f"| {c} | {min(R[c][k]['cos'] for k in ks):.6f} | {R[c][w]['ratio']:.4f} | {w} | {ok(ks)} | {ok([k for k in ks if k in NATIVE])} |")
(HERE / "results.md").write_text("\n".join(L) + "\n")
print("\n".join(L[-12:])); print(R["_meta"])
