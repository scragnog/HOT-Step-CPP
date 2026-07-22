#!/usr/bin/env python3
"""End-to-end acceptance gate for the SA3 ONNX export set.

Runs the FULL SDEdit refine (encode -> 8-step Euler -> decode) twice:
  1. Reference: StableAudioModel.generate() in PyTorch (fp32, CUDA, fixed seed,
     stochastic AE paths zeroed to match the exported graphs)
  2. Harness: ONLY the four ONNX graphs (text enc, seconds embedder, SAME enc,
     DiT, SAME dec) + numpy orchestration that mirrors generate()/sample_diffusion.
     This orchestration is the exact spec the C++ engine implements.

Deterministic by construction: sampler_type=euler (no mid-loop RNG; production
default pingpong differs only by a per-step renoise draw), initial noise drawn
once in torch with the same seed/device as the reference.

Repo pure-math helpers (schedule, effective length) are imported rather than
copied — the C++ port reimplements them with unit tests against these.

Runs in the StableAudio3 uv venv:
  cd d:/Ace-Step-Latest/StableAudio3
  uv run --with onnx --with onnxruntime python \
      d:/Ace-Step-Latest/hot-step-cpp/tools/onnx-export/e2e_sa3_ort.py
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torchaudio

sys.path.insert(0, r"d:/Ace-Step-Latest/StableAudio3")

from stable_audio_3.model import StableAudioModel
from stable_audio_3.inference.sampling import build_schedule
from stable_audio_3.data.utils import compute_effective_seq_len_from_conditioning

SR = 44100
DS = 4096                 # latent downsampling ratio
CHUNK_LATENTS = 128       # SAME chunk graphs are traced at this size
CHUNK_SAMPLES = CHUNK_LATENTS * DS
OVERLAP = 32              # latent-frame overlap for tiling (pipeline default)
STEPS = 8
STRENGTH = 0.30
SEED = 1234
DURATION = 30.0
HEADROOM_SEC = 6.0
PROMPT = ("Instrumental punk rock with distorted electric guitars, driving drums "
          "and punchy melodic bass. Clean modern production. Instrumental only, no vocals.")


def zero_stochastic_paths(ae):
    ae.bottleneck.noise_regularize = False
    for m in ae.modules():
        if hasattr(m, "mask_noise"):
            m.mask_noise = 0


def adapt_sample_size(seconds, encoder_chunk_size=32, encoder_stride=16):
    """Mirror of StableAudioModel._adapt_sample_size for the medium config."""
    target = int((seconds + HEADROOM_SEC) * SR)
    target = ((target + DS - 1) // DS) * DS
    align = DS * (encoder_chunk_size // encoder_stride)
    return ((target + align - 1) // align) * align


# --- ONNX tiling (ports of AudioAutoencoder.encode_audio / decode_audio) -----

def chunk_starts_for(total, size, hop):
    starts = list(range(0, total - size + 1, hop))
    if starts[-1] != total - size:
        starts.append(total - size)
    return starts


def ort_encode_tiled(sess, audio):
    """audio: np [1,2,T_samples] (T multiple of DS) -> latents np [1,256,T//DS]."""
    total_latents = audio.shape[-1] // DS
    if total_latents <= CHUNK_LATENTS:
        raise ValueError("clip shorter than one chunk — pad first")
    hop = (CHUNK_LATENTS - OVERLAP) * DS
    starts = chunk_starts_for(audio.shape[-1], CHUNK_SAMPLES, hop)
    out = np.zeros((1, 256, total_latents), dtype=np.float32)
    half = OVERLAP // 2
    n = len(starts)
    for i, s in enumerate(starts):
        chunk = sess.run(None, {"audio": audio[..., s:s + CHUNK_SAMPLES]})[0]
        first, last = i == 0, i == n - 1
        os_ = (total_latents - CHUNK_LATENTS) if last else s // DS
        left = 0 if first else half
        right = CHUNK_LATENTS if last else CHUNK_LATENTS - half
        out[..., os_ + left:os_ + right] = chunk[..., left:right]
    return out


def ort_decode_tiled(sess, latents):
    """latents: np [1,256,L] -> audio np [1,2,L*DS]."""
    total_latents = latents.shape[-1]
    hop = CHUNK_LATENTS - OVERLAP
    starts = chunk_starts_for(total_latents, CHUNK_LATENTS, hop)
    out = np.zeros((1, 2, total_latents * DS), dtype=np.float32)
    half_s = (OVERLAP // 2) * DS
    n = len(starts)
    for i, s in enumerate(starts):
        chunk = sess.run(None, {"latents": latents[..., s:s + CHUNK_LATENTS]})[0]
        first, last = i == 0, i == n - 1
        os_ = (total_latents - CHUNK_LATENTS) * DS if last else s * DS
        left = 0 if first else half_s
        right = CHUNK_SAMPLES if last else CHUNK_SAMPLES - half_s
        out[..., os_ + left:os_ + right] = chunk[..., left:right]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--audio", default=r"d:/Ace-Step-Latest/sa3-refined/last-call-vfw_inst-original.wav")
    ap.add_argument("--zero-noise", action="store_true",
                    help="Zero all noise (harness only, skips PyTorch ref) — C++ validation mode")
    ap.add_argument("--pad-latents", type=int, default=0,
                    help="Override latent length (match C++ SA3_T_BUCKET padding)")
    ap.add_argument("--dump-tokens", action="store_true",
                    help="Print the padded token ids csv + count for the C++ endpoint")
    args = ap.parse_args()

    clip, in_sr = torchaudio.load(args.audio)
    assert in_sr == SR
    clip = clip[:, : int(DURATION * SR)]

    audio_sample_size = adapt_sample_size(DURATION)
    latent_size = audio_sample_size // DS
    if args.pad_latents > 0:
        latent_size = args.pad_latents
        audio_sample_size = latent_size * DS
    conditioning = [{"prompt": PROMPT, "seconds_total": DURATION}]
    print(f"audio_sample_size={audio_sample_size}  latent_size={latent_size}")

    # ---------------- Reference (PyTorch) ----------------
    print("Reference: loading medium fp32...")
    model = StableAudioModel.from_pretrained("medium", model_half=False)
    zero_stochastic_paths(model.model.pretransform.model)
    if args.dump_tokens:
        tok = model.model.conditioner.conditioners["prompt"].tokenizer
        enc = tok([PROMPT], truncation=True, max_length=256, padding="max_length",
                  return_tensors="np")
        ids = enc["input_ids"][0].tolist()
        n_real = int(enc["attention_mask"][0].sum())
        print("TOKENS_CSV=" + ",".join(str(i) for i in ids))
        print(f"N_TOKENS={n_real}")
        return 0
    if not args.zero_noise:
        t0 = time.time()
        ref = model.generate(
            prompt=PROMPT, duration=DURATION, steps=STEPS, cfg_scale=1.0, seed=SEED,
            sample_size=model.model_config["sample_size"],
            init_audio=(SR, clip), init_noise_level=STRENGTH,
            sampler_type="euler",
        )[0].cpu()
        print(f"Reference done ({time.time()-t0:.0f}s)")

    # ---------------- Harness (ONNX only) ----------------
    import onnxruntime as ort
    sess_opt = ort.SessionOptions()
    load = lambda n: ort.InferenceSession(os.path.join(args.onnx_dir, n),
                                          sess_opt, providers=["CPUExecutionProvider"])
    print("Harness: loading ONNX sessions...")
    s_text = load("sa3-text_encoder.onnx")
    s_sec = load("sa3-seconds_embedder.onnx")
    s_enc = load("sa3-same_encoder.onnx")
    s_dit = load("sa3-dit.onnx")
    s_dec = load("sa3-same_decoder.onnx")

    t0 = time.time()
    # Conditioning
    tok = model.model.conditioner.conditioners["prompt"].tokenizer
    enc = tok([PROMPT], truncation=True, max_length=256, padding="max_length",
              return_tensors="np")
    text_emb = s_text.run(None, {"input_ids": enc["input_ids"].astype(np.int64),
                                 "attention_mask": enc["attention_mask"].astype(np.bool_)})[0]
    sec_emb = s_sec.run(None, {"seconds": np.array([DURATION], dtype=np.float32)})[0]
    cross = np.concatenate([text_emb, sec_emb[:, None, :]], axis=1)  # [1,257,768]

    # Init latents: pad clip to adapted size, tiled ONNX encode
    padded = torch.zeros(1, 2, audio_sample_size)
    padded[0, :, : clip.shape[-1]] = clip
    init_latents = ort_encode_tiled(s_enc, padded.numpy().astype(np.float32))

    # Noise: replicate generate() exactly — manual_seed then randn on CUDA
    if args.zero_noise:
        noise = np.zeros((1, 256, latent_size), dtype=np.float32)
    else:
        torch.manual_seed(SEED)
        noise = torch.randn([1, 256, latent_size], device="cuda").cpu().numpy()
    x = init_latents * (1 - STRENGTH) + noise * STRENGTH

    # Schedule + padding mask (repo helpers = same math as reference)
    eff = compute_effective_seq_len_from_conditioning(conditioning, SR, DS, "cpu")
    sigmas = build_schedule(
        steps=STEPS, sigma_max=STRENGTH,
        dist_shift=model.model.sampling_dist_shift,
        effective_seq_len=eff, fallback_seq_len=latent_size,
        include_endpoint=True, device="cpu",
    ).numpy().astype(np.float32).reshape(-1)
    headroom_tokens = int(HEADROOM_SEC * SR / DS)
    valid = min(int(eff.item()) + headroom_tokens, latent_size)
    padding_mask = np.zeros((1, latent_size), dtype=np.bool_)
    padding_mask[:, :valid] = True

    local_add = np.zeros((1, 257, latent_size), dtype=np.float32)  # no inpaint
    glob = sec_emb.astype(np.float32)

    # Euler loop
    for i in range(STEPS):
        t_curr, t_next = sigmas[i], sigmas[i + 1]
        v = s_dit.run(None, {
            "x": x.astype(np.float32),
            "t": np.array([t_curr], dtype=np.float32),
            "cross_attn_cond": cross.astype(np.float32),
            "global_embed": glob,
            "local_add_cond": local_add,
            "padding_mask": padding_mask,
        })[0]
        x = x + (t_next - t_curr) * v
        print(f"  step {i+1}/{STEPS} t={t_curr:.4f}->{t_next:.4f}")

    # Decode + padding zeroing + trim (mirrors sample_diffusion tail + generate)
    audio = ort_decode_tiled(s_dec, x.astype(np.float32))
    audio_mask = np.repeat(padding_mask, DS, axis=-1)[:, : audio.shape[-1]]
    audio = audio * audio_mask[:, None, :]
    audio = np.clip(audio, -1, 1)[0, :, : int(DURATION * SR)]
    print(f"Harness done ({time.time()-t0:.0f}s)")

    if args.zero_noise:
        out_dir = os.path.dirname(args.onnx_dir)
        path = os.path.join(out_dir, "e2e_ort_zeronoise.wav")
        torchaudio.save(path, torch.tensor(audio), SR)
        print(f"Zero-noise harness output -> {path}")
        return 0

    # ---------------- Compare ----------------
    ref_np = ref.numpy()[:, : int(DURATION * SR)]
    n = min(ref_np.shape[-1], audio.shape[-1])
    a, b = ref_np[..., :n].ravel(), audio[..., :n].ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print(f"E2E cosine={cos:.6f}  max_abs_diff={np.abs(a-b).max():.3e}")

    out_dir = os.path.dirname(args.onnx_dir)
    torchaudio.save(os.path.join(out_dir, "e2e_ref.wav"), torch.tensor(ref_np), SR)
    torchaudio.save(os.path.join(out_dir, "e2e_ort.wav"), torch.tensor(audio), SR)
    print("E2E OK" if cos > 0.99 else "E2E FAILED")
    return 0 if cos > 0.99 else 1


if __name__ == "__main__":
    sys.exit(main())
