"""A-2: frame-aligned audio attributes at the planner LM's 5 Hz, from the SOURCE MIX.

For every song that has ground-truth codes in a 600 s variant, load the original audio
(path from the preprocessed safetensors header), compute librosa features at a fine hop
(512 @ 22050 Hz, ~43 fps) and pool them to 200 ms bins so bin t lines up with code t.

Output: <variant>/lm-attr/<stem>.npz with
    X      [T, F] float32 pooled features
    names  [F]    feature names
    n_codes, n_audio_frames, sr, hop, audio_path, key_index, key_mode

Nothing here renders anything or touches a DiT/VAE/adapter.

    python tools/lm-attr-probe/extract_attrs.py [--workers 8] [--limit N] [--force]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FRAME_S, Variant, attr_path_for, audio_path_for, list_variants, load_codes  # noqa: E402

SR = 22050
HOP = 512
N_FFT = 2048

# Krumhansl-Schmuckler key profiles (major, minor), C-relative.
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _pool(x: np.ndarray, frame_times: np.ndarray, n_bins: int, reduce: str = "mean") -> np.ndarray:
    """x: [F, N] fine-hop features with frame centre times -> [n_bins, F] pooled to 200 ms bins."""
    idx = np.minimum((frame_times / FRAME_S).astype(np.int64), n_bins - 1)
    F = x.shape[0]
    out = np.zeros((n_bins, F), dtype=np.float64)
    cnt = np.bincount(idx, minlength=n_bins).astype(np.float64)
    if reduce == "mean":
        for f in range(F):
            out[:, f] = np.bincount(idx, weights=x[f], minlength=n_bins)
        out /= np.maximum(cnt, 1.0)[:, None]
    elif reduce == "max":
        for f in range(F):
            np.maximum.at(out[:, f], idx, x[f])
    return out.astype(np.float32)


def _estimate_key(chroma_mean: np.ndarray) -> tuple[int, str]:
    best = (-2.0, 0, "major")
    for mode, prof in (("major", _KS_MAJOR), ("minor", _KS_MINOR)):
        for k in range(12):
            r = np.corrcoef(np.roll(prof, k), chroma_mean)[0, 1]
            if r > best[0]:
                best = (r, k, mode)
    return best[1], best[2]


def extract_one(args: tuple[str, str, int, str, bool]) -> tuple[str, str, float]:
    """Returns (stem, status, seconds)."""
    audio_path, out_path, n_codes, stem, force = args
    t0 = time.time()
    out = Path(out_path)
    if out.exists() and not force:
        return stem, "skip", 0.0
    try:
        import librosa

        y, sr = librosa.load(audio_path, sr=SR, mono=True)
        if y.size < SR:
            return stem, "too-short", time.time() - t0
        n_audio = int(math.ceil(y.size / sr / FRAME_S))
        n_bins = n_audio

        S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
        n = S.shape[1]
        times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=HOP)
        mel = librosa.feature.melspectrogram(S=S ** 2, sr=sr, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=1.0)

        feats: dict[str, np.ndarray] = {}
        rms = librosa.feature.rms(S=S, frame_length=N_FFT, hop_length=HOP)[0]
        feats["rms_db"] = 20.0 * np.log10(np.maximum(rms, 1e-6))[None, :]
        feats["rms_delta"] = np.gradient(feats["rms_db"][0])[None, :]
        onset_env = librosa.onset.onset_strength(S=librosa.power_to_db(mel, ref=np.max), sr=sr, hop_length=HOP)
        feats["onset_strength"] = onset_env[None, :n]
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP, units="frames")
        onset_hits = np.zeros(n, dtype=np.float32)
        onset_hits[np.clip(onsets, 0, n - 1)] = 1.0
        feats["onset_count"] = onset_hits[None, :]  # summed per bin below
        feats["spec_centroid"] = librosa.feature.spectral_centroid(S=S, sr=sr)
        feats["spec_bandwidth"] = librosa.feature.spectral_bandwidth(S=S, sr=sr)
        feats["spec_flatness"] = librosa.feature.spectral_flatness(S=S)
        feats["spec_rolloff"] = librosa.feature.spectral_rolloff(S=S, sr=sr)
        feats["spec_flux"] = np.concatenate([[0.0], np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))])[None, :]
        feats["zcr"] = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[:, :n]
        mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=13)
        for i in range(13):
            feats[f"mfcc{i}"] = mfcc[i : i + 1]
        # harmonic / percussive energy split (a cheap "melodic vs rhythmic" proxy)
        H, P = librosa.decompose.hpss(S)
        eh = np.sum(H ** 2, axis=0)
        ep = np.sum(P ** 2, axis=0)
        feats["hp_ratio_db"] = (10.0 * np.log10((eh + 1e-9) / (ep + 1e-9)))[None, :]
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
        chroma = chroma[:, :n] if chroma.shape[1] >= n else np.pad(chroma, ((0, 0), (0, n - chroma.shape[1])))
        chroma_n = chroma / np.maximum(chroma.sum(axis=0, keepdims=True), 1e-9)
        key_idx, key_mode = _estimate_key(chroma.mean(axis=1))
        chroma_rel = np.roll(chroma_n, -key_idx, axis=0)  # tonic at row 0
        for i in range(12):
            feats[f"chroma{i}"] = chroma_n[i : i + 1]
        for i in range(12):
            feats[f"chroma_rel{i}"] = chroma_rel[i : i + 1]
        ent = -np.sum(chroma_n * np.log(np.maximum(chroma_n, 1e-9)), axis=0)
        feats["chroma_entropy"] = ent[None, :]
        # tonal centroid (tonnetz) for a compact harmony descriptor
        tonnetz = librosa.feature.tonnetz(chroma=chroma_n, sr=sr)
        for i in range(6):
            feats[f"tonnetz{i}"] = tonnetz[i : i + 1]

        names = list(feats.keys())
        stack = np.concatenate([feats[k][:, :n] for k in names], axis=0)
        X = _pool(stack, times, n_bins, "mean")
        # onset_count is a count, not a mean
        oc = names.index("onset_count")
        X[:, oc] = _pool(feats["onset_count"][:, :n], times, n_bins, "mean")[:, 0] * (n / n_bins)

        # slow-varying context features computed at 5 Hz: 2 s crest and local tempo stability
        rms5 = X[:, names.index("rms_db")]
        k = 10
        pad = np.pad(rms5, (k // 2, k - k // 2 - 1), mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(pad, k)
        crest = win.max(axis=1) - win.mean(axis=1)
        X = np.concatenate([X, crest[:, None].astype(np.float32)], axis=1)
        names.append("crest_2s")

        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out, X=X.astype(np.float32), names=np.array(names), n_codes=n_codes, n_audio_frames=n_audio,
            sr=sr, hop=HOP, audio_path=str(audio_path), key_index=key_idx, key_mode=key_mode,
        )
        return stem, "ok", time.time() - t0
    except Exception as e:  # noqa: BLE001
        return stem, f"fail: {e!r} {traceback.format_exc()[-300:]}", time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--slug", default="", help="only this artist slug")
    ap.add_argument("--renders", action="store_true",
                    help="instead of source tracks, process every <variant>/lm-attr/renders/*.wav "
                         "(server/scripts/lm-plan-render.ts output); npz lands beside each wav")
    a = ap.parse_args()

    jobs: list[tuple[str, str, int, str, bool]] = []
    missing = 0
    for v in list_variants():
        if a.slug and v.slug != a.slug:
            continue
        if a.renders:
            rd = v.attr_dir / "renders"
            for wav in sorted(rd.glob("*.wav")) if rd.is_dir() else []:
                jobs.append((str(wav), str(wav.with_suffix(".npz")), 0, f"{v.slug}/renders/{wav.stem}", a.force))
            continue
        for r in load_codes(v):
            ap_ = audio_path_for(v, r)
            if ap_ is None:
                missing += 1
                continue
            jobs.append((str(ap_), str(attr_path_for(v, r)), int(r.codes.size), f"{v.slug}/{Path(r.file).stem}", a.force))
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"{len(jobs)} songs to process ({missing} without reachable audio)", flush=True)

    t0 = time.time()
    done = ok = skip = fail = 0
    with Pool(a.workers) as pool:
        for stem, status, secs in pool.imap_unordered(extract_one, jobs, chunksize=1):
            done += 1
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                print(f"  {stem}: {status}", flush=True)
            if done % 25 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"[{done}/{len(jobs)}] ok={ok} skip={skip} fail={fail}  {el:.0f}s elapsed, "
                      f"~{el / done * (len(jobs) - done):.0f}s left", flush=True)
    print(f"done: ok={ok} skip={skip} fail={fail} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
