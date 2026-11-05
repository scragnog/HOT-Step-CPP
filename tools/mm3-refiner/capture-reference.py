"""Capture the pinned torch pipeline at every engine parity boundary, without editing upstream."""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import pyloudnorm as pyln

from convert import HERE, ROOT, contract, sha256, verify_sources


class Capture:
    def __init__(self, directory):
        self.directory = directory
        self.entries = []

    def save(self, name, value, stage):
        path = self.directory / (name + ".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        value = np.ascontiguousarray(value)
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite fixture: {name}")
        np.save(path, value, allow_pickle=False)
        self.entries.append(dict(path=path.relative_to(self.directory).as_posix(), stage=stage,
                                 shape=list(value.shape), dtype=str(value.dtype), sha256=sha256(path)))

    def finish(self, metadata):
        record = dict(**metadata, tensors=self.entries)
        (self.directory / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        reload_fixture(self.directory)


def reload_fixture(directory):
    record = json.loads((directory / "manifest.json").read_text())
    if record["contract_sha256"] != sha256(HERE / "fixtures.json"):
        raise ValueError(f"stale contract: {directory}")
    for entry in record["tensors"]:
        path = directory / entry["path"]
        value = np.load(path, allow_pickle=False, mmap_mode="r")
        if (list(value.shape) != entry["shape"] or str(value.dtype) != entry["dtype"] or
                not np.isfinite(value).all() or sha256(path) != entry["sha256"]):
            raise ValueError(f"fixture reload failed: {path}")
    for entry in record["wav"]:
        path = directory / entry["path"]
        audio, rate = sf.read(path, always_2d=True)
        if (sha256(path) != entry["sha256"] or rate != 44100 or len(audio) != record["original_samples"] or
                audio.shape[1] != 2 or not np.isfinite(audio).all()):
            raise ValueError(f"WAV reload failed: {path}")
    return len(record["tensors"])


class RecordingProcessor:
    def __init__(self, processor, owner, kind):
        self.processor, self.owner, self.kind = processor, owner, kind

    def __call__(self, *args, **kwargs):
        result = self.processor(*args, **kwargs)
        cap, prefix = self.owner.capture, self.owner.prefix
        for key, value in result.items():
            cap.save(f"{prefix}/{self.kind}_processor_{key}", value,
                     "clap_processor" if self.kind == "clap" else "mert")
        audio = kwargs.get("audio", args[0] if args else None)
        cap.save(f"{prefix}/{self.kind}_processor_audio", np.asarray(audio),
                 "clap_processor" if self.kind == "clap" else "mert")
        return result


def load_pipeline(spec, device):
    # Local snapshot directories pin remote MERT code, configs, and processors as well as weights.
    from minimax_music3_latent_refiner import MiniMaxMusic3RefinerPipeline
    from minimax_music3_latent_refiner.model import MiniMaxMusic3LatentRefiner
    from minimax_music3_latent_refiner.dav import MiniMaxMusic3DAV
    from safetensors.torch import load_file
    from transformers import AutoModel, ClapModel, ClapProcessor, Wav2Vec2FeatureExtractor
    models = spec["models"]
    def folder(kind):
        return str((ROOT / models[kind]["weights"]).parent)
    refiner = MiniMaxMusic3LatentRefiner.from_pretrained(folder("refiner"), local_files_only=True).to(device)
    dav = MiniMaxMusic3DAV.from_pretrained(folder("dav"), local_files_only=True).to(device)
    mert = AutoModel.from_pretrained(folder("mert"), trust_remote_code=True, local_files_only=True).to(device)
    clap = ClapModel.from_pretrained(folder("clap"), local_files_only=True).to(device)
    return MiniMaxMusic3RefinerPipeline(refiner, dav, mert,
        Wav2Vec2FeatureExtractor.from_pretrained(folder("mert"), local_files_only=True), clap,
        ClapProcessor.from_pretrained(folder("clap"), local_files_only=True),
        load_file(str(ROOT / models["refiner"]["normalization"]), device=device))


class Recorder:
    def __init__(self, pipe):
        self.pipe = pipe
        self.capture = None
        self.prefix = ""
        self.chunks = []
        self.chunk_index = 0
        self.forward_index = 0
        self.steps = 0
        self.chunk = pipe._refine_chunk
        pipe._refine_chunk = self.record_chunk
        pipe.mert_processor = RecordingProcessor(pipe.mert_processor, self, "mert")
        pipe.clap_processor = RecordingProcessor(pipe.clap_processor, self, "clap")
        self.encoder = pipe.audio_vae.encode
        pipe.audio_vae.encode = self.encode
        self.mert_layers = pipe._mert_layers
        pipe._mert_layers = self.mert
        self.clap_style = pipe._clap_style
        pipe._clap_style = self.clap
        pipe.refiner.register_forward_hook(self.forward)

    def encode(self, waveform):
        result = self.encoder(waveform)
        self.capture.save(self.prefix + "/dav_latents", result, "dav")
        return result

    def mert(self, waveform, frames):
        result = self.mert_layers(waveform, frames)
        self.capture.save(self.prefix + "/mert_interpolated", result, "mert")
        return result

    def clap(self, waveform):
        result = self.clap_style(waveform)
        self.capture.save(self.prefix + "/clap_vector", result, "clap")
        return result

    def record_chunk(self, waveform, steps, audio_encoder):
        self.prefix = f"steps-{steps}/window-{self.chunk_index:02d}"
        self.chunk_index += 1
        self.forward_index = 0
        self.steps = steps
        self.schedule = torch.linspace(1., 0., steps + 1, device=self.pipe.device)
        self.capture.save(self.prefix + "/waveform", waveform, "waveform")
        self.capture.save(self.prefix + "/schedule", self.schedule, "bridge")
        result = self.chunk(waveform, steps, audio_encoder)
        self.capture.save(self.prefix + "/refined_latents", result, "bridge")
        self.chunks.append(result.detach())
        return result

    def forward(self, module, args, output):
        noisy, _, _, layers, _, degraded, _ = args[:7]
        i = self.forward_index
        if i == 0:
            self.capture.save(self.prefix + "/normalized_latents", degraded, "normalized_latents")
            self.capture.save(self.prefix + "/mert_normalized", layers, "mert")
            self.capture.save(self.prefix + "/dit_forward", output, "dit")
            self.capture.save(self.prefix + "/step-00", noisy, "bridge")
        self.capture.save(self.prefix + f"/velocity-{i:02d}", output, "dit")
        updated = noisy - (self.schedule[i] - self.schedule[i + 1]) * output
        self.capture.save(self.prefix + f"/step-{i + 1:02d}", updated, "bridge")
        self.forward_index += 1

    def overlap(self, waveform, latents, spec):
        from minimax_music3_latent_refiner.pipeline import window_starts
        win = max(512, round(spec["window_seconds"] * 44100 / 512) * 512)
        ovl = max(0, round(spec["overlap_seconds"] * 44100 / 512) * 512)
        starts = window_starts(waveform.shape[-1], win, ovl)
        total = math.ceil(waveform.shape[-1] / 512)
        weights = torch.zeros(1, 1, total, device=self.pipe.device)
        accumulated = torch.zeros_like(latents)
        for i, (start, chunk) in enumerate(zip(starts, self.chunks, strict=True)):
            frame = start // 512
            available = min(chunk.shape[-1], total - frame)
            blend = torch.ones(available, device=self.pipe.device)
            if i:
                left = min(available, max(0, starts[i - 1] // 512 + math.ceil(win / 512) - frame))
                if left:
                    blend[:left] = torch.linspace(0., 1., left + 2, device=self.pipe.device)[1:-1]
            if i + 1 < len(starts):
                right = min(available, max(0, frame + available - starts[i + 1] // 512))
                if right:
                    blend[-right:] = torch.minimum(blend[-right:], torch.linspace(1., 0., right + 2, device=self.pipe.device)[1:-1])
            self.capture.save(f"steps-{self.steps}/window-{i:02d}/overlap", blend, "overlap")
            accumulated[..., frame:frame + available] += chunk[..., :available] * blend
            weights[..., frame:frame + available] += blend
        if torch.any(weights == 0) or not torch.equal(accumulated / weights, latents):
            raise ValueError("reconstructed overlap disagrees with the stock pipeline")
        self.capture.save(f"steps-{self.steps}/overlap_weights", weights, "overlap")
        self.capture.save(f"steps-{self.steps}/window_starts", np.asarray(starts, dtype=np.int64), "overlap")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "artifacts/fixtures")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--clips", nargs="+")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    spec = contract()
    selected = [c for c in spec["clips"] if args.clips is None or c["id"] in args.clips]
    if args.clips and set(args.clips) != {c["id"] for c in selected}:
        parser.error("unknown clip id")
    if args.reload:
        for clip in selected:
            print(f"RELOAD {clip['id']}: {reload_fixture(args.out / clip['id'])} tensors", flush=True)
        return
    verify_sources(spec)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(ROOT / "_experiments/mm3-whats-new-2026-11-05/refiner/hf-cache")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    pipe = load_pipeline(spec, args.device)
    recorder = Recorder(pipe)
    by_id = {c["id"]: c for c in spec["clips"]}
    for clip in selected:
        directory = args.out / clip["id"]
        if (directory / "manifest.json").exists():
            print(f"RELOAD {clip['id']}: {reload_fixture(directory)} tensors", flush=True)
            continue
        source = clip if "path" in clip else by_id[clip["derive_from"]]
        data, rate = sf.read(ROOT / source["path"], dtype="float32", always_2d=True)
        if rate != 44100:
            raise ValueError("reference corpus must be 44.1 kHz")
        data = data[:clip.get("samples", len(data))]
        n = len(data)
        padded = np.pad(data, ((0, (-n) % 512), (0, 0)))
        waveform = torch.from_numpy(padded.T.copy())
        cap = Capture(directory)
        recorder.capture = cap
        cap.save("input", data.T, "waveform")
        wavs = []
        for steps in spec["steps"]:
            recorder.chunk_index = 0
            recorder.chunks = []
            print(f"CAPTURE {clip['id']} steps={steps}", flush=True)
            with torch.inference_mode():
                result = pipe(waveform, rate, steps=steps, window_seconds=spec["window_seconds"],
                              overlap_seconds=spec["overlap_seconds"])
                recorder.overlap(waveform, result.latents, spec)
            cap.save(f"steps-{steps}/latents", result.latents, "bridge")
            audio = result.audio[0].T.numpy()[:n]
            cap.save(f"steps-{steps}/decoded", audio.T, "waveform")
            meter = pyln.Meter(rate)
            level_in, level_out = meter.integrated_loudness(data), meter.integrated_loudness(audio)
            gain = level_in - level_out
            if not np.isfinite(gain):
                raise ValueError("non-finite reference loudness gain")
            audio = np.clip(audio * 10 ** (gain / 20), -1, 1)
            cap.save(f"steps-{steps}/final_audio", audio.T, "waveform")
            dest = directory / f"steps-{steps}/final.wav"
            sf.write(dest, audio, rate, subtype="PCM_24")
            wavs.append(dict(path=dest.relative_to(directory).as_posix(), sha256=sha256(dest), gain_db=float(gain)))
        cap.finish(dict(id=clip["id"], original_samples=n, padded_samples=len(padded),
                        source_sha256=sha256(ROOT / source["path"]),
                        contract_sha256=sha256(HERE / "fixtures.json"), wav=wavs))
        print(f"PASS {clip['id']}: {len(cap.entries)} tensors reloaded", flush=True)


if __name__ == "__main__":
    main()
