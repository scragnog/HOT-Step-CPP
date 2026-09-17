"""CPU fixtures that exercise the pinned AI Toolkit YuE2 AR loss bodies.

The complete YuE2 model cannot be loaded in a small fixture, so this module
loads the *actual* ``_ar_inputs`` and ``_ar_losses`` function bodies from the
pinned ``yue2_model.py`` with a deterministic tiny AR double.  It deliberately
does not reimplement the loss equations.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import struct
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REFERENCE = Path(r"D:\Ace-Step-Latest\ai-toolkit\extensions_built_in\audio_models\yue2\yue2_model.py")
CONSTANTS = {"ABC_END": 151848, "CODEC_OFFSET": 152000, "MUSIC_END": 151852, "MUSIC_START": 151851,
             "COT_CODES": {"off": 0, "melody": 1, "full": 2}, "COT_NAMES": {0: "off", 1: "melody", 2: "full"}}


class _FixtureDTO(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def load_reference_bodies(path: str | Path = REFERENCE) -> tuple[dict[str, Any], str]:
    path = Path(path)
    source = path.read_bytes()
    tree = ast.parse(source, filename=str(path))
    wanted = {"_train_prefix", "_ar_inputs", "_ar_losses", "get_noise_prediction",
              "_tag", "_normalize_section", "_number", "parse_caption"}
    nodes = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            node.decorator_list = []
            nodes.append(node)
    if {n.name for n in nodes} != wanted:
        raise RuntimeError(f"reference missing expected bodies: {wanted - {n.name for n in nodes}}")
    namespace = {"torch": torch, "F": F, "random": random, "re": re, "DTO": _FixtureDTO,
                 "flush": lambda: None, "_SONG_SECTION": re.compile(r"^\s*\[[^\]]+\]\s*$"),
                 "_SECTION": re.compile(r"^\s*\[(Tags|Lyrics|Duration)\]\s*$", re.IGNORECASE | re.MULTILINE), **CONSTANTS}
    namespace["__name__"] = "yue2_reference_fixture"
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return {name: namespace[name] for name in wanted}, hashlib.sha256(source).hexdigest()


class TinyAR:
    def __init__(self, network: "TinyNetwork", hidden: int = 8, vocab: int = 152100):
        self.network = network
        self.dtype = torch.float32
        self.embedding = torch.nn.Embedding(vocab, hidden)
        self.model = SimpleNamespace(lm_head=torch.nn.Linear(hidden, vocab, bias=False))
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.model.lm_head.weight.zero_()
            for i in range(hidden):
                self.embedding.weight[i].fill_(0.02 * (i + 1))
                self.model.lm_head.weight[:, i].fill_(0.001 * (i + 1))

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(ids)

    def prefill(self, embeds: torch.Tensor, return_hidden: bool = False):
        self.network.prefill_active.append(bool(self.network.is_active))
        hidden = embeds + (0.125 * embeds if self.network.is_active else 0.0)
        if self.network.is_active:
            hidden = hidden + self.network.adapter_gain * embeds
        self.network.prefill_requires_grad.append(bool(hidden.requires_grad))
        if return_hidden:
            return None, hidden
        return [(hidden, hidden * 2)], None


class TinyNetwork:
    def __init__(self):
        self.is_active = True
        self.prefill_active: list[bool] = []
        self.prefill_requires_grad: list[bool] = []
        self.adapter_gain = torch.tensor(0.05, requires_grad=True)


class TinyTokenizer:
    def prefix_head_ids(self, style: str, lyrics: str, cot: str):
        return [77, 78, 79]


class TinyNAR:
    def __init__(self):
        self.calls = []

    def __call__(self, latent_input, timestep, cache, prompt_len):
        self.calls.append({"latent": latent_input.detach().clone(), "timestep": timestep.detach().clone(),
                           "cache": cache, "prompt_len": int(prompt_len)})
        return torch.zeros_like(latent_input)


class TinyHarness:
    def __init__(self, chunk: int):
        self._network = TinyNetwork()
        self.model = SimpleNamespace(ar=TinyAR(self._network), nar=TinyNAR(), tokenizer=TinyTokenizer(),
                                     device=torch.device("cpu"), to=lambda device: None)
        self.model_config = SimpleNamespace(model_kwargs={"ar_loss_chunk_size": chunk})
        self.ar_kl_weight = 0.2
        self.debug = False
        self._loss_log_every = 0
        self._loss_log_step = 0
        self.cot = "full"
        self.abc_dropout = 1.0
        self.ar_loss_weight = 1.0
        self.ar_max_tokens = 0
        self.device_torch = torch.device("cpu")
        self.torch_dtype = torch.float32
        self._train_calls = 0
        self._last_encode_call = 0


def make_case(name: str, *, abc: bool, start: int, end: int, chunk: int = 512) -> dict[str, Any]:
    bodies, source_sha256 = load_reference_bodies()
    harness = TinyHarness(chunk)
    prefix = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 10
    abc_ids = torch.tensor([101, 102], dtype=torch.long) if abc else torch.zeros(0, dtype=torch.long)
    song = torch.tensor([7, 8, 9, 10, 11, 12], dtype=torch.long)
    conditioning, conditioning_ids = bodies["_ar_inputs"](harness, prefix, abc_ids, song[start:end], end_token=False)
    ar_inputs, ar_ids = bodies["_ar_inputs"](harness, prefix, abc_ids, song, end_token=True)
    ce, kl, _ = bodies["_ar_losses"](harness, ar_inputs, ar_ids, prefix, int(song.numel()), start == 0 and end == song.numel())
    if kl is None:
        raise AssertionError("fixture requires the reference KL path")
    return {
        "name": name, "abc_retained": abc, "window": [start, end], "chunk": chunk,
        "source_sha256": source_sha256, "prefix": prefix, "abc_ids": abc_ids, "song_tokens": song,
        "conditioning_ids": conditioning_ids, "ar_targets": ar_ids, "ce": ce.detach(), "kl": kl.detach(),
        "ce_requires_grad": bool(ce.requires_grad), "kl_requires_grad": bool(kl.requires_grad),
        "prefill_active": harness._network.prefill_active,
        "prefill_requires_grad": harness._network.prefill_requires_grad,
    }


def make_conditioning_case(name: str = "dropout_crop") -> dict[str, Any]:
    bodies, source_sha256 = load_reference_bodies()
    harness = TinyHarness(128)
    harness.tokenizer = TinyTokenizer()
    harness._train_prefix = bodies["_train_prefix"]
    harness._ar_inputs = MethodType(bodies["_ar_inputs"], harness)
    harness._ar_losses = MethodType(bodies["_ar_losses"], harness)
    tokens = torch.tensor([[7, 8, 9, 10, 11, 12]], dtype=torch.long)
    abc = torch.tensor([[101, 102]], dtype=torch.long)
    latents = _FixtureDTO(tokens=tokens, abc_ids=abc, abc_mode=torch.tensor([2], dtype=torch.int32))
    batch = _FixtureDTO(latents=latents, yue2_window=(2, 5))
    batch.get_caption_list = lambda: ["style\n[Lyrics]\nhello"]
    text_embeddings = SimpleNamespace(text_embeds=torch.arange(24, dtype=torch.float32).reshape(1, 3, 8) / 10,
                                      attention_mask=torch.ones((1, 3), dtype=torch.float32))
    latent_input = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    timestep = torch.tensor([500.0])
    old_random = random.random
    random.random = lambda: 0.0
    try:
        prediction = bodies["get_noise_prediction"](harness, latent_input, timestep, text_embeddings, batch=batch)
    finally:
        random.random = old_random
    call = harness.model.nar.calls[0]
    return {"name": name, "source_sha256": source_sha256, "prediction": prediction.detach(),
            "window": [2, 5], "nar_latent": call["latent"], "nar_cache": call["cache"],
            "nar_prompt_len": call["prompt_len"], "ar_prefill_active": harness._network.prefill_active,
            "ar_prefill_requires_grad": harness._network.prefill_requires_grad,
            "abc_dropout": harness.abc_dropout, "abc_mode_used": "off"}


def make_native_loss_case(positions: int = 130, vocab: int = 17, logit_offset: float = 0.0) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(20260917)
    adapted = (torch.randn((positions, vocab), generator=generator, dtype=torch.float32) + logit_offset).requires_grad_()
    base = torch.randn((positions, vocab), generator=generator, dtype=torch.float32) - logit_offset
    targets = torch.randint(vocab, (positions,), generator=generator, dtype=torch.long)
    ce = F.cross_entropy(adapted, targets, reduction="mean")
    base_logp = F.log_softmax(base, -1)
    kl = F.kl_div(F.log_softmax(adapted, -1), base_logp, log_target=True, reduction="sum") / positions
    (ce + 0.2 * kl).backward()
    return {"adapted": adapted.detach(), "base": base, "targets": targets,
            "ce": ce.detach(), "kl": kl.detach(), "gradient": adapted.grad.detach(),
            "positions": positions, "vocab": vocab}


def export_native_loss_case(out_dir: str | Path, chunk: int = 128, positions: int = 130, vocab: int = 17,
                            logit_offset: float = 0.0) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    case = make_native_loss_case(positions=positions, vocab=vocab, logit_offset=logit_offset)
    with (out / "native_loss.bin").open("wb") as f:
        f.write(struct.pack("<QQQ", case["positions"], case["vocab"], chunk))
        f.write(case["adapted"].numpy().tobytes())
        f.write(case["base"].numpy().tobytes())
        f.write(case["targets"].numpy().astype(np.uint32).tobytes())
    np.savez(out / "native_loss_expected.npz", ce=case["ce"].numpy(), kl=case["kl"].numpy(),
             gradient=case["gradient"].numpy())
    return case


def compare_native_output(probe_output: str | Path, expected: dict[str, Any]) -> dict[str, float]:
    raw = Path(probe_output).read_bytes()
    if len(raw) != 16 + expected["gradient"].numel() * 4:
        raise AssertionError("Invalid native loss output length")
    ce, kl = struct.unpack_from("<dd", raw)
    gradient = np.frombuffer(raw, dtype=np.float32, offset=16)
    expected_gradient = expected["gradient"].numpy().reshape(-1)
    result = {"ce_abs": abs(ce - expected["ce"].item()), "kl_abs": abs(kl - expected["kl"].item()),
              "gradient_max_abs": float(np.max(np.abs(gradient - expected_gradient)))}
    if not all(np.isfinite(value) and value <= 3e-5 for value in result.values()):
        raise AssertionError(f"native loss mismatch: {result}")
    return result


def export_fixtures(out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = [make_case("full_windowed", abc=True, start=2, end=5),
             make_case("dropped_windowed", abc=False, start=2, end=5),
             make_case("full_song_real_end", abc=True, start=0, end=6), make_conditioning_case()]
    manifest = {"schema": "yue2_joint_loss_fixture_v1",
                "coverage": {"reference_bodies": ["_ar_inputs", "_ar_losses", "get_noise_prediction", "parse_caption"],
                             "mocked": ["YuE2Model.ar.embed", "YuE2Model.ar.prefill", "lm_head"],
                             "uncovered": ["real tokenizer/caption data and actual model weights", "stochastic rather than forced dropout draw"]},
                "cases": []}
    for case in cases:
        name = case["name"]
        arrays = {k: v.cpu().numpy() for k, v in case.items() if isinstance(v, torch.Tensor)}
        if isinstance(case.get("nar_cache"), list):
            for i, (key, value) in enumerate(case["nar_cache"]):
                arrays[f"nar_cache_{i}_k"] = key.cpu().numpy()
                arrays[f"nar_cache_{i}_v"] = value.cpu().numpy()
        np.savez(out / f"{name}.npz", **arrays)
        summary = {}
        for key, value in case.items():
            if isinstance(value, torch.Tensor):
                summary[key] = value.tolist()
            elif key == "nar_cache":
                summary[key] = [{"k_shape": list(k.shape), "v_shape": list(v.shape),
                                 "detached": not k.requires_grad and not v.requires_grad} for k, v in value]
            else:
                summary[key] = value
        manifest["cases"].append(summary)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    export_native_loss_case(out)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--native-positions", type=int, default=130)
    parser.add_argument("--native-vocab", type=int, default=17)
    parser.add_argument("--compare-native", type=Path)
    args = parser.parse_args()
    export_fixtures(args.out)
    expected = export_native_loss_case(args.out, positions=args.native_positions, vocab=args.native_vocab)
    if args.compare_native:
        print(json.dumps(compare_native_output(args.compare_native, expected), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
