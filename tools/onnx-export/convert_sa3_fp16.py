#!/usr/bin/env python3
"""Convert the SA3 ONNX graph set to fp16 for shipping (halves ~12GB fp32 -> ~6GB).

Weights/compute go fp16, graph I/O stays fp32 (keep_io_types) so the C++
orchestration is precision-agnostic. Validate afterwards by re-running
e2e_sa3_ort.py against the fp16 directory (expect cosine > 0.99 vs the
PyTorch fp32 reference — the production Python pipeline ran fp16 anyway).

Runs in the StableAudio3 uv venv:
  uv run --with onnx --with onnxconverter-common python convert_sa3_fp16.py \
      --input-dir <fp32 dir> --output-dir <fp16 dir>
"""

import argparse
import os

import onnx
from onnxconverter_common import float16

# Text encoder stays fp32: onnxconverter-common emits invalid mixed-dtype casts
# around its bool-mask paths, and the engine keeps text encoders fp32 anyway
# (text-enc-ort.h: "FP32: layernorm overflows in FP16"). Seconds embedder is 0.8MB.
GRAPHS_FP16 = [
    "sa3-same_encoder.onnx",
    "sa3-same_decoder.onnx",
    "sa3-dit.onnx",
]
GRAPHS_COPY_FP32 = [
    "sa3-text_encoder.onnx",
    "sa3-seconds_embedder.onnx",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    import shutil

    for name in GRAPHS_COPY_FP32:
        # NOT a file copy: the export may reference external per-tensor files —
        # load (resolves them) and re-save self-contained (<2GB, fits inline).
        model = onnx.load(os.path.join(args.input_dir, name))
        onnx.save_model(model, os.path.join(args.output_dir, name))
        print(f"Repacked {name} (fp32, self-contained)")

    for name in GRAPHS_FP16:
        src = os.path.join(args.input_dir, name)
        dst = os.path.join(args.output_dir, name)
        print(f"Converting {name}...")
        if name == "sa3-dit.onnx":
            # >2GB: in-memory shape inference hits the protobuf limit — infer on
            # disk. The temp file MUST live next to src: the fp32 export stores
            # weights as per-tensor external files resolved relative to the model.
            inferred = src + ".inferred"
            onnx.shape_inference.infer_shapes_path(src, inferred)
            model = onnx.load(inferred)  # pulls external data fully into memory
            os.remove(inferred)
            model_fp16 = float16.convert_float_to_float16(
                model, keep_io_types=True, disable_shape_infer=True
            )
        else:
            # Shape inference ON — without it the converter misses boundary
            # casts and emits mixed-dtype nodes (invalid graph).
            model = onnx.load(src)
            model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        # Large graphs (DiT) exceed the 2GB protobuf limit even at fp16 with
        # metadata — always save with external data for uniform loading.
        onnx.save_model(
            model_fp16, dst,
            save_as_external_data=(name == "sa3-dit.onnx"),
            all_tensors_to_one_file=True,
            location=os.path.basename(dst) + ".data",
        )
        total = os.path.getsize(dst)
        data = dst + ".data"
        if os.path.exists(data):
            total += os.path.getsize(data)
        print(f"  -> {total/1e9:.2f} GB")

    print("Done.")


if __name__ == "__main__":
    main()
