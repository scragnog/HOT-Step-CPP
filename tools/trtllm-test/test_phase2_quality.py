#!/usr/bin/env python3
"""
Phase 2 Quality Test — verify TRT-LLM generates valid audio tokens.

Sends a Phase 2 prompt (with CoT already filled in from Phase 1)
and checks that the output contains valid audio code tokens.

Audio codes are token IDs >= 151669 (AUDIO_CODE_BASE).

Usage (Docker):
  docker run --rm --gpus all --ipc=host \
    -v D:\Ace-Step-Latest\hot-step-cpp:/workspace \
    nvcr.io/nvidia/tensorrt-llm/release:1.2.1 \
    python3 /workspace/tools/trtllm-test/test_phase2_quality.py
"""

import os
import sys
import time

# Constants from prompt.h
TOKEN_IM_START  = 151644
TOKEN_IM_END    = 151645
TOKEN_THINK     = 151667
TOKEN_THINK_END = 151668
AUDIO_CODE_BASE = 151669
AUDIO_CODE_COUNT = 65535

# Engine path
ENGINE_DIR = "/workspace/models/onnx/lm-4B/trtllm-engine-RTX5090"

# How many tokens to generate (real Phase 2 generates 3000-6000)
MAX_NEW_TOKENS = 500

def main():
    print("=" * 60)
    print("PHASE 2 QUALITY TEST — TRT-LLM Audio Token Generation")
    print("=" * 60)

    if not os.path.exists(ENGINE_DIR):
        print(f"ERROR: Engine not found at {ENGINE_DIR}")
        sys.exit(1)

    # Import TRT-LLM
    from tensorrt_llm._tensorrt_engine import LLM as TrtLLM
    from tensorrt_llm import SamplingParams, BuildConfig
    from transformers import AutoTokenizer

    print(f"\nLoading engine from {ENGINE_DIR}...")
    t0 = time.perf_counter()

    llm = TrtLLM(
        model=ENGINE_DIR,
        kv_cache_config={"free_gpu_memory_fraction": 0.3},
    )
    t_load = time.perf_counter() - t0
    print(f"Engine loaded in {t_load:.1f}s")

    # Load tokenizer for encoding the text parts
    tokenizer = AutoTokenizer.from_pretrained(ENGINE_DIR, trust_remote_code=True)

    # Build Phase 2 prompt matching prompt.h build_lm_prompt_with_cot()
    # This is what the C++ pipeline sends for audio code generation.
    #
    # Format:
    #   <|im_start|>system\n# Instruction\n{instruction}\n\n<|im_end|>\n
    #   <|im_start|>user\n# Caption\n{caption}\n\n# Lyric\n{lyrics}\n<|im_end|>\n
    #   <|im_start|>assistant\n<think>\n{cot_yaml}</think>\n\n

    instruction = "Generate audio semantic tokens based on the given conditions:"

    caption = ("A synthwave electronic track with driving bass and shimmering arpeggios. "
               "Energetic and uplifting mood, 120 BPM, C major, 4/4 time signature. "
               "Features synthesizer, drum machine, and bass.")

    lyrics = """[Verse]
Neon lights are calling
Through the midnight rain
Digital horizons
Breaking every chain

[Chorus]
We're riding on the frequency
Electric hearts align
Through the noise and static
Our signal starts to shine"""

    # CoT YAML (from our Phase 1 test output)
    cot_yaml = """bpm: 111
caption: An energetic synthwave track driven by a powerful four-on-the-floor drum
  machine beat with classic gated reverb on the snare. A pulsating arpeggiated synth
  bassline provides a relentless rhythmic foundation, while layers of shimmering synthesizers
  create an expansive soundscape.
duration: 240
keyscale: C major
language: zxx
timesignature: 4
"""

    # Build token IDs matching the C++ prompt builder exactly
    prompt_ids = []

    # System message
    prompt_ids.append(TOKEN_IM_START)
    prompt_ids.extend(tokenizer.encode(
        f"system\n# Instruction\n{instruction}\n\n", add_special_tokens=False))
    prompt_ids.append(TOKEN_IM_END)
    prompt_ids.extend(tokenizer.encode("\n", add_special_tokens=False))

    # User message
    prompt_ids.append(TOKEN_IM_START)
    prompt_ids.extend(tokenizer.encode(
        f"user\n# Caption\n{caption}\n\n# Lyric\n{lyrics}\n", add_special_tokens=False))
    prompt_ids.append(TOKEN_IM_END)
    prompt_ids.extend(tokenizer.encode("\n", add_special_tokens=False))

    # Assistant turn with pre-filled CoT
    prompt_ids.append(TOKEN_IM_START)
    prompt_ids.extend(tokenizer.encode("assistant\n", add_special_tokens=False))
    prompt_ids.append(TOKEN_THINK)
    prompt_ids.extend(tokenizer.encode(f"\n{cot_yaml}", add_special_tokens=False))
    prompt_ids.append(TOKEN_THINK_END)
    prompt_ids.extend(tokenizer.encode("\n\n", add_special_tokens=False))

    print(f"\nPrompt: {len(prompt_ids)} tokens")
    print(f"Generating up to {MAX_NEW_TOKENS} tokens...")

    sampling_params = SamplingParams(
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        max_tokens=MAX_NEW_TOKENS,
        seed=42,
    )

    t_start = time.perf_counter()
    output = llm.generate(prompt_ids, sampling_params)
    t_end = time.perf_counter()

    gen_token_ids = output.outputs[0].token_ids
    gen_text = output.outputs[0].text
    elapsed = t_end - t_start
    tok_per_sec = len(gen_token_ids) / elapsed

    print(f"\n--- Results ---")
    print(f"  Generated: {len(gen_token_ids)} tokens in {elapsed:.3f}s")
    print(f"  Throughput: {tok_per_sec:.1f} tok/s")

    # Analyze output tokens
    audio_codes = []
    text_tokens = []
    special_tokens = []

    for tid in gen_token_ids:
        if tid >= AUDIO_CODE_BASE:
            audio_codes.append(tid - AUDIO_CODE_BASE)
        elif tid in (TOKEN_IM_START, TOKEN_IM_END, TOKEN_THINK, TOKEN_THINK_END):
            special_tokens.append(tid)
        else:
            text_tokens.append(tid)

    print(f"\n--- Token Analysis ---")
    print(f"  Audio code tokens: {len(audio_codes)}")
    print(f"  Text tokens:       {len(text_tokens)}")
    print(f"  Special tokens:    {len(special_tokens)}")

    if audio_codes:
        print(f"\n  Audio code range: {min(audio_codes)} - {max(audio_codes)} (valid: 0 - {AUDIO_CODE_COUNT-1})")
        print(f"  First 20 audio codes: {audio_codes[:20]}")
        print(f"  Last 10 audio codes:  {audio_codes[-10:]}")

        # Check validity
        valid = all(0 <= c < AUDIO_CODE_COUNT for c in audio_codes)
        print(f"\n  All codes in valid range: {'✅ YES' if valid else '❌ NO'}")
    else:
        print(f"\n  ⚠️  NO AUDIO CODES GENERATED!")

    if text_tokens:
        # Decode any text that appeared before audio codes
        text_before = tokenizer.decode(text_tokens[:50], skip_special_tokens=False)
        print(f"\n  Text before audio codes: {repr(text_before[:200])}")

    # Full text output (may be mixed text + audio code placeholders)
    print(f"\n--- Full Text Output (first 500 chars) ---")
    print(gen_text[:500] if gen_text else "(empty)")
    print("--- End ---")

    # Summary
    print(f"\n{'='*60}")
    if len(audio_codes) > 10:
        print(f"✅ PASS — {len(audio_codes)} valid audio codes generated at {tok_per_sec:.1f} tok/s")
    elif len(audio_codes) > 0:
        print(f"⚠️  PARTIAL — Only {len(audio_codes)} audio codes (expected 100+)")
    else:
        print(f"❌ FAIL — No audio codes generated")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
