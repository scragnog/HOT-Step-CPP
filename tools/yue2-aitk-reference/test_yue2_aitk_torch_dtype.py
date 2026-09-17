"""CPU-only dtype probe for the Toolkit flow-match noise expression.

This intentionally exercises the same expression as
``custom_flowmatch_sampler.py:92-105`` without importing model weights.  It is
an evidence fixture for the native sampler contract, not a parity test for a
CUDA kernel.  Root may run it in the Toolkit venv.
"""
import torch
import json
import sys


def main() -> None:
    torch.set_default_dtype(torch.float32)
    clean32 = torch.tensor([[-1.2345, 0.5033, 2.003]], dtype=torch.float32)
    noise32 = torch.tensor([[0.2547, -0.7549, 1.503]], dtype=torch.float32)
    timestep = torch.tensor([375.0], dtype=torch.float32)
    t01 = (timestep / 1000).to(clean32.device)
    noisy32 = (1.0 - t01) * clean32 + t01 * noise32
    assert noisy32.dtype is torch.float32
    assert torch.equal(noisy32, (1.0 - t01) * clean32 + t01 * noise32)

    clean16 = clean32.to(torch.bfloat16)
    noise16 = noise32.to(torch.bfloat16)
    noisy_mixed = (1.0 - t01) * clean16 + t01 * noise16
    assert noisy_mixed.dtype is torch.float32
    target16 = noise16 - clean16
    assert target16.dtype is torch.bfloat16
    assert not torch.equal(noisy_mixed, noisy32)
    assert torch.equal(target16, noise16 - clean16)
    print("flow-match dtype probe passed:", noisy_mixed.dtype, target16.dtype)
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding='utf-8') as source:
            native = json.load(source)
        clean = torch.tensor(native['clean_bf16'], dtype=torch.bfloat16)
        noise = torch.tensor(native['noise_bf16'], dtype=torch.bfloat16)
        t = torch.tensor([native['timestep']], dtype=torch.float32) / 1000
        noisy = (1-t)*clean+t*noise
        assert torch.equal((noise-clean).float(), torch.tensor(native['target_bf16']))
        assert torch.equal(noisy, torch.tensor(native['noisy_f32_mixed']))
        assert torch.equal(noisy.bfloat16().float(), torch.tensor(native['noisy_bf16']))
        print('Native sampled target and noisy tensors match Torch exactly')


if __name__ == "__main__":
    main()
