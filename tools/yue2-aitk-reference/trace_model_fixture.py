"""Trace the actual frozen checkpoint on the two-frame joint fixture."""
import argparse
from contextlib import contextmanager
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--toolkit', default=r'D:\Ace-Step-Latest\ai-toolkit')
parser.add_argument('--intermediates', action='store_true',
                    help='capture AR layer 5 module inputs/outputs as ar.detail5.*')
parser.add_argument('--detail-layer', type=int, default=5)
parser.add_argument('--sdpa', choices=('auto', 'math', 'cudnn'), default='auto',
                    help='select the PyTorch SDPA backend (default: auto)')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, args.toolkit)
from extensions_built_in.audio_models.yue2.src.model import YuE2Model
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
print('Load frozen checkpoint', flush=True)
model = YuE2Model.load_from_state_dict(load_file(args.checkpoint), dtype=torch.bfloat16).cuda()
values = {}
def capture(name):
    def hook(module, inputs, output):
        x = output[0] if isinstance(output, tuple) else output
        values[name] = x.detach().float().cpu().contiguous()
    return hook

def capture_input(name):
    def hook(module, inputs):
        if inputs:
            values[name] = inputs[0].detach().float().cpu().contiguous()
    return hook

def capture_io(name, module):
    module.register_forward_pre_hook(capture_input(name + '.input'))
    module.register_forward_hook(capture(name + '.output'))

@contextmanager
def sdpa_context(mode):
    if mode == 'auto':
        yield
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel
    backend = SDPBackend.MATH if mode == 'math' else SDPBackend.CUDNN_ATTENTION
    with sdpa_kernel(backends=[backend]):
        yield

for expert in ('ar', 'nar'):
    for i, layer in enumerate(getattr(model, expert).model.layers):
        layer.register_forward_hook(capture(f'{expert}.layer{i}'))
    getattr(model, expert).model.norm.register_forward_hook(capture(f'{expert}.norm'))
if args.intermediates:
    layer = model.ar.model.layers[args.detail_layer]
    capture_io('ar.detail5.input_norm', layer.input_layernorm)
    capture_io('ar.detail5.qkv', layer.self_attn.qkv_proj)
    capture_io('ar.detail5.q_norm', layer.self_attn.q_norm)
    capture_io('ar.detail5.k_norm', layer.self_attn.k_norm)
    capture_io('ar.detail5.attention_pre_oproj', layer.self_attn.o_proj)
    capture_io('ar.detail5.o_proj', layer.self_attn.o_proj)
    capture_io('ar.detail5.post_norm', layer.post_attention_layernorm)
    capture_io('ar.detail5.gate_up', layer.mlp.gate_up_proj)
    capture_io('ar.detail5.down', layer.mlp.down_proj)
with torch.no_grad():
    ids = torch.tensor([[1, 2, 151848, 151851, 151865, 151895, 151852]], device='cuda')
    embeds = model.ar.embed(ids)
    values['embedding'] = embeds.float().cpu()
    with sdpa_context(args.sdpa):
        cache, hidden = model.ar.prefill(embeds, return_hidden=True)
    clean = torch.tensor([(i % 13 - 6) * .03125 for i in range(128)], device='cuda').reshape(1, 2, 64)
    noise = torch.tensor([(i % 17 - 8) * .0625 for i in range(128)], device='cuda').reshape(1, 2, 64)
    noisy = (.625 * clean + .375 * noise).bfloat16()
    model.nar.model.layers[0].register_forward_pre_hook(lambda m, x: values.update({'nar.frontend': x[0].float().cpu()}))
    with sdpa_context(args.sdpa):
        pred = model.nar(noisy, torch.tensor([.375], device='cuda', dtype=torch.bfloat16), cache, 7)
    values['prediction'] = pred.float().cpu().contiguous()
save_file(values, str(args.output / 'trace.safetensors'))
print('Saved frozen AR/NAR layer trace', flush=True)
