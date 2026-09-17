"""Isolate attention output rounding with real layer-five inputs."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.nn.attention import SDPBackend, sdpa_kernel
import argparse
p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
torch.backends.cuda.matmul.allow_tf32=False
r=load_file(str(a.trace/'trace.safetensors'),device='cuda')
q=r['ar.detail5.q_norm.output'].bfloat16();k=r['ar.detail5.k_norm.output'].bfloat16()
angles=torch.arange(7,device='cuda').float()[:,None]/(1000000.**(torch.arange(0,128,2,device='cuda').float()/128))
c,s=angles.cos().bfloat16()[None,None],angles.sin().bfloat16()[None,None]
def rope(x):return torch.cat((x[...,:64]*c-x[...,64:]*s,x[...,64:]*c+x[...,:64]*s),dim=-1)
q,k=rope(q),rope(k)
v=r['ar.detail5.qkv.output'][...,3072:].reshape(1,7,8,128).transpose(1,2).bfloat16()
target=r['ar.detail5.o_proj.input'].bfloat16()
values={'q':q,'k':k,'v':v,'target':target}
report={}
def compare(name,value):
    y=value.transpose(1,2).reshape(1,7,2048)
    report[name]={'different':int((y!=target).sum()),'max_abs':float((y.float()-target.float()).abs().max())}
    values[name]=y
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    z=F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)
compare('auto',z)
report['operators']=[e.key for e in prof.key_averages() if 'attention' in e.key]
for name,backend in [('math',SDPBackend.MATH),('cudnn',SDPBackend.CUDNN_ATTENTION)]:
    try:
        with sdpa_kernel([backend]):z=F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)
        compare(name,z)
    except RuntimeError as e:report[name]={'error':str(e)}
kf=k.float().repeat_interleave(2,dim=1);vf=v.float().repeat_interleave(2,dim=1)
mask=torch.ones(7,7,device='cuda',dtype=torch.bool).tril()
for name,score in [('post_scale',q.float()@kf.transpose(-2,-1)/(128**.5)),
                   ('split_scale',(q.float()/(128**.25))@(kf.transpose(-2,-1)/(128**.25)))]:
    prob=score.masked_fill(~mask,float('-inf')).softmax(-1)
    compare(name,(prob@vf).bfloat16())
save_file({k:v.contiguous().cpu() for k,v in values.items()},str(a.output/'attention.safetensors'))
(a.output/'metrics.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
