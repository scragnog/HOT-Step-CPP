import argparse
import json
from pathlib import Path
import numpy as np
from safetensors.torch import load_file

p=argparse.ArgumentParser()
p.add_argument('reference',type=Path);p.add_argument('native',type=Path)
a=p.parse_args()
ref=load_file(str(a.reference/'trace.safetensors'))
result={}
for key,tensor in ref.items():
    path=a.native/(key+'.f32')
    if not path.exists():continue
    x=tensor.numpy().reshape(-1).astype(np.float64)
    y=np.fromfile(path,dtype='<f4').astype(np.float64)
    assert x.shape==y.shape,(key,x.shape,y.shape)
    result[key]=dict(max_abs=float(np.max(np.abs(y-x))),different=int(np.count_nonzero(y!=x)),
        relative_l2=float(np.linalg.norm(y-x)/max(np.linalg.norm(x),1e-20)))
for key in ['embedding']+[f'ar.layer{i}' for i in range(28)]+['ar.norm','nar.frontend']+[f'nar.layer{i}' for i in range(28)]+['prediction']:
    if key in result:print(key,json.dumps(result[key]))
(a.native/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
