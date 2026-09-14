"""Convert between upstream's AR LoRA .pt (ordered A/B list) and our yue2-ar-lora-v1 safetensors.
usage: ar_lora_convert.py to-st <in.pt> <out.safetensors> [steps]
       ar_lora_convert.py to-pt <in.safetensors> <out.pt>
Order in the .pt list: per layer 0..27: attn q,k,v,o then mlp gate,up,down; each A [r,in] then B [out,r]."""
import sys, torch
from safetensors.torch import load_file, save_file
NAMES=["attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"]
mode,src,dst=sys.argv[1:4]
if mode=="to-st":
    ck=torch.load(src,map_location="cpu",weights_only=False); it=iter(ck["lora"]); out={}
    for L in range(28):
        for n in NAMES:
            A=next(it).float().contiguous(); B=next(it).float().contiguous()
            out[f"yue2.blk.{L}.{n}.lora_A.weight"]=A; out[f"yue2.blk.{L}.{n}.lora_B.weight"]=B
    r=A.shape[0]
    md={"format":"yue2-ar-lora-v1","rank":str(r),"alpha":f"{float(r):.6f}","targets":"attn_mlp","steps":sys.argv[4] if len(sys.argv)>4 else "?","trigger":"albumb","cot":"off","source":"upstream ar_lora_cursor.py .pt converted by ar_lora_convert.py"}
    save_file(out,dst,metadata=md); print("wrote",dst,len(out),"tensors rank",r)
else:
    st=load_file(src); lora=[]
    for L in range(28):
        for n in NAMES:
            lora.append(st[f"yue2.blk.{L}.{n}.lora_A.weight"].float()); lora.append(st[f"yue2.blk.{L}.{n}.lora_B.weight"].float())
    torch.save({"lora":lora,"rank":lora[0].shape[0],"targets":"attn_mlp"},dst); print("wrote",dst,len(lora),"tensors")
