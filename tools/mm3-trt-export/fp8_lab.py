"""MM3 TensorRT DiT lab: profiling, FP8 calibration and renderer parity (dev only).

Runs on the TensorRT Python wheel that matches the engine's nvinfer (10.16.1.11),
plus numpy and cuda-python; no torch or onnx. Work happens in the current
directory, which must hold a copy of mm3-dit-trt.onnx; `placeholder` writes the
4.86 GB external-data file next to it. Set MM3_MODELS_DIR to <models>/mm3 (for
the manifest). Uses the GPU: take the GPU slot first.

Subcommands:
  placeholder               write mm3-dit-trt.onnx.data (placeholder BF16)
  inspect [--all]           print the parsed network's layers (block 0 by default)
  build  --variant V --out PLAN [--norefit] [--gguf G] [--calib J]
                            V = bf16 | fp8mlp | fp8mlpq | nvfp4mlp; --norefit bakes the
                            GGUF's weights instead of building refittable.
                            FP8_SKIP_BLOCKS=0,1,34,35 / FP8_ONLY=ffn_in narrow the variant.
  time   --plan PLAN [--gguf G] [--calib J] [--profile]
                            refit from the GGUF, time B=2 L=689 enqueues, optional
                            per-layer profile bucketed by kernel
  calib  --dump DIR [DIR...] --gguf G --out J [--margin 2]
                            replay MM3_DIT_TRT_DUMP renders through a BF16 engine with
                            the MLP inputs exposed; write per-GEMM activation amax x margin
                            (max over all dumps; per-dump amax kept under "per_dump")
                            (copy J to <models>/mm3/mm3-dit-trt.fp8.json for MM3_DIT_TRT_FP8)
  compare --a DIR --b DIR [--cfg 1.7]
                            per-window final-latent and velocity corr between two dumps
                            of the same seed (e.g. BF16 vs FP8 engine)
"""
import argparse, glob, json, os, struct, sys, time
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cr

HERE = os.getcwd()
MODELS = os.environ.get("MM3_MODELS_DIR", "")
ONNX = os.path.join(HERE, "mm3-dit-trt.onnx")
MANIFEST = os.path.join(MODELS, "mm3-dit-trt.engine.json")
LOG = trt.Logger(trt.Logger.WARNING)


def ck(r):
    err = r[0] if isinstance(r, tuple) else r
    if err != cr.cudaError_t.cudaSuccess:
        raise RuntimeError(err)
    return r[1] if isinstance(r, tuple) and len(r) > 1 else None


def manifest():
    with open(MANIFEST) as f:
        return json.load(f)


# ---------------------------------------------------------------- placeholder
def write_placeholder():
    total = manifest()["onnx_data"]["bytes"]
    path = os.path.join(HERE, "mm3-dit-trt.onnx.data")
    chunk = 32 * 1024 * 1024
    with open(path, "wb") as f:
        for k in range(0, total // 2, chunk):
            n = min(chunk, total // 2 - k)
            h = np.arange(k, k + n, dtype=np.uint64).astype(np.uint32)
            h ^= h >> 16
            h *= np.uint32(0x7FEB352D)
            h ^= h >> 15
            h *= np.uint32(0x846CA68B)
            h ^= h >> 16
            bits = (np.uint32(0x3C00) + ((h >> 7) & np.uint32(0x1FF))) | (((h >> 31) & np.uint32(1)) << 15)
            f.write(bits.astype("<u2").tobytes())
    print("wrote", path, os.path.getsize(path))


# ---------------------------------------------------------------- GGUF
GGML_F32, GGML_F16, GGML_Q8_0, GGML_BF16 = 0, 1, 8, 30


def read_gguf(path, wanted=None):
    """name -> float32 ndarray in torch order (ne reversed). Only F32/F16/BF16/Q8_0."""
    mm = np.memmap(path, np.uint8, "r")
    pos = 0

    def rd(fmt):
        nonlocal pos
        v = struct.unpack_from(fmt, mm, pos)
        pos += struct.calcsize(fmt)
        return v[0]

    def rstr():
        nonlocal pos
        n = rd("<Q")
        s = bytes(mm[pos:pos + n]).decode()
        pos += n
        return s

    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    fmts = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def rval(t):
        nonlocal pos
        if t == 8:
            return rstr()
        if t == 9:
            it = rd("<I")
            n = rd("<Q")
            if it == 8:
                return [rstr() for _ in range(n)]
            pos += sizes[it] * n
            return None
        return rd(fmts[t])

    assert bytes(mm[:4]) == b"GGUF"
    pos = 4
    rd("<I")
    nt, nkv = rd("<Q"), rd("<Q")
    align = 32
    for _ in range(nkv):
        k = rstr()
        v = rval(rd("<I"))
        if k == "general.alignment":
            align = v
    infos = []
    for _ in range(nt):
        name = rstr()
        nd = rd("<I")
        ne = [rd("<Q") for _ in range(nd)]
        t = rd("<I")
        off = rd("<Q")
        infos.append((name, ne, t, off))
    base = (pos + align - 1) // align * align
    out = {}
    for name, ne, t, off in infos:
        if wanted is not None and name not in wanted:
            continue
        n = int(np.prod(ne))
        o = base + off
        if t == GGML_F32:
            a = np.frombuffer(mm, np.float32, n, o).copy()
        elif t == GGML_F16:
            a = np.frombuffer(mm, np.float16, n, o).astype(np.float32)
        elif t == GGML_BF16:
            a = (np.frombuffer(mm, np.uint16, n, o).astype(np.uint32) << 16).view(np.float32)
        elif t == GGML_Q8_0:
            blk = np.frombuffer(mm, np.uint8, n // 32 * 34, o).reshape(-1, 34)
            d = blk[:, :2].copy().view(np.float16).astype(np.float32)
            q = blk[:, 2:].copy().view(np.int8).astype(np.float32)
            a = (q * d).reshape(-1)
        else:
            raise ValueError(f"{name}: ggml type {t} not supported here")
        out[name] = a.reshape(list(reversed(ne)))
    return out


def bf16_bits(a):
    """float32 -> bf16 bit pattern, round to nearest even (same as mm3-dit-trt.h bf16())."""
    b = np.ascontiguousarray(a, np.float32).view(np.uint32)
    r = ((b + np.uint32(0x7FFF) + ((b >> 16) & np.uint32(1))) >> 16).astype(np.uint16)
    nan = (b & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    r[nan] = ((b[nan] >> 16) | np.uint32(0x40)).astype(np.uint16)
    return r


def e4m3(x):
    """float32 -> E4M3FN bytes, saturating at 448, round to nearest even."""
    x = np.asarray(x, np.float32)
    a = np.minimum(np.abs(x), np.float32(448))
    f, E = np.frexp(a)                       # a = f * 2^E, f in [0.5, 1)
    e = E.astype(np.int32) - 1               # a = (2f) * 2^e
    normal = e >= -6
    m = np.rint((2 * f - 1) * 8).astype(np.int32)
    e = np.where(m == 8, e + 1, e)
    m = np.where(m == 8, 0, m)
    code_n = ((e + 7) << 3) | m
    code_s = np.rint(a * np.float32(512)).astype(np.int32)   # subnormal: a / 2^-9 (8 -> 0x08 = 2^-6)
    code = np.where(normal, code_n, code_s)
    code = np.where(a == 0, 0, code)
    return (code | (np.signbit(x).astype(np.int32) << 7)).astype(np.uint8)


def e4m3_decode(b):
    b = b.astype(np.int32)
    e, m = (b >> 3) & 15, b & 7
    v = np.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * 2.0 ** (e - 7))
    return np.where(b & 128, -v, v).astype(np.float32)


FP4_MID = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], np.float32)


def nvfp4(W):
    """W [K,N] -> (packed E2M1 bytes [K*N/2], E4M3 block scales [K/16,N], FP32 global). Block 16 along K."""
    K, N = W.shape
    amax = np.abs(W.reshape(K // 16, 16, N)).max(1)
    s_hp = np.maximum(amax / 6.0, 1e-12).astype(np.float32)
    g = np.float32(s_hp.max() / 448.0)
    s8 = e4m3(s_hp / g)
    s_eff = np.repeat(e4m3_decode(s8) * g, 16, axis=0)
    s_eff[s_eff == 0] = 1
    v = W / s_eff
    idx = np.searchsorted(FP4_MID, np.minimum(np.abs(v), 6.0)).astype(np.uint8)
    nib = (idx | ((v < 0).astype(np.uint8) << 3)).reshape(-1)
    return (nib[0::2] | (nib[1::2] << 4)).astype(np.uint8), s8, g


def engine_weights(gguf):
    """engine_name -> float32 array in engine layout, from the manifest + GGUF."""
    man = manifest()
    src = read_gguf(gguf, {e["gguf_name"] for e in man["weights"]})
    out = {}
    for e in man["weights"]:
        a = src[e["gguf_name"]]
        if e["transpose"]:
            a = np.ascontiguousarray(a.reshape(a.shape[-2], a.shape[-1]).T)
        out[e["engine_name"]] = a.reshape(e["dims"]).astype(np.float32)
    return out


# ---------------------------------------------------------------- network
def parse():
    b = trt.Builder(LOG)
    net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    p = trt.OnnxParser(net, LOG)
    if not p.parse_from_file(ONNX):
        raise RuntimeError([p.get_error(i).desc() for i in range(p.num_errors)])
    return b, net, p


def layers(net):
    for i in range(net.num_layers):
        yield net.get_layer(i)


def consumers(net):
    c = {}
    for L in layers(net):
        for i in range(L.num_inputs):
            t = L.get_input(i)
            if t is not None:
                c.setdefault(t.name, []).append((L, i))
    return c


def producers(net):
    p = {}
    for L in layers(net):
        for i in range(L.num_outputs):
            p[L.get_output(i).name] = L
    return p


def mlp_gemms(net):
    """[(weight name, const layer, broadcast shuffle, matmul, kind)] for the 72 ff_in/ff_out MatMuls."""
    init_ff_names()
    cons = consumers(net)
    found = []
    for L in layers(net):
        if L.type == trt.LayerType.CONSTANT and L.name in FF_NAMES:
            (S, _), = cons[L.get_output(0).name]
            assert S.type == trt.LayerType.SHUFFLE
            (M, idx), = cons[S.get_output(0).name]
            assert M.type == trt.LayerType.MATRIX_MULTIPLY and idx == 1
            K = L.get_output(0).shape[0]
            found.append((L.name, L, S, M, "ffn_in" if K == 2048 else "ffn_out"))
    assert len(found) == 72, len(found)
    return found


def named_const(net, arr, dt, name, keep):
    arr = np.ascontiguousarray(arr)
    keep.append(arr)
    w = trt.Weights(dt, arr.ctypes.data, arr.size)
    c = net.add_constant(arr.shape, w)
    assert net.set_weights_name(w, name), name
    return c.get_output(0)


def add_fp8(net, keep, sw_values=None, act_amax=None, silu32=True, prequant=False):
    """Q/DQ the 72 MLP GEMMs: per-output-channel FP8 weight scale (fp8w.<w>, BF16 [N]) and a
    per-tensor activation scale (fp8a.<w>; BF16, or FP32 for ff_out when the GLU runs in FP32).
    With silu32 the GLU (value * silu(gate)) is recomputed in FP32 and quantized from FP32."""
    prod = producers(net)
    only = os.environ.get("FP8_ONLY")          # ffn_in | ffn_out
    skip = {int(b) for b in os.environ.get("FP8_SKIP_BLOCKS", "").split(",") if b}
    gem = mlp_gemms(net)
    for bi, (wname, C, S, M, kind) in enumerate(gem):
        if (only and kind != only) or (bi // 2) in skip:
            continue
        K, N = C.get_output(0).shape
        sw = sw_values[wname] if sw_values is not None else np.full(N, 1.0 / 448, np.float32)
        swc = named_const(net, bf16_bits(sw), trt.DataType.BF16, "fp8w." + wname, keep)
        if prequant:
            q8 = np.full((K, N), 0x30, np.uint8)  # placeholder 1.0; refit sets fp8q.<w>
            q = named_const(net, q8, trt.DataType.FP8, "fp8q." + wname, keep)
        else:
            q = net.add_quantize(C.get_output(0), swc, trt.DataType.FP8)
            q.axis = 1
            q = q.get_output(0)
        dw = net.add_dequantize(q, swc, trt.DataType.BF16)
        dw.axis = 1
        S.set_input(0, dw.get_output(0))
        a = M.get_input(0)
        amax = act_amax[wname] if act_amax else 64.0
        if kind == "ffn_out" and silu32:
            mul = prod[a.name]
            value, silu = mul.get_input(0), mul.get_input(1)
            gate = prod[silu.name].get_input(0)
            g = net.add_cast(gate, trt.DataType.FLOAT).get_output(0)
            sg = net.add_activation(g, trt.ActivationType.SIGMOID).get_output(0)
            si = net.add_elementwise(g, sg, trt.ElementWiseOperation.PROD).get_output(0)
            v = net.add_cast(value, trt.DataType.FLOAT).get_output(0)
            a = net.add_elementwise(v, si, trt.ElementWiseOperation.PROD).get_output(0)
            # Myelin needs both matmul operand scales in one type: back to BF16 before Q
            a = net.add_cast(a, trt.DataType.BF16).get_output(0)
        if True:
            sac = named_const(net, bf16_bits(np.array(amax / 448.0, np.float32)), trt.DataType.BF16,
                              "fp8a." + wname, keep)
        qa = net.add_quantize(a, sac, trt.DataType.FP8)
        da = net.add_dequantize(qa.get_output(0), sac, trt.DataType.BF16)
        M.set_input(0, da.get_output(0))


def add_nvfp4(net, keep, act_amax=None):
    """NVFP4 on the MLP GEMMs: pre-quantized FP4 weights (fp4q/fp4s/fp4g, set at refit) and dynamic
    FP4 activations (block 16, FP8 block scales, global fp4a)."""
    only = os.environ.get("FP8_ONLY")
    skip = {int(b) for b in os.environ.get("FP8_SKIP_BLOCKS", "").split(",") if b}
    for bi, (wname, C, S, M, kind) in enumerate(mlp_gemms(net)):
        if (only and kind != only) or (bi // 2) in skip:
            continue
        K, N = C.get_output(0).shape
        w4 = np.full(K * N // 2, 0x22, np.uint8)               # placeholder 1.0
        keep.append(w4)
        ww = trt.Weights(trt.DataType.FP4, w4.ctypes.data, K * N)
        c4 = net.add_constant((K, N), ww)
        assert net.set_weights_name(ww, "fp4q." + wname)
        s8 = named_const(net, np.full((K // 16, N), 0x38, np.uint8), trt.DataType.FP8, "fp4s." + wname, keep)
        g = named_const(net, np.array(1.0, np.float32), trt.DataType.FLOAT, "fp4g." + wname, keep)
        sw = net.add_dequantize(s8, g, trt.DataType.BF16).get_output(0)  # FP32 here fails type inference with a BF16 output
        dw = net.add_dequantize(c4.get_output(0), sw, trt.DataType.BF16)
        dw.axis = 0
        S.set_input(0, dw.get_output(0))
        a = M.get_input(0)
        amax = act_amax[wname] if act_amax else 64.0
        ga = named_const(net, bf16_bits(np.array(amax / 6.0 / 448.0, np.float32)), trt.DataType.BF16,
                         "fp4a." + wname, keep)
        dq = net.add_dynamic_quantize(a, 2, 16, trt.DataType.FP4, trt.DataType.FP8)
        dq.set_input(1, ga)
        sx = net.add_dequantize(dq.get_output(1), ga, trt.DataType.BF16).get_output(0)
        ax = net.add_dequantize(dq.get_output(0), sx, trt.DataType.BF16)
        ax.axis = 2
        M.set_input(0, ax.get_output(0))


def fp8_weight_scales(W):
    return {k: np.maximum(np.abs(v).max(0), 1e-8) / 448.0 for k, v in W.items() if k in FF_NAMES}


FF_NAMES = set()


def init_ff_names():
    for e in manifest()["weights"]:
        if ".ffn_in.weight" in e["gguf_name"] or ".ffn_out.weight" in e["gguf_name"]:
            FF_NAMES.add(e["engine_name"])


def bake_weights(net, W, keep):
    """Replace every manifest weight's values (no-refit build)."""
    n = 0
    for L in layers(net):
        if L.type == trt.LayerType.CONSTANT and L.name in W:
            L.__class__ = trt.IConstantLayer
            arr = bf16_bits(W[L.name])
            keep.append(arr)
            L.weights = trt.Weights(trt.DataType.BF16, arr.ctypes.data, arr.size)
            n += 1
        elif L.type == trt.LayerType.CONVOLUTION:
            L.__class__ = trt.IConvolutionLayer
            pre = "preprocess_conv" if L.num_output_maps == 2304 else "postprocess_conv"
            k = bf16_bits(W[pre + ".weight"])
            keep.append(k)
            L.kernel = trt.Weights(trt.DataType.BF16, k.ctypes.data, k.size)
            n += 1
    assert n == 368, n


def config(b, refit, detailed=True):
    c = b.create_builder_config()
    c.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    c.set_flag(trt.BuilderFlag.TF32)
    if refit:
        c.set_flag(trt.BuilderFlag.REFIT)
    if detailed:
        c.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    p = b.create_optimization_profile()
    for name, mn, opt in (("x", (1, 128, 3), (2, 128, 689)), ("cond", (1, 2048, 3), (2, 2048, 689)),
                          ("timestep_fourier", (1, 256), (2, 256)), ("rope_cos", (4, 32), (690, 32)),
                          ("rope_sin", (4, 32), (690, 32))):
        p.set_shape(name, mn, opt, opt)
    c.add_optimization_profile(p)
    return c


# ---------------------------------------------------------------- runtime
class Runner:
    def __init__(self, plan):
        self.rt = trt.Runtime(LOG)
        self.eng = self.rt.deserialize_cuda_engine(plan)
        assert self.eng is not None
        self.ctx = None
        self.stream = ck(cr.cudaStreamCreate())
        self.buf = {}

    def refit(self, W, act_amax=None):
        r = trt.Refitter(self.eng, LOG)
        names = set(r.get_all_weights())
        hold = []
        fp4_cache = {}
        for name in sorted(names):
            if name in W and not name.startswith("fp8"):
                arr = W[name]
            elif name.startswith("fp8w."):
                arr = np.maximum(np.abs(W[name[5:]]).max(0), 1e-8) / 448.0
            elif name.startswith("fp8q."):
                w = W[name[5:]]
                arr = e4m3(w / (np.maximum(np.abs(w).max(0), 1e-8) / 448.0))
            elif name[:5] in ("fp4q.", "fp4s.", "fp4g."):
                key = name[5:]
                if key not in fp4_cache:
                    fp4_cache.clear()
                    fp4_cache[key] = nvfp4(W[key])
                q, s8, g = fp4_cache[key]
                arr = {"fp4q.": q, "fp4s.": s8, "fp4g.": np.array([g], np.float32)}[name[:5]]
            elif name.startswith("fp4a."):
                arr = np.array([act_amax[name[5:]] / 6.0 / 448.0], np.float32)
            elif name.startswith("fp8a."):
                arr = np.array([act_amax[name[5:]] / 448.0], np.float32)
            else:
                continue  # val_8 / val_137: real scalars from the ONNX
            proto = r.get_weights_prototype(name)
            if proto.dtype == trt.DataType.FP4:
                b = np.ascontiguousarray(arr, np.uint8).reshape(-1)
                hold.append(b)
                assert r.set_named_weights(name, trt.Weights(proto.dtype, b.ctypes.data, b.size * 2)), name
                continue
            if proto.dtype == trt.DataType.FP8:
                b = np.ascontiguousarray(arr, np.uint8).reshape(-1)
            elif proto.dtype == trt.DataType.FLOAT:
                b = np.ascontiguousarray(arr, np.float32).reshape(-1)
            else:
                assert proto.dtype == trt.DataType.BF16, (name, proto.dtype)
                b = bf16_bits(arr).reshape(-1)
            assert b.size == proto.size, (name, b.size, proto.size)
            hold.append(b)
            assert r.set_named_weights(name, trt.Weights(proto.dtype, b.ctypes.data, b.size)), name
        skipped = sorted(set(W) - names)
        if skipped:
            print(f"  {len(skipped)} manifest weights not refittable (folded away), e.g. {skipped[:2]}")
        assert not r.get_missing_weights(), r.get_missing_weights()
        t = time.perf_counter()
        assert r.refit_cuda_engine()
        return time.perf_counter() - t

    def ensure_ctx(self):
        if self.ctx is None:
            self.ctx = self.eng.create_execution_context()

    def set(self, name, arr):
        arr = np.ascontiguousarray(arr, np.float32)
        if name not in self.buf or self.buf[name][1] < arr.nbytes:
            self.buf[name] = (ck(cr.cudaMalloc(max(arr.nbytes, 4))), arr.nbytes)
        ck(cr.cudaMemcpy(self.buf[name][0], arr.ctypes.data, arr.nbytes, cr.cudaMemcpyKind.cudaMemcpyHostToDevice))
        if name in ("x", "cond", "timestep_fourier", "rope_cos", "rope_sin"):
            self.ctx.set_input_shape(name, arr.shape)
        self.ctx.set_tensor_address(name, self.buf[name][0])

    def inputs(self, x, cond, t, basis, B=2):
        """x [128,L] chan-major, cond [2048,L] chan-major (already gated), row 1 = zeros cond."""
        self.ensure_ctx()
        L = x.shape[1]
        bx = np.stack([x] * B)
        bc = np.zeros((B, 2048, L), np.float32)
        bc[0] = cond
        ang = 2 * np.pi * np.float64(t) * basis.astype(np.float64)
        f = np.concatenate([np.cos(ang), np.sin(ang)]).astype(np.float32)
        p = np.arange(L + 1)[:, None] / np.power(10000.0, np.arange(0, 32, 2)[None, :16] / 32)
        cos = np.concatenate([np.cos(p)] * 2, 1).astype(np.float32)
        sin = np.concatenate([np.sin(p)] * 2, 1).astype(np.float32)
        self.set("x", bx)
        self.set("cond", bc)
        self.set("timestep_fourier", np.stack([f] * B))
        self.set("rope_cos", cos)
        self.set("rope_sin", sin)
        for i in range(self.eng.num_io_tensors):
            n = self.eng.get_tensor_name(i)
            if self.eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.ctx.get_tensor_shape(n))
                isz = 4 if self.eng.get_tensor_dtype(n) == trt.DataType.FLOAT else 2
                nbytes = int(np.prod(shape)) * isz
                if n not in self.buf or self.buf[n][1] < nbytes:
                    self.buf[n] = (ck(cr.cudaMalloc(nbytes)), nbytes)
                self.ctx.set_tensor_address(n, self.buf[n][0])

    def run(self):
        assert self.ctx.execute_async_v3(self.stream)
        ck(cr.cudaStreamSynchronize(self.stream))

    def get(self, name):
        shape = tuple(self.ctx.get_tensor_shape(name))
        tdt = self.eng.get_tensor_dtype(name)
        dt = {trt.DataType.FLOAT: np.float32, trt.DataType.HALF: np.float16, trt.DataType.BF16: np.uint16}[tdt]
        out = np.empty(shape, dt)
        ck(cr.cudaMemcpy(out.ctypes.data, self.buf[name][0], out.nbytes, cr.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        if self.eng.get_tensor_dtype(name) == trt.DataType.BF16:
            out = (out.astype(np.uint32) << 16).view(np.float32)
        return out

    def time(self, n=50):
        for _ in range(5):
            self.ctx.execute_async_v3(self.stream)
        ck(cr.cudaStreamSynchronize(self.stream))
        e0, e1 = ck(cr.cudaEventCreate()), ck(cr.cudaEventCreate())
        ck(cr.cudaEventRecord(e0, self.stream))
        for _ in range(n):
            self.ctx.execute_async_v3(self.stream)
        ck(cr.cudaEventRecord(e1, self.stream))
        ck(cr.cudaEventSynchronize(e1))
        return ck(cr.cudaEventElapsedTime(e0, e1)) / n


class Prof(trt.IProfiler):
    def __init__(self):
        trt.IProfiler.__init__(self)
        self.t = {}

    def report_layer_time(self, name, ms):
        self.t[name] = self.t.get(name, 0.0) + ms


def basis_from(gguf):
    return read_gguf(gguf, {"dit.time_fourier.weight"})["dit.time_fourier.weight"].reshape(-1)[:128]


# ---------------------------------------------------------------- commands
def cmd_inspect(a):
    b, net, p = parse()
    print("layers", net.num_layers)
    for L in layers(net):
        ins = [net and (L.get_input(i).name if L.get_input(i) is not None else None) for i in range(L.num_inputs)]
        outs = [(L.get_output(i).name, L.get_output(i).dtype, tuple(L.get_output(i).shape)) for i in range(L.num_outputs)]
        if "blocks.0." in L.name or any("blocks.0" in (n or "") for n in ins) or a.all:
            print(L.type, L.name, ins, outs)


def cmd_build(a):
    init_ff_names()
    b, net, p = parse()
    keep = []
    W = engine_weights(a.gguf) if a.gguf else None
    act = json.load(open(a.calib))["act_amax"] if a.calib else None
    if a.norefit and W is not None:
        bake_weights(net, W, keep)
    if a.variant == "nvfp4mlp":
        add_nvfp4(net, keep, act)
    elif a.variant.startswith("fp8"):
        add_fp8(net, keep, fp8_weight_scales(W) if (W is not None and a.norefit) else None, act,
                prequant=a.variant == "fp8mlpq")
    cfg = config(b, refit=not a.norefit)
    t = time.perf_counter()
    plan = b.build_serialized_network(net, cfg)
    assert plan is not None, "build failed"
    with open(a.out, "wb") as f:
        f.write(plan)
    print(f"built {a.out} {plan.nbytes/1e9:.2f} GB in {time.perf_counter()-t:.0f} s")


def cmd_time(a):
    plan = open(a.plan, "rb").read()
    r = Runner(plan)
    if a.gguf and r.eng.refittable and "mm3-trt-cache" not in a.plan:
        init_ff_names()
        W = engine_weights(a.gguf)
        act = json.load(open(a.calib))["act_amax"] if a.calib else None
        print(f"refit {r.refit(W, act):.1f} s")
    rng = np.random.default_rng(0)
    L = 689
    basis = basis_from(a.gguf) if a.gguf else rng.standard_normal(128).astype(np.float32)
    r.inputs(rng.standard_normal((128, L)).astype(np.float32), rng.standard_normal((2048, L)).astype(np.float32) * 0.5,
             0.5, basis)
    r.run()
    ms = [r.time(a.n) for _ in range(3)]
    print(f"B=2 L=689: {' '.join(f'{m:.2f}' for m in ms)} ms/enqueue  -> {min(ms)*30/1000:.3f} s per 30-step window")
    if a.profile:
        prof = Prof()
        r.ctx.profiler = prof
        for _ in range(a.n):
            r.run()
        tot = sum(prof.t.values())
        info = json.loads(r.eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
        kinds = {}
        for L in info["Layers"]:
            if isinstance(L, dict):
                kinds[L.get("Name")] = (L.get("LayerType", "?"), L.get("TacticName", L.get("Metadata", "")))
        groups = {}
        for name, t in prof.t.items():
            lt, tac = kinds.get(name, ("?", ""))
            g = lt
            if lt in ("gemm", "CaskGemmMatrixMultiply", "MatrixMultiply") or "gemm" in str(tac).lower():
                g = "GEMM"
            groups.setdefault(g, 0.0)
            groups[g] += t
        print(f"profiled total {tot/a.n:.2f} ms/enqueue (profiler adds sync per layer)")
        for g, t in sorted(groups.items(), key=lambda x: -x[1]):
            print(f"  {g:40s} {t/a.n:7.2f} ms  {100*t/tot:5.1f}%")
        top = sorted(prof.t.items(), key=lambda x: -x[1])[:25]
        for name, t in top:
            print(f"  {t/a.n:6.3f} ms  {kinds.get(name, ('?', ''))[0]:24s} {name[:150]}")
        with open(a.plan + ".layers.json", "w") as f:
            json.dump({"per_layer_ms": {k: v / a.n for k, v in prof.t.items()}, "info": info}, f)


# ---------------------------------------------------------------- dumps (MM3_DIT_TRT_DUMP)
def read_dump(path):
    """One C++ call: dict L, B, t, gate, x [128,L], cond [L,2048] or None, out [B,128,L]."""
    raw = open(path, "rb").read()
    L, B = struct.unpack_from("<ii", raw, 0)
    t, gate = struct.unpack_from("<ff", raw, 8)
    has = struct.unpack_from("<i", raw, 16)[0]
    o = 20
    x = np.frombuffer(raw, np.float32, 128 * L, o).reshape(128, L); o += 128 * L * 4
    cond = None
    if has:
        cond = np.frombuffer(raw, np.float32, 2048 * L, o).reshape(L, 2048); o += 2048 * L * 4
    out = np.frombuffer(raw, np.float32, B * 128 * L, o).reshape(B, 128, L)
    return dict(L=L, B=B, t=t, gate=gate, x=x, cond=cond, out=out)


def dump_calls(d):
    files = sorted(glob.glob(os.path.join(d, "call_*.bin")))
    assert files, d
    return files


def windows(d):
    """[[call dict...] per window], a window starts at a call that carries cond."""
    ws = []
    for f in dump_calls(d):
        c = read_dump(f)
        if c["cond"] is not None:
            ws.append([])
        ws[-1].append(c)
    return ws


def final_latent(win, cfg):
    last = win[-1]
    c, u = last["out"][0], last["out"][1]
    return last["x"] + (1.0 - last["t"]) * (u + cfg * (c - u))


def corr(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def cmd_compare(a):
    A, B = windows(a.a), windows(a.b)
    assert len(A) == len(B), (len(A), len(B))
    cs, rs = [], []
    for k, (wa, wb) in enumerate(zip(A, B)):
        fa, fb = final_latent(wa, a.cfg), final_latent(wb, a.cfg)
        c = corr(fa, fb)
        r = float(np.linalg.norm(fa - fb) / np.linalg.norm(fa))
        v = [corr(ca["out"], cb["out"]) for ca, cb in zip(wa, wb)]
        cs.append(c); rs.append(r)
        print(f"window {k:2d} L={wa[0]['L']} steps={len(wa)} latent corr {c:.6f} rel {r:.4f}  "
              f"velocity corr step0 {v[0]:.6f} min {min(v):.6f}")
    print(f"ALL windows: latent corr min {min(cs):.6f} mean {np.mean(cs):.6f}; rel max {max(rs):.4f}")
    fa = np.concatenate([final_latent(w, a.cfg) for w in A], 1)
    fb = np.concatenate([final_latent(w, a.cfg) for w in B], 1)
    print(f"concatenated full-clip latent corr {corr(fa, fb):.6f}")


def calib_network():
    """BF16 refit network with max |input| of each of the 72 MLP GEMMs as extra
    scalar outputs (reduced on the GPU: copying the full activations to the host
    took ~1.3 s per call)."""
    b, net, p = parse()
    for wname, C, S, M, kind in mlp_gemms(net):
        t = M.get_input(0)
        a = net.add_unary(t, trt.UnaryOperation.ABS).get_output(0)
        r = net.add_reduce(a, trt.ReduceOperation.MAX, (1 << len(t.shape)) - 1, False).get_output(0)
        r.name = "act." + wname
        net.mark_output(r)
    return b, net


def cmd_calib(a):
    init_ff_names()
    plan_path = os.path.join(HERE, "calib-bf16-amax.plan")
    if not os.path.exists(plan_path):
        b, net = calib_network()
        plan = b.build_serialized_network(net, config(b, refit=True, detailed=False))
        assert plan is not None
        open(plan_path, "wb").write(plan)
    r = Runner(open(plan_path, "rb").read())
    W = engine_weights(a.gguf)
    print(f"refit {r.refit(W):.1f} s")
    basis = basis_from(a.gguf)
    amax = {}
    per_dump = {}
    worst = 1.0
    calls = 0
    for d in a.dump:
        da = per_dump[os.path.basename(os.path.normpath(d))] = {}
        for w in windows(d):
            cond = w[0]["cond"].T  # later calls of a window reuse its condition
            for c in w[::a.stride]:
                r.inputs(c["x"], cond, c["t"], basis, B=c["B"])
                r.run()
                worst = min(worst, corr(r.get("output"), c["out"]))
                for n in FF_NAMES:
                    da[n] = max(da.get(n, 0.0), float(r.get("act." + n)))
                calls += 1
        for n, v in da.items():
            amax[n] = max(amax.get(n, 0.0), v)
        print(f"{d}: done, {calls} calls so far")
    print(f"{calls} calls replayed; min velocity corr vs C++ dump {worst:.6f}")
    vals = np.array(list(amax.values()))
    print(f"act amax over MLP inputs: min {vals.min():.2f} median {np.median(vals):.2f} max {vals.max():.2f}")
    json.dump({"version": 1, "source": os.path.basename(a.gguf), "calls": calls, "margin": a.margin,
               "dumps": sorted(per_dump),
               "act_amax": {k: v * a.margin for k, v in sorted(amax.items())},
               "per_dump": {d: {k: v for k, v in sorted(m.items())} for d, m in sorted(per_dump.items())}},
              open(a.out, "w"), indent=1)
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("placeholder")
    s = sp.add_parser("inspect")
    s.add_argument("--all", action="store_true")
    s = sp.add_parser("build")
    s.add_argument("--variant", default="bf16", choices=["bf16", "fp8mlp", "fp8mlpq", "nvfp4mlp"])
    s.add_argument("--out", required=True)
    s.add_argument("--norefit", action="store_true")
    s.add_argument("--gguf")
    s.add_argument("--calib")
    s = sp.add_parser("time")
    s.add_argument("--plan", required=True)
    s.add_argument("--gguf")
    s.add_argument("--calib")
    s.add_argument("--profile", action="store_true")
    s.add_argument("--n", type=int, default=30)
    s = sp.add_parser("compare")
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--cfg", type=float, default=1.7)
    s = sp.add_parser("calib")
    s.add_argument("--dump", required=True, nargs="+")
    s.add_argument("--gguf", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--stride", type=int, default=1)
    s.add_argument("--margin", type=float, default=2.0)
    a = ap.parse_args()
    {"placeholder": lambda a: write_placeholder(), "inspect": cmd_inspect, "build": cmd_build, "time": cmd_time,
     "compare": cmd_compare, "calib": cmd_calib}[a.cmd](a)


if __name__ == "__main__":
    main()
