#pragma once
// dit-train-graph.h — the trainable DiT forward and the §4.5 product loss.
//
// Productionised from spike-dit2-graph.h (d2_layer / d2_build_forward), which
// the adversarial verifier audited op-for-op against dit-graph.h. The spike
// headers are reference material only and are NOT included here.
//
// Properties reproduced exactly, each load-bearing (plan §3.4):
//  1. NO flash attention — GGML_OP_FLASH_ATTN_EXT has no backward. dit_attn_f32
//     (soft_max_ext) is exact: its backward uses the softmax OUTPUT, so mask and
//     scale are baked in and masked positions get exactly zero gradient. The
//     retained [S,S,Nh] softmax output is the S^2 term in the VRAM model.
//  2. REAL masks, both. sa_mask only when layer_type == 0 (sliding window);
//     layer_type == 1 is full attention (dit.h:656). ca_mask always. Spike round 1
//     passed nullptr and half the layers silently ran full attention.
//  3. ggml_swiglu_split — the fused ggml_swiglu has no backward.
//  4. Cross-attn Q/K norm BEFORE the permute is the same op as the inference
//     graph's norm-after-permute: rms_norm acts on ne0 == D in both layouts.
//  5. AdaLN: adaln = reshape_1d(scale_shift_table, 6H) + tproj, six [H] views in
//     order shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff. Output head
//     adds temb to BOTH out_shift and out_scale (dit-graph.h:883-884).
//  6. Cross-attention has NO gate — plain residual add.
//
// docs/plans/2026-07-28-dit-trainer-implementation.md §3.4 / §3.5

#include "dit-graph.h"
#include "dit.h"
#include "train/dit-adapter.h"
#include "train/dit-mirror.h"

#include <string>
#include <utility>
#include <vector>

// ─── attention mode (docs/plans/2026-09-01-flash-attn-backward.md) ──────────
//
// DIT_ATTN_EXACT is the shipped graph and the default: dit_attn_f32's
// mul_mat -> soft_max_ext -> mul_mat -> cont(permute) chain, retained
// [S_kv,S,Nh,B] softmax and all. DIT_ATTN_FLASH routes BOTH attention sites
// through the fused GGML_OP_FLASH_ATTN_TRAIN / _BACK pair, which never
// materialises that softmax (docs/plans/fattn-train-spec.md §2, §3).
//
// Exact mode must emit the byte-identical graph it emitted before this flag
// existed — the §2.3.1 anchor and the SC1-SC3 rungs are the proof — so the mode
// is tested at the two dit_attn_* call sites and NOWHERE else. A ternary whose
// false arm is the original call leaves the original node sequence untouched.
enum DitAttnMode { DIT_ATTN_EXACT = 0, DIT_ATTN_FLASH = 1 };

// The fused drop-in for dit_attn_f32. Same arguments, and the SAME result shape
// and layout: ggml_flash_attn_train_get_o is a CONTIGUOUS [D,Nh,S,B] view, which
// is what dit_attn_f32's closing ggml_cont(ggml_permute(...)) produces too, so
// the ggml_reshape_3d(attn, Nh*D, S, B) at both call sites accepts it unchanged.
//
// Note what is NOT here: no ggml_cont on q/k/v. They arrive as permuted views
// and the op reads them through nb[1..3] on purpose — materialising them would
// hand back part of the saving (spec §1.1.3).
//
// HOT-Step patch: flash-attn-train — `prec` is the ARITHMETIC request, set on
// the forward node only. ggml_compute_backward copies it onto the _BACK node it
// builds, which is load-bearing: the backward recomputes S and reuses the
// forward's LSE, so a backward rounding differently from its forward is not
// rounding error, it is a different function (tf32 design §3.2).
static ggml_tensor * dit_attn_flash(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                    ggml_tensor * mask, float scale, ggml_prec prec) {
    ggml_tensor * packed = ggml_flash_attn_train(ctx, q, k, v, mask, scale);
    ggml_flash_attn_train_set_prec(packed, prec);
    return ggml_flash_attn_train_get_o(ctx, packed);  // [D,Nh,S,B]
}

// ─── spec §9.8: the supports_op probe ───────────────────────────────
//
// A `false` from ggml_backend_supports_op is not a failure the trainer would
// ever see: backend_sched_new registers the CPU backend alongside the GPU one,
// so the scheduler would simply SPLIT the graph — Q/K/V and the F16 mask copied
// across PCIe for every layer of every step. Correct, unusably slow, and with
// LOW VRAM and a silent NVML tripwire, i.e. indistinguishable from a pass on
// every number the run reports. So flash mode asks the question explicitly, at
// init, before the first graph, and refuses to start rather than fall back.
//
// The probe uses the run's real D / Nh / Nkv / S / S_kv / B because a capability
// check is allowed to be shape-dependent, and ours is (D must be 64 or 128).
// Contiguous probe tensors are faithful enough: the only stride the CUDA check
// looks at is nb[0], which is sizeof(float) for the real permuted views too.
static bool dit_flash_probe(ggml_backend_t backend, int D, int Nh, int Nkv, int S, int S_kv, int B, float scale,
                            bool * fwd_ok, bool * bwd_ok) {
    *fwd_ok = false;
    *bwd_ok = false;
    ggml_context * probe;
    {
        ggml_init_params p = { 64 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/true };
        probe              = ggml_init(p);
    }
    if (!probe) {
        return false;
    }
    ggml_tensor * pq = ggml_new_tensor_4d(probe, GGML_TYPE_F32, D, S, Nh, B);
    ggml_tensor * pk = ggml_new_tensor_4d(probe, GGML_TYPE_F32, D, S_kv, Nkv, B);
    ggml_tensor * pv = ggml_new_tensor_4d(probe, GGML_TYPE_F32, D, S_kv, Nkv, B);
    ggml_tensor * pm = ggml_new_tensor_2d(probe, GGML_TYPE_F16, S_kv, S);
    ggml_tensor * pf = ggml_flash_attn_train(probe, pq, pk, pv, pm, scale);
    *fwd_ok          = ggml_backend_supports_op(backend, pf);
    // The back node needs a forward-shaped `fwd` and a same-sized `dfwd`; the
    // forward node itself is exactly the former.
    ggml_tensor * pd = ggml_new_tensor_1d(probe, GGML_TYPE_F32, ggml_flash_attn_train_nelements(pq));
    ggml_tensor * pb = ggml_flash_attn_train_back(probe, pq, pk, pv, pm, pf, pd, scale);
    *bwd_ok          = ggml_backend_supports_op(backend, pb);
    ggml_free(probe);
    return *fwd_ok && *bwd_ok;
}

// HOT-Step patch: flash-attn-train
//
// Which arithmetic the backend's LAST fused-attention launch actually used
// (dir 0 = forward, 1 = backward): "tf32", or "f32 (<reason>)" when the request
// was overridden, or "n/a" when the backend has no such kernels or none has run.
//
// This is not a restatement of --attn. The request is a request: the CUDA
// dispatch drops to v1's scalar kernels on pre-Ampere devices, at D != 128, and
// on an 8-byte-unaligned view (tf32 design §3.3), so two runs whose logs both
// say "flash" can differ in arithmetic depending on the GPU they landed on.
// Resolved through the backend registry rather than linked, because ggml-cuda is
// a loadable module and the trainer must build without it.
static const char * dit_flash_last_prec(ggml_backend_t backend, int dir) {
    typedef const char * (*fa_last_prec_fn)(int);
    ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg) {
        return "n/a";
    }
    fa_last_prec_fn fn =
        (fa_last_prec_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_fattn_train_last_prec");
    return fn ? fn(dir) : "n/a";
}

// The two directions collapsed into one log field. They resolve through the same
// helper and normally agree; when they do not, say so rather than pick one.
static std::string dit_flash_prec_label(ggml_backend_t backend) {
    const std::string f = dit_flash_last_prec(backend, 0);
    const std::string b = dit_flash_last_prec(backend, 1);
    return (f == b) ? f : ("fwd " + f + " / bwd " + b);
}

// ─── graph inputs (every one re-uploaded EVERY micro-step, §3.0) ────────────
//
// Every shape below carries an OPTIONAL trailing batch axis B (design B1). B == 1
// reproduces the pre-batching layout exactly: a [X, Y, 1] view has the same ne/nb
// as the [X, Y] tensor it replaces, so ggml dispatches identically and the §2.3.1
// byte-identity anchor survives. See dit_train_forward's `B` parameter.
struct DitInputs {
    ggml_tensor * t_input = nullptr;  // [in_ch, T,  B]     concat(context, xt) per frame
    ggml_tensor * t_enc   = nullptr;  // [enc_H, enc_S, B]
    ggml_tensor * t_pos   = nullptr;  // [S*B] i32, 0..S-1 repeated per element
    ggml_tensor * t_temb  = nullptr;  // [H,  1, B]  precomputed
    ggml_tensor * t_tproj = nullptr;  // [6H, 1, B]  precomputed
    // [S,S] f16 sliding-window mask, SHARED over heads AND batch (nothing padded
    // — the fast path), or [S,S,1,B] per element when any element in the batch is
    // padded, so the pad can be masked out as attention KEYS (dit-data.h).
    ggml_tensor * t_sa    = nullptr;
    // The same per-element pad mask WITHOUT the sliding window, for layer_type 1
    // (full attention), which otherwise takes no self-attention mask at all and
    // would let every padded frame straight back into the valid queries. nullptr
    // whenever nothing is padded, which restores the exact pre-batching graph.
    ggml_tensor * t_sa_pad = nullptr;
    ggml_tensor * t_ca    = nullptr;  // [enc_S, S, 1, B] f16 encoder-padding mask, per element
    ggml_tensor * t_vtgt  = nullptr;  // [Oc, T, B] velocity target
    ggml_tensor * t_cw    = nullptr;  // [Oc] channel weights (mean 1)
    // Design B4. [1, T, B] per-frame loss weights carrying the valid-frame mask
    // (padded frames are exactly 0) and, when B > 1, the per-element flow_snr
    // weight. nullptr = the pre-batching loss (no mask multiply at all).
    ggml_tensor * t_lw = nullptr;
    // Same shape, weight-free: the UNWEIGHTED masked mean, for the `rawLoss`
    // report only. Never flagged as a loss, so the backward pass skips it.
    ggml_tensor * t_lwu = nullptr;
    // NOT a tensor. Which attention the graph builds (--attn), carried on the
    // input struct so that threading it costs zero changes to dit_train_layer /
    // _stack / _forward / the checkpoint driver's signatures — the exact-mode
    // graph is textually the graph it always was. Default EXACT: every existing
    // caller keeps today's behaviour without touching a line.
    DitAttnMode attn_mode = DIT_ATTN_EXACT;
    // HOT-Step patch: flash-attn-train — the arithmetic REQUESTED of the fused
    // ops (--attn flash => GGML_PREC_DEFAULT, i.e. TF32 tensor cores where the
    // backend has them; --attn flash-f32 => GGML_PREC_F32, v1's scalar kernels).
    // Named for the request because the backend may override it (pre-sm_80,
    // D != 128, unaligned view) and dit_train_log.json records the RESOLVED
    // answer separately (tf32 design §3.5). Ignored entirely in exact mode.
    ggml_prec attn_prec_req = GGML_PREC_DEFAULT;
};

// Named probes for the §6.2 V1 forward diff. Populated only when non-null, so
// training pays nothing (ggml_set_output pins a tensor for the whole graph).
struct DitTaps {
    std::vector<std::pair<std::string, ggml_tensor *>> v;
};

static void dit_tap(DitTaps * taps, const char * name, ggml_tensor * t) {
    if (!taps) {
        return;
    }
    ggml_set_name(t, name);
    ggml_set_output(t);
    // A reshape/permute is a VIEW. ggml_gallocr only refuses to free a node that
    // carries GGML_TENSOR_FLAG_OUTPUT itself — marking the view does NOT protect
    // its parent's storage, so the parent gets recycled and the probe reads back
    // whatever landed there next. Mark the whole view chain.
    for (ggml_tensor * p = t->view_src; p; p = p->view_src) {
        ggml_set_output(p);
    }
    taps->v.push_back({ std::string(name), t });
}

// Same protection for a graph we did not build (the inference graph's own debug
// outputs are views for exactly the same reason).
static void dit_protect_output_views(ggml_cgraph * gf) {
    const int n = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n; i++) {
        ggml_tensor * nd = ggml_graph_node(gf, i);
        if (!(nd->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            continue;
        }
        for (ggml_tensor * p = nd->view_src; p; p = p->view_src) {
            ggml_set_output(p);
        }
    }
}

// ─── grouped-query head expansion (spike amendment A1) ──────────────────────
//
// Byte-identical to the mapping ggml's own mul_mat broadcast performs (query head
// h reads kv head h / (Nh/Nkv)): a [D,Nkv,S,B] tensor is viewed as [D,1,Nkv,S*B],
// tiled to [D,G,Nkv,S*B] and re-viewed as [D,Nh,S,B], so the head index is
// g + G*kv and h/G == kv exactly.
//
// WHY it exists at all: with the batch on ne3 and Nkv < Nh, ggml's mul_mat
// BACKWARD builds `tmp = out_prod(src1, grad)` for the broadcast src0 and then
// asserts `tmp->ne[3] == 1` (ggml.c:6585-6588) before folding the head repetition
// with ggml_repeat_back — that assert is `B == 1`, and the process aborts.
// Expanding K/V first leaves no broadcast to fold. Proven in spike-batch.h
// (sb_expand_heads); run `ace-train spike batch --naive-bwd` for the abort.
//
// Costs: K/V activations grow x(Nh/Nkv) — dit-vram.h charges it — and the CUDA
// REPEAT_BACK kernel caps grad->ne[2]*ne[3] at 32768 (ggml-cuda.cu:5305). That
// product is Nkv*S*B for self-attention and Nkv*enc_S*B for CROSS-attention, so
// the binding constraint is Nkv*max(S,enc_S)*B; dit_vram_max_batch() enforces it
// and dit_train_stage re-checks it as a hard pre-flight.
static ggml_tensor * dit_expand_heads(ggml_context * ctx, ggml_tensor * x, int D, int Nkv, int Nh, int S, int B) {
    if (Nh == Nkv) {
        return x;
    }
    const int64_t G  = (int64_t) Nh / (int64_t) Nkv;
    const int64_t SB = (int64_t) S * (int64_t) B;
    ggml_tensor * a  = ggml_reshape_4d(ctx, x, D, 1, Nkv, SB);
    ggml_tensor * r  = ggml_repeat_4d(ctx, a, D, G, Nkv, SB);
    return ggml_reshape_4d(ctx, r, D, Nh, S, B);
}

// HOT-Step patch: flash-attn-train (investigation B2) — WHO still needs it.
//
// Everything the comment above describes is a workaround for ONE thing: ggml's
// MUL_MAT backward aborts on a broadcast src0 at B > 1. The fused ops are not
// mul_mats. They take native GQA directly — gate 1's grid runs Nkv 8 / Nh 32 and
// the backward folds the G query heads into each KV head inside the dkdv kernel
// — so in flash mode the expansion buys nothing and costs three things: K/V
// activations x(Nh/Nkv) (dit_vram_kv_expand_bytes), a repeat + repeat_back node
// per attention site per layer, and DIT_REPEAT_BACK_MAX's hard cap on B, which
// is a constraint of the REPEAT_BACK kernel and of nothing else.
//
// Exact mode is untouched: `B > 1` is the original condition, unmoved, so the
// --attn exact graph is still the graph it always was (gate T3's byte-identity
// anchor covers it).
static bool dit_attn_needs_kv_expand(const DitInputs & in, int B) {
    return B > 1 && in.attn_mode != DIT_ATTN_FLASH;
}

// One [H,1,B] adaLN slot out of the [6H,1,B] adaln tensor. ggml_cont because the
// stride-6H view is not contiguous and every consumer broadcasts it against
// [H,S,B]; the copy is H*B floats and carries no gradient (neither
// scale_shift_table nor t_tproj is a parameter).
static ggml_tensor * dit_adaln_slot(ggml_context * ctx, ggml_tensor * adaln, int k, int H, int B) {
    return ggml_cont(ctx, ggml_view_3d(ctx, adaln, H, 1, B, adaln->nb[1], adaln->nb[2],
                                       (size_t) k * (size_t) H * sizeof(float)));
}

// ─── one decoder layer (unfused, manual attention, batch on the last axis) ──

static ggml_tensor * dit_train_layer(ggml_context * ctx, DitTrainModel * M, int li, ggml_tensor * hidden,
                                     ggml_tensor * enc, const DitInputs & in, const DitAdapter * ad, int S, int enc_S,
                                     int B, DitTaps * taps) {
    const DiTGGMLConfig & c  = M->m.cfg;
    DiTGGMLLayer *        ly = &M->m.layers[li];
    const int             H = c.hidden_size, D = c.head_dim, Nh = c.n_heads, Nkv = c.n_kv_heads;
    const float           scale = 1.0f / sqrtf((float) D);
    const bool            tap0  = (taps != nullptr && li == 0);

    // [6H,1,B] + [6H]: t_tproj must be src0 so the broadcast direction is legal
    // for B > 1. At B == 1 float addition is commutative and exact, so the value
    // is identical to the pre-batching `add(ss_flat, t_tproj)`.
    ggml_tensor * ss_flat = ggml_reshape_1d(ctx, ly->scale_shift_table, 6 * H);
    ggml_tensor * adaln   = ggml_add(ctx, in.t_tproj, ss_flat);  // [6H,1,B]
    ggml_tensor * shift_sa = dit_adaln_slot(ctx, adaln, 0, H, B);
    ggml_tensor * scale_sa = dit_adaln_slot(ctx, adaln, 1, H, B);
    ggml_tensor * gate_sa  = dit_adaln_slot(ctx, adaln, 2, H, B);
    ggml_tensor * shift_ff = dit_adaln_slot(ctx, adaln, 3, H, B);
    ggml_tensor * scale_ff = dit_adaln_slot(ctx, adaln, 4, H, B);
    ggml_tensor * gate_ff  = dit_adaln_slot(ctx, adaln, 5, H, B);

    // ── self-attention ──
    ggml_tensor * res = hidden;
    ggml_tensor * nsa = dit_ggml_rms_norm_weighted(ctx, hidden, ly->self_attn_norm, c.rms_norm_eps);
    nsa               = dit_ggml_adaln(ctx, nsa, scale_sa, shift_sa, M->m.scalar_one);
    if (tap0) {
        dit_tap(taps, "layer0_sa_input", nsa);
    }

    ggml_tensor * q = ad->applyP(ctx, ly->sa_q_proj, li, DIT_SA_Q, nsa);
    ggml_tensor * k = ad->applyP(ctx, ly->sa_k_proj, li, DIT_SA_K, nsa);
    ggml_tensor * v = ad->applyP(ctx, ly->sa_v_proj, li, DIT_SA_V, nsa);

    q = ggml_reshape_4d(ctx, q, D, Nh, S, B);
    k = ggml_reshape_4d(ctx, k, D, Nkv, S, B);
    v = ggml_reshape_4d(ctx, v, D, Nkv, S, B);
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, c.rms_norm_eps), dit_ggml_f32(ctx, ly->sa_q_norm));
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, c.rms_norm_eps), dit_ggml_f32(ctx, ly->sa_k_norm));
    // positions are [S*B]; merge S and B for rope (its position index is ne1 in
    // 3-D), restore 4-D after. At B == 1 both reshapes are no-ops on the layout.
    q = ggml_reshape_3d(ctx, q, D, Nh, (int64_t) S * B);
    k = ggml_reshape_3d(ctx, k, D, Nkv, (int64_t) S * B);
    q = ggml_rope_ext(ctx, q, in.t_pos, NULL, D, 2 /*NEOX*/, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, in.t_pos, NULL, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    if (tap0) {
        dit_tap(taps, "layer0_q_after_rope", q);
        dit_tap(taps, "layer0_k_after_rope", k);
    }
    q = ggml_reshape_4d(ctx, q, D, Nh, S, B);
    k = ggml_reshape_4d(ctx, k, D, Nkv, S, B);
    if (dit_attn_needs_kv_expand(in, B)) {
        k = dit_expand_heads(ctx, k, D, Nkv, Nh, S, B);
        v = dit_expand_heads(ctx, v, D, Nkv, Nh, S, B);
    }
    // layer_type 0 = sliding window, 1 = full attention (dit.h:656). Full
    // attention still takes the PAD mask when one exists (t_sa_pad is null unless
    // the batch has a padded element) — masking the pad only in the windowed
    // layers would leave every full-attention layer leaking it.
    ggml_tensor * sa_mask = (ly->layer_type == 0) ? in.t_sa : in.t_sa_pad;
    ggml_tensor * attn    = nullptr;
    {
        // Node-profiler site (dit-node-profile.h). It starts at the permutes, not
        // at the attention call, because the permuted q/k/v ARE srcs of the
        // attention backward in exact mode — tagging them is what lets those
        // out_prod nodes be attributed to attention instead of `other`. The scope
        // creates no tensors and reorders none: sa_mask above is a plain pointer
        // choice, so --attn exact still emits the identical node sequence.
        // Windowed and full layers are named apart because they are not the same
        // work: the fused kernel skips tiles a window mask killed and the manual
        // chain computes the dense S^2 either way (plan investigation B4).
        DnpScope _dnp(ctx, ly->layer_type == 0 ? "attn.self.win" : "attn.self.full");
        q = ggml_permute(ctx, q, 0, 2, 1, 3);
        k = ggml_permute(ctx, k, 0, 2, 1, 3);
        v = ggml_permute(ctx, v, 0, 2, 1, 3);
        // One of the two mode checks in the whole graph build. The false arm is
        // the original call, unmoved, so --attn exact emits the identical nodes.
        attn = (in.attn_mode == DIT_ATTN_FLASH) ? dit_attn_flash(ctx, q, k, v, sa_mask, scale, in.attn_prec_req)
                                                : dit_attn_f32(ctx, q, k, v, sa_mask, scale);  // [D,Nh,S,B]
        attn = ggml_reshape_3d(ctx, attn, (int64_t) Nh * D, S, B);
    }
    if (tap0) {
        dit_tap(taps, "layer0_attn_out", attn);
    }
    ggml_tensor * sa_out = ad->applyP(ctx, ly->sa_o_proj, li, DIT_SA_O, attn);
    if (tap0) {
        dit_tap(taps, "layer0_sa_output", sa_out);
    }
    hidden = dit_ggml_gated_add(ctx, res, sa_out, gate_sa);
    if (tap0) {
        dit_tap(taps, "layer0_after_self_attn", hidden);
    }

    // ── cross-attention (no gate, plain residual) ──
    ggml_tensor * nca = dit_ggml_rms_norm_weighted(ctx, hidden, ly->cross_attn_norm, c.rms_norm_eps);
    ggml_tensor * cq  = ad->applyP(ctx, ly->ca_q_proj, li, DIT_CA_Q, nca);
    ggml_tensor * ck  = ad->applyP(ctx, ly->ca_k_proj, li, DIT_CA_K, enc);
    ggml_tensor * cv  = ad->applyP(ctx, ly->ca_v_proj, li, DIT_CA_V, enc);
    // QK-norm BEFORE the permute. The inference graph (dit-graph.h:534-547) and
    // the spike both normalise AFTER it; rms_norm acts on ne0 == D, which stays
    // contiguous under permute(0,2,1,3), so the FORWARD is bit-identical either
    // way (gate T3 confirms it). The BACKWARD is not: ggml's rms_norm_back on a
    // non-contiguous src produced a systematically wrong gradient — measured by
    // gate T4 as fd/analytic ~= 0.16-0.24 on cross_attn.k_proj and ~0.9 on
    // cross_attn.q_proj, while every site without a QK-norm (v/o) was clean to
    // 1e-4 and self-attention (which normalises pre-permute already) was clean.
    // Normalising pre-permute hands rms_norm a contiguous tensor and the
    // discrepancy disappears. Self-attention above uses the same ordering.
    cq                = ggml_reshape_4d(ctx, cq, D, Nh, S, B);
    ck                = ggml_reshape_4d(ctx, ck, D, Nkv, enc_S, B);
    cv                = ggml_reshape_4d(ctx, cv, D, Nkv, enc_S, B);
    cq                = ggml_mul(ctx, ggml_rms_norm(ctx, cq, c.rms_norm_eps), dit_ggml_f32(ctx, ly->ca_q_norm));
    ck                = ggml_mul(ctx, ggml_rms_norm(ctx, ck, c.rms_norm_eps), dit_ggml_f32(ctx, ly->ca_k_norm));
    if (dit_attn_needs_kv_expand(in, B)) {
        ck = dit_expand_heads(ctx, ck, D, Nkv, Nh, enc_S, B);
        cv = dit_expand_heads(ctx, cv, D, Nkv, Nh, enc_S, B);
    }
    ggml_tensor * cattn = nullptr;
    {
        DnpScope _dnp(ctx, "attn.cross");  // see the self-attention scope above
        cq = ggml_permute(ctx, cq, 0, 2, 1, 3);
        ck = ggml_permute(ctx, ck, 0, 2, 1, 3);
        cv = ggml_permute(ctx, cv, 0, 2, 1, 3);
        // The second mode check. Cross-attention is the same op with S_kv = enc_S:
        // at crop 1250 its retained softmax (enc_S x S) is already LARGER than
        // self-attention's S^2, so flashing only self-attention would leave the
        // bigger of the two walls standing (spec §0.1).
        cattn = (in.attn_mode == DIT_ATTN_FLASH) ? dit_attn_flash(ctx, cq, ck, cv, in.t_ca, scale, in.attn_prec_req)
                                                 : dit_attn_f32(ctx, cq, ck, cv, in.t_ca, scale);
        cattn = ggml_reshape_3d(ctx, cattn, (int64_t) Nh * D, S, B);
    }
    hidden              = ggml_add(ctx, hidden, ad->applyP(ctx, ly->ca_o_proj, li, DIT_CA_O, cattn));
    if (tap0) {
        dit_tap(taps, "layer0_after_cross_attn", hidden);
    }

    // ── MLP ──
    res               = hidden;
    ggml_tensor * nff = dit_ggml_rms_norm_weighted(ctx, hidden, ly->mlp_norm, c.rms_norm_eps);
    nff               = dit_ggml_adaln(ctx, nff, scale_ff, shift_ff, M->m.scalar_one);
    ggml_tensor * g   = ad->applyP(ctx, ly->gate_proj, li, DIT_MLP_GATE, nff);
    ggml_tensor * u   = ad->applyP(ctx, ly->up_proj, li, DIT_MLP_UP, nff);
    ggml_tensor * ff  = ggml_swiglu_split(ctx, g, u);  // fused ggml_swiglu has NO backward
    ggml_tensor * fo  = ad->applyP(ctx, ly->down_proj, li, DIT_MLP_DOWN, ff);
    return dit_ggml_gated_add(ctx, res, fo, gate_ff);
}

// ─── pre / post stacks (design C2: they belong to the first / last segment) ──
//
// Split out of dit_train_forward so the checkpointing driver (dit-train-ckpt.h)
// can compose them one segment at a time. dit_train_forward below still calls
// them in exactly the original order and builds exactly the original graph — the
// §2.3.1 byte-identity anchor depends on that, and the T3/T7 rungs measure it.

static ggml_tensor * dit_train_proj_in(ggml_context * ctx, DitTrainModel * M, const DitInputs & in, int T, int B,
                                       DitTaps * taps) {
    const DiTGGMLConfig & c       = M->m.cfg;
    const int             S       = T / c.patch_size;
    ggml_tensor *         patched = ggml_reshape_3d(ctx, in.t_input, (int64_t) c.in_channels * c.patch_size, S, B);
    ggml_tensor *         hidden  = ggml_add(ctx, ggml_mul_mat(ctx, M->m.proj_in_w, patched), M->m.proj_in_b);
    dit_tap(taps, "hidden_after_proj_in", hidden);
    return hidden;  // [H,S,B]
}

// Amendment A4: every checkpoint segment RECOMPUTES this from t_enc instead of
// storing an encoder boundary. cond_emb is frozen and enc carries no useful
// gradient, so the recompute is exact and cheaper than another [enc_H,enc_S,B]
// buffer.
static ggml_tensor * dit_train_cond(ggml_context * ctx, DitTrainModel * M, const DitInputs & in, DitTaps * taps) {
    ggml_tensor * enc = ggml_add(ctx, ggml_mul_mat(ctx, M->m.cond_emb_w, in.t_enc), M->m.cond_emb_b);
    dit_tap(taps, "enc_after_cond_emb", enc);
    return enc;  // [H,enc_S,B]
}

// Decoder layers [lo, hi) applied to `hidden`. This is the segment seam (C2).
static ggml_tensor * dit_train_stack(ggml_context * ctx, DitTrainModel * M, const DitAdapter * ad,
                                     const DitInputs & in, ggml_tensor * hidden, ggml_tensor * enc, int lo, int hi,
                                     int S, int enc_S, int B, DitTaps * taps) {
    for (int i = lo; i < hi; i++) {
        hidden = dit_train_layer(ctx, M, i, hidden, enc, in, ad, S, enc_S, B, taps);
        if (taps && (i == 0 || i == 6 || i == 12 || i == 18 || i == M->m.cfg.n_layers - 1)) {
            char nm[64];
            snprintf(nm, sizeof(nm), "hidden_after_layer%d", i);
            dit_tap(taps, nm, hidden);
        }
    }
    return hidden;
}

static ggml_tensor * dit_train_head(ggml_context * ctx, DitTrainModel * M, ggml_tensor * hidden, const DitInputs & in,
                                    int T, int B, DitTaps * taps) {
    const DiTGGMLConfig & c = M->m.cfg;
    const int             H = c.hidden_size;
    // t_temb is src0 so the [H,1,B] broadcast direction is legal; at B == 1 the
    // swap is a commutative float add and the value is unchanged.
    ggml_tensor * oss_flat  = ggml_reshape_1d(ctx, M->m.out_scale_shift, 2 * H);
    ggml_tensor * out_shift = ggml_add(ctx, in.t_temb, ggml_view_1d(ctx, oss_flat, H, 0));
    ggml_tensor * out_scale = ggml_add(ctx, in.t_temb, ggml_view_1d(ctx, oss_flat, H, (size_t) H * sizeof(float)));
    ggml_tensor * nout      = dit_ggml_rms_norm_weighted(ctx, hidden, M->m.norm_out, c.rms_norm_eps);
    nout                    = dit_ggml_adaln(ctx, nout, out_scale, out_shift, M->m.scalar_one);
    ggml_tensor * out       = ggml_add(ctx, ggml_mul_mat(ctx, M->m.proj_out_w, nout), M->m.proj_out_b);
    ggml_tensor * vel       = ggml_reshape_3d(ctx, out, c.out_channels, T, B);  // [Oc, T, B]
    dit_tap(taps, "velocity", vel);
    return vel;
}

// ─── full trainable forward -> velocity [Oc, T, B] ──────────────────────────
//
// `layer_first`/`layer_last` run a SUB-STACK; the finite-difference gate (T4)
// uses [n_layers-1, n_layers) so the graph stays small. -1/-1 = the whole stack.
//
// `B` is the micro-batch size (design B1). B == 1 is the pre-batching graph: the
// trailing-1 reshapes have the same ne/nb as the 2-D/3-D originals, the K/V head
// expansion is skipped so mul_mat keeps its GQA broadcast, and every added op
// (the adaLN cont, the loss mask multiply) is exactly value-preserving. That is
// what makes invariant §2.3.1 hold.
//
// This is the MONOLITHIC path only — the code `--ckpt 0` keeps. The segmented
// path (dit-train-ckpt.h) composes dit_train_proj_in / dit_train_cond /
// dit_train_stack / dit_train_head itself, one segment at a time, so it needs no
// entry-point parameters here. (It briefly had `hidden_in`/`with_head` for that;
// the segment driver never used them and they are gone.)
static ggml_tensor * dit_train_forward(ggml_context * ctx, DitTrainModel * M, const DitAdapter * ad,
                                       const DitInputs & in, int T, int enc_S, DitTaps * taps = nullptr,
                                       int layer_first = -1, int layer_last = -1, int B = 1) {
    const DiTGGMLConfig & c = M->m.cfg;
    const int             S = T / c.patch_size;
    const int             lo = (layer_first < 0) ? 0 : layer_first;
    const int             hi = (layer_last < 0) ? c.n_layers : layer_last;

    ggml_tensor * hidden = dit_train_proj_in(ctx, M, in, T, B, taps);
    ggml_tensor * enc    = dit_train_cond(ctx, M, in, taps);
    hidden               = dit_train_stack(ctx, M, ad, in, hidden, enc, lo, hi, S, enc_S, B, taps);
    return dit_train_head(ctx, M, hidden, in, T, B, taps);
}

// ─── the §4.5 product loss (D8) + the design-B4 batch reduction ─────────────
//
//   err[c,i,b] = v_pred - v_tgt
//   per_elem   = err^2
//   if channel_balance: per_elem[c,i,b] *= cw[c]     (cw = (1/std)/mean(1/std))
//   if t_lw:            per_elem[c,i,b] *= lw[0,i,b]
//   loss = gscale * sum(per_elem)
//
// `t_lw` [1,T,B] carries the valid-frame mask (padded frames exactly 0, so they
// contribute nothing to the loss OR to the gradients — §2.3.5) and, at B > 1, the
// per-element flow_snr weight, which a single scalar t_lossgrad cannot express
// once elements differ. `gscale` <= 0 means the pre-batching 1/(Oc*len).
//
// BYTE-IDENTITY (§2.3.1): at B == 1 the caller passes an all-ones mask and
// gscale = 1/(Oc*len), so the mask multiply is x*1.0f (exact for every finite
// float, and exact through the mul backward too) and the value is bit-identical
// to the pre-batching `scale(sum(pe), 1/(Oc*len))`. The flow_snr weight then
// stays where it always was, in the scalar t_lossgrad.
//
// `report_out` (B > 1 only) is the same sum against the WEIGHT-FREE mask, for the
// `rawLoss` JSONL field. It is never ggml_set_loss'd and never gets a grad
// accumulator, so ggml_compute_backward skips it outright (ggml.c:6417-6421).
static ggml_tensor * dit_train_loss(ggml_context * ctx, ggml_tensor * vpred, const DitInputs & in, int Oc, int len,
                                    bool channel_balance, float gscale = 0.0f,
                                    ggml_tensor ** report_out = nullptr) {
    ggml_tensor * diff = ggml_sub(ctx, vpred, in.t_vtgt);  // [Oc, T, B]
    ggml_tensor * pe   = ggml_sqr(ctx, diff);
    if (channel_balance && in.t_cw) {
        pe = ggml_mul(ctx, pe, in.t_cw);  // [Oc,T,B] * [Oc] row-broadcast
    }
    if (report_out && in.t_lwu) {
        ggml_tensor * rep = ggml_sum(ctx, ggml_mul(ctx, pe, in.t_lwu));
        ggml_set_name(rep, "flow_loss_raw");
        ggml_set_output(rep);
        *report_out = rep;
    }
    ggml_tensor * pew = in.t_lw ? ggml_mul(ctx, pe, in.t_lw) : pe;  // [Oc,T,B] * [1,T,B]
    const float   gs  = (gscale > 0.0f) ? gscale : (1.0f / (float) ((int64_t) Oc * (int64_t) len));
    ggml_tensor * loss = ggml_scale(ctx, ggml_sum(ctx, pew), gs);
    ggml_set_name(loss, "flow_loss");
    ggml_set_output(loss);
    return loss;
}

// ─── temb / tproj precompute (substitution S4, equivalence class EXACT) ─────
//
// ggml_timestep_embedding has no backward and nothing upstream of temb is
// trainable, so temb [H] / tproj [6H] are computed in a separate tiny graph on
// the TRAINING backend from the MIRRORED time_embed weights and uploaded as
// constants. r == t (D12), so time_embed_r sees t - t_r == 0.
//
// Called once per optimizer window (grad_accum timesteps at a time) because
// §3.5's loss weighting already needs the whole window's t values up front.
static bool dit_train_temb(DitTrainModel * M, const std::vector<float> & ts, std::vector<std::vector<float>> * temb_out,
                           std::vector<std::vector<float>> * tproj_out) {
    const int H = M->m.cfg.hidden_size;

    ggml_context * ctxs;
    {
        ggml_init_params p = { 8 * ggml_tensor_overhead(), nullptr, true };
        ctxs               = ggml_init(p);
    }
    if (!ctxs) {
        return false;
    }
    ggml_tensor * t_t = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_tensor * t_r = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_set_input(t_t);
    ggml_set_input(t_r);
    ggml_backend_buffer_t bufs = ggml_backend_alloc_ctx_tensors(ctxs, M->backend);
    if (!bufs) {
        ggml_free(ctxs);
        return false;
    }

    std::vector<uint8_t> arena((size_t) 16 << 20);
    bool                 ok = true;
    temb_out->assign(ts.size(), {});
    tproj_out->assign(ts.size(), {});

    for (size_t i = 0; i < ts.size() && ok; i++) {
        ggml_backend_tensor_set(t_t, &ts[i], 0, sizeof(float));
        ggml_backend_tensor_set(t_r, &ts[i], 0, sizeof(float));  // r == t, so t - t_r == 0

        ggml_init_params ip  = { arena.size(), arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 2048, false);

        ggml_tensor *tproj_t = nullptr, *tproj_r = nullptr;
        ggml_tensor * temb_t =
            dit_ggml_build_temb(ctx, &M->m, &M->m.time_embed, t_t, &tproj_t, nullptr, nullptr, nullptr, "_t");
        ggml_tensor * tdiff = ggml_sub(ctx, t_t, t_r);
        ggml_tensor * temb_r =
            dit_ggml_build_temb(ctx, &M->m, &M->m.time_embed_r, tdiff, &tproj_r, nullptr, nullptr, nullptr, "_r");
        ggml_tensor * temb  = ggml_add(ctx, temb_t, temb_r);
        ggml_tensor * tproj = ggml_add(ctx, tproj_t, tproj_r);
        ggml_set_output(temb);
        ggml_set_output(tproj);
        ggml_build_forward_expand(gf, temb);
        ggml_build_forward_expand(gf, tproj);

        ggml_backend_sched_reset(M->sched);
        ok = ggml_backend_sched_graph_compute(M->sched, gf) == GGML_STATUS_SUCCESS;
        if (ok) {
            (*temb_out)[i].resize((size_t) H);
            (*tproj_out)[i].resize((size_t) 6 * H);
            ggml_backend_tensor_get(temb, (*temb_out)[i].data(), 0, (size_t) H * sizeof(float));
            ggml_backend_tensor_get(tproj, (*tproj_out)[i].data(), 0, (size_t) 6 * H * sizeof(float));
        }
        ggml_free(ctx);
    }

    ggml_backend_buffer_free(bufs);
    ggml_free(ctxs);
    return ok;
}
