#pragma once
// lm-train-run.h — `ace-train train-lm` stage driver (plan §3.5.3, §3.5.4).
//
// extract -> train -> export, each with its own teardown. The training loop is
// the hand-rolled one from §3.5: our own persistent grad accumulators driven by
// ggml_build_backward_expand, and our own merged norm+clip+AdamW graph.
//
// Emission contract (§2.2): `start` is the first JSONL line, `done` is the
// last; a `fatal` line replaces `done` and precedes a non-zero exit.

#include "backend.h"
#include "bpe.h"
#include "train/lm-ckpt.h"
#include "train/lm-common.h"
#include "train/lm-data.h"
#include "train/lm-export.h"
#include "train/lm-extract.h"
#include "train/lm-graph.h"
#include "train/lm-optim.h"
#include "train/lm-prior.h"
#include "train/lm-resume.h"
#include "train/lm-selftest.h"
#include "train/lm-vram.h"
#include "version.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

struct LmTrainArgs {
    std::vector<std::string> stages{ "extract", "train", "export" };

    std::string tensors_dir, codes_path, out_dir, models_dir;
    std::string dit_path, lm_path, lm_name, lm_size;  // already resolved by cmd_train_lm

    int   rank = 16, alpha = 32;
    // Adapter parameterization (2026-07-30). "lora" is the shipped path and the
    // default. "lokr" writes lokr_weights.safetensors in the SAME LyCORIS layout
    // the DiT trainer writes, so adapter-merge.h's kron reader applies.
    std::string adapter_type        = "lora";
    int         lokr_dim            = 512;
    float       lokr_alpha          = 512.0f;
    int         lokr_factor         = 6;
    bool        lokr_decompose_both = true;
    float lr = 1e-4f;
    int   epochs = 75, grad_accum = 2;   // GA2 = Side-Step parity (GA4 halves optimizer steps/epoch)
    float warmup_ratio = 0.05f, grad_clip = 1.0f, weight_decay = 0.01f;
    int   seed         = 42;
    float target_loss  = 4.0f;   // parity: Side-Step's own electriccallboy run used 4.0 and stopped at epoch 29
    std::string order  = "shuffle";

    int  max_len         = 0;
    int  vram_reserve_mb = 1024;
    bool loss_on_cot     = true;

    // 4B low-VRAM path (2026-07-28 plan §2.1)
    std::string low_vram        = "auto";  // auto|on|off
    int         attn_head_block = -1;      // -1 = engine picks (lm_ckpt_default_head_block)
    int         chunk           = 128;

    // Speed levers (2026-07-28 plan §2.1). Both DEFAULT TO THE SHIPPED
    // BEHAVIOUR — §6.0 requires a flags-off run to be byte-identical.
    std::string weights = "f32-window";  // f32-window|bf16  (Lever A)
    std::string batch   = "1";           // 1..8|auto        (Lever B)

    // MUL_MAT activation-gradient formulation (engine/patches/mm-backward.patch):
    //   "outprod" = ggml upstream, out_prod(src0, transpose(grad)) — F32-only on CUDA
    //   "mm"      = mul_mat(cont(transpose(src0)), grad) — dtype-agnostic, BF16
    //               tensor cores. ~1.7-1.8x per layer per step on an RTX 5090.
    // ENGINE DEFAULT IS "outprod" so a bare ace-train invocation is unchanged; the
    // server passes --bwd mm. Selected by setting GGML_BACKWARD_MM before any
    // backward graph is built (ace-train.cpp).
    std::string bwd = "outprod";

    // Attention formulation (docs/plans/2026-09-02-lm-flash-attn.md, R2 of the
    // flash-attn roadmap):
    //   "exact"     = qwen3_attn_f32 — mul_mat -> soft_max_ext -> mul_mat ->
    //                 cont(permute), retained [S_kv,S,Nh] softmax and all. The
    //                 shipped graph, byte-identical to pre-flag runs (D2/G0).
    //   "flash"     = the fused GGML_OP_FLASH_ATTN_TRAIN / _BACK pair at the
    //                 whole-head attention site, so that softmax is never
    //                 materialised and attention memory is linear in S. On the
    //                 naive 0.6B/1.7B path that is the difference between
    //                 2034*S^2 bytes (29 GB at S 3800) and a per-token term.
    //   "flash-f32" = the same fused ops pinned to GGML_PREC_F32, i.e. the
    //                 scalar v1 kernels instead of the TF32 tensor-core ones
    //                 "flash" selects. Slower; it separates "did fusion move the
    //                 training" from "did TF32 move it".
    //
    // ENGINE DEFAULT IS "exact" (D8). Flash gradients are NOT bit-identical to
    // exact — same drift family as --bwd mm and --mirror bf16 — so the default
    // must stay the shipped arithmetic until the ear test (G7) says otherwise.
    std::string attn = "exact";

    // Optimizer (2026-07-30, ported from the DiT trainer — lm-optim.h is shared,
    // so the rule split, batched Newton-Schulz and shape bucketing were already
    // sitting underneath this trainer unused). "adamw" is the default and the
    // shipped path. "muon" puts every 2-D parameter whose SHORT side is >=
    // muon_min_dim on orthogonalized-momentum updates.
    //
    // NOTE FOR LoRA: A is [H, r], so the SHORT SIDE IS THE RANK. At the default
    // rank 16 that is exactly muon_min_dim, and a rank-8 adapter would fall
    // entirely through to AdamW — lm_optim_init logs the split, so check it.
    std::string optimizer     = "adamw";
    float       muon_lr_scale = 1.0f;
    float       muon_momentum = 0.95f;
    int         muon_ns_steps = 5;
    bool        muon_nesterov = true;
    int         muon_min_dim  = 16;
    int         muon_bucket   = 16;

    // Trigger word embedded in the exported adapter. Empty = fall back to the
    // variant's preprocess_meta.json custom_tag/tag_position; still empty after
    // that = no trigger keys written and the adapter is byte-identical to a
    // pre-trigger build. docs/plans/2026-07-28-adapter-trigger-embedding.md T5
    std::string trigger, trigger_position;

    float milestone_step = 1.0f;
    int   milestone_keep = 6;

    // Resume (--init-adapter, 2026-08-09): dir of an exported run whose
    // factors seed this one. Identity hyperparams are adopted from its
    // lm_train_log.json by lm_resume_prepare (cmd_train_lm); an explicit CLI
    // contradiction is exit 2. init_from_loss carries the source's saved_loss
    // for the epoch-1 honesty log and the train-log provenance keys.
    std::string init_adapter;
    double      init_from_loss = -1.0;

    // ── Lever C: caption dropout (2026-09-02) ────────────────────────────
    //
    // Probability that a training micro-step swaps the sample's full caption
    // for the TRIGGER WORD ALONE. 0 = off, and off is the default: with both
    // this and prior preservation at their defaults the trainer emits exactly
    // the graphs, consumes exactly the RNG and produces exactly the loss
    // trajectory it did before they existed (the standing Lever A/B rule).
    //
    // Ported from the MM3 LM trainer (mm3-lm-train-run.h `caption_dropout`).
    // The point is to make the trigger carry the style BY ITSELF, which is the
    // only way a bare-trigger prompt works at inference — a model that has only
    // ever seen "president" wedged into a 400-token MOSS caption has no reason
    // to have learned what the word means on its own. Deliberately not
    // 1.0-by-another-name: training trigger-only rows EXCLUSIVELY would leave
    // the descriptor path untrained, so the adapter could summon the album but
    // never be steered. Mixing keeps both.
    //
    // ACE DIFFERENCE FROM MM3. MM3's prompt is caption + lyrics and nothing
    // else, so its trigger-only variant is a pure prompt swap. ACE's prompt
    // carries a CoT YAML built by the engine's own build_cot_yaml(), whose
    // `caption:` line is the caption — so the trigger-only variant is rebuilt
    // end to end by lm_build_sequence() and its CoT says "caption: president"
    // too. That is the honest thing: the variant is exactly the sequence the
    // model would face at inference from a bare trigger, CoT included, and its
    // n_masked/s_tr differ from the full-caption row accordingly.
    float caption_dropout = 0.0f;

    // ── Lever D: prior preservation (2026-09-02) ─────────────────────────
    //
    // Port of train/mm3-lm-prior.h + the reg path in mm3-lm-train-run.h. Every
    // reg_every'th micro-step trains on ANOTHER artist's song against the
    // frozen base's own cached next-token distribution instead of the album's
    // ground-truth codes, so the adapter is punished for changing its mind
    // about material that has nothing to do with this artist.
    //
    // OFF by default: it needs a corpus the user has to supply, and silently
    // training a different objective than the one asked for would be worse than
    // not offering it.
    std::vector<std::string> reg_codes;      // other artists' lm_codes.jsonl
    int                      reg_songs = 24; // rows sampled across those files
    int                      reg_every = 0;  // 0 = off; 3 = bghira's 1 prior : 2 style
    int                      reg_topk  = 64; // classes kept per position (<= LM_CAPTURE_K_MAX)
    std::string              reg_prior_dir;  // default <out>/prior

    bool overwrite = false;
    int  limit     = 0;
    bool self_test = false;
};

// Resolve the trigger to embed: explicit CLI flags win, else the variant's
// preprocess_meta.json, else nothing. `replace` positions are dropped with a
// warn (the tag was never applied to those captions — T4).
static void lm_resolve_trigger(const std::string & tensors_dir, std::string * trigger, std::string * position) {
    if (trigger->empty() && !tensors_dir.empty()) {
        lm_read_variant_tag(tensors_dir, trigger, position);
    }
    std::string why;
    if (!lm_trigger_normalize(trigger, position, &why) && !why.empty()) {
        jl("{\"type\":\"log\",\"level\":\"warn\",\"message\":\"%s\"}", lm_json_escape(why).c_str());
        fprintf(stderr, "[train-lm] %s\n", why.c_str());
    }
}

static inline bool lm_has_stage(const LmTrainArgs & a, const char * s) {
    for (size_t i = 0; i < a.stages.size(); i++) {
        if (a.stages[i] == s) {
            return true;
        }
    }
    return false;
}

// ─── the training stage ─────────────────────────────────────────────────────

struct LmTrainOutcome {
    int       epochs_run       = 0;
    double    final_loss       = -1.0;
    double    best_loss        = -1.0;
    int       best_epoch       = 0;
    // Which epoch's adapter is actually in out_dir (2026-07-30). Since the
    // export became best-only, final_loss is no longer what shipped.
    double      saved_loss   = -1.0;
    int         saved_epoch  = 0;
    std::string saved_reason;
    bool      stopped_on_target = false;
    int       samples          = 0;
    int       skipped_long     = 0;
    bool      exported         = false;
    int       export_tensors   = 0;
    long long ms               = 0;
};

static int lm_train_stage(const LmTrainArgs & a, LmExportMeta * meta, LmTrainOutcome * out) {
    const int64_t t_stage0 = ggml_time_ms();
    jl("{\"type\":\"stage\",\"stage\":\"train\",\"state\":\"begin\",\"total\":%d}", a.epochs);

    // ── D11: the hard `hidden_size >= 2560 || n_layers >= 36` refusal is GONE.
    // Refusal is now purely the generic VRAM refusal against whichever footprint
    // model the selected mode uses; `fatal reason:"unsupported-size"` is retired.

    // ── load the LM, unfused (L18) ───────────────────────────────────────
    const int64_t t_lm0 = ggml_time_ms();
    g_qwen3_load_no_fuse = true;
    Qwen3LM    lm;
    const bool loaded    = qw3lm_load(&lm, a.lm_path.c_str(), /*max_seq_len=*/64, /*n_kv_sets=*/1);
    g_qwen3_load_no_fuse = false;
    if (!loaded) {
        lm_fatal("model-load", "cannot load the LM base " + a.lm_path);
        return 1;
    }
    const Qwen3LMConfig & c = lm.cfg;
    const int             H = c.hidden_size, V = c.vocab_size;
    jl("{\"type\":\"model\",\"stage\":\"lm\",\"path\":\"%s\",\"ms\":%lld,\"layers\":%d,\"hidden\":%d,\"vocab\":%d}",
       lm_json_escape(a.lm_path).c_str(), (long long) (ggml_time_ms() - t_lm0), c.n_layers, H, V);

    // ── --attn: resolve the mode and the arithmetic request (D1/D5) ──────
    //
    // Resolved HERE, before the head-block decision and before either footprint
    // model runs, because both depend on it: flash forces head blocking off (D3)
    // and prices attention as a per-token linear term instead of c1*Nh*S^2 (D9).
    //
    // attn_prec_req is a REQUEST. GGML_PREC_DEFAULT is 0, which is also what
    // op_params zero-init gives, and on sm_80+ it resolves to the TF32 kernels —
    // so "flash" already means TF32 unless the user says otherwise. What
    // actually ran is read back from the backend after the first epoch and
    // logged as attn_prec; the flag alone cannot say.
    const bool      attn_flash    = (a.attn == "flash" || a.attn == "flash-f32");
    const ggml_prec attn_prec_req = (a.attn == "flash-f32") ? GGML_PREC_F32 : GGML_PREC_DEFAULT;
    if (meta) {
        meta->attn_mode = a.attn;
    }

    // ── mode selection (§3.2) ────────────────────────────────────────────
    //
    // DEVIATION vs §3.2's step order, with justification. The plan evaluates the
    // naive fit BEFORE the mirror exists, relying on the lemma that the mirror
    // term cancels out of the budget. That lemma is arithmetically right but its
    // inputs are not stable: `free` is a live CUDA reading that moves with every
    // other process on the card, and allocator padding makes
    // free_after_mirror + mirror_bytes only approximately free0 + base_bytes.
    // Gate (e) demands the naive maxLen INTEGER be unchanged, so instead of
    // re-deriving it we simply run the shipped code path untouched whenever the
    // mode could still be naive, and only switch afterwards. 4B (and
    // --low-vram on) skip the mirror entirely and never enter that path.
    const std::string size_label = lm_size_label_from_config(c);
    const bool        auto_mode  = (a.low_vram != "on" && a.low_vram != "off");
    LmVramMode        mode       = lm_vram_pick_mode(a.low_vram, size_label, /*naive_max_len=*/0);

    // ── Lever A gating (§3.4), before the mirror decision ────────────────
    //
    // bf16 requires BOTH the CUDA backend and a BF16-native base. Neither
    // condition used to have a graceful path (a fatal exit-1, or on CPU an
    // outright GGML_ABORT deep in the backward pass — lm-graph.h §property
    // 5 — since ggml_out_prod is F32-only on CUDA and aborts for BF16 on
    // CPU). Now that the server defaults `weights` to 'bf16' (2026-07-29)
    // and CPU/Vulkan release builds exist, both checks warn and fall back to
    // 'f32-window' instead — mirroring the DiT trainer's --mirror bf16 CUDA
    // gate (dit-train-run.h). `weights_used` (not `a.weights`) drives every
    // downstream report (vram JSONL, dit/lm_train_log.json) so a fallback is
    // never mislabelled as the bf16 run it wasn't.
    bool        weights_bf16 = (a.weights == "bf16");
    std::string weights_used = a.weights;
    if (weights_bf16) {
        std::string fallback_reason;
        if (strncmp(ggml_backend_name(lm.backend), "CUDA", 4) != 0) {
            // Gate BEFORE the graph is ever built: only ggml-cuda's out_prod
            // carries the BF16 patch (engine/patches/bf16-out-prod.patch);
            // CPU/Vulkan would GGML_ABORT mid-backward-pass instead of
            // failing cleanly.
            char b[192];
            snprintf(b, sizeof(b), "BF16 weights require CUDA — falling back to f32-window (this run picked '%s')",
                      ggml_backend_name(lm.backend));
            fallback_reason = b;
        } else if (!lm_bf16_base_is_bf16(lm)) {
            char b[256];
            snprintf(b, sizeof(b),
                     "BF16 weights require a BF16-native LM base — falling back to f32-window (%s loads its "
                     "projections as %s)",
                     a.lm_name.c_str(), lm_bf16_base_proj_type_name(lm));
            fallback_reason = b;
        }
        if (!fallback_reason.empty()) {
            lm_log("warn", fallback_reason);
            weights_bf16 = false;
            weights_used = "f32-window";
        }
    }
    if (weights_bf16) {
        // bf16 is meaningless on the naive path: it mirrors every weight to F32
        // and releases the BF16 buffer, so there is nothing left to run a BF16
        // GEMM on. --low-vram off + --weights bf16 was already rejected at exit
        // 2 in cmd_train_lm, so forcing the mode here can never contradict the
        // user. Only reached when bf16 survived both fallback checks above.
        mode = LM_VRAM_LOWVRAM;
    }

    size_t base_bytes = lm_base_weight_bytes(lm);

    // ── F32 mirror; a quantized base is refused here ─────────────────────
    LmF32Mirror mirror;
    if (mode == LM_VRAM_NAIVE) {
        std::string err;
        if (!lm_build_f32_mirror(&lm, &mirror, &err)) {
            lm_fatal("model-load", err);
            lm_mirror_free(&mirror);
            qw3lm_free(&lm);
            return 1;
        }
    } else {
        std::string err;
        if (!lm_ckpt_check_base(&lm, &err)) {
            lm_fatal("model-load", err);
            qw3lm_free(&lm);
            return 1;
        }
        fprintf(stderr, "[train-lm] low-vram mode: base stays %s resident (%.1f MB), no F32 mirror\n",
                ggml_type_name(lm.embed_tokens->type), base_bytes / 1048576.0);
    }

    // ── tokenizer ────────────────────────────────────────────────────────
    BPETokenizer bpe;
    if (!load_bpe_from_gguf(&bpe, a.lm_path.c_str())) {
        lm_fatal("model-load", "no BPE tokenizer in " + a.lm_path);
        lm_mirror_free(&mirror);
        qw3lm_free(&lm);
        return 1;
    }

    // ── VRAM auto-fit (L8) ───────────────────────────────────────────────
    LmVramModel vm;
    vm.n_layers     = c.n_layers;
    vm.hidden       = H;
    vm.ffn          = c.intermediate_size;
    vm.n_heads      = c.n_heads;
    vm.vocab        = V;
    vm.mirror_bytes = mirror.bytes;
    // D9. head_dim / n_kv_heads are read ONLY by the flash branch; with
    // attn_flash false the polynomial is the shipped one to the byte.
    vm.attn_flash   = attn_flash;
    vm.head_dim     = c.head_dim;
    vm.n_kv_heads   = c.n_kv_heads;
    {
        // The fit charges 4 buffers per parameter (param + acc + m + v), so this
        // count has to match the parameterization actually built or the budget
        // is wrong in the dangerous direction: a dim-512 LoKr on 0.6B is 22.0M
        // parameters against a rank-16 LoRA's 10.1M.
        const bool is_lokr = (a.adapter_type == "lokr");
        size_t     np      = 0;
        for (int s = 0; s < QW_LORA_NSLOTS; s++) {
            int in_dim = 0, out_dim = 0;
            lm_slot_dims(c, s, &in_dim, &out_dim);
            if (is_lokr) {
                int64_t out_l, out_k, in_m, in_n;
                lokr_factorization(out_dim, a.lokr_factor, &out_l, &out_k);
                lokr_factorization(in_dim, a.lokr_factor, &in_m, &in_n);
                np += (size_t) (in_m * out_l);
                np += lokr_w2_mono(a.lokr_dim, out_k, in_n)
                          ? (size_t) (in_n * out_k)
                          : (size_t) ((int64_t) a.lokr_dim * out_k + in_n * (int64_t) a.lokr_dim);
            } else {
                np += (size_t) in_dim * (size_t) a.rank + (size_t) a.rank * (size_t) out_dim;
            }
        }
        vm.lora_params = np * (size_t) c.n_layers;
    }

    // D3: flash forces head blocking off. `--attn flash --attn-head-block N>0`
    // already exited 2 in cmd_train_lm, so a.attn_head_block is -1 or 0 here and
    // the default resolver returns 0 for flash — the explicit zero below is the
    // belt to that braces, and it is what keeps t_zero_attn unallocated.
    int head_block =
        attn_flash ? 0 : ((a.attn_head_block >= 0) ? a.attn_head_block : lm_ckpt_default_head_block(c, attn_flash));
    if (!lm_ckpt_head_block_ok(c, head_block)) {
        char b[192];
        snprintf(b, sizeof(b), "--attn-head-block %d is not valid for n_heads %d / n_kv_heads %d — falling back to %d",
                 head_block, c.n_heads, c.n_kv_heads, lm_ckpt_default_head_block(c, attn_flash));
        lm_log("warn", b);
        head_block = lm_ckpt_default_head_block(c, attn_flash);
    }
    LmVramLowCfg lc;
    lc.attn_head_block = head_block;
    lc.chunk           = a.chunk;
    lc.head_dim        = c.head_dim;
    lc.base_bytes      = base_bytes;
    lc.emb_t_bytes     = (size_t) V * (size_t) H * ggml_type_size(lm.embed_tokens->type);
    lc.layer_w_bytes   = lm_layer_weight_bytes(c);
    lc.layer_wt_bytes  = lm_layer_proj_bytes(c, GGML_TYPE_BF16);  // §3.5
    lc.weights_bf16    = weights_bf16;
    lc.attn_flash      = attn_flash;                              // D9
    lc.n_kv_heads      = c.n_kv_heads;

    LmVramFit fit;
    if (mode == LM_VRAM_NAIVE) {
        fit = lm_vram_fit(vm, lm.backend, a.vram_reserve_mb, a.max_len);
        // §3.2 step 6, second clause: auto turns low-vram ON for smaller bases
        // when the naive path would have to skip full-song samples. On this
        // machine the shipped fit yields ~3200 (0.6B) and ~2515 (1.7B), so this
        // branch is unreachable there and Rob's behaviour is unchanged.
        //
        // `a.max_len <= 0` is load-bearing: D5 words the rule as "the shipped
        // naive AUTO-FIT yields max_len < 2048", but lm_vram_fit() returns
        // `user_len` VERBATIM when the user pinned --max-len (lm-vram.h:97-102).
        // Without this guard any user value in [512, 2047] — a plain
        // Training-Studio field, always emitted by buildTrainLmArgs — silently
        // flipped 0.6B/1.7B onto the checkpointed path, violating D5's "Do
        // 0.6B/1.7B change? No." A pinned length is a user statement about
        // sequence length, not about VRAM.
        if (auto_mode && a.max_len <= 0 && fit.max_len < 2048) {
            char b[192];
            snprintf(b, sizeof(b), "naive auto-fit yields max_len %d (< 2048) — switching to low-VRAM mode",
                     fit.max_len);
            lm_log("info", b);
            lm_mirror_free(&mirror);
            qw3lm_free(&lm);
            g_qwen3_load_no_fuse = true;
            const bool re        = qw3lm_load(&lm, a.lm_path.c_str(), /*max_seq_len=*/64, /*n_kv_sets=*/1);
            g_qwen3_load_no_fuse = false;
            if (!re) {
                lm_fatal("model-load", "cannot re-load the LM base for low-VRAM mode " + a.lm_path);
                return 1;
            }
            std::string err;
            if (!lm_ckpt_check_base(&lm, &err)) {
                lm_fatal("model-load", err);
                qw3lm_free(&lm);
                return 1;
            }
            base_bytes      = lm_base_weight_bytes(lm);
            lc.base_bytes   = base_bytes;
            vm.mirror_bytes = 0;
            mode            = LM_VRAM_LOWVRAM;
        }
    }
    if (mode == LM_VRAM_LOWVRAM) {
        fit = lm_vram_fit_lowvram(vm, lc, lm.backend, a.vram_reserve_mb, a.max_len);
    }

    if (!fit.ok) {
        char extra[224];
        snprintf(extra, sizeof(extra), ",\"needMb\":%lld,\"freeMb\":%lld,\"lmSize\":\"%s\",\"mode\":\"%s\"",
                 (long long) (fit.est_bytes / 1048576.0), (long long) fit.free_mb, a.lm_size.c_str(),
                 mode == LM_VRAM_LOWVRAM ? "lowvram" : "naive");
        lm_fatal("vram",
                 "not enough free VRAM for LM training: need ~" + std::to_string((long long) (fit.est_bytes / 1048576.0)) +
                     " MB, " + std::to_string((long long) fit.free_mb) + " MB free",
                 extra);
        lm_mirror_free(&mirror);
        qw3lm_free(&lm);
        return 1;
    }
    const int max_len = fit.max_len;

    // ── samples ──────────────────────────────────────────────────────────
    std::vector<LmCodeRow> rows;
    {
        std::string err;
        if (!lm_load_codes(a.codes_path.c_str(), &rows, &err)) {
            lm_fatal("no-samples", err);
            lm_mirror_free(&mirror);
            qw3lm_free(&lm);
            return 1;
        }
    }
    if (a.limit > 0 && (int) rows.size() > a.limit) {
        rows.resize((size_t) a.limit);
    }

    // ── Lever C: the trigger-only twin of every sample ───────────────────
    //
    // Built here, once, rather than per micro-step: rebuilding the prompt inside
    // the training loop would put the BPE tokenizer on the hot path, and the
    // variant is a pure function of the row. `caption_dropout_variants` stays
    // EMPTY when the lever is off, which is what keeps this whole block
    // unreachable in a default run.
    const bool  cd_on   = a.caption_dropout > 0.0f;
    const std::string cd_trigger = (cd_on && meta) ? meta->trigger : std::string();

    std::vector<LmSample> samples;
    std::vector<LmSample> cd_variants;      // parallel to `samples`; empty when off
    int                   skipped_long = 0;  // §2.2: OVER-LENGTH only (the L5 skip)
    int                   skipped_bad  = 0;  // structural rejections — malformed rows
    int                   min_len = 0, max_seq = 0;
    long long             trained_tokens = 0;
    for (size_t i = 0; i < rows.size(); i++) {
        LmSample    s;
        std::string why;
        bool        over = false;
        if (!lm_build_sequence(bpe, rows[i], a.loss_on_cot, max_len, &s, &why, &over)) {
            // Counting a malformed row as "too long" would point the user at
            // --max-len instead of at the data.
            if (over) {
                skipped_long++;
            } else {
                skipped_bad++;
            }
            lm_log("warn", "SKIP " + rows[i].file + ": " + why);
            continue;
        }
        const int S = (int) s.tokens.size();
        if (samples.empty() || S < min_len) {
            min_len = S;
        }
        if (S > max_seq) {
            max_seq = S;
        }
        trained_tokens += s.s_tr;
        samples.push_back(s);

        if (cd_on) {
            // Same row, caption replaced by the trigger word alone. Everything
            // else — lyrics, bpm/key/duration/language, the codes, the im_end
            // decision — comes from the same LmCodeRow, and the sequence is
            // built by the SAME lm_build_sequence() the real sample used, so
            // the prompt and its CoT are whatever the engine's own prompt code
            // says they are. A shorter caption can only shorten the sequence,
            // so the variant never needs more VRAM than its parent.
            LmCodeRow   tr_row = rows[i];
            tr_row.caption     = cd_trigger;
            LmSample    tv;
            std::string tw;
            if (lm_build_sequence(bpe, tr_row, a.loss_on_cot, max_len, &tv, &tw)) {
                cd_variants.push_back(tv);
            } else {
                // Falling back to the full-caption sample keeps the two vectors
                // index-aligned and makes the failure a no-op for that row
                // instead of a crash or a silent index skew.
                lm_log("warn", "caption dropout: no trigger-only variant for " + rows[i].file + " (" + tw +
                                   ") — that row will always train on its full caption");
                cd_variants.push_back(s);
            }
        }
    }
    if (samples.empty()) {
        lm_fatal("no-samples", "no usable training samples (every song was skipped; max_len " +
                                   std::to_string(max_len) + ")");
        lm_mirror_free(&mirror);
        qw3lm_free(&lm);
        return 1;
    }

    int s_tr_max = 0;
    for (size_t i = 0; i < samples.size(); i++) {
        s_tr_max = std::max(s_tr_max, samples[i].s_tr);
    }

    // ── Lever D: the regularisation corpus ───────────────────────────────
    //
    // Loaded HERE, before anything is allocated, because a reg row is a whole
    // other artist's song and can be longer than every song on this album. The
    // buffers, the checkpoint state and the high-water probe are all sized from
    // `alloc_seq`/`alloc_s_tr` below, which fold the reg set in. The REPORTED
    // data/vram numbers stay style-only, so `data.maxLen` still means "the
    // longest song being trained on".
    const bool reg_on = (a.reg_every > 0 && !a.reg_codes.empty());
    std::vector<LmSample>     reg_samples;
    std::vector<LmPriorCache> reg_priors;
    std::vector<std::string>  reg_ids;
    if (reg_on) {
        if (mode != LM_VRAM_LOWVRAM) {
            lm_fatal("unsupported",
                     "prior preservation (--reg-every) is implemented on the low-VRAM/checkpointed path only — "
                     "the capture reuses the chunked head's top-K readback (lm_ckpt_capture_topk), which the "
                     "naive path does not have. Add --low-vram on, or train the 4B base (which uses it anyway).");
            lm_mirror_free(&mirror);
            qw3lm_free(&lm);
            return 2;
        }
        // --reg-every LARGER THAN THE ALBUM schedules ZERO reg micro-steps, and
        // does it silently. `micro_per_ep` below starts at `n` and only grows
        // while `micro_per_ep - micro_per_ep / reg_every < n`; when
        // reg_every > n that condition is already false, so micro_per_ep stays
        // n, floor(n / reg_every) is 0, and `t % reg_every` never comes round.
        // The run would still pay for the capture and still print the coverage
        // line and the low-coverage warning, so every visible signal would say
        // the lever is on while nothing regularises — the dead-knob failure mode
        // this codebase has been burned by before (adapter group-scale sliders,
        // the LM echo sideband). Refused here, BEFORE the corpus load and the
        // capture, rather than warned about afterwards.
        //
        // reg_every == n is fine: micro_per_ep grows to n+1 and exactly one
        // micro-step per epoch is a reg step. The refusal is strictly
        // reg_every > n.
        if (a.reg_every > (int) samples.size()) {
            lm_fatal("unsupported",
                     "--reg-every " + std::to_string(a.reg_every) + " with only " +
                         std::to_string((int) samples.size()) +
                         " usable song(s) schedules ZERO prior-preservation steps per epoch — every "
                         "reg_every'th micro-step of " +
                         std::to_string((int) samples.size()) +
                         " never comes round. Use --reg-every <= " + std::to_string((int) samples.size()) +
                         " (or drop --reg-every to train without prior preservation).");
            lm_mirror_free(&mirror);
            qw3lm_free(&lm);
            return 2;
        }
        // Every candidate row across every --reg-codes file, then a
        // seed-derived shuffle, then the first --reg-songs that BUILD. Walking
        // the shuffled order (rather than sampling N and hoping) keeps the
        // selection deterministic even when some rows are over-length.
        std::vector<std::pair<size_t, LmCodeRow>> pool;
        for (size_t f = 0; f < a.reg_codes.size(); f++) {
            std::vector<LmCodeRow> rr;
            std::string            rerr;
            if (!lm_load_codes(a.reg_codes[f].c_str(), &rr, &rerr)) {
                lm_fatal("no-samples", "--reg-codes: " + rerr);
                lm_mirror_free(&mirror);
                qw3lm_free(&lm);
                return 1;
            }
            for (size_t i = 0; i < rr.size(); i++) {
                pool.push_back(std::make_pair(f, rr[i]));
            }
        }
        std::vector<int> pick;
        lm_epoch_order(&pick, (int) pool.size(), /*shuffle=*/a.order != "fixed",
                       (uint64_t) a.seed ^ 0x5EE9ull, /*epoch=*/0);
        int reg_over = 0, reg_bad = 0;
        for (size_t k = 0; k < pick.size() && (int) reg_samples.size() < a.reg_songs; k++) {
            const std::pair<size_t, LmCodeRow> & pr = pool[(size_t) pick[k]];
            LmSample    s;
            std::string why;
            bool        over = false;
            if (!lm_build_sequence(bpe, pr.second, a.loss_on_cot, max_len, &s, &why, &over)) {
                if (over) {
                    reg_over++;
                } else {
                    reg_bad++;
                }
                continue;
            }
            reg_samples.push_back(s);
            reg_ids.push_back(lm_prior_row_id(a.reg_codes[pr.first], pr.second.file, pr.second.id));
        }
        if (reg_samples.empty()) {
            lm_fatal("no-samples",
                     "--reg-every was asked for but no regularisation song survived (" +
                         std::to_string(reg_over) + " over max_len " + std::to_string(max_len) + ", " +
                         std::to_string(reg_bad) + " malformed)");
            lm_mirror_free(&mirror);
            qw3lm_free(&lm);
            return 1;
        }
        char b[256];
        snprintf(b, sizeof(b), "prior preservation: %d regularisation song(s) from %d file(s) (%d over max_len, %d malformed)",
                 (int) reg_samples.size(), (int) a.reg_codes.size(), reg_over, reg_bad);
        lm_log("info", b);
    }

    // Allocation-side worst case: style ∪ reg. Identical to the style figures
    // whenever the reg set is empty, which is what keeps a flags-off run
    // byte-identical.
    int alloc_seq = max_seq, alloc_s_tr = s_tr_max;
    for (size_t i = 0; i < reg_samples.size(); i++) {
        alloc_seq  = std::max(alloc_seq, (int) reg_samples[i].tokens.size());
        alloc_s_tr = std::max(alloc_s_tr, reg_samples[i].s_tr);
    }

    // ── --attn flash: capability probe (D6, spec §9.8) ───────────────────
    //
    // Asked HERE: after alloc_seq is known (so the geometry is the one this run
    // will actually emit) and before a single graph is built or a single buffer
    // allocated. A `false` from ggml_backend_supports_op is NOT an error this
    // trainer would otherwise notice — backend_sched_new registers the CPU
    // backend alongside CUDA, so the scheduler would silently SPLIT the fused
    // ops onto the CPU: Q/K/V and the F16 mask over PCIe for every layer of
    // every micro-step. Correct, unusably slow, LOW VRAM, and a quiet NVML
    // tripwire — i.e. indistinguishable from a pass on every number this run
    // reports. So we ask, and refuse rather than fall back.
    //
    // ONE attention shape, unlike the DiT's two: the LM has no cross-attention,
    // so S_kv == S == alloc_seq. Nkv is the NATIVE GQA width (Qwen3 4B is 32/8)
    // — the fused op takes GQA directly and lm_train_layer never expands heads,
    // so probing Nh would ask about a geometry this run never builds.
    //
    // NOTE the frozen KV prefix (lm-kvprefix.h) makes S_kv = kv->n + S, which is
    // LARGER than what is probed here. That path is MM3-only in R2 (D12: MM3
    // gains no flag), so no flash graph in this trainer ever emits it. R3 must
    // extend this probe before wiring the prefix to flash.
    if (attn_flash) {
        const float ascale = 1.0f / sqrtf((float) c.head_dim);
        bool        pf = false, pb = false;
        dit_flash_probe(lm.backend, c.head_dim, c.n_heads, c.n_kv_heads, alloc_seq, alloc_seq, /*B=*/1, ascale, &pf,
                        &pb);
        if (!(pf && pb)) {
            char extra[224];
            snprintf(extra, sizeof(extra), ",\"attn\":\"%s\",\"fwd\":%s,\"bwd\":%s", a.attn.c_str(),
                     pf ? "true" : "false", pb ? "true" : "false");
            char b[512];
            snprintf(b, sizeof(b),
                     "--attn %s: backend %s does not support the fused attention ops at this geometry "
                     "(D %d, Nh %d, Nkv %d, S %d, S_kv %d, B 1) — fwd %s / bwd %s. Refusing to start: the "
                     "scheduler would silently run them on the CPU instead, which is correct, unusably slow, "
                     "and looks like a pass on every number this run reports. Use --attn exact.",
                     a.attn.c_str(), ggml_backend_name(lm.backend), c.head_dim, c.n_heads, c.n_kv_heads, alloc_seq,
                     alloc_seq, pf ? "yes" : "NO", pb ? "yes" : "NO");
            lm_fatal("attn-unsupported", b, extra);
            lm_mirror_free(&mirror);
            qw3lm_free(&lm);
            return 1;
        }
        char b[448];
        snprintf(b, sizeof(b),
                 "--attn %s: %s supports FLASH_ATTN_TRAIN and FLASH_ATTN_TRAIN_BACK at D %d, Nh %d, Nkv %d, "
                 "S %d, B 1 — no CPU split. Requested arithmetic: %s (the backend resolves it per launch; "
                 "lm_train_log.json records what actually ran)",
                 a.attn.c_str(), ggml_backend_name(lm.backend), c.head_dim, c.n_heads, c.n_kv_heads, alloc_seq,
                 attn_prec_req == GGML_PREC_F32 ? "strict f32" : "tf32 where available");
        lm_log("info", b);
        fprintf(stderr, "[train-lm] %s\n", b);
    }

    const int n = (int) samples.size();
    // Micro-steps per epoch. With prior preservation on, every reg_every'th
    // micro-step is a reg step INSERTED into the pass rather than replacing a
    // song: an ACE epoch is defined as one teacher-forced pass over the album,
    // and target-loss/best-epoch selection both read that mean, so an epoch
    // that quietly saw two-thirds of the songs would corrupt the one number the
    // run is steered by. (MM3 is step-driven and has no such contract, which is
    // why it can simply replace.) The cost is honest and visible: at
    // --reg-every 3 an epoch runs 1.5x the micro-steps.
    //
    // reg_every <= n is guaranteed by the refusal in the Lever D block above, so
    // this loop always runs at least once when the lever is on. Two invariants
    // fall out of that and both are load-bearing below: micro_per_ep > n (there
    // is at least one reg step per epoch), and micro_per_ep is NEVER a multiple
    // of reg_every — a multiple m = k*reg_every yields the same style count as
    // m-1 (m - k == (m-1) - (k-1)), so the loop would have stopped at m-1. The
    // second invariant is why the final, possibly short optimizer window always
    // ends on a STYLE micro-step.
    int micro_per_ep = n;
    if (reg_on) {
        while (micro_per_ep - micro_per_ep / a.reg_every < n) {
            micro_per_ep++;
        }
    }
    const int steps_per_ep = (micro_per_ep + a.grad_accum - 1) / a.grad_accum;
    const int total_steps  = std::max(1, steps_per_ep * a.epochs);

    // lm_lr_lambda(0, ...) is exactly 0 by design (L6b), so a warmup that covers
    // every optimizer step trains NOTHING: AdamW updates only m/v, B stays 0 and
    // the exported adapter is an identity. The plan's `max(1, ...)` does exactly
    // that whenever total_steps == 1 (`--epochs 1` with n <= grad_accum, the
    // natural quick-test invocation) and also makes `--warmup-ratio 0`
    // inoperable. Hence the half-run cap below, and ratio 0 == "no warmup".
    //
    // The 50-step FLOOR mirrors dit-train-run.h (kept in lockstep deliberately):
    // percentage-only warmup is a hyperparameter-porting hazard, because the
    // reference recipes these learning rates come from (Side-Step) warm up over
    // a fixed ~50 of ~400 optimizer steps. With our smaller effective batch, 5 %
    // compressed to ~7 steps on a short run and the LR reached full scale before
    // the adapter had settled — reproducibly blowing up three DiT LoKR runs at
    // the end of the ramp (2026-07-29).
    int warmup_steps = 0;
    if (a.warmup_ratio > 0.0f) {
        warmup_steps = std::max(50, (int) ((double) total_steps * (double) a.warmup_ratio));
        warmup_steps = std::min(warmup_steps, total_steps / 2);
        if (warmup_steps < 1) {
            warmup_steps = 0;  // only reachable at total_steps == 1
        }
    }

    // ── vram + data events ───────────────────────────────────────────────
    // estMb is the predicted PEAK, i.e. the footprint at the LONGEST ACCEPTED
    // sequence — not at maxLen, which is only the skip threshold. That is the
    // quantity G4 compares against the observed peak.
    const bool   low  = (mode == LM_VRAM_LOWVRAM);
    const double est_bytes =
        low ? lm_vram_bytes_lowvram(vm, lc, alloc_seq, alloc_s_tr) : lm_vram_bytes(vm, alloc_seq, alloc_s_tr);
    const double ckpt_bytes = low ? (double) c.n_layers * (double) H * (double) alloc_seq * 4.0 : 0.0;
    const double seg_bytes  = low ? lm_vram_lowvram_transient(vm, lc, alloc_seq) : 0.0;

    // The attention share of estMb, printed beside the total. Skill §5: an
    // arena line that showed only the total once cost a whole refit chasing an
    // under-prediction that did not exist.
    const double attn_bytes = lm_vram_attn_term_bytes(vm, lc, alloc_seq, low);
    // D5. `attnPrec` here is the REQUESTED arithmetic, not the resolved one:
    // nothing has launched a fused kernel yet at this point in the run (the
    // capability probe above builds no_alloc nodes and computes nothing), so
    // asking the backend what it last ran would print "n/a" every time. The
    // RESOLVED answer is read back after each epoch and lands in
    // lm_train_log.json's attn_prec — that pair, requested here and resolved
    // there, is the whole point of logging two fields (skill §2 point 6).
    // Same wording as dit-node-profile.h uses for the requested label.
    const char * attn_prec_req_label =
        attn_flash ? (attn_prec_req == GGML_PREC_F32 ? "f32" : "tf32-where-available") : "n/a";

    // §2.2: five additive fields. In "naive" mode they read "naive", the mirror
    // size and three zeros, so the event stays byte-compatible with [P] §2.9.
    jl("{\"type\":\"vram\",\"freeMb\":%lld,\"totalMb\":%lld,\"reserveMb\":%d,\"mirrorMb\":%lld,\"maxLen\":%d,"
       "\"estMb\":%lld,\"source\":\"%s\",\"mode\":\"%s\",\"baseMb\":%lld,\"ckptMb\":%lld,\"segPeakMb\":%lld,"
       "\"attnHeadBlock\":%d,\"chunk\":%d,\"weights\":\"%s\",\"batch\":%d,\"batchSource\":\"%s\","
       "\"attn\":\"%s\",\"attnPrec\":\"%s\",\"attnMb\":%lld}",
       (long long) fit.free_mb, (long long) fit.total_mb, a.vram_reserve_mb,
       (long long) (low ? 0 : mirror.bytes / 1048576), max_len, (long long) (est_bytes / 1048576.0),
       a.max_len > 0 ? "user" : "auto", low ? "lowvram" : "naive",
       (long long) ((low ? (double) base_bytes : (double) mirror.bytes) / 1048576.0),
       (long long) (ckpt_bytes / 1048576.0), (long long) (seg_bytes / 1048576.0), low ? lc.attn_head_block : 0,
       low ? lc.chunk : 0,
       // §2.2: three additive fields. In a default run they read
       // "f32-window", 1, "user", so the event stays byte-compatible in meaning
       // and [P] §2.9's SSE mapping needs no change. weights_used (not
       // a.weights) so a CUDA/base-dtype fallback reports "f32-window", the
       // mode actually run, not the "bf16" the caller requested.
       weights_used.c_str(), 1, "user",
       // D5: three more additive fields. A default run reads "exact", "n/a" and
       // the exact model's own c1*Nh*S^2 attention term.
       a.attn.c_str(), attn_prec_req_label, (long long) (attn_bytes / 1048576.0));

    // `skippedBad` is additive (§2.2: consumers ignore unknown fields) and keeps
    // `skippedLong` meaning what its name says.
    jl("{\"type\":\"data\",\"samples\":%d,\"skippedLong\":%d,\"skippedBad\":%d,\"minLen\":%d,\"maxLen\":%d,"
       "\"maxLenCap\":%d,\"trainedTokens\":%lld,\"stepsPerEpoch\":%d,\"totalSteps\":%d,\"warmupSteps\":%d,"
       "\"loraParams\":%lld,\"batches\":%d,\"padTokens\":%lld,\"padPct\":%.1f}",
       n, skipped_long, skipped_bad, min_len, max_seq, max_len, trained_tokens, steps_per_ep, total_steps,
       warmup_steps, (long long) vm.lora_params,
       // §2.2: at --batch 1 there is one batch per sample and no padding at all,
       // so these read `samples`, 0 and 0.0 — the honest cost side of Lever B's
       // ledger, reported even when the lever is off.
       n, (long long) 0, 0.0);
    if (skipped_bad > 0) {
        char sb[128];
        snprintf(sb, sizeof(sb), "%d song(s) rejected for malformed prompt/codes — not a length problem", skipped_bad);
        lm_log("warn", sb);
    }

    // ── persistent allocations (§3.5.1) ──────────────────────────────────
    ggml_context * ctx_static = nullptr;
    {
        ggml_init_params p = { 32 * ggml_tensor_overhead(), nullptr, true };
        ctx_static         = ggml_init(p);
    }
    ggml_tensor * t_tok = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, alloc_seq);
    ggml_tensor * t_pos = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, alloc_seq);
    // D4: F32 under --attn exact (byte-identical to the shipped allocation),
    // F16 under flash — the fused op asserts an F16 contiguous mask. Same flat
    // [S_max*S_max] layout either way, so every view below is unchanged apart
    // from sizing its row stride with ggml_element_size().
    ggml_tensor * t_msk = lm_mask_alloc(ctx_static, (int64_t) alloc_seq * alloc_seq, attn_flash);
    // The [V, s_tr_max] label buffer is a naive-path structure: 1,184 MiB at
    // 4B / s_tr 1556. Low-vram sparse-writes a [V, chunk] buffer instead (D4).
    ggml_tensor * t_lab      = low ? nullptr : ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, V, alloc_s_tr);
    ggml_tensor * t_adamw    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 7);
    ggml_tensor * t_lossgrad = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_clip     = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_eps      = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_gnorm2   = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_gs       = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_one      = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_set_input(t_tok);
    ggml_set_input(t_pos);
    ggml_set_input(t_msk);
    if (t_lab) {
        ggml_set_input(t_lab);
    }

    ggml_backend_buffer_t buf_static = ggml_backend_alloc_ctx_tensors(ctx_static, lm.backend);
    if (!buf_static) {
        lm_fatal("vram", "static input buffer allocation failed");
        lm_mirror_free(&mirror);
        qw3lm_free(&lm);
        return 1;
    }
    ggml_backend_buffer_clear(buf_static, 0);

    {  // constants uploaded once
        const float lg = 1.0f / (float) a.grad_accum;  // == Side-Step's loss/(n_tok*grad_accum)
        const float cl = a.grad_clip;
        const float ep = 1e-6f;
        // D9: the low-vram trunk surrogate's loss gradient is EXACTLY 1.0 —
        // 1/grad_accum is already folded into the per-chunk `gs`. Using
        // t_lossgrad here would scale every gradient by 1/grad_accum^2.
        const float on = 1.0f;
        ggml_backend_tensor_set(t_lossgrad, &lg, 0, sizeof(float));
        ggml_backend_tensor_set(t_clip, &cl, 0, sizeof(float));
        ggml_backend_tensor_set(t_eps, &ep, 0, sizeof(float));
        ggml_backend_tensor_set(t_one, &on, 0, sizeof(float));
    }

    LmLora lora;
    {
        std::string err;
        const bool init_ok =
            (a.adapter_type == "lokr")
                ? lm_lokr_init(&lora, &lm, 0, c.n_layers, a.lokr_dim, a.lokr_alpha, a.lokr_factor,
                               a.lokr_decompose_both, (uint64_t) a.seed, &err)
                : lm_lora_init(&lora, &lm, 0, c.n_layers, a.rank, (float) a.alpha, (uint64_t) a.seed, /*b_sigma=*/0.0f,
                               &err);
        if (!init_ok) {
            lm_fatal("vram", err);
            return 1;
        }
    }
    // Resume: overwrite the fresh init with the source run's factors. After
    // lm_lora_init/lm_lokr_init so the tensors exist and the RNG stream stays
    // identical to a scratch run (a resumed run that falls back to scratch
    // would otherwise train from different noise than a bare run).
    if (!a.init_adapter.empty()) {
        std::string err;
        int         n_loaded = 0;
        if (!lm_resume_load(&lora, a.init_adapter, &n_loaded, &err)) {
            lm_fatal("resume", err);
            return 1;
        }
        char b[320];
        snprintf(b, sizeof(b), "resumed %d tensors from %s (source saved_loss %.4f) — expect epoch 1 near that loss",
                 n_loaded, a.init_adapter.c_str(), a.init_from_loss);
        lm_log("info", b);
        jl("{\"type\":\"resume\",\"initAdapter\":\"%s\",\"tensors\":%d,\"sourceLoss\":%.6f}",
           lm_json_escape(a.init_adapter).c_str(), n_loaded, a.init_from_loss);
    }
    if (lora.n_params != vm.lora_params) {
        char b[192];
        snprintf(b, sizeof(b), "LoRA parameter count %zu != predicted %zu", lora.n_params, vm.lora_params);
        lm_log("warn", b);
    }

    LmOptim opt;
    {
        std::string err;
        // BEFORE init: the rule split and the optimizer-state allocation are both
        // decided there (a Muon parameter gets no second momentum buffer).
        opt.optimizer     = a.optimizer;
        opt.muon.lr_scale = a.muon_lr_scale;
        opt.muon.momentum = a.muon_momentum;
        opt.muon.ns_steps = a.muon_ns_steps;
        opt.muon.nesterov = a.muon_nesterov;
        opt.muon.min_dim  = a.muon_min_dim;
        opt.muon.bucket   = a.muon_bucket;
        if (!lm_optim_init(&opt, lora.params, lm.backend, &err)) {
            lm_fatal("vram", err);
            return 1;
        }
    }
    // The rule split is only knowable AFTER lm_optim_init classifies, and it is
    // the number that says whether Muon did anything at all: a rank-8 LoRA puts
    // zero parameters on it and silently trains as AdamW.
    if (meta) {
        meta->muon_params  = opt.n_muon;
        meta->muon_buckets = (int) opt.muon_buckets.size();
    }
    opt.t_adamw      = t_adamw;
    opt.t_lossgrad   = t_lossgrad;
    opt.t_clip       = t_clip;
    opt.t_eps        = t_eps;
    opt.t_gnorm2     = t_gnorm2;
    opt.base_lr      = a.lr;
    opt.weight_decay = a.weight_decay;
    opt.grad_clip    = a.grad_clip;
    opt.total_steps  = total_steps;
    opt.warmup_steps = warmup_steps;

    // ── low-vram persistent state (§3.3) ─────────────────────────────────
    LmCkptState ckpt;
    LmCkptRun   run;
    if (low) {
        LmCkptCfg cc;
        cc.chunk           = lc.chunk;
        cc.attn_head_block = lc.attn_head_block;
        cc.s_max           = alloc_seq;
        cc.layer_lo        = 0;
        cc.layer_hi        = c.n_layers;
        cc.weights_bf16    = weights_bf16;  // Lever A
        cc.attn_flash      = attn_flash;    // D1 — reaches P2/P3/P7 and the node
        cc.attn_prec       = attn_prec_req; //      probe via lm_ckpt_layer_opts
        std::string err;
        if (!lm_ckpt_alloc(&ckpt, &lm, cc, &err) || !lm_ckpt_build_embed_t(&ckpt, &err)) {
            lm_fatal("vram", err.empty() ? std::string("low-vram allocation failed") : err);
            return 1;
        }
        run.lm         = &lm;
        run.opt        = &opt;
        run.st         = &ckpt;
        run.t_tok      = t_tok;
        run.t_pos      = t_pos;
        run.t_msk      = t_msk;
        run.t_gs       = t_gs;
        run.t_one      = t_one;
        run.grad_accum = a.grad_accum;
    }

    // ── graph arena + scheduler sized from a real node count ─────────────
    std::vector<uint8_t> arena((size_t) 128 << 20);

    // The naive trunk's layer options. EVERY field is at its default when
    // --attn exact, so lm_build_trunk's opts overload emits exactly the node
    // sequence the no-opts overload emits (the two builder bodies differ only in
    // whether they forward `opts`, and lm_train_layer's 7-argument form
    // constructs precisely this default). That equality is what G0 checks and is
    // the reason the naive path can carry the flag without moving its graph.
    LmLayerOpts naive_opts;
    naive_opts.attn_flash = attn_flash;
    naive_opts.attn_prec  = attn_prec_req;

    int  last_mask_S = 0;
    auto upload_mask = [&](int S) {
        if (S == last_mask_S) {
            return;
        }
        std::vector<float> m;
        lm_causal_mask(S, &m);
        lm_mask_set(t_msk, m);  // converts to F16 when t_msk is F16 (D4)
        last_mask_S = S;
    };

    // Build (but do not run) the largest graph to size the scheduler.
    int graph_nodes = 0, graph_leafs = 0;
    if (low) {
        // The worst low-vram graph is one backward segment at S = max_seq: the
        // trunk is never built whole, so sizing from it would over-allocate the
        // sched by ~L x. Reuse of a single sched across ~110 graph computes per
        // micro-step is what keeps the allocator arena stable (T13).
        upload_mask(alloc_seq);
        run.sched   = nullptr;
        graph_nodes = lm_ckpt_probe_segment_nodes(run, alloc_seq);
        graph_leafs = 0;
    } else {
        ggml_init_params ip  = { arena.size(), arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, /*grads=*/true);

        const LmSample & s      = samples[0];
        ggml_tensor *    hidden = lm_build_trunk(ctx, &lm, t_tok, t_pos, t_msk, alloc_seq, 0, c.n_layers, naive_opts);
        ggml_tensor *    hd     = ggml_cont(
            ctx, ggml_view_2d(ctx, hidden, H, s.s_tr, hidden->nb[1], (size_t) (s.n_masked - 1) * hidden->nb[1]));
        ggml_tensor * logits = ggml_mul_mat(ctx, lm.embed_tokens, hd);
        ggml_tensor * labv   = ggml_view_2d(ctx, t_lab, V, s.s_tr, t_lab->nb[1], 0);
        ggml_tensor * loss   = ggml_cross_entropy_loss(ctx, logits, labv);
        ggml_set_loss(loss);
        ggml_set_output(loss);
        ggml_build_forward_expand(gf, loss);
        std::vector<ggml_tensor *> gacc;
        lm_optim_fill_gacc(&opt, gf, &gacc);
        ggml_build_backward_expand(ctx, gf, gacc.data());
        graph_nodes = ggml_graph_n_nodes(gf);
        graph_leafs = 0;  // ggml_cgraph is opaque; size the sched from nodes + headroom
        ggml_free(ctx);
    }
    fprintf(stderr, "[train-lm] %s graph: %d nodes\n", low ? "worst segment fwd+bwd" : "fwd+bwd", graph_nodes);

    BackendPair bp;
    bp.backend     = lm.backend;
    bp.cpu_backend = lm.cpu_backend;
    bp.has_gpu     = lm.backend != lm.cpu_backend;
    // THE SCHEDULER IS SHARED WITH THE OPTIMIZER STEP, so it must be sized for
    // whichever of the two graphs is larger — not just the training one.
    //
    // This bit a real 4B Muon run (2026-07-30): the low-VRAM path segments the
    // model, so `graph_nodes` was 569 and the sched took the 8192 floor, while
    // Muon's optimizer graph for 612 parameters is ~7-9k nodes. ggml asserts
    // `hash_set.size >= n_nodes + n_leafs` (ggml-backend.cpp:1866) and aborts
    // mid-run. It did NOT show on the 0.6B, where the graph is unsegmented and
    // big enough that the sched was large enough by accident — which is exactly
    // the kind of luck that makes a smoke test lie.
    //
    // opt.est_nodes is the optimizer's own cap (AdamW ~8/param, Muon ~(20+10*ns)
    // per bucket plus ~10/param) and already carries 25%; the extra quarter here
    // covers leafs, which the assert counts and est_nodes does not.
    const int sched_nodes = std::max(std::max(8192, graph_nodes + graph_nodes / 2 + 2048),
                                     opt.est_nodes + opt.est_nodes / 4 + 1024);
    ggml_backend_sched_t sched = backend_sched_new(bp, sched_nodes);
    if (low) {
        run.sched = sched;
    }

    // ── one micro-step ───────────────────────────────────────────────────
    LmVramTracker tracker;
    // The shipped naive lambda, MOVED but not edited.
    auto micro_step_naive = [&](const LmSample & s, bool count_loss, double * ce_out) -> bool {
        const int S    = (int) s.tokens.size();
        const int s_tr = s.s_tr;

        upload_mask(S);
        ggml_backend_tensor_set(t_tok, s.tokens.data(), 0, (size_t) S * 4);
        {
            std::vector<int32_t> ip((size_t) S);
            for (int i = 0; i < S; i++) {
                ip[(size_t) i] = i;
            }
            ggml_backend_tensor_set(t_pos, ip.data(), 0, (size_t) S * 4);
        }

        ggml_init_params gip = { arena.size(), arena.data(), true };
        ggml_context *   ctx = ggml_init(gip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, /*grads=*/true);

        ggml_tensor * hidden = lm_build_trunk(ctx, &lm, t_tok, t_pos, t_msk, S, 0, c.n_layers, naive_opts);
        ggml_tensor * hd =
            ggml_cont(ctx, ggml_view_2d(ctx, hidden, H, s_tr, hidden->nb[1], (size_t) (s.n_masked - 1) * hidden->nb[1]));
        ggml_tensor * logits = ggml_mul_mat(ctx, lm.embed_tokens, hd);  // [V, s_tr] (tied head)
        ggml_tensor * labv   = ggml_view_2d(ctx, t_lab, V, s_tr, t_lab->nb[1], 0);
        ggml_tensor * loss   = ggml_cross_entropy_loss(ctx, logits, labv);
        ggml_set_loss(loss);
        ggml_set_output(loss);
        ggml_build_forward_expand(gf, loss);

        std::vector<ggml_tensor *> gacc;
        lm_optim_fill_gacc(&opt, gf, &gacc);
        ggml_build_backward_expand(ctx, gf, gacc.data());

        bool ok = false;
        {
            LmLabelGuard guard(t_lab, s.targets.data(), s_tr, V);
            ggml_backend_sched_reset(sched);
            ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
            if (ok && count_loss) {
                float ce = 0.0f;
                ggml_backend_tensor_get(loss, &ce, 0, sizeof(float));
                *ce_out = (double) ce;
            }
        }
        ggml_free(ctx);
        return ok;
    };

    auto micro_step = [&](const LmSample & s, bool count_loss, double * ce_out) -> bool {
        return low ? lm_ckpt_micro_step(run, s, count_loss, ce_out) : micro_step_naive(s, count_loss, ce_out);
    };

    // ── high-water probe (§3.7) ──────────────────────────────────────────
    //
    // The probe must bound the run's PEAK allocation, not merely its longest
    // sequence. The two biggest transients — `logits` and its gradient, each
    // V*s_tr*4 (1.66 MB per trained token at 0.6B) — scale with s_tr, and
    // argmax(S) != argmax(s_tr) in general because n_masked (the caption+lyrics
    // prompt length) varies independently of the code count. Probing only the
    // longest sequence lets ggml_gallocr grow mid-epoch, which transiently holds
    // the old and new arenas at once — exactly the "40 songs in, OOM" §3.7 says
    // the probe exists to prevent — and it invalidates the L10 leak baseline.
    //
    // §3.7 already sanctions synthetic tokens, so probe the JOINT worst case
    // (max_seq, s_tr_max) even if no single real sample exhibits it.
    {
        // The longest sequence over EVERY set this run will feed the graph. With
        // prior preservation on, a regularisation row from another artist can be
        // the longest thing here, and probing only the album would leave the
        // gallocr to grow on the first reg step.
        const LmSample * plong = &samples[0];
        for (size_t i = 1; i < samples.size(); i++) {
            if (samples[i].tokens.size() > plong->tokens.size()) {
                plong = &samples[i];
            }
        }
        for (size_t i = 0; i < reg_samples.size(); i++) {
            if (reg_samples[i].tokens.size() > plong->tokens.size()) {
                plong = &reg_samples[i];
            }
        }
        LmSample probe = *plong;                      // real token ids, longest length
        probe.soft_k   = 0;                           // hard labels: the probe is a shape test
        probe.soft_idx.clear();
        probe.soft_p.clear();
        probe.n_masked = alloc_seq - alloc_s_tr;      // >= 1: every sample has s_tr <= S-1
        probe.s_tr     = alloc_s_tr;
        GGML_ASSERT(probe.n_masked >= 1 && (int) probe.tokens.size() == alloc_seq);
        probe.targets.assign(probe.tokens.begin() + probe.n_masked, probe.tokens.end());
        GGML_ASSERT((int) probe.targets.size() == alloc_s_tr);

        double dummy = 0.0;
        if (!micro_step(probe, false, &dummy)) {
            lm_fatal("vram", "the high-water probe failed to run — not enough VRAM for the worst-case sequence");
            return 1;
        }
        lm_optim_zero_grad(&opt);

        // Trainer-owned footprint: the buffers we allocated plus the scheduler's
        // own compute arena. Device-wide (total - free) would fold in every other
        // process on the card and is not comparable with estMb.
        size_t fixed = low ? (base_bytes + ckpt.fixed_bytes())
                           : (mirror.buf ? ggml_backend_buffer_get_size(mirror.buf) : mirror.bytes);
        fixed += ggml_backend_buffer_get_size(buf_static);
        if (lora.buf) {
            fixed += ggml_backend_buffer_get_size(lora.buf);
        }
        if (opt.buf_grad) {
            fixed += ggml_backend_buffer_get_size(opt.buf_grad);
        }
        if (opt.buf_mom) {
            fixed += ggml_backend_buffer_get_size(opt.buf_mom);
        }
        tracker.probe_baseline(lm.backend, sched, fixed);
        // BOTH TERMS, not just the total (skill §5). The number this line exists
        // to let you read is `measured / est`; when that ratio looks wrong the
        // very next question is "which term", and a line that printed only the
        // total once sent a whole DiT refit chasing an under-prediction that was
        // really a 73 % OVER-prediction in a term nobody had printed. `attn` is
        // the c1*Nh*S^2 retained softmax in exact mode and the fused op's
        // per-token state under --attn flash; `rest` is everything else.
        fprintf(stderr,
                "[train-lm] high-water probe at S=%d s_tr=%d: trainer VRAM %zu MB (est %lld MB = attn %lld MB + "
                "rest %lld MB, --attn %s, device %zu MB)\n",
                alloc_seq, alloc_s_tr, tracker.base_mb, (long long) (est_bytes / 1048576.0),
                (long long) (attn_bytes / 1048576.0), (long long) ((est_bytes - attn_bytes) / 1048576.0),
                a.attn.c_str(), lm_vram_used_mb(lm.backend));
    }

    // ── Lever D: capture the frozen base's own answers ───────────────────
    //
    // MUST run before the first optimizer step and after the machinery exists.
    // lm_lora_init/lm_lokr_init both leave the delta at EXACTLY zero, so right
    // now a forward pass IS the frozen base; one step from now it is not, and
    // the teacher would quietly become the student. The high-water probe above
    // ran a graph but zeroed the gradients and never stepped, so the adapter is
    // still inert here.
    //
    // A RESUME CANNOT REGENERATE. --init-adapter loads trained factors before
    // this point, so the model is no longer the base and a fresh capture would
    // teach the adapter its own output. The cache must be found on disk, and
    // this says so rather than capturing something contaminated.
    if (reg_on) {
        std::string prior_dir = a.reg_prior_dir.empty() ? lm_join(a.out_dir, "prior") : a.reg_prior_dir;
        if (!pm_mkdir_p(prior_dir)) {
            lm_fatal("export", "cannot create the prior cache dir " + prior_dir);
            return 1;
        }
        const int W = lm_ckpt_head_width(&lm, ckpt.cfg);  // scored width the capture spans
        reg_priors.resize(reg_samples.size());

        int           made = 0, n_cached = 0;
        double        cov_sum = 0.0;
        const int64_t cap_t0  = ggml_time_ms();
        for (size_t i = 0; i < reg_samples.size(); i++) {
            const LmSample &  rs   = reg_samples[i];
            const std::string path = lm_prior_path(prior_dir, reg_ids[i], a.lm_path, a.reg_topk);
            std::string       lerr;
            if (lm_prior_load(path, a.reg_topk, rs.s_tr, W, &reg_priors[i], &lerr)) {
                n_cached++;
                cov_sum += lm_prior_coverage(reg_priors[i]);
                continue;
            }
            if (!a.init_adapter.empty()) {
                lm_fatal("resume", "resuming from " + a.init_adapter + ", but the prior cache for " + reg_ids[i] +
                                       " is missing or stale (" + lerr +
                                       "). It can only be captured before the first optimizer step, so the "
                                       "adapter is no longer inert and a fresh capture would teach the student "
                                       "its own output. Point --reg-prior-dir at the original cache, or train "
                                       "from scratch.");
                return 1;
            }

            LmPriorCache pc;
            pc.k     = a.reg_topk;
            pc.n_pos = rs.s_tr;
            pc.width = W;
            pc.idx.reserve((size_t) rs.s_tr * (size_t) a.reg_topk);
            pc.p.reserve((size_t) rs.s_tr * (size_t) a.reg_topk);
            // Forward-only through the machinery training already allocated, so
            // the capture costs compute and NOT VRAM.
            run.forward_only = true;
            run.capture_k    = a.reg_topk;
            run.capture_idx  = &pc.idx;
            run.capture_p    = &pc.p;
            const bool ok    = lm_ckpt_micro_step(run, rs, false, nullptr);
            run.capture_k    = 0;
            run.capture_idx  = nullptr;
            run.capture_p    = nullptr;
            run.forward_only = false;
            if (!ok || (int) pc.idx.size() != pc.n_pos * pc.k) {
                lm_fatal("vram", "prior capture failed for " + reg_ids[i]);
                return 1;
            }
            std::string serr;
            if (!lm_prior_save(path, pc, &serr)) {
                lm_log("warn", "prior cache not written (" + serr + ") — continuing in memory");
            }
            cov_sum += lm_prior_coverage(pc);
            reg_priors[i] = pc;
            made++;
        }
        // Point every reg sample at its teacher, RENORMALISING each row to sum
        // to 1 on the way in.
        //
        // THIS IS NOT COSMETIC. ggml's cross-entropy backward is hard-wired to
        // grad = (softmax(z) - labels) * d / nr (ggml-cpu/ops.cpp
        // ggml_compute_forward_cross_entropy_loss_back_f32, and the same
        // expression in ggml-cuda/cross-entropy-loss.cu). The true gradient of
        // the forward loss L = -SUM_j y_j log softmax_j is (S*softmax_j - y_j)
        // with S = SUM_j y_j, so the two agree ONLY when S == 1. Feed it rows
        // that sum to S < 1 and the implemented gradient is the true one PLUS
        // (1 - S) * softmax_j — the gradient of (1 - S) * logsumexp(z), a pull
        // on the raw logit SCALE that changes no probability at all and is
        // unbounded below along a uniform shift. It is the cheapest direction
        // the adapter can descend, and the reported CE would not even be the
        // quantity being minimised.
        //
        // On MM3 that error is invisible (S ~ 0.99, a 1% perturbation), which
        // is why the ported code got away with it there. On ACE the top-K
        // covers 18.5-24% of the mass, so ~80% of the gradient would be the
        // spurious logit-shrink term. Normalising here makes ggml's expression
        // the EXACT gradient of the renormalised-top-K cross-entropy, which is
        // then also what regCe reports and what the run is actually steered by.
        //
        // Normalisation happens at THIS call site, not in the capture: the
        // cached .prior file and `coverage` above both stay the raw captured
        // mass (the honest measure of how much of the base we recorded), and
        // lm-ckpt.h's capture stays byte-identical for MM3.
        //
        // HOW the row is brought to 1 (2026-09-02, after the coverage numbers
        // below were measured): NOT by rescaling the captured top-K. At 18-24%
        // coverage that would hold the adapter to a target far SHARPER than the
        // base — every code outside the K asked for 0, the K asked for 4-5x
        // the mass the base gave them — and sharpening is precisely the loop
        // attractor this term exists to resist. Instead the captured entries
        // keep their TRUE probabilities and the leftover mass (1 - S) is spread
        // uniformly over the audio-code token range by lm_soft_labels_write
        // (LmSample::soft_tail). A flat tail is an approximation of the base's
        // real tail, but its entropy is on the right side: the target is at
        // least as flat as the base, never sharper. Coverage still reads raw.
        for (size_t i = 0; i < reg_samples.size(); i++) {
            const LmPriorCache & pc = reg_priors[i];
            std::vector<float>   tail((size_t) std::max(0, pc.n_pos), 0.0f);
            if (pc.k > 0) {
                for (int r = 0; r < pc.n_pos; r++) {
                    const float * row = pc.p.data() + (size_t) r * (size_t) pc.k;
                    double        s   = 0.0;
                    for (int k = 0; k < pc.k; k++) {
                        s += (double) row[k];
                    }
                    // Numerics can put S a hair above 1; clamp so the tail is
                    // never negative. An all-zero row (degenerate head) gets a
                    // uniform tail of 1 — "no opinion", every code equally.
                    tail[(size_t) r] = (float) std::max(0.0, 1.0 - s);
                }
            }
            reg_samples[i].soft_k       = pc.k;
            reg_samples[i].soft_idx     = pc.idx;
            reg_samples[i].soft_p       = pc.p;
            reg_samples[i].soft_tail    = std::move(tail);
            reg_samples[i].soft_tail_lo = AUDIO_CODE_BASE;
            reg_samples[i].soft_tail_hi = AUDIO_CODE_BASE + AUDIO_CODE_COUNT;
        }
        const int n_ok = made + n_cached;
        char      pb[384];
        snprintf(pb, sizeof(pb),
                 "prior preservation: %d song(s) (%d captured, %d cached), top-%d covering %.2f%% of the base's "
                 "probability mass, %lld s",
                 n_ok, made, n_cached, a.reg_topk, n_ok ? 100.0 * cov_sum / (double) n_ok : 0.0,
                 (long long) ((ggml_time_ms() - cap_t0) / 1000));
        lm_log("info", pb);
        fprintf(stderr, "[train-lm] %s\n", pb);
        // COVERAGE IS THE NUMBER TO READ, and on ACE it is not MM3's number.
        // MM3 scores 16,385 classes and its top-64 routinely covers >99%; ACE
        // scores 217,204 and its base sits at CE ~9 on another artist's codes,
        // i.e. thousands of effective classes, so the same K covers a fraction
        // of the mass. What that costs is RESOLUTION, not a broken gradient and
        // not sharpening: the captured K keep their true probabilities and the
        // rest of the mass is a flat floor over the code range, so the target
        // is the base's head plus a uniform tail. At 99% coverage that is the
        // base. At 18-24% most of the target IS the floor — a weak anchor that
        // says "stay flat" more than "stay the base". Raising --reg-topk (up
        // to 256) sharpens the picture a little; the real fix is a live
        // teacher forward (no cache, full distribution), not a bigger K.
        const double cov = n_ok ? cov_sum / (double) n_ok : 0.0;
        if (cov < 0.5) {
            char wb[448];
            snprintf(wb, sizeof(wb),
                     "prior preservation: top-%d captures only %.1f%% of the base's probability mass; the other "
                     "%.1f%% of each target row is a uniform floor over the %d audio-code tokens. The term keeps "
                     "the adapter FLAT rather than holding it to the base's exact distribution. Raise --reg-topk "
                     "(max %d) for a slightly sharper picture.",
                     a.reg_topk, 100.0 * cov, 100.0 * (1.0 - cov), AUDIO_CODE_COUNT, LM_CAPTURE_K_MAX);
            lm_log("warn", wb);
            fprintf(stderr, "[train-lm] %s\n", wb);
        }
        fprintf(stderr,
                "[train-lm] every %d%s micro-step trains against the frozen base instead of the artist — %d of "
                "%d micro-steps per epoch are the album\n",
                a.reg_every, a.reg_every == 2 ? "nd" : a.reg_every == 3 ? "rd" : "th", n, micro_per_ep);
        jl("{\"type\":\"prior\",\"songs\":%d,\"captured\":%d,\"cached\":%d,\"topK\":%d,\"coverage\":%.6f,"
           "\"regEvery\":%d,\"microPerEpoch\":%d,\"styleMicroPerEpoch\":%d}",
           n_ok, made, n_cached, a.reg_topk, n_ok ? cov_sum / (double) n_ok : 0.0, a.reg_every, micro_per_ep, n);
    }

    // A previous run into the same <out> leaves its milestone dirs behind; they
    // are absent from this run's log and would be offered by the UI picker.
    lm_milestone_reset(a.out_dir);

    // ── epoch loop (§3.5.4) ──────────────────────────────────────────────
    meta->max_len        = max_len;
    meta->max_len_source = a.max_len > 0 ? "user" : "auto";
    meta->samples        = n;
    meta->skipped_long   = skipped_long;
    meta->vram_free_mb   = fit.free_mb;
    meta->vram_total_mb  = fit.total_mb;
    meta->vram_mirror_mb = mirror.bytes / 1048576;
    meta->vram_est_mb    = (size_t) (est_bytes / 1048576.0);

    meta->low_vram        = low;
    meta->attn_head_block = low ? lc.attn_head_block : 0;
    meta->chunk           = low ? lc.chunk : 0;
    meta->weights         = weights_used;   // §2.3 — recorded so a resume can refuse (S6); ACTUAL mode, not requested
    meta->batch           = 1;
    meta->init_adapter    = a.init_adapter;
    meta->init_from_loss  = a.init_from_loss;
    meta->vram_mode       = low ? "lowvram" : "naive";
    meta->vram_base_mb    = (size_t) ((low ? (double) base_bytes : (double) mirror.bytes) / 1048576.0);
    meta->vram_ckpt_mb    = (size_t) (ckpt_bytes / 1048576.0);
    meta->vram_seg_peak_mb = (size_t) (seg_bytes / 1048576.0);

    double    ladder      = 0.0;
    bool      ladder_seed = false;
    long long ep_ms_sum   = 0;
    int       global_step = 0;
    std::string export_err;
    int       rc = 0;

    // Lever D bookkeeping. `reg_seen` is a run-global counter of reg
    // micro-steps, so the reg pass order is derived from it rather than reset
    // per epoch — every song in the corpus gets its turn before any is repeated.
    std::vector<int> reg_order;
    int              reg_epoch_cur = -1;
    int              reg_seen      = 0;

    for (int epoch = 0; epoch < a.epochs; epoch++) {
        const int64_t t_ep0 = ggml_time_ms();
        std::vector<int> order;
        lm_epoch_order(&order, n, a.order != "fixed", (uint64_t) a.seed, epoch);

        // Lever C's draw stream. Seeded PER EPOCH from the run seed rather than
        // carried across epochs, so a given (seed, epoch) always makes the same
        // choices no matter what else consumed randomness first. That is what
        // makes it reproducible across a --init-adapter resume, whose epoch
        // counter starts over. Never constructed when the lever is off, so it
        // cannot perturb a default run.
        LmRng cd_rng;
        if (cd_on) {
            lm_rng_seed(&cd_rng, ((uint64_t) a.seed ^ 0xCA97D40Bull) + (uint64_t) epoch);
        }
        int cd_used = 0;

        double running = 0.0;
        int    n_micro = 0;
        double reg_running = 0.0;
        int    reg_micro   = 0;
        lm_optim_zero_grad(&opt);
        LmStepStats last_stats;

        // §2.2 defines `step.micro` as the micro-batches folded into THIS
        // optimizer step and `step.ms` as that step's own wall time. Both need
        // window-local state: the epoch accumulators would report a cumulative
        // count and a smoothed epoch mean, and timing from inside the sample loop
        // measures only the final micro-batch (~1/grad_accum of the truth).
        double  win_loss  = 0.0;
        int     win_micro = 0;
        int64_t t_win0    = ggml_time_ms();

        // `t` is the micro-step index within the epoch, 1-based. With both levers
        // off it equals j+1 and micro_per_ep equals n, so every decision below
        // collapses to the shipped loop exactly.
        int j = 0;  // cursor into `order`, advanced only by STYLE micro-steps
        for (int t = 1; t <= micro_per_ep; t++) {
            const bool is_reg = reg_on && (t % a.reg_every == 0);
            double     ce     = 0.0;

            const LmSample * sp = nullptr;
            if (is_reg) {
                // The reg pass has its OWN shuffled order, so a reg step never
                // consumes a style epoch, and its position comes from the
                // run-global counter rather than a cursor a resume could
                // silently reset.
                const int nreg      = (int) reg_samples.size();
                const int reg_epoch = reg_seen / nreg;
                if (reg_epoch != reg_epoch_cur || reg_order.empty()) {
                    lm_epoch_order(&reg_order, nreg, a.order != "fixed", (uint64_t) a.seed ^ 0x9E37ull, reg_epoch);
                    reg_epoch_cur = reg_epoch;
                }
                sp = &reg_samples[(size_t) reg_order[(size_t) (reg_seen % nreg)]];
                reg_seen++;
            } else {
                sp = &samples[(size_t) order[(size_t) j]];
                // Caption dropout. A reg step NEVER drops: its cached teacher was
                // captured against that row's own full caption and no other, so a
                // trigger-only prompt there would score the base's answer to a
                // question it was never asked.
                if (cd_on && lm_rng_uniform(&cd_rng) < a.caption_dropout) {
                    sp = &cd_variants[(size_t) order[(size_t) j]];
                    cd_used++;
                }
                j++;
            }

            if (!micro_step(*sp, true, &ce)) {
                lm_fatal("vram", "graph compute failed mid-epoch");
                rc = 1;
                break;
            }
            if (is_reg) {
                // Kept out of the style loss on purpose: it is a different
                // objective (soft targets against the frozen base), and folding
                // it into the epoch mean would move target-loss and best-epoch
                // selection for reasons that have nothing to do with the album.
                reg_running += ce;
                reg_micro++;
            } else {
                running += ce;
                n_micro++;
                win_loss += ce;
                win_micro++;
            }

            // `win_micro` is the count of STYLE micro-steps in this window, and
            // it is what `step.loss` is divided by. It can never be 0:
            //   * lever off  — every micro-step is a style step;
            //   * lever on   — --reg-every >= 2 and --grad-accum >= 2 are both
            //                  refused otherwise (ace-train.cpp), so no window
            //                  of >= 2 consecutive t can be all reg (two
            //                  consecutive integers cannot both be multiples of
            //                  reg_every >= 2), and the one window that can be
            //                  1 wide — the final t == micro_per_ep — is a style
            //                  step by the "never a multiple" invariant above.
            // Without that guarantee a style-free window would publish a
            // FABRICATED 0.0 into the primary telemetry stream the Training
            // Studio charts, which reads as the run collapsing. The std::max
            // below is a divide-by-zero backstop, not a licence to emit 0.0.
            if (t % a.grad_accum == 0 || t == micro_per_ep) {
                if (!lm_optim_step(&opt, sched, &last_stats)) {
                    lm_fatal("vram", "optimizer step failed");
                    rc = 1;
                    break;
                }
                global_step++;
                const size_t vram_mb = tracker.sample();
                jl("{\"type\":\"step\",\"epoch\":%d,\"step\":%d,\"totalSteps\":%d,\"micro\":%d,\"loss\":%.6f,"
                   "\"lr\":%.9g,\"gradNorm\":%.6f,\"clipScale\":%.6f,\"ms\":%lld,\"vramMb\":%lld,"
                   "\"samples\":%d}",
                   epoch + 1, global_step, total_steps, win_micro, win_loss / std::max(1, win_micro),
                   (double) last_stats.lr, (double) last_stats.grad_norm, (double) last_stats.clip,
                   (long long) (ggml_time_ms() - t_win0), (long long) vram_mb,
                   // §4.8: samples == micro * B_cur summed over the window. At
                   // B == 1 that is exactly `micro`.
                   win_micro);
                win_loss  = 0.0;
                win_micro = 0;
                t_win0    = ggml_time_ms();
            }
        }
        if (rc != 0) {
            break;
        }
        // Per-epoch lever telemetry, emitted as their OWN event types rather
        // than as extra keys on `step`, so a flags-off run's step stream stays
        // byte-for-byte what it always was.
        if (cd_on) {
            fprintf(stderr,
                    "[train-lm] caption dropout: %d of %d micro-steps used the trigger-only prompt (p=%.3g)\n",
                    cd_used, n, (double) a.caption_dropout);
            jl("{\"type\":\"caption_dropout\",\"epoch\":%d,\"used\":%d,\"styleMicro\":%d,\"p\":%.6g}",
               epoch + 1, cd_used, n, (double) a.caption_dropout);
        }
        if (reg_on) {
            const double reg_avg = reg_running / std::max(1, reg_micro);
            fprintf(stderr, "[train-lm] prior preservation: %d reg micro-step(s), reg CE %.6f\n", reg_micro,
                    reg_avg);
            jl("{\"type\":\"reg_epoch\",\"epoch\":%d,\"micro\":%d,\"regCe\":%.6f}", epoch + 1, reg_micro,
               reg_avg);
        }

        const double avg   = running / std::max(1, n_micro);
        const long long ems = (long long) (ggml_time_ms() - t_ep0);
        ep_ms_sum += ems;

        // Resume honesty gate (observational): epoch 1 of a resumed run is a
        // teacher-forced pass from the loaded factors, so it should land near
        // the source's saved_loss. A large gap = wrong init mapping, or the
        // dataset/prompt format changed since the source run.
        if (epoch == 0 && !a.init_adapter.empty() && a.init_from_loss > 0.0) {
            const double gap = fabs(avg - a.init_from_loss);
            char b[224];
            snprintf(b, sizeof(b), "resume check: epoch 1 loss %.4f vs source saved_loss %.4f (|gap| %.4f)%s", avg,
                     a.init_from_loss, gap, gap > 0.35 ? " — LARGER THAN EXPECTED, verify dataset/init" : "");
            lm_log(gap > 0.35 ? "warn" : "info", b);
        }

        const bool best = (out->best_loss < 0.0) || (avg < out->best_loss);
        if (best) {
            out->best_loss  = avg;
            out->best_epoch = epoch + 1;
        }
        out->final_loss = avg;
        out->epochs_run = epoch + 1;

        LmEpochRec rec;
        rec.epoch     = epoch + 1;
        rec.loss      = avg;
        rec.lr        = last_stats.lr;
        rec.grad_norm = last_stats.grad_norm;
        rec.ms        = ems;
        meta->epoch_log.push_back(rec);

        // D5 — the RESOLVED arithmetic, asked of the backend that has just run
        // an epoch's worth of fused launches rather than restated from the flag.
        // Refreshed every epoch so the field is populated even if the run is
        // cancelled after the first one. Mirrors dit-train-run.h.
        if (attn_flash) {
            meta->attn_prec = dit_flash_prec_label(lm.backend);
        }

        // Export the BEST epoch, not the last one (2026-07-30) — mirrors the
        // same change in dit-train-run.h. This used to export unconditionally
        // every epoch, so a run that never reached its target shipped whatever
        // the final epoch produced. Exporting only on improvement still leaves
        // a usable adapter on disk from epoch 1 onward (the original reason for
        // the unconditional export) and writes strictly less.
        //
        // Selection is on the per-epoch mean, NOT an MA5 as on the DiT: an LM
        // epoch is a full teacher-forced pass over every sample with no random
        // timestep draw, so its loss is far less noisy than a DiT epoch's and
        // needs no smoothing. It is also the metric the target test below uses,
        // so the two stay consistent.
        //
        // hit_target forces the export even when it is not an improvement: a
        // run that stops on target must ship the epoch that tripped it.
        meta->epochs_run = out->epochs_run;
        meta->final_loss = out->final_loss;
        meta->best_loss  = out->best_loss;
        meta->best_epoch = out->best_epoch;
        const bool lm_hit_target = (a.target_loss > 0.0f && avg <= (double) a.target_loss);
        if (best || lm_hit_target) {
            out->saved_loss   = best ? avg : out->saved_loss;
            out->saved_epoch  = epoch + 1;
            out->saved_reason = lm_hit_target ? "target" : "best";
            meta->saved_loss   = out->saved_loss;
            meta->saved_epoch  = out->saved_epoch;
            meta->saved_reason = out->saved_reason;

            LmExportResult xr;
            const bool xok = lora.is_lokr ? lm_export_lokr(lora, *meta, a.out_dir, &xr, &export_err)
                                          : lm_export_peft(lora, c, *meta, a.out_dir, &xr, &export_err);
            if (!xok) {
                lm_fatal("export", export_err);
                rc = 1;
                break;
            }
            out->exported       = true;
            out->export_tensors = xr.tensors;
        }
        // milestones (L20)
        if (a.milestone_step > 0.0f) {
            if (!ladder_seed) {
                ladder      = floor(avg / (double) a.milestone_step) * (double) a.milestone_step;
                ladder_seed = true;
            }
            while (avg <= ladder + 1e-12) {
                const std::string label = lm_fmt1(ladder);          // "7.3"
                const double      lval  = atof(label.c_str());       // exactly what the dir says
                const std::string rel   = "milestones/loss_" + label;
                const std::string mdir  = lm_join(a.out_dir, rel);
                LmExportResult    mr;
                std::string       merr;
                const bool mok = lora.is_lokr ? lm_export_lokr(lora, *meta, mdir, &mr, &merr)
                                              : lm_export_peft(lora, c, *meta, mdir, &mr, &merr);
                if (mok) {
                    LmMilestoneRec ms;
                    ms.loss  = lval;
                    ms.epoch = epoch + 1;
                    ms.path  = rel;
                    meta->milestones.push_back(ms);
                    jl("{\"type\":\"milestone\",\"loss\":%.4g,\"epoch\":%d,\"path\":\"%s\",\"bytes\":%lld}", lval,
                       epoch + 1, lm_json_escape(mdir).c_str(), mr.bytes);
                    lm_milestone_prune(a.out_dir, &meta->milestones, a.milestone_keep);
                } else {
                    lm_log("warn", "milestone snapshot failed: " + merr);
                }
                ladder -= (double) a.milestone_step;
            }
        }

        // §3.5.4 exports every epoch so a cancelled/crashed run leaves a usable
        // adapter — but the server reads lm_train_log.json out of <out> (§4.3),
        // so the log has to land alongside it, not only at the end of the export
        // stage. Written after the milestone block so it lists this epoch's
        // snapshots. `total_ms` is the stage elapsed here; the export stage
        // overwrites it with the whole-run figure.
        meta->vram_peak_mb = tracker.peak_mb;
        meta->total_ms     = (long long) (ggml_time_ms() - t_stage0);
        if (!lm_write_train_log(a.out_dir, *meta)) {
            lm_log("warn", "cannot write lm_train_log.json in " + a.out_dir);
        }

        const long long eta = (long long) ((double) ep_ms_sum / (double) (epoch + 1) * (double) (a.epochs - epoch - 1));
        jl("{\"type\":\"epoch\",\"epoch\":%d,\"epochs\":%d,\"loss\":%.6f,\"lr\":%.9g,\"gradNorm\":%.6f,\"ms\":%lld,"
           "\"etaMs\":%lld,\"best\":%s}",
           epoch + 1, a.epochs, avg, (double) last_stats.lr, (double) last_stats.grad_norm, ems, eta,
           best ? "true" : "false");
        jl("{\"type\":\"progress\",\"completed\":%d,\"total\":%d,\"phase\":\"train\"}", epoch + 1, a.epochs);
        fprintf(stderr, "[train-lm] epoch %d/%d loss=%.6f lr=%.3e gnorm=%.3f (%lld ms)\n", epoch + 1, a.epochs, avg,
                (double) last_stats.lr, (double) last_stats.grad_norm, ems);

        if (a.target_loss > 0.0f && avg <= (double) a.target_loss) {
            out->stopped_on_target  = true;
            meta->target_stop       = true;
            meta->target_stop_epoch = epoch + 1;
            meta->target_stop_loss  = avg;
            jl("{\"type\":\"target_stop\",\"epoch\":%d,\"loss\":%.6f,\"targetLoss\":%.6g}", epoch + 1, avg,
               (double) a.target_loss);
            break;
        }
    }

    meta->vram_peak_mb = tracker.peak_mb;
    out->samples       = n;
    out->skipped_long  = skipped_long;
    out->ms            = (long long) (ggml_time_ms() - t_stage0);

    char b[224];
    snprintf(b, sizeof(b), "leak counter: baseline %zu MB, peak %zu MB, max delta %lld MB over %d optimizer steps",
             tracker.base_mb, tracker.peak_mb, tracker.max_delta, global_step);
    fprintf(stderr, "[train-lm] %s\n", b);
    jl("{\"type\":\"leak\",\"baselineMb\":%lld,\"peakMb\":%lld,\"deltaMb\":%lld,\"steps\":%d}",
       (long long) tracker.base_mb, (long long) tracker.peak_mb, tracker.max_delta, global_step);

    if (rc == 0) {
        // savedEpoch/savedLoss/savedReason say which epoch's adapter is in the
        // run dir. finalLoss is the LAST epoch and is usually NOT it.
        fprintf(stderr, "[train-lm] adapter saved from epoch %d (loss %.4f, %s) of %d run\n",
                out->saved_epoch, out->saved_loss, out->saved_reason.c_str(), out->epochs_run);
        jl("{\"type\":\"stage\",\"stage\":\"train\",\"state\":\"end\",\"epochsRun\":%d,\"finalLoss\":%.6f,"
           "\"bestLoss\":%.6f,\"savedEpoch\":%d,\"savedLoss\":%.6f,\"savedReason\":\"%s\","
           "\"stoppedOnTarget\":%s,\"ms\":%lld}",
           out->epochs_run, out->final_loss, out->best_loss, out->saved_epoch, out->saved_loss,
           out->saved_reason.c_str(), out->stopped_on_target ? "true" : "false", out->ms);
    }

    // teardown (free the GPU before the export stage does its file writes)
    ggml_backend_sched_free(sched);
    lm_optim_free(&opt);
    lm_lora_detach(&lora, &lm);
    lm_lora_free(&lora);
    lm_ckpt_free(&ckpt);
    ggml_backend_buffer_free(buf_static);
    ggml_free(ctx_static);
    lm_mirror_free(&mirror);
    qw3lm_free(&lm);
    return rc;
}

// ─── main entry ─────────────────────────────────────────────────────────────

static int lm_train_main(const LmTrainArgs & a) {
    if (a.self_test) {
        return lm_self_test(a.lm_path, a.codes_path, (uint64_t) a.seed);
    }

    const int64_t t_run0 = ggml_time_ms();

    std::string stage_csv;
    for (size_t i = 0; i < a.stages.size(); i++) {
        if (i) {
            stage_csv += ",";
        }
        stage_csv += "\"" + a.stages[i] + "\"";
    }

    jl("{\"type\":\"start\",\"stages\":[%s],\"tensors\":\"%s\",\"codes\":\"%s\",\"out\":\"%s\",\"lm\":\"%s\","
       "\"lmSize\":\"%s\",\"rank\":%d,\"alpha\":%d,\"lr\":%.9g,\"epochs\":%d,\"gradAccum\":%d,\"targetLoss\":%.9g,"
       "\"gradClip\":%.9g,\"seed\":%d,\"lossOnCot\":%s,\"bwd\":\"%s\",\"captionDropout\":%.6g}",
       stage_csv.c_str(), lm_json_escape(a.tensors_dir).c_str(), lm_json_escape(a.codes_path).c_str(),
       lm_json_escape(a.out_dir).c_str(), lm_json_escape(a.lm_name).c_str(), a.lm_size.c_str(), a.rank, a.alpha,
       (double) a.lr, a.epochs, a.grad_accum, (double) a.target_loss, (double) a.grad_clip, a.seed,
       a.loss_on_cot ? "true" : "false", a.bwd.c_str(), (double) a.caption_dropout);
    if (a.reg_every > 0) {
        std::string csv;
        for (size_t i = 0; i < a.reg_codes.size(); i++) {
            if (i) {
                csv += ",";
            }
            csv += "\"" + lm_json_escape(a.reg_codes[i]) + "\"";
        }
        jl("{\"type\":\"prior_config\",\"regEvery\":%d,\"regSongs\":%d,\"regTopK\":%d,\"regPriorDir\":\"%s\","
           "\"regCodes\":[%s]}",
           a.reg_every, a.reg_songs, a.reg_topk,
           lm_json_escape(a.reg_prior_dir.empty() ? lm_join(a.out_dir, "prior") : a.reg_prior_dir).c_str(),
           csv.c_str());
    }

    // ── extract ──────────────────────────────────────────────────────────
    if (lm_has_stage(a, "extract")) {
        LmExtractOpts eo;
        eo.tensors_dir = a.tensors_dir;
        eo.codes_path  = a.codes_path;
        eo.dit_path    = a.dit_path;
        eo.overwrite   = a.overwrite;
        eo.limit       = a.limit;
        std::string err;
        if (!lm_extract_run(eo, &err)) {
            lm_fatal("model-load", err);
            return 1;
        }
    }

    LmExportMeta   meta;
    LmTrainOutcome out;
    meta.producer       = std::string("ace-train ") + ACE_VERSION;
    meta.created_at     = pm_iso8601_utc_now();
    meta.lm_path        = a.lm_path;
    meta.lm_size        = a.lm_size;
    meta.codes_path     = a.codes_path;
    meta.tensors_dir    = a.tensors_dir;
    meta.order          = a.order;
    meta.rank           = a.rank;
    meta.alpha          = a.alpha;
    meta.lr             = a.lr;
    meta.grad_clip      = a.grad_clip;
    meta.weight_decay   = a.weight_decay;
    meta.warmup_ratio   = a.warmup_ratio;
    meta.target_loss    = a.target_loss;
    meta.epochs         = a.epochs;
    meta.grad_accum     = a.grad_accum;
    meta.seed           = a.seed;
    meta.loss_on_cot    = a.loss_on_cot;
    meta.bwd            = a.bwd;
    // D5. The REQUESTED mode; lm_train_stage fills meta.attn_prec with what the
    // backend actually launched once an epoch has run.
    meta.attn_mode      = a.attn;
    meta.adapter_type   = a.adapter_type;
    meta.lokr_dim       = a.lokr_dim;
    meta.lokr_alpha     = a.lokr_alpha;
    meta.lokr_factor    = a.lokr_factor;
    meta.optimizer      = a.optimizer;
    meta.muon_lr_scale  = a.muon_lr_scale;
    meta.muon_ns_steps  = a.muon_ns_steps;
    meta.caption_dropout = a.caption_dropout;
    meta.reg_every       = a.reg_every;
    meta.reg_songs       = a.reg_songs;
    meta.reg_topk        = a.reg_topk;
    meta.reg_codes       = a.reg_codes;
    meta.reg_prior_dir   = a.reg_prior_dir.empty() ? lm_join(a.out_dir, "prior") : a.reg_prior_dir;

    // Trigger word (T5): CLI flags win, else the variant's preprocess_meta.json.
    // `--codes` always sits in the variant dir, so its parent is the fallback
    // when `--tensors` was not passed (the train-only stage does not need it).
    {
        meta.trigger          = a.trigger;
        meta.trigger_position = a.trigger_position;
        std::string vdir      = a.tensors_dir.empty() ? lm_dirname(a.codes_path) : a.tensors_dir;
        lm_resolve_trigger(vdir, &meta.trigger, &meta.trigger_position);
        if (!meta.trigger.empty()) {
            fprintf(stderr, "[train-lm] trigger \"%s\" (%s) will be embedded in the adapter\n",
                    meta.trigger.c_str(), meta.trigger_position.c_str());
        }
        // Lever C has nothing to drop TO without a trigger: an empty caption is
        // far out of distribution for this prompt format, and silently training
        // that instead would be a different experiment wearing this flag's name.
        // Refused here rather than at parse time because the trigger can come
        // from the variant's preprocess_meta.json, which is only read now.
        if (a.caption_dropout > 0.0f && meta.trigger.empty()) {
            fprintf(stderr,
                    "ace-train train-lm: --caption-dropout %.3g needs a trigger word, and none resolved.\n"
                    "  Pass --trigger <word>, or preprocess the dataset with a custom tag so the variant's\n"
                    "  preprocess_meta.json carries custom_tag/tag_position. (tag_position \"replace\" does\n"
                    "  not count — that tag was never applied to the captions.)\n",
                    (double) a.caption_dropout);
            return 2;
        }
    }

    // ── train ────────────────────────────────────────────────────────────
    if (lm_has_stage(a, "train")) {
        const int rc = lm_train_stage(a, &meta, &out);
        if (rc != 0) {
            return rc;
        }
    }

    // ── export ───────────────────────────────────────────────────────────
    if (lm_has_stage(a, "export")) {
        jl("{\"type\":\"stage\",\"stage\":\"export\",\"state\":\"begin\"}");
        const int64_t t_x0 = ggml_time_ms();
        if (!out.exported) {
            lm_log("warn", "nothing to export — the train stage did not run or produced no epoch");
            jl("{\"type\":\"stage\",\"stage\":\"export\",\"state\":\"end\",\"ms\":%lld}",
               (long long) (ggml_time_ms() - t_x0));
        } else {
            meta.total_ms = (long long) (ggml_time_ms() - t_run0);
            if (!lm_write_train_log(a.out_dir, meta)) {
                lm_fatal("export", "cannot write lm_train_log.json in " + a.out_dir);
                return 1;
            }
            long long bytes = 0;
            pm_stat_file(lm_join(a.out_dir, "adapter_model.safetensors"), &bytes, NULL);
            jl("{\"type\":\"export\",\"path\":\"%s\",\"tensors\":%d,\"bytes\":%lld,\"ms\":%lld}",
               lm_json_escape(a.out_dir).c_str(), out.export_tensors, bytes,
               (long long) (ggml_time_ms() - t_x0));
            jl("{\"type\":\"stage\",\"stage\":\"export\",\"state\":\"end\",\"ms\":%lld}",
               (long long) (ggml_time_ms() - t_x0));
        }
    }

    const long long run_ms = (long long) (ggml_time_ms() - t_run0);
    jl("{\"type\":\"done\",\"stages\":[%s],\"epochsRun\":%d,\"finalLoss\":%.6f,\"stoppedOnTarget\":%s,"
       "\"adapter\":\"%s\",\"samples\":%d,\"skippedLong\":%d,\"ms\":%lld}",
       stage_csv.c_str(), out.epochs_run, out.final_loss, out.stopped_on_target ? "true" : "false",
       lm_json_escape(a.out_dir).c_str(), out.samples, out.skipped_long, run_ms);

    fprintf(stderr, "[train-lm] done in %.1f s\n", (double) run_ms / 1000.0);
    return 0;
}
