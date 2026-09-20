#pragma once
// yue2/yue2-adapter.h — YuE2 NAR LoRA merge at load time.
//
// HOT-Step file (does not exist upstream, no acestep.cpp analog). Included
// only from inside engine/src/yue2/ (by yue2-model.h), per the fork's hook
// doctrine. It includes nothing from minimax/ — mm3-adapter.h is the model it
// was written from, not a dependency.
//
// SCOPE (phase 3 of docs/plans/yue2/08-nar-lora-trainer.md, designed off the
// survey in docs/plans/yue2/10-adapter-merge-notes.md): merge one or more YuE2
// NAR LoRAs into the LM GGUF's weights in the seam between
// yue2_load_lm_tensors() and wctx_alloc() — the same seam MM3 uses
// (mm3-model.h:1766) and ACE uses in dit.h. Patching the staged
// WeightCtx::PendingCopy sources there costs no extra VRAM and needs no
// second pass: wctx_alloc then uploads already-adapted weights.
//
// Deliberately NOT in scope: LoKr, DoRA, basin re-base, runtime (unmerged)
// deltas, and an importer for the upstream ComfyUI-YuE2-Trainer's fused-qkv
// layout (plan §4 calls that "a nice-to-have, not v1" — see "Fused qkv" below
// for what this file does when it meets one).
//
// Everything expensive is REUSED from adapter-merge.h: the safetensors reader
// (via safetensors.h), adapter_to_f32, alpha handling, and
// adapter_merge_on_backend()'s single-graph
// upload/dequant/BF16-round/add/re-encode/download pipeline. What is genuinely
// YuE2-specific, and nearly all this file adds, is the KEY MAPPING and the
// QUANTIZED-BASE GUARD.
//
// ── The on-disk format ─────────────────────────────────────────────────────
//
// Ours, and only ours: plain safetensors written by phase 4's exporter through
// train/st-write.h. Keys are the trainer's own site tags
// (yue2_nt_make_adapters, train/yue2-nar-train-graph.h:751-802) with a `yue2.`
// prefix and a LoRA factor suffix:
//
//   yue2.blk.7.nar_attn_q.lora_A.weight        A, PyTorch shape [rank, in]
//   yue2.blk.7.nar_attn_q.lora_B.weight        B, PyTorch shape [out, rank]
//   yue2.time_embd.1.lora_A.weight
//
// so the whole mapping is: strip a leading `yue2.`, strip the factor suffix,
// append `.weight`. One substr and one concatenation — markedly less work than
// mm3_lora_target, which has two upstream naming conventions to reconcile.
//
// Two spellings in plan §4 as written are WRONG against both the converter and
// the trainer, and this file is the place the disagreement gets settled (see
// 10-adapter-merge-notes.md §3 for the full derivation):
//
//   * `.lora_a` / `.lora_b` — no reader in this tree parses that. We accept it
//     anyway (yue2_lora_suffix below), because refusing an adapter over a
//     letter case is a stupid way to lose a training run, but the canonical
//     spelling the exporter should write is `.lora_A.weight` / `.lora_B.weight`.
//     Case matters on the TRAINER side for a different reason:
//     yue2_nt_init_adapters keys kaiming-vs-zero init on the tensor name's last
//     character (graph.h:872-895), so a lowercased round-trip would zero the
//     wrong factor.
//   * `time_embed.{0,2}` — the GGUF, the converter and the trainer all say
//     `time_embd.{0,1}`. `time_embed.2` is PyTorch's `time_embedder.mlp.2`,
//     which convert-yue2.py:507-512 renames to `time_embd.1`. We remap and warn
//     rather than refuse: the mapping is unambiguous, and the shape check below
//     would catch a wrong guess anyway ([256,2048] vs [2048,2048]).
//
// The merge targets are enumerated in yue2_lora_site_family. That allow-list is
// deliberate, not a leftover: `latent_pos_embed` is read by ggml_get_rows and
// must never carry a delta, and the norms/biases are 1-D. A key outside the
// list is counted and reported, never merged on the strength of "the tensor
// exists".
//
// ── Two families, and why they cannot be told apart by key alone ────────────
//
// There are now TWO trainable halves, so two families of site:
//
//   NAR  196 block sites (28 x `nar_attn_{q,k,v,output}`, `nar_ffn_{gate,up,down}`)
//        + 4 flat flow heads (`vae2llm`, `llm2vae`, `time_embd.{0,1}`) = 200
//   AR   196 block sites (28 x `attn_{q,k,v,output}`, `ffn_{gate,up,down}`)
//        and nothing flat — the flow heads have no part in the AR forward
//        (docs/plans/yue2/14-ar-lora-contract.md §2.2)
//
// AR and NAR are a Mixture of Transformers with ZERO weight sharing
// (yue2-model.h:155-157), so `blk.7.attn_q.weight` and `blk.7.nar_attn_q.weight`
// are disjoint tensors: an AR adapter and a NAR adapter stack cleanly into one
// model through the caller's multi-spec list, with independent scales, and
// neither touches the other's half.
//
// The cost of accepting both is that this file lost the guarantee its original
// comment named — "merging into `blk.N.attn_q` because an adapter happened to
// name it would quietly adapt the frozen half of the model". One mis-spelled
// key in a NAR export (`attn_q` for `nar_attn_q`) is now a legal site, and a
// silent landing on the other half is exactly the failure this loader exists to
// prevent. So the family is GATED ON `__metadata__.format` (contract §7.3):
//
//   format "yue2-nar-lora-v1"  -> only NAR block sites and the flat heads are
//                                 legal; an AR site is a REFUSAL, not a warning
//   format "yue2-ar-lora-v1"   -> only AR block sites are legal; a `nar_*` or a
//                                 flat site is a REFUSAL
//   format absent/unrecognised -> either family is accepted, but a file naming
//                                 BOTH is refused, with one key from each named
//
// A file spanning both halves is either a joint adapter nobody has designed or a
// corrupted export; refusing it is the same posture as two modules resolving to
// one tensor below.
//
// ── Fused qkv: YuE2 has none ───────────────────────────────────────────────
//
// `blk.N.nar_attn_{q,k,v}.weight` are three separate tensors (yue2-model.h:932-934;
// convert-yue2.py never fuses, and a grep for `qkv` over either file returns
// nothing), so every MM3QkvSlot mechanism — the slot enum, the sliced branch,
// the `out_feat % 3` check, the ggml_concat of three sub-deltas, the zero block
// standing in for an untrained projection — drops out. Each site is one A, one
// B, one base tensor, one adapter_merge_on_backend call. Note only that q and
// k/v have DIFFERENT output widths (2048 vs 1024 — GQA, 16 query heads over 8
// KV heads), so nothing here may assume a square weight.
//
// There is no SwiGLU-orientation question either: MM3's `ffn_in` emits both
// halves of a SwiGLU from one tensor, so a gate-first adapter merged
// value-first is silently wrong; YuE2's `nar_ffn_gate` and `nar_ffn_up` are
// separate tensors (yue2-model.h:939-940), so there are no halves to swap.
//
// The mirror-image problem is real but out of scope: the upstream
// ComfyUI-YuE2-Trainer exports ComfyUI-native keys with q/k/v fused
// block-diagonally. Converting one of those needs SLICING (the opposite
// direction from MM3's diffusers path). Until that converter exists, a fused
// key is REFUSED BY NAME here rather than merged as a wrong-shaped delta.
//
// ── Quantized bases: which ones can actually take a merge ──────────────────
//
// 196 of the 200 targets are block-quantized on a K-quant LM. The four flow
// heads never are, on any file — quantize.cpp's should_quantize() excludes
// them by literal name (:249-258, YUE2 POLICY at :47-56), so a `proj`-preset
// adapter merges cleanly into every LM we ship.
//
// adapter_merge_on_backend takes one of three paths per tensor type, and they
// are not equally safe:
//
//   BF16 / F16 / Q8_0  — GPU encode round-trip. Zero VRAM growth, zero host
//                        work, the same arithmetic PEFT does. The clean case,
//                        and the two packs that matter (yue2-full-precision,
//                        yue2-recommended). Merged silently.
//   K-quants, IQ3/IQ4  — GPU decode, HOST requant. adapter_requant calls
//                        ggml_quantize_chunk with imatrix = NULL
//                        (adapter-merge.h:437), so merging into a `-imat` file
//                        silently throws its importance matrix away on every
//                        tensor it touches. The quant ladder measured that gap
//                        directly (rel-L2 0.116 with imatrix vs 0.154 without,
//                        07-quant-ladder.md §9). It still works; we warn,
//                        because invisible is the problem, not fatal.
//   NVFP4 / MXFP4      — host decode, then PROMOTION TO BF16 rather than
//                        requant (adapter-merge.h:718-731). Works, roughly
//                        doubles the LM's VRAM. We warn.
//   IQ2_XXS/IQ2_XS/IQ1_S — ggml_quantize_chunk hard-asserts without an imatrix
//                        (ggml.c:8310-8313), which takes ace-server down and
//                        leaves the Node layer respawning it. We ship
//                        yue2-lm-IQ2_XS-imat. REFUSED, before anything is
//                        patched, with a message naming the type.
//
// The refusal is a PREFLIGHT over every resolved target, not a per-tensor
// skip, precisely so a refusal can never leave a half-merged model: nothing is
// patched until every target is known to exist in the base, to have factors
// that fit it, to have those factors actually present in the file
// (yue2a_extent_ok — safetensors.h bounds-checks nothing below the header),
// and to be of a type the merge can put back.
//
// An adapter should be auditioned against the same quant it will ship on. The
// merge is not quant-neutral.

#include "adapter-merge.h"
#include "hot-step-fsutf8.h"  // hs_stat / HS_STAT_T
#include "safetensors.h"
#include "timer.h"
#include "weight-ctx.h"
#include "weight-source.h"

#include <cctype>
#include <map>
#include <set>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

// How hard one adapter pushes, broken out by where in the model it lands.
//
// Mirrors MM3LmAdapterScales (minimax/mm3-lm-adapter.h) field for field, and
// for the same reason: one number for the whole file is too blunt a dial. An
// album adapter that carries the timbre in attention and the phrasing in the
// MLPs can be held at full attention strength and half MLP; a run that only
// went wrong in the late blocks can be pulled back there alone.
//
// Every field defaults to 1.0 and multiplies independently — `global` is the
// master, `attn`/`mlp` pick by module kind, and `early`/`mid`/`late` pick by
// which third of the block stack the module sits in. They are NOT derived from
// each other, so a caller that sets only `global` gets exactly today's uniform
// behaviour.
struct Yue2LmAdapterScales {
    float global = 1.0f;
    float attn   = 1.0f;
    float mlp    = 1.0f;
    float early  = 1.0f;
    float mid    = 1.0f;
    float late   = 1.0f;

    bool operator==(const Yue2LmAdapterScales & o) const {
        return global == o.global && attn == o.attn && mlp == o.mlp && early == o.early && mid == o.mid &&
               late == o.late;
    }
    bool operator!=(const Yue2LmAdapterScales & o) const { return !(*this == o); }
};

// One requested adapter. Lives here rather than in yue2-model.h so the server
// (which parses it off the wire) and the loader (which consumes it) share one
// definition without the server having to know how a merge works.
struct Yue2AdapterSpec {
    std::string         path;
    Yue2LmAdapterScales scales;
};

// Canonical rendering of a request, one entry per adapter, "; "-joined. Used as
// the CHANGE KEY by POST /yue2/select-model (a repeat selection must be a
// no-op, a different one must force a reload) and echoed in /yue2/props, so it
// has to be stable and total — same string in, same string out.
//
// EVERY dial is in the key, not just the master. The merge happens once at
// load, so a scale the key does not mention is a scale that silently fails to
// take effect until something else happens to evict the model. The Node
// backend mirrors this format byte for byte (services/backends/yue2/index.ts,
// yue2AdapterKey) to detect drift after an engine restart; the two have to be
// changed together or the drift check starts firing on every poll.
static std::string yue2_adapter_key(const std::vector<Yue2AdapterSpec> & specs) {
    std::string out;
    for (const Yue2AdapterSpec & s : specs) {
        if (!out.empty()) {
            out += "; ";
        }
        char buf[128];
        snprintf(buf, sizeof(buf), "%.4f,a%.4f,m%.4f,e%.4f,i%.4f,l%.4f", (double) s.scales.global,
                 (double) s.scales.attn, (double) s.scales.mlp, (double) s.scales.early, (double) s.scales.mid,
                 (double) s.scales.late);
        out += s.path + "@" + buf;
    }
    return out;
}

// ── Key parsing ─────────────────────────────────────────────────────────────

// Strip the LoRA factor suffix. Returns 'A', 'B', 'a' (a baked alpha scalar)
// or 0.
//
// The first entry of each list is the canonical spelling phase 4's exporter
// should write (and the one adapter-merge.h's own lora_is_a/lora_is_b already
// accept); the rest are tolerated so a PEFT-shaped file, or one written to
// plan §4's original lowercase spelling, still loads rather than reading as an
// adapter that matched nothing. No entry is a suffix of another, so the scan
// order does not change any answer.
static char yue2_lora_suffix(const std::string & key, std::string & module_out) {
    static const char * a_sfx[] = { ".lora_A.weight", ".lora_down.weight", ".lora.down.weight", ".lora_A",
                                    ".lora_a.weight", ".lora_a" };
    static const char * b_sfx[] = { ".lora_B.weight", ".lora_up.weight", ".lora.up.weight", ".lora_B",
                                    ".lora_b.weight", ".lora_b" };
    for (const char * s : a_sfx) {
        size_t n = strlen(s);
        if (key.size() > n && key.compare(key.size() - n, n, s) == 0) {
            module_out = key.substr(0, key.size() - n);
            return 'A';
        }
    }
    for (const char * s : b_sfx) {
        size_t n = strlen(s);
        if (key.size() > n && key.compare(key.size() - n, n, s) == 0) {
            module_out = key.substr(0, key.size() - n);
            return 'B';
        }
    }
    for (const char * s : { ".alpha", ".lora_alpha" }) {
        size_t n = strlen(s);
        if (key.size() > n && key.compare(key.size() - n, n, s) == 0) {
            module_out = key.substr(0, key.size() - n);
            return 'a';
        }
    }
    return 0;
}

static bool yue2a_starts(const std::string & s, const char * p) {
    size_t n = strlen(p);
    return s.size() >= n && s.compare(0, n, p) == 0;
}

// Which trainable half a site belongs to. YUE2_FAM_NONE is "not a site we
// merge"; YUE2_FAM_ANY is only ever a REQUEST (an adapter that declares no
// format), never the answer for a concrete module.
enum Yue2LoraFamily { YUE2_FAM_NONE = 0, YUE2_FAM_NAR, YUE2_FAM_AR, YUE2_FAM_ANY };

static const char * yue2_family_name(Yue2LoraFamily f) {
    switch (f) {
        case YUE2_FAM_NAR: return "nar";
        case YUE2_FAM_AR:  return "ar";
        case YUE2_FAM_ANY: return "any";
        default:           return "none";
    }
}

// Is this module one of the sites a YuE2 trainer can produce, and which half?
//
// The two lists are the trainers', tag for tag — NAR from
// yue2-nar-train-graph.h:780-803, AR from yue2-ar-train-graph.h's own site table
// (contract §2.1) — and they are the whole reason this file does not merge on
// the strength of "ws.exists says the tensor is there". Both halves' tensors
// exist in every LM GGUF, so the key alone cannot say which half an adapter
// MEANT; the format gate in yue2_adapter_merge_st is what answers that.
//
// The FLAT sites are NAR-only and stay that way: `vae2llm`, `llm2vae` and
// `time_embd.{0,1}` belong to the flow head and have no part in the AR forward,
// so an AR adapter that names one is malformed (contract §2.2).
// Which part of a block a site sits in, for the per-group scales. FLAT is the
// handful of NAR sites that are not in the block stack at all (vae2llm,
// llm2vae, time_embd.*) — they have neither an attention/MLP identity nor a
// block index, so they take the master scale and nothing else.
enum Yue2LoraGroup { YUE2_GRP_FLAT = 0, YUE2_GRP_ATTN, YUE2_GRP_FFN };

static Yue2LoraFamily yue2_lora_site_family(const std::string & mod,
                                            Yue2LoraGroup *     grp_out = nullptr,
                                            int *               blk_out = nullptr) {
    // Defaults describe a site that is not in the block stack; every early
    // return below leaves them alone, so a caller never reads a stale group.
    if (grp_out) {
        *grp_out = YUE2_GRP_FLAT;
    }
    if (blk_out) {
        *blk_out = -1;
    }
    if (mod == "vae2llm" || mod == "llm2vae" || mod == "time_embd.0" || mod == "time_embd.1") {
        return YUE2_FAM_NAR;
    }
    if (!yue2a_starts(mod, "blk.")) {
        return YUE2_FAM_NONE;
    }
    size_t d = 4, e = 4;
    while (e < mod.size() && isdigit((unsigned char) mod[e])) {
        e++;
    }
    if (e == d || e >= mod.size() || mod[e] != '.') {
        return YUE2_FAM_NONE;
    }
    // Block index is decimal and unpadded on BOTH sides (yue2_fmt("blk.%d..."),
    // model.h:932; "blk." + std::to_string(i), graph.h:756) — the kind of thing
    // that would otherwise bite at block 10.
    const int         blk  = atoi(mod.c_str() + d);
    const std::string tail = mod.substr(e + 1);
    if (tail == "nar_attn_q" || tail == "nar_attn_k" || tail == "nar_attn_v" || tail == "nar_attn_output" ||
        tail == "nar_ffn_gate" || tail == "nar_ffn_up" || tail == "nar_ffn_down") {
        if (grp_out) {
            *grp_out = yue2a_starts(tail, "nar_attn") ? YUE2_GRP_ATTN : YUE2_GRP_FFN;
        }
        if (blk_out) {
            *blk_out = blk;
        }
        return YUE2_FAM_NAR;
    }
    if (tail == "attn_q" || tail == "attn_k" || tail == "attn_v" || tail == "attn_output" ||
        tail == "ffn_gate" || tail == "ffn_up" || tail == "ffn_down") {
        if (grp_out) {
            *grp_out = yue2a_starts(tail, "attn") ? YUE2_GRP_ATTN : YUE2_GRP_FFN;
        }
        if (blk_out) {
            *blk_out = blk;
        }
        return YUE2_FAM_AR;
    }
    return YUE2_FAM_NONE;
}

// The per-tensor strength one adapter actually merges at.
//
// Same formula as MM3's MM3LmAdapter::effective (minimax/mm3-lm-adapter.h):
// master, times the module-kind dial, times the depth-band dial. The one
// deliberate difference is the band thresholds: MM3 hardcodes layer < 12 and
// layer < 24 because its layer count is a compile-time constant, while YuE2's
// block count comes off the GGUF (yue2.block_count, 28 today). Copying MM3's
// literals here would put the whole of a 28-block model's back third in "mid",
// so the thirds are computed from L instead.
//
// A flat site (block < 0) takes the master only: it is neither attention nor
// MLP and sits in no third, so applying either dial to it would be inventing a
// meaning the dial does not have.
static float yue2_adapter_effective(const Yue2LmAdapterScales & s, Yue2LoraGroup grp, int block, int n_blocks) {
    float v = s.global;
    if (block < 0 || grp == YUE2_GRP_FLAT) {
        return v;
    }
    v *= (grp == YUE2_GRP_ATTN) ? s.attn : s.mlp;
    if (n_blocks > 0) {
        v *= (block < n_blocks / 3) ? s.early : (block < (2 * n_blocks) / 3) ? s.mid : s.late;
    }
    return v;
}

// Why a key was not mapped. The caller reports these separately: an unknown
// site is noise worth one summary line, a fused qkv is a wrong-format file that
// deserves its own message.
enum Yue2LoraReject { YUE2_LR_OK = 0, YUE2_LR_UNKNOWN, YUE2_LR_FUSED_QKV, YUE2_LR_FOREIGN };

struct Yue2LoraTarget {
    std::string     gguf_name;              // empty => not merged
    Yue2LoraReject  why = YUE2_LR_UNKNOWN;
    Yue2LoraFamily  family = YUE2_FAM_NONE;  // meaningful only when gguf_name is set
    Yue2LoraGroup   group = YUE2_GRP_FLAT;   // ditto — which dial this site answers to
    int             block = -1;              // block index, -1 for a flat site
    bool            renamed_time_embed = false;  // came in as plan §4's `time_embed.`
};

// Map one adapter module path onto a GGUF tensor name.
static Yue2LoraTarget yue2_lora_target(const std::string & raw_module) {
    Yue2LoraTarget t;

    std::string m = raw_module;
    if (yue2a_starts(m, "yue2.")) {
        m = m.substr(5);
    }
    // PEFT can wrap modules; ACE and MM3 both hit this.
    for (const char * infix : { ".original_module.", ".base_layer." }) {
        size_t p = m.find(infix);
        if (p != std::string::npos) {
            m = m.substr(0, p) + "." + m.substr(p + strlen(infix));
        }
    }

    // A fused q/k/v key means a ComfyUI-native export, which needs a SLICING
    // converter we deliberately have not written. Merging it would take a
    // [H, 4096]-ish delta at a [H, 2048] target (or worse, pass the shape check
    // on some other site) — name it instead.
    if (m.find("to_qkv") != std::string::npos || m.find("attn_qkv") != std::string::npos ||
        m.find(".qkv") != std::string::npos) {
        t.why = YUE2_LR_FUSED_QKV;
        return t;
    }
    // Anything still carrying a foreign namespace is not ours. Diffusers/ComfyUI
    // prefixes are not stripped on purpose: an adapter wearing one was trained
    // against a different module tree and a "helpful" strip would merge it.
    for (const char * pfx : { "diffusion_model.", "model.diffusion_model.", "base_model.model.", "transformer.",
                              "lycoris_", "lora_unet_" }) {
        if (yue2a_starts(m, pfx)) {
            t.why = YUE2_LR_FOREIGN;
            return t;
        }
    }

    // Plan §4's `time_embed.{0,2}` — see the file header. Unambiguous remap,
    // and the shape check downstream is the backstop if it ever stops being so.
    if (yue2a_starts(m, "time_embed.")) {
        const std::string idx = m.substr(strlen("time_embed."));
        if (idx == "0") {
            m                     = "time_embd.0";
            t.renamed_time_embed  = true;
        } else if (idx == "2") {
            m                     = "time_embd.1";
            t.renamed_time_embed  = true;
        }
    }

    Yue2LoraGroup        grp = YUE2_GRP_FLAT;
    int                  blk = -1;
    const Yue2LoraFamily fam = yue2_lora_site_family(m, &grp, &blk);
    if (fam == YUE2_FAM_NONE) {
        t.why = YUE2_LR_UNKNOWN;
        return t;
    }
    t.gguf_name = m + ".weight";
    t.family    = fam;
    t.group     = grp;
    t.block     = blk;
    t.why       = YUE2_LR_OK;
    return t;
}

// ── Header metadata ─────────────────────────────────────────────────────────

// Plan §4 specifies `format`, `rank`, `alpha`, `targets`, `steps`,
// `clip_frames`, `trigger`, `base_sha` in the safetensors `__metadata__`.
// NOTHING in adapter-merge.h reads __metadata__ — its alpha convention is a
// baked per-module `.alpha` scalar, else `lora_alpha` in an adapter_config.json
// sidecar (adapter_read_alpha, :190). Rather than have our own exporter write a
// PEFT-shaped sidecar that would be fiction, we parse the header ourselves, the
// way mm3_read_swiglu_gate_first does (mm3-adapter.h:206): re-read the JSON
// header out of `st.mapping + 8`, length `st.data_offset - 8`.
//
// st-write.h writes every metadata value as a STRING (st_write_file takes
// pair<string,string>), so the numeric reads accept both spellings.
struct Yue2AdapterMeta {
    std::string format;
    std::string targets;
    std::string trigger;
    std::string base_sha;
    float       alpha = 0.0f;   // 0 => absent
    int64_t     rank  = 0;      // 0 => absent (informational only; rank always
                                //      comes from the tensor shapes)
    int         steps = 0;
};

static bool yue2a_meta_str(yyjson_val * meta, const char * key, std::string * out) {
    yyjson_val * v = yyjson_obj_get(meta, key);
    if (!v) {
        return false;
    }
    if (yyjson_is_str(v)) {
        *out = yyjson_get_str(v);
        return true;
    }
    return false;
}

static bool yue2a_meta_num(yyjson_val * meta, const char * key, double * out) {
    yyjson_val * v = yyjson_obj_get(meta, key);
    if (!v) {
        return false;
    }
    if (yyjson_is_real(v)) {
        *out = yyjson_get_real(v);
        return true;
    }
    if (yyjson_is_int(v)) {
        *out = (double) yyjson_get_sint(v);
        return true;
    }
    if (yyjson_is_str(v)) {
        *out = atof(yyjson_get_str(v));
        return true;
    }
    return false;
}

static Yue2AdapterMeta yue2_adapter_read_meta(const STFile & st) {
    Yue2AdapterMeta md;
    if (st.data_offset <= 8) {
        return md;
    }
    yyjson_doc * doc = yyjson_read((const char *) st.mapping + 8, st.data_offset - 8, 0);
    if (!doc) {
        return md;
    }
    yyjson_val * meta = yyjson_obj_get(yyjson_doc_get_root(doc), "__metadata__");
    if (meta && yyjson_is_obj(meta)) {
        yue2a_meta_str(meta, "format", &md.format);
        yue2a_meta_str(meta, "targets", &md.targets);
        yue2a_meta_str(meta, "trigger", &md.trigger);
        yue2a_meta_str(meta, "base_sha", &md.base_sha);
        double d = 0.0;
        if (yue2a_meta_num(meta, "alpha", &d)) {
            md.alpha = (float) d;
        }
        if (yue2a_meta_num(meta, "rank", &d)) {
            md.rank = (int64_t) d;
        }
        if (yue2a_meta_num(meta, "steps", &d)) {
            md.steps = (int) d;
        }
    }
    yyjson_doc_free(doc);
    return md;
}

// ── Quantized-base guard ────────────────────────────────────────────────────

// Mirrors adapter_merge_on_backend's own decode probe (adapter-merge.h:641-649)
// so the preflight answers the same question the merge will: can the backend
// ggml_cast this native type to F32? A `false` here means the whole merge for
// that tensor runs on the host and ends with a BF16 promotion.
static bool yue2_adapter_can_decode(ggml_backend_t backend, enum ggml_type type) {
    if (type == GGML_TYPE_F32 || type == GGML_TYPE_BF16 || type == GGML_TYPE_F16) {
        return true;
    }
    size_t                  meta   = ggml_tensor_overhead() * 4 + 1024;
    struct ggml_init_params params = { meta, NULL, true };
    struct ggml_context *   ctx    = ggml_init(params);
    struct ggml_tensor *    src    = ggml_new_tensor_1d(ctx, type, 64);
    struct ggml_tensor *    dst    = ggml_cast(ctx, src, GGML_TYPE_F32);
    const bool              ok     = ggml_backend_supports_op(backend, dst);
    ggml_free(ctx);
    return ok;
}

// ── The merge ───────────────────────────────────────────────────────────────

// Does this entry's declared byte range actually lie inside the mapping, and
// does it hold exactly the elements its shape claims?
//
// Nothing else asks. st_open validates one thing — that the JSON header fits
// in the file (safetensors.h:277-283) — and st_parse stores each entry's
// `data_offsets` verbatim, never comparing them against the shape or the end
// of the mapping (:198-215). st_data then hands back
// `mapping + data_offset + data_start` whatever that is.
//
// Which makes a TRUNCATED EXPORT — the exact failure the orphan-A check above
// says it refuses — read past the end of the mapping instead. st_write_file
// computes every offset and serialises the complete JSON header before it
// streams a single byte of tensor data (train/st-write.h: offsets :146, header
// :160, data last at :218), so an interrupted write leaves an intact header
// naming every A/B pair: the orphan check cannot fire, and the shape check
// passes because the shapes it reads are the header's, not the file's — the
// bytes those offsets point at are simply not there. The convert in the merge
// loop would then
// fault on a pointer past the mapping — and per the IQ2 reasoning above, a
// SIGSEGV here is not an error message, it is ace-server dying and the Node
// layer respawning it in a loop.
//
// Checked in the preflight, with everything else, so the answer is a refusal
// and not a half-merged model.
static bool yue2a_extent_ok(const STFile & st, const STEntry & e, int64_t nelem, size_t elem_size) {
    if (nelem <= 0 || e.data_end <= e.data_start || st.data_offset > st.file_size) {
        return false;
    }
    if (e.data_end > st.file_size - st.data_offset) {
        return false;
    }
    return (e.data_end - e.data_start) == (size_t) nelem * elem_size;
}

// Bytes per element for the three dtypes adapter_to_f32 accepts. Only call it
// once the dtype has been checked against that list.
static size_t yue2a_elem_size(const std::string & dtype) {
    return dtype == "F32" ? 4u : 2u;
}

// One (A, B, alpha) triple bound for one GGUF tensor. No slot field: YuE2 has
// no fused projection, so a target never collects more than one triple.
struct Yue2LoraFactor {
    const STEntry * a     = nullptr;
    const STEntry * b     = nullptr;
    float           alpha = 0.0f;  // 0 => scaling 1.0
    // Carried from the target so the merge loop can price each tensor without
    // re-parsing its name: the loop iterates the gguf-name map, and the module
    // path that classified it is long gone by then.
    Yue2LoraGroup   group = YUE2_GRP_FLAT;
    int             block = -1;
};

// Merge every LoRA tensor in `st` into `wctx`, which must be the YuE2 LM
// WeightCtx after yue2_load_lm_tensors() has staged it and BEFORE wctx_alloc().
// `scales` is the user-facing strength, priced per tensor by
// yue2_adapter_effective (master x module kind x depth third).
//
// Returns the number of GGUF tensors patched, or -1 on a refusal (which writes
// `err_out` and patches NOTHING). Zero means the adapter matched nothing; the
// caller decides whether that is fatal. `family_out`, when non-null, is set to
// the half this file actually merged into ("ar" or "nar") — /yue2/props reports
// it next to the count so a user who loaded the wrong file can see it without
// reading stderr.
static int yue2_adapter_merge_st(WeightCtx *                 wctx,
                                 const GGUFModel &           gf,
                                 const STFile &              st,
                                 const std::string &         cfg_dir,
                                 const Yue2LmAdapterScales & scales,
                                 ggml_backend_t              backend,
                                 std::string *               err_out,
                                 std::string *               family_out = nullptr) {
    WeightSource ws = {};
    ws.is_st        = false;
    ws.gf           = const_cast<GGUFModel *>(&gf);

    const Yue2AdapterMeta md = yue2_adapter_read_meta(st);

    // `format` is a guard, not decoration, and since the AR family landed it
    // does TWO jobs.
    //
    // The old one: an MM3 or ACE adapter pointed at this path would otherwise
    // fail only as "0 tensors matched", which reads like an empty adapter
    // rather than a wrong one.
    //
    // The new one: it decides which half is legal. Both halves' sites are now
    // accepted by yue2_lora_site_family, so `blk.7.attn_q` in a NAR export —
    // one missing `nar_` — would land on the frozen AR half and change the
    // audio in a way nothing reports. The format says which family the trainer
    // MEANT, and a key from the other one is an error (contract §7.3).
    Yue2LoraFamily want_fam = YUE2_FAM_ANY;
    if (md.format.empty()) {
        fprintf(stderr, "[YuE2-Adapter] NOTE: no `format` in __metadata__ — merging on the strength of the "
                        "key names alone, and refusing the file if it names both halves\n");
    } else if (yue2a_starts(md.format, "yue2-nar-lora")) {
        want_fam = YUE2_FAM_NAR;
    } else if (yue2a_starts(md.format, "yue2-ar-lora")) {
        want_fam = YUE2_FAM_AR;
    } else if (!yue2a_starts(md.format, "yue2-")) {
        // Not ours at all. Refuse by name rather than by an empty merge.
        if (err_out) {
            *err_out = "adapter __metadata__ says format=\"" + md.format +
                       "\"; this loader merges yue2-nar-lora-v1 / yue2-ar-lora-v1 files only";
        }
        return -1;
    } else {
        // A `yue2-` format this build does not know (a later revision, say).
        // Treated as unlabelled: either family, never both.
        fprintf(stderr,
                "[YuE2-Adapter] NOTE: unrecognised format=\"%s\" — merging on the key names alone, and "
                "refusing the file if it names both halves\n",
                md.format.c_str());
    }
    if (!md.base_sha.empty()) {
        // Nothing here knows the resident LM's sha, so this is a breadcrumb for
        // a human reading the log after an adapter sounds wrong, not a check.
        fprintf(stderr, "[YuE2-Adapter] trained against base_sha %s\n", md.base_sha.c_str());
    }

    const int cfg_alpha = adapter_read_alpha(cfg_dir.c_str());

    // Pass 1: baked per-module .alpha scalars (highest-precedence alpha source,
    // per adapter_merge_lora's own convention at adapter-merge.h:886-890).
    std::map<std::string, float> alpha_by_module;
    for (const auto & e : st.entries) {
        std::string mod;
        if (yue2_lora_suffix(e.name, mod) == 'a' && e.dtype == "F32" && e.n_dims == 0) {
            // Extent-checked like everything else, because this read happens
            // BEFORE the preflight and four bytes off the end of the mapping
            // fault exactly as hard as four megabytes. Dropping a truncated
            // alpha is safe rather than silent: the same truncation lands on
            // the (far larger) A/B factors, which the preflight then refuses.
            if (!yue2a_extent_ok(st, e, 1, sizeof(float))) {
                continue;
            }
            float v = 0.0f;
            memcpy(&v, st_data(st, e), sizeof(float));
            alpha_by_module[mod] = v;
        }
    }

    // Pass 2: the A/B factors.
    std::map<std::string, const STEntry *> a_by_module, b_by_module;
    for (const auto & e : st.entries) {
        std::string mod;
        const char  which = yue2_lora_suffix(e.name, mod);
        if (which == 'A') {
            a_by_module[mod] = &e;
        } else if (which == 'B') {
            b_by_module[mod] = &e;
        }
    }

    std::map<std::string, Yue2LoraFactor> targets;  // gguf name -> factors
    int                                   n_unknown = 0, n_fused = 0, n_foreign = 0, n_time_embed = 0;
    std::string                           orphan;
    // First key seen from each half, for the mixed-file refusal below. Kept as
    // names rather than counts because the message has to be actionable: "this
    // file names both halves" without saying which keys is a bug report nobody
    // can act on.
    std::string                           first_ar, first_nar;

    for (const auto & kv : a_by_module) {
        const std::string & mod = kv.first;
        auto                bit = b_by_module.find(mod);
        if (bit == b_by_module.end()) {
            // A LoRA factor can only ever be exported in pairs, so a lone A
            // means a truncated or interrupted write. Refusing beats merging
            // the 199 modules that did survive.
            if (orphan.empty()) {
                orphan = mod;
            }
            continue;
        }
        const Yue2LoraTarget t = yue2_lora_target(mod);
        if (t.gguf_name.empty()) {
            switch (t.why) {
                case YUE2_LR_FUSED_QKV: n_fused++;   break;
                case YUE2_LR_FOREIGN:   n_foreign++; break;
                default:                n_unknown++; break;
            }
            continue;
        }
        if (t.renamed_time_embed) {
            n_time_embed++;
        }
        // The family gate. A declared format makes the other half an ERROR, not
        // an ignored key: the whole hazard is that the wrong half merges
        // cleanly, renders audio and tells nobody.
        if (want_fam != YUE2_FAM_ANY && t.family != want_fam) {
            if (err_out) {
                *err_out = std::string("adapter __metadata__ says format=\"") + md.format + "\" (the " +
                           yue2_family_name(want_fam) + " half), but module \"" + mod + "\" belongs to the " +
                           yue2_family_name(t.family) + " half (it targets " + t.gguf_name +
                           "). Merging it would adapt the half the trainer never touched, which is the exact "
                           "failure this check exists to prevent — refusing the whole file";
            }
            return -1;
        }
        if (t.family == YUE2_FAM_AR && first_ar.empty()) {
            first_ar = mod;
        } else if (t.family == YUE2_FAM_NAR && first_nar.empty()) {
            first_nar = mod;
        }
        if (targets.count(t.gguf_name)) {
            // Two adapter modules resolving to one tensor can only mean a
            // malformed file (e.g. both `time_embed.2` and `time_embd.1`).
            // Which one wins would be alphabetical accident, so refuse.
            if (err_out) {
                *err_out = "two adapter modules target " + t.gguf_name + " (one of them is \"" + mod +
                           "\") — ambiguous, refusing rather than picking one";
            }
            return -1;
        }
        Yue2LoraFactor f;
        f.a      = kv.second;
        f.b      = bit->second;
        f.group  = t.group;
        f.block  = t.block;
        auto ait = alpha_by_module.find(mod);
        f.alpha  = (ait != alpha_by_module.end()) ? ait->second
                 : (md.alpha > 0.0f)              ? md.alpha
                                                  : (float) cfg_alpha;
        targets[t.gguf_name] = f;
    }

    if (!orphan.empty()) {
        if (err_out) {
            *err_out = "adapter module \"" + orphan +
                       "\" has an A factor with no matching B (truncated or half-written export)";
        }
        return -1;
    }
    if (n_fused) {
        if (err_out) {
            *err_out = "this adapter has fused q/k/v keys (ComfyUI-YuE2-Trainer layout). YuE2's GGUF keeps "
                       "nar_attn_q/k/v as three separate tensors, so merging it would apply a wrong-shaped "
                       "delta. A slicing converter is not built (plan 08 §4); retrain or convert first";
        }
        return -1;
    }
    if (n_foreign) {
        if (err_out) {
            *err_out = "this adapter's keys carry a foreign namespace (diffusers/ComfyUI/LyCORIS) — it was "
                       "not trained against the YuE2 module tree";
        }
        return -1;
    }
    // An unlabelled file naming both halves. Nothing in this tree trains both at
    // once, so it is a corrupted export or a joint adapter nobody has designed;
    // either way, guessing which half the author meant is not this loader's job.
    if (!first_ar.empty() && !first_nar.empty()) {
        if (err_out) {
            *err_out = "this adapter names BOTH halves of the model (\"" + first_ar + "\" is an AR site, \"" +
                       first_nar +
                       "\" is a NAR site) and carries no __metadata__ format saying which it meant. Nothing "
                       "trains both halves at once, so this is a corrupted export — refusing rather than "
                       "adapting a half the trainer never touched";
        }
        return -1;
    }
    if (n_time_embed) {
        fprintf(stderr, "[YuE2-Adapter] WARNING: %d key(s) spelled `time_embed.{0,2}` (plan 08 §4's "
                        "uncorrected spelling) remapped to `time_embd.{0,1}` — fix the exporter\n",
                n_time_embed);
    }
    if (n_unknown) {
        fprintf(stderr, "[YuE2-Adapter] %d adapter module(s) name sites we do not merge (norms, biases, "
                        "latent_pos_embed) — ignored\n", n_unknown);
    }
    if (targets.empty()) {
        return 0;
    }
    // Resolved, not declared: this is the half the merge is about to touch, and
    // it is what /yue2/props reports next to the tensor count.
    const Yue2LoraFamily fam = first_ar.empty() ? YUE2_FAM_NAR : YUE2_FAM_AR;
    if (family_out) {
        *family_out = yue2_family_name(fam);
    }

    // ── Preflight: shapes, dtypes, extents, and the quantized-base guard ─────
    //
    // One pass over every resolved target BEFORE anything is patched, checking
    // every way a merge can go wrong. Doing it up front is the whole design: a
    // per-tensor skip mid-merge leaves an LM that is adapted in some layers and
    // pristine in others, which renders audio, sounds nearly right, and tells
    // nobody — the "wrong but quiet" outcome this path exists to avoid. And on
    // the IQ2 types it is not even a skip: it is a GGML_ASSERT inside
    // ggml_quantize_chunk that takes the process down (ggml.c:8310-8313,
    // reached from adapter_requant's imatrix = NULL at adapter-merge.h:437),
    // after which the Node layer respawns ace-server in a loop.
    //
    // NOTHING here is a per-tensor skip, a target the base does not have
    // included. An earlier draft warned and dropped that one, reasoning that a
    // `proj`-preset adapter on some future LM that had dropped a flow head
    // should still merge what it can. But there is exactly one YuE2
    // architecture, so the hypothetical buys nothing, while the live cost is
    // real: an adapter trained against a different block count (yue2_lora_site_ok
    // accepts ANY decimal block index) would merge blocks 0..27, drop the rest
    // into stderr, and hand back a number nothing downstream can compare
    // against anything — yue2_apply_adapters only treats n == 0 as fatal, and
    // /yue2/props reports `adapter.tensors` as a bare count. Refuse instead.
    {
        std::set<int> seen_type;
        for (const auto & tgt : targets) {
            const std::string &    name = tgt.first;
            const Yue2LoraFactor & f    = tgt.second;
            if (!ws.exists(name.c_str())) {
                if (err_out) {
                    *err_out = name +
                               ": the base model has no such tensor — this adapter was trained against a "
                               "different YuE2 LM (a different block count, or a different module tree)";
                }
                return -1;
            }
            enum ggml_type ttype  = GGML_TYPE_F32;
            int            n_dims = 0;
            int64_t        ne[4]  = { 1, 1, 1, 1 };
            ws.data(name.c_str(), ttype);
            ws.shape(name.c_str(), n_dims, ne);

            // safetensors shapes are PyTorch order: A is [rank, in], B is
            // [out, rank]. Compared against the ACTUAL base shape, never
            // against an assumed square — nar_attn_q is [2048, 2048] but
            // nar_attn_k/v are [2048, 1024] (GQA, 16 query heads over 8 KV
            // heads).
            if (f.a->n_dims != 2 || f.b->n_dims != 2 || f.a->shape[0] <= 0 || f.a->shape[1] != ne[0] ||
                f.b->shape[0] != ne[1] || f.b->shape[1] != f.a->shape[0]) {
                char buf[320];
                snprintf(buf, sizeof(buf),
                         "%s: adapter factors do not fit the base tensor (A [%lld,%lld] %d-D, B [%lld,%lld] "
                         "%d-D vs base in=%lld out=%lld)",
                         name.c_str(), (long long) f.a->shape[0], (long long) f.a->shape[1], f.a->n_dims,
                         (long long) f.b->shape[0], (long long) f.b->shape[1], f.b->n_dims, (long long) ne[0],
                         (long long) ne[1]);
                if (err_out) {
                    *err_out = buf;
                }
                return -1;
            }
            // adapter_to_f32 handles exactly these three (adapter-merge.h:62-74),
            // and st-write.h can only emit these three (STWDType, :133-144).
            if ((f.a->dtype != "F32" && f.a->dtype != "BF16" && f.a->dtype != "F16") ||
                (f.b->dtype != "F32" && f.b->dtype != "BF16" && f.b->dtype != "F16")) {
                if (err_out) {
                    *err_out = name + ": unsupported factor dtype (A=" + f.a->dtype + " B=" + f.b->dtype +
                               "), expected F32/BF16/F16";
                }
                return -1;
            }
            // Shapes agreeing with the base says nothing about whether the
            // bytes are there — see yue2a_extent_ok. Both factors, now that
            // their dtypes (and so their element sizes) are known.
            const int64_t rank_a = f.a->shape[0];
            if (!yue2a_extent_ok(st, *f.a, rank_a * ne[0], yue2a_elem_size(f.a->dtype)) ||
                !yue2a_extent_ok(st, *f.b, ne[1] * rank_a, yue2a_elem_size(f.b->dtype))) {
                char buf[384];
                snprintf(buf, sizeof(buf),
                         "%s: adapter factor data is outside the file (A %zu..%zu, B %zu..%zu, data section is "
                         "%zu bytes) — a truncated or interrupted export",
                         name.c_str(), f.a->data_start, f.a->data_end, f.b->data_start, f.b->data_end,
                         st.file_size - st.data_offset);
                if (err_out) {
                    *err_out = buf;
                }
                return -1;
            }

            if (ggml_quantize_requires_imatrix(ttype)) {
                if (err_out) {
                    *err_out = std::string("cannot merge an adapter into a ") + ggml_type_name(ttype) +
                               " LM: re-quantizing a merged tensor to this type needs an importance matrix "
                               "that the merge path does not have, and ggml_quantize_chunk aborts the process "
                               "rather than returning. Use the bf16 or Q8_0 LM (yue2-full-precision / "
                               "yue2-recommended) for adapter work, or a K-quant if you accept the "
                               "imatrix loss";
                }
                return -1;
            }
            if (seen_type.insert((int) ttype).second) {
                if (!yue2_adapter_can_decode(backend, ttype)) {
                    fprintf(stderr,
                            "[YuE2-Adapter] NOTE: %s tensors have no backend dequant cast — the merge runs on "
                            "the host and PROMOTES them to BF16 (adapter-merge.h:718-731). The LM's VRAM "
                            "footprint will grow to roughly 2x native.\n",
                            ggml_type_name(ttype));
                } else if (ggml_is_quantized(ttype) && !adapter_backend_can_encode(backend, ttype)) {
                    fprintf(stderr,
                            "[YuE2-Adapter] WARNING: %s tensors are re-quantized on the host WITHOUT an "
                            "importance matrix. On an `-imat` LM the merged tensors come back out as plain "
                            "%s, so the merge partially undoes the imatrix (07-quant-ladder.md §9 measured "
                            "rel-L2 0.116 with vs 0.154 without). Audition the adapter on the quant you will "
                            "ship it on.\n",
                            ggml_type_name(ttype), ggml_type_name(ttype));
                }
            }
        }
    }

    // pending lookups: by src pointer for adapter_merge_on_backend, and by
    // tensor NAME so a second adapter in the same load stacks on the first
    // one's merged bytes rather than on the pristine GGUF bytes (ACE does the
    // same in dit.h, MM3 at mm3-adapter.h:322-331). Both are rebuilt per
    // adapter file, which is what keeps the pointer keys valid after a prior
    // merge swapped a PendingCopy's src to a staging buffer.
    std::unordered_map<const void *, size_t> pending_idx;
    std::unordered_map<std::string, size_t>  pending_by_name;
    pending_idx.reserve(wctx->pending.size());
    for (size_t i = 0; i < wctx->pending.size(); i++) {
        pending_idx[wctx->pending[i].src] = i;
        if (wctx->pending[i].tensor) {
            pending_by_name[ggml_get_name(wctx->pending[i].tensor)] = i;
        }
    }

    int        merged         = 0;
    const bool verify         = getenv("YUE2_ADAPTER_VERIFY") != nullptr;
    int        verify_changed = 0;
    bool       warned_alpha   = false;

    // Block count for the depth thirds, read off the same GGUF the merge is
    // patching rather than assumed: an adapter and a base that disagree about
    // the block count is already refused above (the "no such tensor" check), so
    // by here this is the one true L. 0 would mean a base without the KV at
    // all, and yue2_adapter_effective treats that as "no depth banding" rather
    // than dividing by zero.
    const int n_blocks = (int) gf_get_u32(gf, "yue2.block_count");

    for (const auto & kv : targets) {
        const std::string &    gguf_name = kv.first;
        const Yue2LoraFactor & f         = kv.second;

        // Per-tensor strength. A dial set to 0 means "do not adapt this part of
        // the model", and skipping is not just an optimisation: a merge of a
        // zero-scaled delta still round-trips the base weight through the
        // backend and back into a quantised type, which is a lossy no-op.
        const float eff = yue2_adapter_effective(scales, f.group, f.block, n_blocks);
        if (eff == 0.0f) {
            continue;
        }

        enum ggml_type ttype    = GGML_TYPE_F32;
        const void *   base_ptr = ws.data(gguf_name.c_str(), ttype);
        int            n_dims   = 0;
        int64_t        ne[4]    = { 1, 1, 1, 1 };
        ws.shape(gguf_name.c_str(), n_dims, ne);

        // Stack onto the running merged value when a prior adapter in this same
        // load already patched this tensor.
        auto pn = pending_by_name.find(gguf_name);
        if (pn != pending_by_name.end()) {
            base_ptr = wctx->pending[pn->second].src;
            if (wctx->pending[pn->second].tensor) {
                ttype = wctx->pending[pn->second].tensor->type;
            }
        }

        const int64_t in_feat  = ne[0];
        const int64_t out_feat = ne[1];

        // Shapes, dtypes and byte extents were all validated in the preflight
        // above, which is why nothing in this loop can bail out half way and
        // leave a partly adapted LM — and why the reads below are in-bounds.
        // A is [rank, in] and B is [out, rank] in PyTorch order.
        const int64_t rank = f.a->shape[0];

        std::vector<float> av((size_t) (rank * in_feat));
        std::vector<float> bv((size_t) (out_feat * rank));
        if (!adapter_to_f32(st_data(st, *f.a), av.data(), rank * in_feat, f.a->dtype) ||
            !adapter_to_f32(st_data(st, *f.b), bv.data(), out_feat * rank, f.b->dtype)) {
            if (err_out) {
                *err_out = gguf_name + ": factor conversion failed (A=" + f.a->dtype + " B=" + f.b->dtype + ")";
            }
            return -1;
        }

        // scaling = alpha / rank, rank ALWAYS from the tensor shapes and never
        // from metadata (adapter_read_alpha's own note, adapter-merge.h:188-189).
        // No alpha anywhere means scaling 1.0, which is what alpha == rank gives
        // — the plan's own starting point (rank 16, alpha 16) — but say so once
        // rather than let a missing alpha look like a deliberate 1.0.
        const float scaling = (f.alpha > 0.0f) ? (f.alpha / (float) rank) : 1.0f;
        if (f.alpha <= 0.0f && !warned_alpha) {
            warned_alpha = true;
            fprintf(stderr, "[YuE2-Adapter] NOTE: no alpha found (no baked .alpha scalar, no __metadata__ "
                            "alpha, no adapter_config.json) — using scaling 1.0, i.e. alpha == rank\n");
        }

        auto build = [&](struct ggml_context * ctx) {
            // The (tensor, data) pairs are captured EXPLICITLY rather than
            // re-derived at upload time. A is [rank, in] and B is [out, rank],
            // so on a SQUARE weight — nar_attn_q and nar_attn_output are both
            // [2048, 2048] — they have identical element counts and any
            // size-based disambiguation silently uploads A into B. That bug
            // compiled clean in MM3 and surfaced only as a zero-delta adapter
            // that still changed the audio (mm3-adapter.h:446-457). It is why
            // YUE2_ADAPTER_VERIFY exists below.
            struct ggml_tensor * ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_feat, rank);
            struct ggml_tensor * tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, out_feat);
            // PEFT rounds A and B through BF16 before the GEMM; match it on the
            // backend so our merge equals theirs bit for bit.
            struct ggml_tensor * ta_br = ggml_cast(ctx, ggml_cast(ctx, ta, GGML_TYPE_BF16), GGML_TYPE_F32);
            struct ggml_tensor * tb_br = ggml_cast(ctx, ggml_cast(ctx, tb, GGML_TYPE_BF16), GGML_TYPE_F32);
            struct ggml_tensor * ta_t  = ggml_cont(ctx, ggml_transpose(ctx, ta_br));

            adapter_delta_build db;
            db.tdelta = ggml_scale(ctx, ggml_mul_mat(ctx, ta_t, tb_br), scaling);
            db.upload = [ta, tb, &av, &bv]() {
                GGML_ASSERT(ggml_nelements(ta) == (int64_t) av.size());
                GGML_ASSERT(ggml_nelements(tb) == (int64_t) bv.size());
                ggml_backend_tensor_set(ta, av.data(), 0, av.size() * sizeof(float));
                ggml_backend_tensor_set(tb, bv.data(), 0, bv.size() * sizeof(float));
            };
            return db;
        };

        // promote_f32 = false, for the same reason MM3 gives (mm3-adapter.h:508-512):
        // the shipped bases are BF16/Q8_0, CUDA can encode F32 -> both, so the
        // merged weight goes back in its native type and the LM's VRAM is
        // unchanged. Promoting the NAR half to F32 would cost several GB.
        const size_t base_nb = ggml_row_size(ttype, in_feat) * (size_t) out_feat;
        if (!adapter_merge_on_backend(wctx, pending_idx, base_ptr, ttype, in_feat, out_feat,
                                      /*ds=*/nullptr, eff, backend, gguf_name.c_str(), build,
                                      /*promote_f32=*/false)) {
            // Fatal, not a skip: by this point some tensors are already
            // patched, so continuing would produce a partly adapted LM. The
            // caller throws the whole staged WeightCtx away (yue2_load_parts's
            // all-or-nothing contract), which is the only safe answer.
            if (err_out) {
                *err_out = gguf_name + ": backend merge failed after " + std::to_string(merged) +
                           " tensor(s) — see the [Adapter] line above";
            }
            return -1;
        }
        merged++;

        // YUE2_ADAPTER_VERIFY=1: byte-compare each merged result against the
        // base it was built from. EXPECT A NON-ZERO COUNT even when everything
        // is right, and do not "fix" it: `base + 0` normalises IEEE -0.0 to
        // +0.0, flipping exactly one byte per negative zero. MM3 measured the
        // differing-byte count equal to the -0.0 count tensor for tensor, with
        // bit-identical audio. The real gate is the audio (plan 08 §6 phase 3:
        // a zero adapter renders identically, a trained one does not); this
        // counter only localises a failure once the audio has moved.
        if (verify) {
            auto vp = pending_idx.find(base_ptr);
            if (vp != pending_idx.end()) {
                const WeightCtx::PendingCopy & pc = wctx->pending[vp->second];
                if (pc.nbytes != base_nb) {
                    fprintf(stderr, "[YuE2-Adapter] VERIFY %s: size changed %zu -> %zu (type promoted)\n",
                            gguf_name.c_str(), base_nb, pc.nbytes);
                    verify_changed++;
                } else if (memcmp(pc.src, base_ptr, base_nb) != 0) {
                    size_t ndiff = 0;
                    for (size_t bi = 0; bi < base_nb; bi++) {
                        if (((const uint8_t *) pc.src)[bi] != ((const uint8_t *) base_ptr)[bi]) {
                            ndiff++;
                        }
                    }
                    if (verify_changed < 3) {
                        fprintf(stderr, "[YuE2-Adapter] VERIFY %s: %zu/%zu bytes differ\n", gguf_name.c_str(),
                                ndiff, base_nb);
                    }
                    verify_changed++;
                }
            }
        }
    }

    if (verify) {
        fprintf(stderr, "[YuE2-Adapter] VERIFY: %d of %d merged tensor(s) differ from their base\n",
                verify_changed, merged);
    }

    return merged;
}

// Resolve an adapter path (a single safetensors file, or a directory holding
// one), open it, and merge. Returns the number of tensors patched, or -1 on
// failure with `err_out` set.
// Which half a LoRA file targets, read WITHOUT merging: "ar", "nar", or ""
// when the file does not say (no __metadata__.format and no key names either
// way). yue2_apply_adapters uses it to hand each adapter to the half that is
// actually being loaded (doc 30 #5) — a merge against a half that is not
// staged is fatal by design. Accepts the same path shapes yue2_adapter_merge
// does (a file, or a directory holding one of the exporters' default names).
static std::string yue2_adapter_probe_family(const std::string & path) {
    HS_STAT_T sb;
    if (hs_stat(path, &sb) != 0) return "";
    std::string sf_path = path;
    if (S_ISDIR(sb.st_mode)) {
        const char * cands[] = { "/adapter_model.safetensors", "/yue2-nar-lora.safetensors",
                                 "/yue2-ar-lora.safetensors" };
        sf_path.clear();
        for (const char * c : cands) {
            if (hs_stat(path + c, &sb) == 0) {
                sf_path = path + c;
                break;
            }
        }
        if (sf_path.empty()) return "";
    }
    STFile st = {};
    if (!st_open(&st, sf_path.c_str())) return "";
    std::string fam;
    const Yue2AdapterMeta md = yue2_adapter_read_meta(st);
    if (yue2a_starts(md.format, "yue2-nar-lora")) fam = "nar";
    else if (yue2a_starts(md.format, "yue2-ar-lora")) fam = "ar";
    else {
        bool has_nar = false, has_ar = false;
        for (const STEntry & e : st.entries) {
            if (e.name.find(".nar_") != std::string::npos) has_nar = true;
            else if (e.name.find(".attn_") != std::string::npos || e.name.find(".ffn_") != std::string::npos) has_ar = true;
        }
        if (has_nar != has_ar) fam = has_nar ? "nar" : "ar";
    }
    st_close(&st);
    return fam;
}

static int yue2_adapter_merge(WeightCtx *                 wctx,
                              const GGUFModel &           gf,
                              const char *                path,
                              const Yue2LmAdapterScales & scales,
                              ggml_backend_t              backend,
                              std::string *               err_out,
                              std::string *               family_out = nullptr) {
    // hs_stat, not stat: MSVC's narrow stat is _stat64i32, whose 32-bit st_size
    // returns -1 for any file >= 2 GiB — so a large adapter reads as MISSING
    // rather than as itself (commit ae64b19c). It is also the UTF-8-correct
    // call, which matters for a path that arrived as JSON off the wire.
    HS_STAT_T sb;
    if (hs_stat(std::string(path), &sb) != 0) {
        if (err_out) {
            *err_out = std::string("adapter path does not exist: ") + path;
        }
        return -1;
    }

    std::string sf_path;
    std::string cfg_dir;
    if (S_ISDIR(sb.st_mode)) {
        // A directory is accepted for symmetry with MM3/PEFT layouts, but our
        // own exporters write one file each — the NAR trainer's and the AR
        // trainer's default names are both probed, since either half can be the
        // thing in the directory.
        const char * cands[] = { "/adapter_model.safetensors", "/yue2-nar-lora.safetensors",
                                 "/yue2-ar-lora.safetensors" };
        bool         found   = false;
        for (const char * c : cands) {
            sf_path = std::string(path) + c;
            if (hs_stat(sf_path, &sb) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            if (err_out) {
                *err_out = std::string("no adapter_model.safetensors (or yue2-nar-lora.safetensors, or "
                                       "yue2-ar-lora.safetensors) in ") +
                           path;
            }
            return -1;
        }
        cfg_dir = path;
    } else {
        sf_path      = path;
        size_t slash = sf_path.find_last_of("/\\");
        cfg_dir      = (slash == std::string::npos) ? "." : sf_path.substr(0, slash);
    }

    STFile st = {};
    if (!st_open(&st, sf_path.c_str())) {
        if (err_out) {
            *err_out = "cannot open adapter safetensors: " + sf_path;
        }
        return -1;
    }

    Timer       t;
    std::string fam;
    const int   merged = yue2_adapter_merge_st(wctx, gf, st, cfg_dir, scales, backend, err_out, &fam);
    st_close(&st);

    if (merged < 0) {
        fprintf(stderr, "[YuE2-Adapter] %s: REFUSED — %s\n", sf_path.c_str(),
                (err_out && !err_out->empty()) ? err_out->c_str() : "unknown reason");
        return -1;
    }
    if (family_out) {
        *family_out = fam;
    }
    // The family is on this line on purpose: "196 tensors" alone reads the same
    // whichever half it landed on, and which half it landed on is the thing a
    // wrong file gets wrong.
    // Every dial is on this line, not just the master: a merge that came out
    // quiet because the MLP dial sat at 0.1 reads identically to one that
    // merged at full strength if the log only ever quotes one number.
    fprintf(stderr,
            "[YuE2-Adapter] %s: merged %d tensor(s) into the %s half at scale %.3f "
            "(attn %.3f, mlp %.3f, thirds %.3f/%.3f/%.3f) in %.1f ms\n",
            sf_path.c_str(), merged, fam.empty() ? "?" : fam.c_str(), (double) scales.global,
            (double) scales.attn, (double) scales.mlp, (double) scales.early, (double) scales.mid,
            (double) scales.late, t.ms());
    return merged;
}
