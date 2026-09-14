#pragma once
// train/yue2-ar-train-run.h — the runner around yue2-ar-train-graph.h.
//
// HOT-Step file, TRAINING-SIDE (ace-train only). Implements
// docs/plans/yue2/14-ar-lora-contract.md §§5-8: the dataset, the FD gate, the
// training loop, checkpoint/resume and the safetensors exporter. The template
// for every one of those is train/yue2-nar-train-run.h, which shipped days
// earlier; where this file differs from it, the comment says why.
//
// ── What this trains, and why it could not be trained before ───────────────
//
// The AR half of YuE2 is the COMPOSER. It writes the semantic token stream that
// fixes melody, phrasing and structure; the NAR half and the VAE only render
// what the AR decided. Training it needed ground-truth codec tokens for real
// audio, which did not exist until the encoder landed in 0b8c3f1c / b95540c0.
// `ace-train yue2-tokenize` now fills the per-source `codec_ids` slot, and that
// file — the WHOLE-SONG raw code array, one i32 per latent frame — is precisely
// this trainer's target.
//
// ── What is in here ────────────────────────────────────────────────────────
//
//   `--fd-check N`  the gradient gate. Builds one prefix, a synthetic codec
//                   stream, runs one forward + backward and finite-differences
//                   the analytic gradient against the measured loss change. The
//                   synthetic stream is correct for a gradient check and wrong
//                   for anything else: FD tests that the backward agrees with
//                   the forward, which is a property of the GRAPH, not the data.
//   (no --fd-check) the training loop: manifest -> whole-song sequences ->
//                   next-token CE over the codec positions only, AdamW with
//                   warmup + cosine, grad accumulation, global-norm clipping,
//                   `--save-every` checkpoints, `--resume`, and a safetensors
//                   export in the key scheme contract §7.1 specifies.
//
// ── A falling loss is not evidence of a correct backward ───────────────────
//
// The DiT-trainer delta fiasco is this codebase's own proof. `--fd-check` is
// the only gradient gate this trainer has, and it has to return a VERDICT
// rather than a note — which is what `--ar-layers K` buys (§6.1 below).
//
// ── Determinism, and what it costs ─────────────────────────────────────────
//
// Every random decision — artist-or-minted, which song, the FD stand-in stream
// — is drawn from a stream seeded by (seed, micro-step index, purpose), NOT
// from one long-lived generator. That is what makes `--resume` exact: the data
// stream at micro-step k is a pure function of k, so a run that pauses at 300
// and resumes sees exactly the draws the uninterrupted run would have seen.
//
// ── THE MINTED REGULARIZER ─────────────────────────────────────────────────
//
// `minted_regularizer_pack.pt` (Mothersuperior/yue2-minted-corpus, dataset repo,
// path `regularizer/minted_regularizer_pack.pt`, 101,878,861 bytes, sha256
// bdd9b978…dc4e) is the other half of upstream's recipe: 4,732 YuE2-generated
// songs, each {name, src, style, lyrics, codec} with `codec` an int32 array of
// RAW codes in [0, 32768). 4,516 are `minted` and 216 are `minted_val`, held
// out by md5(name) % 20 == 0 (verified against all 4,732 records).
//
// Downloaded and converted 2026-09-14 to K:/yue2/models/yue2-minted/ by
// engine/tools/convert-yue2-minted.py; format spec at
// docs/plans/yue2/15-minted-pack.md. `--minted` takes the converted
// minted_manifest.json, never the .pt.
//
// Its ABSENCE remains an explicit state, and contract §5.3's refusal is
// unchanged and deliberately loud:
//
//   1. `--minted <path>` is accepted; its ABSENCE is an explicit state.
//   2. Without it the run REFUSES TO START unless `--allow-no-minted` is also
//      passed.
//   3. With `--allow-no-minted` a banner prints at the top of the run and again
//      at the end, and the exported adapter's __metadata__ carries
//      minted: "absent".
//   4. There is NO minted_val eval without the pack, and the trainer says so
//      rather than substituting the artist hold-out — that answers a different
//      question (has the artist been memorised) from the one minted_val answers
//      (has YuE2's token grammar been damaged).
//
// WHY IT IS STRUCTURAL, not garnish: docs/plans/yue2/14-tokenizer-truth-gate.md
// §4 measured our encoder's adjacent-repeat rate at 3.3-6.6% against YuE2's own
// 0.02-0.17%, because YuE2's semantic stage samples with repetition_penalty 1.2
// over a 50-frame window and the encoder has no such constraint. That is out of
// distribution in exactly the direction that produces LOOPING, and the 50/50
// minted mix — whose codes are true YuE2 tokens — is the counterweight.
//
// The pack is a torch pickle, so it needs a converter. `--minted` therefore
// takes a JSON manifest in the same shape yue2-preprocess writes (sources[]
// with style/lyrics/codec_ids and a per-source `src` of "minted"/"minted_val"),
// and a `.pt` path is refused with a message saying so rather than half-read.
// The converted pack adds two optional per-source fields, `codec_ids_offset`
// and `codec_ids_frames` (both in FRAMES), so all 4,732 songs slice out of one
// 86 MB blob instead of 4,732 small files. Absent, they mean "the whole file",
// which is what yue2-tokenize writes.
//
// Two things the regularizer set does NOT inherit from the artist set, because
// upstream does not give them to it either (`ar_prep.py` takes the minted style
// from request.json verbatim): --trigger is not prepended to a minted style,
// and --style / --lyrics are not used as its fallbacks. Teaching the trigger
// word on 4,516 songs that are not the artist would undo the one thing the
// trigger is for. And the minted_val hold-out is excluded from the TRAINING
// pool, not merely reported: training on it would make the number this run
// watches flat for the wrong reason.
//
// ── The other honest gap: lyrics (contract §5.2) ───────────────────────────
//
// The yue2-preprocess manifest carries NO lyrics field and its captions are
// empty on the greenday run (caption_mode "none"). Without a style and a tagged
// lyric sheet the AR prefix is not the prefix inference builds, and the README
// is emphatic that truncated lyrics ruin structure. Three routes, in order of
// preference, and the trainer says which one it took:
//
//   1. a per-source `lyrics` (and non-empty `caption`) in the manifest —
//      contract §5.2's phase-4 change to yue2-preprocess. Preferred; wins
//      whenever present.
//   2. `--sidecars on` (DEFAULT): read HOT-Step's own ACE dataset sidecar
//      `<stem>.txt` next to the source audio, with sidecarIO.ts's exact rule —
//      ONCE `lyrics:` STARTS, EVERY SUBSEQUENT LINE IS LYRICS, colons included.
//      This is a DEVIATION from the contract, which puts that parser in
//      yue2-preprocess; it is additive, touches no producer, and is the only
//      reason this trainer can build a real prefix before phase 4 lands.
//   3. `--style` / `--lyrics` on the command line, applied to every song.
//
// With none of the three the run still starts and prints a banner saying the
// prefix is style-less and lyric-less, because that trains and it is wrong, and
// silence would be the worst of the three outcomes.

#include "train/gpu-mem.h"
#include "train/lm-optim.h"
#include "train/preprocess-io.h"  // pm_mkdir_p, pm_file_exists, pm_stat_file
#include "train/st-write.h"       // the exporter
#include "train/yue2-ar-train-graph.h"
#include "train/yue2-sidecar.h"  // the ACE sidecar parser, shared with yue2-preprocess

#include "hot-step-fsutf8.h"
#include "yue2/yue2-model.h"
#include "yue2/yue2-tokenizer.h"

#include <algorithm>
#include <cfloat>  // FLT_EPSILON — the hard floor under the FD gate's resolution estimate
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// ── Arguments (contract §8) ────────────────────────────────────────────────

struct Yue2ArTrainArgs {
    // Discovery is directory-based (yue2_discover scans <dir>/yue2 then <dir>),
    // so --lm is split into a search directory and a type token and handed to
    // the normal loader rather than opening the file behind its back.
    std::string lm_path;
    std::string models_dir;

    std::string manifest;   // yue2_preprocess.json, the artist set
    std::string minted;     // the converted regularizer manifest; EMPTY is a state, not a default
    bool        allow_no_minted = false;
    std::string out_dir;
    std::string name = "yue2_ar_lora";  // export stem: <out>/<name>.safetensors

    // Fallback conditioning. Real runs take these per song (manifest, then ACE
    // sidecar); these apply to every song when neither does, and they are what
    // the FD gate builds its prefix from.
    std::string style;
    std::string lyrics;
    bool        sidecars = true;  // parse <stem>.txt beside the source audio

    // upstream's `trigger_word`: the phrase the trained style is addressed by at
    // generation time, prepended to every caption. Written into the exported
    // metadata as well, so half a run under a different one is a different
    // adapter.
    std::string trigger;
    // "upstream" (default): the artist style string is upstream's training
    // template, trigger + "in the style of" + caption + genre/BPM/key tail
    // (yue2_style_string). "bare": the old "<trigger>, <caption>". Written to
    // the adapter's metadata so generation composes the same string.
    std::string style_template = "upstream";

    std::string target = "attn_mlp";  // attn | attn_mlp   (attn_mlp_embed: refused, contract §2.3)
    int64_t     rank   = 64;          // upstream's rank, and the FD gate's (contract §6.4)
    float       alpha  = 64.0f;
    uint64_t    seed   = 42;

    // ── the loop: upstream's numbers, unchanged, because nothing has been
    //    heard yet (ar_lora_cursor.py:12-13, 30, 79, 87, 93) ──
    float       lr            = 1e-4f;
    int64_t     steps         = 1600;
    int64_t     warmup        = 50;
    std::string lr_scheduler  = "cosine";  // cosine | constant
    int64_t     sched_steps   = 3000;      // upstream's SCHED_STEPS: the cosine HORIZON, not --steps
    int64_t     grad_accum    = 2;         // upstream's ACC
    float       max_grad_norm = 1.0f;
    float       weight_decay  = 0.0f;
    // upstream's AdamW betas (ar_lora_cursor.py:31). Overridable so the
    // divergence that used to be silent can now be A/B'd on purpose.
    float       adam_beta1    = 0.9f;
    float       adam_beta2    = 0.95f;
    double      artist_frac   = 0.5;       // one draw per micro-step: random() < frac ? artist : minted
    int64_t     max_len       = 12288;     // upstream's MAXLEN
    bool        allow_overtrain = false;   // past ~1500 steps the model memorises the songs

    std::string attn        = "exact";  // exact | flash | flash-f32
    bool        weights_f32 = true;
    // The TRAINING loss is always the 32,769-row slice (contract §3.3) — the
    // label buffer is what makes the full vocabulary unaffordable, at 3.3 GB
    // against 32 MB per chunk. `--ce-full` / `--ce-slice` control only whether
    // each EVAL also reports the full-vocab number, which is the one that is
    // comparable to upstream's README. Default on: evals are infrequent.
    bool    eval_full_vocab = true;
    int64_t chunk           = 256;  // supervised rows per CE chunk

    int64_t     save_every = 200;  // upstream's CK_EVERY
    // upstream's CK_FROM is 600. Lowered: on a twelve-song set the artist loss
    // is already 0.63 at step 500 and the coherent-sounding region of the
    // ladder (Rob, 2026-09-14, on upstream's own renders) is 600 and EARLIER.
    // A floor of 600 exports only the far side of that. Snapshots are cheap;
    // a rung you never rendered is not.
    int64_t     ckpt_from  = 200;
    int64_t     eval_every = 100;
    int64_t     log_every  = 20;
    bool        resume     = false;

    // The lyric-cursor auxiliary loss (contract §3.4), upstream's CUR_W = 0.08.
    // Its targets are per-source `cursor_words` files named in the manifest:
    // little-endian f32 [n_words, 5] = (start_s, end_s, score, char0, char1),
    // upstream's cursor_prep.py layout, char offsets in codepoints into the
    // manifest's own `lyrics` string. Produced by a forced aligner outside this
    // binary for now (the Python bridge; a native one is the follow-up).
    //
    // Why it is not optional: run side by side with the term on and off on the
    // same twelve songs, the model's frame-to-lyric alignment loss FALLS from
    // 12.3 to 1.6 with it and RISES from 10.8 to 16.3 without it — the AR
    // spends its capacity memorising token streams and loses track of which
    // words it is singing. That is the "structure broken, vocals incoherent,
    // timbre fine" verdict on the first C++ ladder, which trained without it.
    //
    // > 0 with NO bound artist song is REFUSED: a run that binds nothing would
    // print "cursor nan" forever and finish, which is the gate that cannot
    // fail. 0 switches the term off explicitly.
    double cursor_weight = 0.08;

    // --fd-check N: run the gradient gate over N probes instead of training.
    // --forward-check N: the two checks the FD gate cannot see (contract
    // §§6.5-6.6), over a REAL manifest song rather than the gate's synthetic
    // stream. N is how many supervised rows to compare against yue2_ar_forward.
    int     fwd_check = 0;
    int64_t fc_song   = 0;  // which manifest source to run them on

    int    fd_check  = 0;
    double fd_eps    = 1e-2;
    int    ar_layers = 2;    // F32 isolation depth / stack truncation; 0 = no isolation (report only)
    int64_t fd_frames = 128;  // synthetic codec stream length for the gate
};

// ── F32 isolation of the AR stack (contract §6.1) ──────────────────────────
//
// mm3-f32-isolate.h's surgery, scaled to what this gate reasons about.
// MIRRORED: the first `n_layers` AR blocks (11 tensors each) and the shared
// `output_norm`. LEFT ALONE: `token_embd` (consumed only by ggml_get_rows, an
// exact widening) and the output head — the head the trainer scores is ALREADY
// an F32 copy (`Yue2AtState::t_head`, built once at alloc), so mirroring it
// would be a second copy of the same numbers.
//
// WORTH BEING PRECISE ABOUT WHAT THIS BUYS, because `Yue2AtOpts::weights_f32`
// (on by default) already widens every frozen projection IN GRAPH and those two
// widenings produce identical values. What the mirror adds is (a) the 11 cast
// kernels per graph go away for the probed layers, and (b) a QUANTIZED BASE IS
// REFUSED BY NAME instead of quietly dequantized — and dequantizing to measure
// would measure the quantizer. What actually turns the gate into a verdict is
// the pair of them plus `--ar-layers K` TRUNCATING the stack, so 28 layers of
// accumulated rounding do not swamp the defect being looked for.
//
// The original weight buffer is NOT freed: the untouched layers still live in
// it, so freeing it would dangle 26 layers' worth of pointers.
struct Yue2AtF32Slice {
    ggml_context *        ctx       = nullptr;
    ggml_backend_buffer_t buf       = nullptr;
    size_t                bytes     = 0;
    int                   n_tensors = 0;
    int                   n_layers  = 0;
};

static void yue2_at_f32_free(Yue2AtF32Slice * M) {
    if (M->buf) {
        ggml_backend_buffer_free(M->buf);
    }
    if (M->ctx) {
        ggml_free(M->ctx);
    }
    *M = Yue2AtF32Slice{};
}

static bool yue2_at_f32_isolate(Yue2Model * m, int n_layers, Yue2AtF32Slice * M, std::string * err) {
    if (n_layers < 1 || n_layers > (int) m->lm.blk.size()) {
        *err = "f32-isolate: --ar-layers out of range";
        return false;
    }

    struct Slot {
        ggml_tensor ** field;
        ggml_tensor *  src;
        ggml_tensor *  dst;
    };
    std::vector<Slot> slots;

    const int        budget = n_layers * 11 + 8;
    ggml_init_params p      = { (size_t) budget * ggml_tensor_overhead() + 4096, nullptr, /*no_alloc*/ true };
    M->ctx                  = ggml_init(p);
    if (!M->ctx) {
        *err = "f32-isolate: context alloc failed";
        return false;
    }

    bool        bad_type = false;
    std::string bad_name, bad_type_name;
    auto        add = [&](ggml_tensor ** field) {
        if (!field || !*field) {
            return;
        }
        ggml_tensor * s = *field;
        if (s->type != GGML_TYPE_F32 && s->type != GGML_TYPE_F16 && s->type != GGML_TYPE_BF16) {
            // A quantized base is a legitimate thing to TRAIN (the in-graph cast
            // handles it) but not a thing to ISOLATE.
            if (!bad_type) {
                bad_type      = true;
                bad_name      = ggml_get_name(s);
                bad_type_name = ggml_type_name(s->type);
            }
            return;
        }
        ggml_tensor * d = ggml_new_tensor_2d(M->ctx, GGML_TYPE_F32, s->ne[0], s->ne[1]);
        ggml_set_name(d, ggml_get_name(s));
        M->bytes += ggml_nbytes(d);
        slots.push_back({ field, s, d });
    };

    for (int i = 0; i < n_layers; i++) {
        Yue2LmLayer & ly = m->lm.blk[(size_t) i];
        add(&ly.attn_norm);
        add(&ly.attn_q);
        add(&ly.attn_k);
        add(&ly.attn_v);
        add(&ly.attn_output);
        add(&ly.attn_q_norm);
        add(&ly.attn_k_norm);
        add(&ly.ffn_norm);
        add(&ly.ffn_gate);
        add(&ly.ffn_up);
        add(&ly.ffn_down);
    }
    add(&m->lm.output_norm);

    if (bad_type) {
        *err = "f32-isolate: '" + bad_name + "' is " + bad_type_name +
               " — the gate needs an F16/BF16/F32 base (run it on bf16 even when training on a quant); "
               "pass --ar-layers 0 to skip isolation and get a report instead of a verdict";
        return false;
    }

    M->buf = ggml_backend_alloc_ctx_tensors(M->ctx, m->backend);
    if (!M->buf) {
        char b[160];
        snprintf(b, sizeof(b), "f32-isolate: allocation failed (%.1f MB)", M->bytes / 1048576.0);
        *err = b;
        return false;
    }
    ggml_backend_buffer_set_usage(M->buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<uint8_t> raw;
    std::vector<float>   f32;
    for (Slot & sl : slots) {
        ggml_tensor * s      = sl.src;
        const size_t  nbytes = ggml_nbytes(s);
        const size_t  ne     = (size_t) ggml_nelements(s);
        raw.resize(nbytes);
        ggml_backend_tensor_get(s, raw.data(), 0, nbytes);
        if (s->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(sl.dst, raw.data(), 0, nbytes);
        } else if (s->type == GGML_TYPE_F16) {
            f32.resize(ne);
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), f32.data(), (int64_t) ne);
            ggml_backend_tensor_set(sl.dst, f32.data(), 0, ne * sizeof(float));
        } else {  // BF16 — widen by hand, same as mm3-f32-isolate.h:174-183
            f32.resize(ne);
            const uint16_t * u = (const uint16_t *) raw.data();
            for (size_t j = 0; j < ne; j++) {
                const uint32_t bits = (uint32_t) u[j] << 16;
                float          val;
                memcpy(&val, &bits, 4);
                f32[j] = val;
            }
            ggml_backend_tensor_set(sl.dst, f32.data(), 0, ne * sizeof(float));
        }
        *sl.field = sl.dst;
    }

    M->n_tensors = (int) slots.size();
    M->n_layers  = n_layers;
    fprintf(stderr,
            "[yue2-ar-fd] F32 isolation: %d AR layers + output_norm, %d tensors, %.1f MB (token_embd left "
            "native — get_rows is an exact widening; the scored head is already an F32 copy)\n",
            n_layers, M->n_tensors, M->bytes / 1048576.0);
    return true;
}

// ── Model discovery ────────────────────────────────────────────────────────

static bool yue2_at_discover(Yue2Model * m, const Yue2ArTrainArgs & a, std::string * err) {
    if (!a.lm_path.empty()) {
        std::string path = a.lm_path;
        for (char & ch : path) {
            if (ch == '\\') {
                ch = '/';
            }
        }
        const size_t      slash = path.find_last_of('/');
        const std::string dir   = (slash == std::string::npos) ? std::string(".") : path.substr(0, slash);
        const std::string base  = (slash == std::string::npos) ? path : path.substr(slash + 1);
        if (base.size() <= 13 || base.compare(0, 8, "yue2-lm-") != 0 ||
            base.compare(base.size() - 5, 5, ".gguf") != 0) {
            *err = "--lm must name a yue2-lm-<type>.gguf (got '" + base + "')";
            return false;
        }
        yue2_discover(m, dir.c_str(), base.substr(8, base.size() - 13).c_str());
    } else if (!a.models_dir.empty()) {
        yue2_discover(m, a.models_dir.c_str());
    } else {
        *err = "yue2-ar-train: one of --lm <yue2-lm-*.gguf> or --models <dir> is required";
        return false;
    }
    if (!yue2_available(*m)) {
        *err = m->meta_errors.empty() ? "YuE2 LM GGUF not found or its metadata probe failed"
                                      : m->meta_errors[0];
        return false;
    }
    return true;
}

static bool yue2_at_open_model(Yue2Model * m, const Yue2ArTrainArgs & a, const char * tag, std::string * err) {
    if (!yue2_at_discover(m, a, err)) {
        return false;
    }
    if (!yue2_load_parts(m, /*want_lm=*/true, /*want_vae=*/false, YUE2_VAE_STANDARD, /*want_encoder=*/false,
                         err)) {
        *err = "load: " + *err;
        return false;
    }
    // 2.8 GiB of NAR weights are resident and never touched by this trainer:
    // `lm_resident` is all-or-nothing over AR + NAR + flow heads
    // (yue2-model.h:33). Not needed to fit, but worth knowing exists
    // (contract §4.5).
    fprintf(stderr, "[%s] loaded %s (the NAR half is resident and unused — ~2.7 GiB a want_ar_only loader "
                    "would free)\n",
            tag, m->lm_file.path.c_str());
    return true;
}

// ── The dataset ────────────────────────────────────────────────────────────
//
// The AR trainer reads the manifest's `sources` array, NOT its `clips`: the
// source-level `codec_ids` file is the WHOLE-SONG raw code array, one i32 per
// latent frame, resized to exactly `frames` by yue2-tokenize. That is precisely
// the AR training target, and the 241 per-clip slices are the NAR trainer's.
//
// WHOLE SONGS, AND NO CROP POLICY. Contract §4.4: the greenday corpus runs
// 4,077-7,647 frames and every song fits whole in either attention mode. MM3
// paid twice for the alternative — 5.12 s crops against a 204 s median produced
// renders that "start mid-song and never resolve", because a random crop
// teaches that a song may legitimately begin at position c0 and because EOS is
// supervised at exactly one position per example. There are no --crop-* flags
// here on purpose.
struct Yue2ArSong {
    std::string          name;
    std::string          source;   // the audio file the codes came from
    std::string          style;    // what goes in [Tags]
    std::string          lyrics;   // what goes in [Lyrics]
    std::string          genre, bpm, key;  // sidecar fields for the style template (artist rows only)
    std::string          codes_path;
    int64_t              codes_off = 0;  // FRAMES into codes_path; blob-packed sets only
    int64_t              codes_n   = 0;  // FRAMES to read; 0 = the whole file
    std::string          cursor_words;  // path to the f32 [n_words, 5] spans; empty = none
    std::vector<float>   words5;        // those spans, loaded
    Yue2AtCursor         cursor;        // the per-frame target block, built once
    std::vector<int32_t> codec;   // RAW ids, [0, 32768); lazily loaded for the minted set
    std::vector<int32_t> prefix;  // tokenized once
    bool                 minted     = false;
    bool                 minted_val = false;
    bool                 loaded     = false;
};

struct Yue2ArSet {
    std::vector<Yue2ArSong> songs;
    std::string             dir;
    std::string             format;
    std::string             caption_format;
    bool                    codec_ids_present = false;
};

static bool yue2_at_is_abs(const std::string & p) {
    return (p.size() > 1 && p[1] == ':') || (!p.empty() && (p[0] == '/' || p[0] == '\\'));
}

static std::string yue2_at_join(const std::string & dir, const std::string & p) {
    if (p.empty() || dir.empty() || yue2_at_is_abs(p)) {
        return p;
    }
    return dir + "/" + p;
}

static std::string yue2_at_dirname(const std::string & p) {
    const size_t s = p.find_last_of("/\\");
    return (s == std::string::npos) ? std::string(".") : p.substr(0, s);
}

// ── The ACE dataset sidecar (contract §5.2, and the deviation named above) ──
//
// `<stem>.txt` beside the source audio, in HOT-Step's own Option-A format.
// The rules — and the one that matters, `lyrics:` running to end of file — live
// in train/yue2-sidecar.h, which `yue2-preprocess --caption-mode ace` parses
// with as well. ONE implementation, because two copies of a parser whose
// failure mode is "trains at full speed on a mangled prefix" is the drift
// contract §5.2 is written against.
static bool yue2_at_parse_sidecar(const std::string & text, std::string * caption, std::string * lyrics) {
    return yue2_sidecar_parse(text, caption, lyrics);
}

static bool yue2_at_read_text(const std::string & path, std::string * out) {
    FILE * f = hs_fopen(path, "rb");
    if (!f) {
        return false;
    }
    char   buf[8192];
    size_t n;
    out->clear();
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        out->append(buf, n);
    }
    fclose(f);
    return true;
}

static std::string yue2_at_sidecar_path(const std::string & audio) {
    return yue2_sidecar_path(audio);
}

// One song's raw codes. `n_frames == 0` means "the whole file", which is the
// yue2-preprocess case: one .i32 per source, written by `ace-train
// yue2-tokenize`. A non-zero `n_frames` reads a slice out of a shared blob at
// `off_frames`, which is how the converted minted pack stores 4,732 songs in
// one 86 MB file instead of 4,732 small ones (docs/plans/yue2/15-minted-pack.md).
// Both offsets are in FRAMES — i32 elements — never bytes.
// The cursor spans: raw little-endian f32, five per word, upstream's
// cursor_words.npy body without the .npy header. Checked, not trusted: a
// non-multiple-of-five file, a negative time, a word ending before it starts, a
// char span outside the lyrics, or start times that run backwards each name
// the file and refuse — a silently-wrong span shifts every target after it.
static bool yue2_at_read_words5(const std::string & path, int64_t n_lyric_chars, std::vector<float> * out,
                                std::string * err) {
    FILE * f = hs_fopen(path, "rb");
    if (!f) {
        *err = "cannot open cursor_words file " + path;
        return false;
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0 || (n % (5 * 4)) != 0) {
        fclose(f);
        *err = path + " is not a whole number of (start, end, score, char0, char1) f32 rows (" +
               std::to_string((long long) n) + " bytes)";
        return false;
    }
    out->assign((size_t) (n / 4), 0.0f);
    const size_t got = fread(out->data(), 1, (size_t) n, f);
    fclose(f);
    if (got != (size_t) n) {
        *err = "short read on " + path;
        return false;
    }
    const size_t nW = out->size() / 5;
    float        prev = -1.0f;
    for (size_t w = 0; w < nW; w++) {
        const float * r = out->data() + w * 5;
        if (!(r[0] >= 0.0f) || !(r[1] >= r[0]) || !(r[3] >= 0.0f) || !(r[4] >= r[3]) ||
            (n_lyric_chars > 0 && r[4] > (float) n_lyric_chars) || r[0] < prev) {
            char b[200];
            snprintf(b, sizeof(b), "word %zu is malformed (start %.3f end %.3f chars [%.0f, %.0f) of %lld)",
                     w, (double) r[0], (double) r[1], (double) r[3], (double) r[4], (long long) n_lyric_chars);
            *err = path + ": " + b;
            return false;
        }
        prev = r[0];
    }
    return true;
}

// Codepoints in a UTF-8 string: what Python's len() returns for it, and the
// unit the cursor spans' char offsets are in.
static int64_t yue2_at_utf8_len(const std::string & s) {
    int64_t n = 0;
    for (unsigned char c : s) {
        if ((c & 0xC0) != 0x80) {
            n++;
        }
    }
    return n;
}

static bool yue2_at_read_i32_file(const std::string & path, int64_t off_frames, int64_t n_frames,
                                  std::vector<int32_t> * out, std::string * err) {
    FILE * f = hs_fopen(path, "rb");
    if (!f) {
        *err = "cannot open codec_ids file " + path;
        return false;
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0 || (n % 4) != 0) {
        fclose(f);
        *err = path + " is not a whole number of i32 codec ids (" + std::to_string((long long) n) + " bytes)";
        return false;
    }
    const int64_t have = (int64_t) (n / 4);
    if (n_frames <= 0) {
        off_frames = 0;
        n_frames   = have;
    } else if (off_frames < 0 || off_frames + n_frames > have) {
        fclose(f);
        // A blob index that points past the end is a converter bug, and reading
        // a neighbouring song's codes instead would train silently on the wrong
        // target. Refuse rather than clamp.
        *err = path + ": codec_ids_offset " + std::to_string((long long) off_frames) + " + " +
               std::to_string((long long) n_frames) + " frames runs past the file's " +
               std::to_string((long long) have) + " frames";
        return false;
    }
    if (off_frames && fseek(f, (long) (off_frames * 4), SEEK_SET) != 0) {
        fclose(f);
        *err = path + ": cannot seek to frame " + std::to_string((long long) off_frames);
        return false;
    }
    out->assign((size_t) n_frames, 0);
    const bool ok = out->empty() || fread(out->data(), 4, out->size(), f) == out->size();
    fclose(f);
    if (!ok) {
        *err = "short read on " + path;
        return false;
    }
    return true;
}

// upstream train-side convention: the trigger word IS the style when there is
// no caption, and prefixes it when there is.
static std::string yue2_at_style(const std::string & trigger, const std::string & caption) {
    if (caption.empty()) {
        return trigger;
    }
    if (trigger.empty()) {
        return caption;
    }
    return trigger + ", " + caption;
}

// upstream caps the style string at 1500 characters (ar_prep.py:30) and
// collapses whitespace. Both matter: the style is part of the prefix, and the
// prefix is part of the VRAM model.
static std::string yue2_at_squash(const std::string & s, size_t cap) {
    std::string out;
    out.reserve(s.size());
    bool sp = false;
    for (char ch : s) {
        const bool ws = (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n');
        if (ws) {
            sp = true;
            continue;
        }
        if (sp && !out.empty()) {
            out += ' ';
        }
        sp = false;
        out += ch;
    }
    if (out.size() > cap) {
        out.resize(cap);
    }
    return out;
}

// Reads a yue2-preprocess-v1 manifest's `sources` array (or a converted minted
// manifest in the same shape). `want_minted` tags every song and honours a
// per-source "src" of "minted_val".
static bool yue2_at_load_manifest(const std::string & path, bool want_minted, const Yue2ArTrainArgs & a,
                                  Yue2ArSet * out, std::string * err) {
    yyjson_read_err rerr;
    yyjson_doc *    doc = yyjson_read_file(path.c_str(), 0, nullptr, &rerr);
    if (!doc) {
        *err = "cannot parse " + path + ": " + (rerr.msg ? rerr.msg : "unknown error");
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!root || !yyjson_is_obj(root)) {
        yyjson_doc_free(doc);
        *err = path + ": root is not an object";
        return false;
    }
    auto jstr = [](yyjson_val * o, std::initializer_list<const char *> keys) {
        for (const char * k : keys) {
            yyjson_val * v = o ? yyjson_obj_get(o, k) : nullptr;
            if (v && yyjson_is_str(v)) {
                return std::string(yyjson_get_str(v));
            }
        }
        return std::string();
    };
    auto jint = [](yyjson_val * o, const char * k) -> int64_t {
        yyjson_val * v = o ? yyjson_obj_get(o, k) : nullptr;
        return (v && yyjson_is_int(v)) ? (int64_t) yyjson_get_sint(v) : 0;
    };

    out->dir            = yue2_at_dirname(path);
    out->format         = jstr(root, { "format" });
    out->caption_format = jstr(root, { "caption_format" });
    {
        yyjson_val * cp        = yyjson_obj_get(root, "codec_ids_present");
        out->codec_ids_present = cp && yyjson_is_bool(cp) && yyjson_get_bool(cp);
    }

    yyjson_val * arr = yyjson_obj_get(root, "sources");
    if (!arr || !yyjson_is_arr(arr)) {
        arr = yyjson_obj_get(root, "songs");
    }
    if (!arr || !yyjson_is_arr(arr)) {
        yyjson_doc_free(doc);
        *err = path + ": no sources[] array. The AR trainer trains WHOLE SONGS, so it reads the "
                      "source-level codec_ids, not the per-clip slices the NAR trainer uses.";
        return false;
    }

    size_t       idx = 0, max = 0;
    yyjson_val * it = nullptr;
    std::string  ferr;
    int          n_sidecar = 0, n_manifest_lyrics = 0;
    yyjson_arr_foreach(arr, idx, max, it) {
        if (!yyjson_is_obj(it)) {
            continue;
        }
        Yue2ArSong s;
        s.name   = jstr(it, { "name", "id" });
        s.source = jstr(it, { "source", "file", "audio", "path" });
        const std::string cap = jstr(it, { "caption", "style", "text", "prompt" });
        s.lyrics              = jstr(it, { "lyrics" });
        s.genre               = jstr(it, { "genre" });
        s.bpm                 = jstr(it, { "bpm" });
        s.key                 = jstr(it, { "key" });
        s.cursor_words        = jstr(it, { "cursor_words" });
        const std::string cd  = jstr(it, { "codec_ids", "codec", "codes" });
        if (cd.empty()) {
            ferr = "source \"" + s.name + "\" carries no codec_ids — run `ace-train yue2-tokenize` over "
                                          "this manifest first";
            break;
        }
        s.codes_path = yue2_at_join(out->dir, cd);
        // Blob-packed sets (the converted minted pack) point every source at
        // one shared .i32 and slice it. Absent fields mean "whole file", which
        // is what yue2-tokenize writes.
        s.codes_off = jint(it, "codec_ids_offset");
        s.codes_n   = jint(it, "codec_ids_frames");
        if (s.codes_n < 0 || s.codes_off < 0) {
            ferr = "source \"" + s.name + "\" has a negative codec_ids_offset/codec_ids_frames";
            break;
        }
        if (!s.cursor_words.empty()) {
            s.cursor_words = yue2_at_join(out->dir, s.cursor_words);
        }

        std::string caption = cap;
        if (!s.lyrics.empty()) {
            n_manifest_lyrics++;
        } else if (a.sidecars && !s.source.empty()) {
            std::string text;
            if (yue2_at_read_text(yue2_at_sidecar_path(s.source), &text)) {
                std::string sc_cap, sc_lyr;
                if (yue2_at_parse_sidecar(text, &sc_cap, &sc_lyr)) {
                    if (caption.empty()) {
                        caption = sc_cap;
                    }
                    s.lyrics = sc_lyr;
                    n_sidecar++;
                }
            }
        }
        // Contract §5.2 item 3: a caption that CONTAINS a lyric sheet is the one
        // wrong guess nothing downstream could catch — it would train at full
        // speed on a mangled prefix with `bpm: 121` inside [Tags] and an empty
        // [Lyrics]. Refuse rather than infer.
        if (out->caption_format != "ace-sidecar" && caption.find("lyrics:") != std::string::npos) {
            ferr = "source \"" + s.name + "\" has a caption containing a `lyrics:` line while the manifest's "
                                          "caption_format is \"" + (out->caption_format.empty() ? std::string("(absent)")
                                                                                                : out->caption_format) +
                   "\". That is a whole ACE sidecar fed in as a style string: it would put bpm/key and the "
                   "entire lyric sheet inside [Tags] and leave [Lyrics] empty. Re-run yue2-preprocess with "
                   "--caption-mode ace, or strip the caption.";
            break;
        }
        // --style / --lyrics / --trigger describe THE ARTIST. Applying any of
        // them to the regularizer would teach the trigger word on 4,516 songs
        // that are not the artist, which is the one thing the trigger exists to
        // avoid — and upstream does not do it: `ar_prep.py` takes the minted
        // style from `request.json` verbatim, and only the artist caption
        // carries the trigger phrase.
        if (!want_minted) {
            if (s.lyrics.empty()) {
                s.lyrics = a.lyrics;
            }
            if (caption.empty()) {
                caption = a.style;
            }
        }
        // Artist rows get the style TEMPLATE (yue2_style_string, sidecar.h):
        // upstream's "<trigger>, in the style of <trigger>. <caption>. <genre>,
        // <bpm> BPM, key of <key>." unless --style-template bare. Minted rows
        // keep their own style verbatim, as before.
        s.style  = yue2_at_squash(want_minted ? caption
                                              : yue2_style_string(a.trigger, caption, s.genre, s.bpm, s.key,
                                                                  a.style_template != "bare"),
                                  1500);
        s.minted = want_minted;
        if (want_minted) {
            const std::string src = jstr(it, { "src" });
            s.minted_val          = (src == "minted_val");
        }
        if (s.name.empty()) {
            s.name = "song" + std::to_string((long long) out->songs.size());
        }
        out->songs.push_back(std::move(s));
    }
    const std::string blob         = jstr(root, { "codec_ids_blob" });
    const int64_t     total_frames = jint(root, "total_frames");
    yyjson_doc_free(doc);
    if (!ferr.empty()) {
        *err = path + ": " + ferr;
        return false;
    }
    if (out->songs.empty()) {
        *err = path + " holds no sources";
        return false;
    }
    // A blob-packed set says how big its blob should be. Check it ONCE, here,
    // rather than discovering a truncated or stale blob when song 4,000 is
    // first drawn several hundred steps in — and a blob that is the wrong file
    // entirely reads as valid i32 at every offset, so the size is the only
    // cheap thing that can catch it.
    if (!blob.empty() && total_frames > 0) {
        int64_t sum = 0;
        for (const Yue2ArSong & s : out->songs) {
            sum += s.codes_n;
        }
        const std::string bpath = yue2_at_join(out->dir, blob);
        int64_t           have  = -1;
        if (FILE * bf = hs_fopen(bpath, "rb")) {
            fseek(bf, 0, SEEK_END);
            have = (int64_t) ftell(bf);
            fclose(bf);
        }
        if (have != total_frames * 4 || sum != total_frames) {
            char b[512];
            snprintf(b, sizeof(b),
                     "%s: codec blob %s is %lld bytes but the manifest claims %lld frames "
                     "(%lld bytes), and the sources index %lld. The manifest and the blob are not "
                     "from the same conversion — re-run engine/tools/convert-yue2-minted.py.",
                     path.c_str(), bpath.c_str(), (long long) have, (long long) total_frames,
                     (long long) total_frames * 4, (long long) sum);
            *err = b;
            return false;
        }
    }
    if (!want_minted) {
        fprintf(stderr, "[yue2-ar-train] %zu song(s); lyrics from the manifest for %d, from ACE sidecars "
                        "for %d, from --lyrics for the rest\n",
                out->songs.size(), n_manifest_lyrics, n_sidecar);
    }
    return true;
}

// Codes are loaded on first use and kept. The artist set is a few hundred KB;
// a 4,732-song minted pack is ~95 MB of host RAM, which is the cheapest part of
// this trainer.
static bool yue2_at_song_codes(Yue2ArSong * s, std::string * err) {
    if (s->loaded) {
        return true;
    }
    if (!yue2_at_read_i32_file(s->codes_path, s->codes_off, s->codes_n, &s->codec, err)) {
        return false;
    }
    s->loaded = true;
    return true;
}

// ── Trainable state ────────────────────────────────────────────────────────

struct Yue2AtTrainCtx {
    ggml_context *             actx = nullptr;
    ggml_backend_buffer_t      abuf = nullptr;
    Yue2AtAdapters             ad;
    std::vector<ggml_tensor *> params;
    ggml_tensor *              cursor_w   = nullptr;  // the [H, H] lyric-cursor head; null = term off
    ggml_tensor *              t_lossgrad = nullptr;
    ggml_tensor *              t_adamw    = nullptr;
    ggml_tensor *              t_clip     = nullptr;
    ggml_tensor *              t_eps      = nullptr;
    ggml_tensor *              t_gnorm2   = nullptr;
    LmOptim                    opt;
    ggml_backend_sched_t       sched  = nullptr;
    ggml_backend_sched_t       osched = nullptr;
    // The HOST copy of dL/dloss. `t_lossgrad` above is the device scalar the
    // optimizer owns, and in THIS trainer nothing in the backward reads it —
    // the seed has to reach the chunked head's `gs` instead
    // (Yue2AtRun::lossgrad). Keeping the host value on the context is what lets
    // every Yue2AtRun be initialised from one place rather than from whichever
    // local the caller happened to keep.
    float                      lossgrad = 1.0f;
};

static void yue2_at_train_ctx_free(Yue2AtTrainCtx * C) {
    if (C->sched) {
        ggml_backend_sched_free(C->sched);
        C->sched = nullptr;
    }
    if (C->osched) {
        ggml_backend_sched_free(C->osched);
        C->osched = nullptr;
    }
    lm_optim_free(&C->opt);
    if (C->abuf) {
        ggml_backend_buffer_free(C->abuf);
        C->abuf = nullptr;
    }
    if (C->actx) {
        ggml_free(C->actx);
        C->actx = nullptr;
    }
    C->params.clear();
    C->ad = Yue2AtAdapters{};
}

// `b_sigma` is the gate's escape from LoRA's zero-init trap; a real run passes
// 0 so a fresh adapter is an exact no-op. `lossgrad` is dL/dloss — and note
// that it is 1.0 HERE for a real run too, not 1/grad_accum: the grad-accum
// scaling lives in the chunked head's `gs` (lm-ckpt.h's D9), and seeding it
// twice would square it.
//
// It is stored on the context as well as uploaded to `t_lossgrad`, because the
// device scalar is DEAD in this trainer: no segment graph reads it (the loss
// node the optimizer would seed lives in the hand-written head, not in an
// autodiffed graph). Yue2AtRun::lossgrad, multiplied into `gs`, is the live
// path — the upload survives only so lm_optim_step's unconditional read finds
// a real value.
//
// `cursor_head` adds upstream's [H, H] cursor_head (ar_lora_cursor.py:29:
// `nn.Linear(H, H, bias=False)`, `nn.init.eye_`) to the trainable set — same
// optimizer, same lr, same clip, wd 0, exactly as upstream's single param
// group. It is named so as NOT to end in 'A' (yue2_nt_init_adapters keys
// kaiming init on that) and is set to the identity after that init runs.
static bool yue2_at_train_ctx_init(const Yue2Model & m, int n_layers, int64_t rank, float alpha,
                                   Yue2AtTarget target, uint64_t seed, float b_sigma, float lossgrad,
                                   float grad_clip, const char * tag, Yue2AtTrainCtx * C, std::string * err,
                                   bool cursor_head = false) {
    const size_t     n_tensors = yue2_at_adapter_tensor_count(n_layers, target);
    ggml_init_params aip       = { (n_tensors + 16) * ggml_tensor_overhead(), nullptr, /*no_alloc*/ true };
    C->actx                    = ggml_init(aip);
    if (!C->actx || !yue2_at_make_adapters(C->actx, m, n_layers, rank, alpha, target, &C->ad, &C->params) ||
        C->params.size() != n_tensors) {
        char b[160];
        snprintf(b, sizeof(b), "adapter allocation failed (%zu of %zu tensors)", C->params.size(), n_tensors);
        *err = b;
        return false;
    }
    if (cursor_head) {
        const int64_t H = (int64_t) m.lm_cfg.embedding_length;
        C->cursor_w     = ggml_new_tensor_2d(C->actx, GGML_TYPE_F32, H, H);
        if (!C->cursor_w) {
            *err = "cursor head allocation failed";
            return false;
        }
        ggml_set_name(C->cursor_w, "cursor_head.W");
        ggml_set_param(C->cursor_w);
        C->params.push_back(C->cursor_w);
    }
    // The five host-owned optimizer scalars. LmOptim declares every one nullptr
    // and lm_optim_init creates NONE — each is written or read unconditionally
    // inside lm_optim_step, so a missing one is a null dereference AFTER the
    // graph has computed, which reads as a fault in the optimizer maths.
    C->t_lossgrad = ggml_new_tensor_1d(C->actx, GGML_TYPE_F32, 1);
    C->t_adamw    = ggml_new_tensor_1d(C->actx, GGML_TYPE_F32, 7);
    C->t_clip     = ggml_new_tensor_1d(C->actx, GGML_TYPE_F32, 1);
    C->t_eps      = ggml_new_tensor_1d(C->actx, GGML_TYPE_F32, 1);
    C->t_gnorm2   = ggml_new_tensor_1d(C->actx, GGML_TYPE_F32, 1);
    ggml_set_name(C->t_lossgrad, "lossgrad");
    ggml_set_name(C->t_adamw, "adamw_params");
    ggml_set_name(C->t_clip, "grad_clip");
    ggml_set_name(C->t_eps, "eps");
    ggml_set_name(C->t_gnorm2, "gnorm2");

    C->abuf = ggml_backend_alloc_ctx_tensors(C->actx, m.backend);
    if (!C->abuf) {
        *err = "adapter buffer allocation failed";
        return false;
    }
    // Without the WEIGHTS usage hint the scheduler cuts the graph once per LoRA
    // parameter — 293 splits and 47.6 s/step on MM3, GPU at 9 %.
    ggml_backend_buffer_set_usage(C->abuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    yue2_nt_init_adapters(C->params, seed, b_sigma);
    if (C->cursor_w) {
        // nn.init.eye_: ggml [H_in, H_out], element (h, o) at o*H + h.
        const int64_t      H = C->cursor_w->ne[0];
        std::vector<float> I((size_t) (H * H), 0.0f);
        for (int64_t h = 0; h < H; h++) {
            I[(size_t) (h * H + h)] = 1.0f;
        }
        ggml_backend_tensor_set(C->cursor_w, I.data(), 0, I.size() * sizeof(float));
    }
    C->lossgrad = lossgrad;
    {
        const float epsv = 1e-6f;
        ggml_backend_tensor_set(C->t_lossgrad, &lossgrad, 0, sizeof(float));
        ggml_backend_tensor_set(C->t_clip, &grad_clip, 0, sizeof(float));
        ggml_backend_tensor_set(C->t_eps, &epsv, 0, sizeof(float));
    }

    C->opt.optimizer  = "adamw";
    C->opt.t_lossgrad = C->t_lossgrad;
    C->opt.t_adamw    = C->t_adamw;
    C->opt.t_clip     = C->t_clip;
    C->opt.t_eps      = C->t_eps;
    C->opt.t_gnorm2   = C->t_gnorm2;
    C->opt.grad_clip  = grad_clip;
    if (!lm_optim_init(&C->opt, C->params, m.backend, err)) {
        *err = "optimizer init: " + *err;
        return false;
    }

    BackendPair bp{};
    bp.backend     = m.backend;
    bp.cpu_backend = m.cpu_backend;
    bp.has_gpu     = m.backend != m.cpu_backend;
    C->sched       = backend_sched_new(bp, 65536);
    C->osched      = backend_sched_new(bp, 16384);
    if (!C->sched || !C->osched) {
        *err = "scheduler alloc failed";
        return false;
    }
    const int64_t n_cur = C->cursor_w ? ggml_nelements(C->cursor_w) : 0;
    fprintf(stderr, "[%s] %zu LoRA tensors over %d AR layers, rank %lld, alpha %.1f, target %s (%.1fM "
                    "trainable parameters%s)\n",
            tag, C->params.size() - (C->cursor_w ? 1 : 0), n_layers, (long long) rank, (double) alpha,
            yue2_at_target_name(target),
            (double) (yue2_at_param_count(m.lm_cfg, n_layers, rank, target) + n_cur) / 1e6,
            C->cursor_w ? ", of which 4.2M is the [H, H] cursor head" : "");
    return true;
}

// ── Deterministic per-micro-step draws ─────────────────────────────────────

static inline uint64_t yue2_at_seed_mix(uint64_t seed, uint64_t k, uint64_t tag) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ull * (k + 1) + 0xD1B54A32D192ED03ull * (tag + 1);
    z          = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z          = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

enum {
    YUE2_AT_TAG_MIX  = 1,  // artist vs minted
    YUE2_AT_TAG_SONG = 2,  // which song within the chosen pool
};

// ── Learning rate ──────────────────────────────────────────────────────────
//
// upstream's own schedule, to the step (ar_lora_cursor.py:79):
//
//     lr = LR * min(1, st/50) * (0.2 + 0.8 * 0.5 * (1 + cos(pi * min(st, SCHED)/SCHED)))
//
// Two things about it that are easy to lose in translation and both change the
// numbers: the warmup arm is `st/warmup` with st 1-BASED (so step 1 runs at
// lr/50, not at 0), and the cosine bottoms at 0.2 of base rather than at zero,
// over a horizon of SCHED_STEPS — which is 3000 while the run is 1600, so the
// cosine never reaches its floor.
//
// LmOptim's own schedule is NEUTRALISED rather than approximated: setting
// {lr_floor 1, total 1, warmup 0} makes lm_lr_lambda identically 1.0, and
// `base_lr`, set per step from here, carries the whole schedule exactly.
static double yue2_at_lr_at(const Yue2ArTrainArgs & a, int64_t step1) {
    const double lr   = (double) a.lr;
    const double warm = (a.warmup > 0) ? std::min(1.0, (double) step1 / (double) a.warmup) : 1.0;
    if (a.lr_scheduler != "cosine") {
        return lr * warm;
    }
    const double sched = (double) std::max<int64_t>(1, a.sched_steps);
    const double ratio = std::min((double) step1, sched) / sched;
    return lr * warm * (0.2 + 0.8 * 0.5 * (1.0 + std::cos(3.14159265358979 * ratio)));
}

// ── Checkpoint / resume (contract §7.4) ────────────────────────────────────
//
// IS RESUME BIT-EXACT? Yes, within one machine + build.
//
//   * PERSISTED: every LoRA factor, every AdamW moment, the optimizer's step
//     and iteration counters (opt_iter drives the bias correction, so dropping
//     it would silently re-warm Adam) and the step counter.
//   * NOT persisted: the RNG. Every draw is a pure function of (seed, micro-step
//     index, purpose), so the data stream after a resume is exactly the stream
//     the uninterrupted run would have seen.
//   * BREAKS IT: a different GPU, driver, ggml build or backend (reductions
//     reassociate), a different base GGUF or quant, or a changed manifest (song
//     order is part of the data stream). None of those is visible from inside
//     this file, so they are listed here and nowhere else.
//   * The FINGERPRINT catches everything else argv can change, and `cond_hash`
//     folds in the manifest path, the minted path, --trigger, --style,
//     --lyrics, --sidecars, --artist-frac, --max-len and --attn. Those look
//     like "just conditioning" and are not: the first several ARE the prefix
//     the model is trained to answer to, and the rest re-parameterise draws off
//     the same seeded stream, which moves the data a resume sees while leaving
//     the loss curve looking untouched.
//   * NOTE rather than refusal: --steps, --lr, --warmup, --lr-scheduler,
//     --sched-steps, --weight-decay, --max-grad-norm. Each only reshapes the
//     schedule or the update rule from here on.
//   * The state file is single-machine, host-endian and NOT a distribution
//     format.
static const char     YUE2_AT_CKPT_MAGIC[8] = { 'Y', '2', 'A', 'R', 'C', 'K', '1', '\0' };
// 2: AdamW betas joined the state. They were (0.9, 0.999) by omission and are
// now (0.9, 0.95) to match upstream, which changes the update rule — so a v1
// state must refuse rather than resume into different optimizer dynamics.
static const uint32_t YUE2_AT_CKPT_VERSION  = 2;

struct Yue2AtCkptState {
    int32_t  rank = 0, n_params = 0, n_layers = 0, target = 0;
    int32_t  grad_accum = 0, total_steps = 0;
    float    alpha = 0.0f, lr = 0.0f;
    uint64_t seed      = 0;
    uint64_t cond_hash = 0;
    int32_t  warmup = 0, lr_sched = 0, sched_steps = 0;
    float    weight_decay = 0.0f, max_grad_norm = 0.0f;
    float    adam_beta1 = 0.0f, adam_beta2 = 0.0f;
    int32_t  steps_done = 0;
    double   loss_sum   = 0.0;
    int64_t  n_micro    = 0;
    int32_t  opt_step = 0, opt_iter = 0;
};

// FNV-1a over the knobs that decide WHAT is trained rather than how fast.
// Floats go in by their exact bits, not a formatted string.
static uint64_t yue2_at_cond_hash(const Yue2ArTrainArgs & a) {
    uint64_t h         = 1469598103934665603ull;
    auto     mix_bytes = [&](const void * p, size_t n) {
        const uint8_t * b = (const uint8_t *) p;
        for (size_t i = 0; i < n; i++) {
            h ^= (uint64_t) b[i];
            h *= 1099511628211ull;
        }
    };
    auto mix_str = [&](const std::string & s) {
        const uint32_t n = (uint32_t) s.size();
        mix_bytes(&n, sizeof(n));
        mix_bytes(s.data(), s.size());
        const uint8_t sep = 0xFF;
        mix_bytes(&sep, 1);
    };
    mix_str(a.manifest);
    mix_str(a.minted);
    mix_str(a.trigger);
    mix_str(a.style_template);
    mix_str(a.style);
    mix_str(a.lyrics);
    mix_str(a.attn);
    mix_bytes(&a.sidecars, sizeof(a.sidecars));
    mix_bytes(&a.artist_frac, sizeof(a.artist_frac));
    mix_bytes(&a.max_len, sizeof(a.max_len));
    return h;
}

// `yue2_at_put`, not `yue2_at_w`: the graph header already owns `yue2_at_w` as
// the frozen-weight F32 cast, and two unrelated things under one name in one
// translation unit is how an overload set starts deciding things for you.
template <typename T>
static bool yue2_at_put(FILE * f, const T & v) {
    return fwrite(&v, sizeof(T), 1, f) == 1;
}

template <typename T>
static bool yue2_at_get(FILE * f, T * v) {
    return fread(v, sizeof(T), 1, f) == 1;
}

static bool yue2_at_put_str(FILE * f, const std::string & s) {
    const uint32_t n = (uint32_t) s.size();
    return yue2_at_put(f, n) && (n == 0 || fwrite(s.data(), 1, n, f) == n);
}

static bool yue2_at_get_str(FILE * f, std::string * s) {
    uint32_t n = 0;
    if (!yue2_at_get(f, &n) || n > 4096) {
        return false;
    }
    s->assign(n, '\0');
    return n == 0 || fread(&(*s)[0], 1, n, f) == n;
}

static bool yue2_at_put_tensor(FILE * f, ggml_tensor * t, std::vector<uint8_t> * scratch) {
    const size_t bytes = ggml_nbytes(t);
    scratch->resize(bytes);
    ggml_backend_tensor_get(t, scratch->data(), 0, bytes);
    const uint64_t n64 = (uint64_t) bytes;
    return yue2_at_put_str(f, std::string(ggml_get_name(t))) && yue2_at_put(f, n64) &&
           fwrite(scratch->data(), 1, bytes, f) == bytes;
}

static std::string yue2_at_ckpt_path(const std::string & out_dir) {
    return out_dir + "/yue2_ar_ckpt.bin";
}

// .tmp + rename, so a crash mid-write cannot leave a truncated file a resume
// would half-believe.
static bool yue2_at_ckpt_save(const std::string & path, const Yue2AtCkptState & st, const Yue2AtTrainCtx & C,
                              std::string * err) {
    const std::string tmp = path + ".tmp";
    FILE *            f   = hs_fopen(tmp, "wb");
    if (!f) {
        *err = "cannot open " + tmp + " for writing";
        return false;
    }
    std::vector<uint8_t> scratch;
    bool ok = fwrite(YUE2_AT_CKPT_MAGIC, 1, sizeof(YUE2_AT_CKPT_MAGIC), f) == sizeof(YUE2_AT_CKPT_MAGIC);
    ok      = ok && yue2_at_put(f, YUE2_AT_CKPT_VERSION);
    ok      = ok && yue2_at_put(f, st.rank) && yue2_at_put(f, st.n_params) && yue2_at_put(f, st.n_layers);
    ok      = ok && yue2_at_put(f, st.target) && yue2_at_put(f, st.grad_accum) && yue2_at_put(f, st.total_steps);
    ok      = ok && yue2_at_put(f, st.alpha) && yue2_at_put(f, st.lr) && yue2_at_put(f, st.seed);
    ok      = ok && yue2_at_put(f, st.cond_hash) && yue2_at_put(f, st.warmup) && yue2_at_put(f, st.lr_sched);
    ok      = ok && yue2_at_put(f, st.sched_steps) && yue2_at_put(f, st.weight_decay) &&
         yue2_at_put(f, st.max_grad_norm);
    ok = ok && yue2_at_put(f, st.adam_beta1) && yue2_at_put(f, st.adam_beta2);
    ok = ok && yue2_at_put(f, st.steps_done) && yue2_at_put(f, st.loss_sum) && yue2_at_put(f, st.n_micro);
    ok = ok && yue2_at_put(f, st.opt_step) && yue2_at_put(f, st.opt_iter);
    {
        const uint32_t n = (uint32_t) C.params.size();
        ok               = ok && yue2_at_put(f, n);
        for (size_t j = 0; ok && j < C.params.size(); j++) {
            ok = yue2_at_put_tensor(f, C.params[j], &scratch);
        }
    }
    {
        uint32_t n = 0;
        for (size_t j = 0; j < C.opt.mom_m.size(); j++) {
            if (C.opt.mom_m[j]) {
                n++;
            }
            if (j < C.opt.mom_v.size() && C.opt.mom_v[j]) {
                n++;
            }
        }
        ok = ok && yue2_at_put(f, n);
        for (size_t j = 0; ok && j < C.opt.mom_m.size(); j++) {
            if (C.opt.mom_m[j]) {
                ok = ok && yue2_at_put_tensor(f, C.opt.mom_m[j], &scratch);
            }
            if (ok && j < C.opt.mom_v.size() && C.opt.mom_v[j]) {
                ok = yue2_at_put_tensor(f, C.opt.mom_v[j], &scratch);
            }
        }
    }
    const uint32_t eof_marker = 0xD09EF00Du;
    ok                        = ok && yue2_at_put(f, eof_marker);
    if (fclose(f) != 0) {
        ok = false;
    }
    if (!ok) {
        *err = "write failed (disk full?) — " + tmp;
        hs_remove(tmp);
        return false;
    }
    hs_remove(path);
    if (hs_rename(tmp, path) != 0) {
        *err = "cannot rename " + tmp + " into place";
        return false;
    }
    return true;
}

static bool yue2_at_ckpt_load(const std::string & path, const Yue2AtCkptState & want, Yue2AtCkptState * got,
                              Yue2AtTrainCtx * C, std::string * err) {
    FILE * f = hs_fopen(path, "rb");
    if (!f) {
        *err = "cannot open " + path + " — --resume needs the checkpoint --save-every wrote";
        return false;
    }
    auto fail = [&](const std::string & why) {
        *err = path + ": " + why;
        fclose(f);
        return false;
    };
    char magic[sizeof(YUE2_AT_CKPT_MAGIC)];
    if (fread(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        memcmp(magic, YUE2_AT_CKPT_MAGIC, sizeof(magic)) != 0) {
        return fail("not a yue2-ar-train checkpoint");
    }
    uint32_t ver = 0;
    if (!yue2_at_get(f, &ver) || ver != YUE2_AT_CKPT_VERSION) {
        return fail("checkpoint version " + std::to_string(ver) + ", this build writes " +
                    std::to_string(YUE2_AT_CKPT_VERSION));
    }
    Yue2AtCkptState st;
    bool ok = yue2_at_get(f, &st.rank) && yue2_at_get(f, &st.n_params) && yue2_at_get(f, &st.n_layers) &&
              yue2_at_get(f, &st.target) && yue2_at_get(f, &st.grad_accum) && yue2_at_get(f, &st.total_steps) &&
              yue2_at_get(f, &st.alpha) && yue2_at_get(f, &st.lr) && yue2_at_get(f, &st.seed) &&
              yue2_at_get(f, &st.cond_hash) && yue2_at_get(f, &st.warmup) && yue2_at_get(f, &st.lr_sched) &&
              yue2_at_get(f, &st.sched_steps) && yue2_at_get(f, &st.weight_decay) &&
              yue2_at_get(f, &st.max_grad_norm) && yue2_at_get(f, &st.adam_beta1) &&
              yue2_at_get(f, &st.adam_beta2) &&
              yue2_at_get(f, &st.steps_done) && yue2_at_get(f, &st.loss_sum) &&
              yue2_at_get(f, &st.n_micro) && yue2_at_get(f, &st.opt_step) && yue2_at_get(f, &st.opt_iter);
    if (!ok) {
        return fail("truncated header");
    }
    if (st.rank != want.rank || st.n_params != want.n_params || st.n_layers != want.n_layers ||
        st.target != want.target || st.grad_accum != want.grad_accum || st.seed != want.seed ||
        st.alpha != want.alpha) {
        char b[320];
        snprintf(b, sizeof(b),
                 "was written by a different run (rank %d/%d, params %d/%d, layers %d/%d, target %d/%d, "
                 "grad-accum %d/%d, seed %llu/%llu, alpha %.3f/%.3f) — start a new --out rather than "
                 "resuming into it",
                 st.rank, want.rank, st.n_params, want.n_params, st.n_layers, want.n_layers, st.target,
                 want.target, st.grad_accum, want.grad_accum, (unsigned long long) st.seed,
                 (unsigned long long) want.seed, (double) st.alpha, (double) want.alpha);
        return fail(b);
    }
    if (st.cond_hash != want.cond_hash) {
        return fail("was written with different conditioning or a different data stream — one of "
                    "--manifest, --minted, --trigger, --style, --lyrics, --sidecars, --artist-frac, "
                    "--max-len or --attn has changed since. Put them back, or start a new --out; resuming "
                    "across them trains an adapter neither set describes");
    }
    if (st.total_steps != want.total_steps || st.lr != want.lr || st.warmup != want.warmup ||
        st.lr_sched != want.lr_sched || st.sched_steps != want.sched_steps ||
        st.weight_decay != want.weight_decay || st.max_grad_norm != want.max_grad_norm ||
        st.adam_beta1 != want.adam_beta1 || st.adam_beta2 != want.adam_beta2) {
        fprintf(stderr,
                "[yue2-ar-train] NOTE: resuming with a different schedule (--steps %d, --lr %.3g, --warmup "
                "%d, --sched-steps %d, betas %.3g/%.3g) into a checkpoint written at (%d, %.3g, %d, %d, "
                "%.3g/%.3g). The schedule and the update rule change from here; the weights do not, and "
                "the resumed run is no longer bit-identical to the uninterrupted one. Note that the AdamW "
                "moments carried in this state were ACCUMULATED under the old betas — beta2 in particular "
                "sets how long the second moment remembers, so the first steps after a beta change are "
                "running on a history that does not match the new decay.\n",
                want.total_steps, (double) want.lr, want.warmup, want.sched_steps,
                (double) want.adam_beta1, (double) want.adam_beta2, st.total_steps,
                (double) st.lr, st.warmup, st.sched_steps,
                (double) st.adam_beta1, (double) st.adam_beta2);
    }

    std::unordered_map<std::string, ggml_tensor *> live;
    for (ggml_tensor * t : C->params) {
        live[ggml_get_name(t)] = t;
    }
    for (size_t j = 0; j < C->opt.mom_m.size(); j++) {
        if (C->opt.mom_m[j]) {
            live[ggml_get_name(C->opt.mom_m[j])] = C->opt.mom_m[j];
        }
        if (j < C->opt.mom_v.size() && C->opt.mom_v[j]) {
            live[ggml_get_name(C->opt.mom_v[j])] = C->opt.mom_v[j];
        }
    }
    std::vector<uint8_t> scratch;
    int                  restored = 0;
    for (int block = 0; block < 2; block++) {
        uint32_t n = 0;
        if (!yue2_at_get(f, &n)) {
            return fail("truncated tensor block header");
        }
        for (uint32_t i = 0; i < n; i++) {
            std::string name;
            uint64_t    bytes = 0;
            if (!yue2_at_get_str(f, &name) || !yue2_at_get(f, &bytes)) {
                return fail("truncated tensor record");
            }
            auto lit = live.find(name);
            if (lit == live.end()) {
                return fail("holds tensor \"" + name + "\", which this run does not have");
            }
            if (ggml_nbytes(lit->second) != (size_t) bytes) {
                return fail("tensor \"" + name + "\" is the wrong size");
            }
            scratch.resize((size_t) bytes);
            if (fread(scratch.data(), 1, (size_t) bytes, f) != (size_t) bytes) {
                return fail("truncated data for \"" + name + "\"");
            }
            ggml_backend_tensor_set(lit->second, scratch.data(), 0, (size_t) bytes);
            restored++;
        }
    }
    uint32_t eof_marker = 0;
    if (!yue2_at_get(f, &eof_marker) || eof_marker != 0xD09EF00Du) {
        return fail("no end marker — the file is truncated");
    }
    fclose(f);
    if (restored != (int) live.size()) {
        *err = path + ": restored " + std::to_string(restored) + " of " + std::to_string(live.size()) +
               " live tensors — a partial restore is indistinguishable from training on garbage";
        return false;
    }
    C->opt.opt_step = st.opt_step;
    C->opt.opt_iter = st.opt_iter;
    *got            = st;
    return true;
}

// ── Export (contract §7.1) ─────────────────────────────────────────────────
//
// THE HALF THAT MUST NOT BE WRONG. The key spellings are the ones
// yue2-adapter.h parses, checked against THAT FILE rather than against the plan
// (10-adapter-merge-notes.md §3 exists because the plan and the code disagreed
// on both sides once already):
//
//   * `yue2.` prefix     -> stripped by yue2_lora_target (yue2-adapter.h:264)
//   * `.lora_A.weight` /
//     `.lora_B.weight`   -> the FIRST entry of each list in yue2_lora_suffix
//                           (:183-186), i.e. the canonical spelling. The
//                           lowercase `.lora_a` form is tolerated legacy; do
//                           not write it.
//   * CAPITALISATION IS LOAD-BEARING on both sides: yue2_nt_init_adapters keys
//     kaiming-vs-zero on the tensor name's last character, and the parser's A/B
//     lists are case-sensitive.
//   * module names       -> `blk.N.attn_{q,k,v,output}`, `blk.N.ffn_{gate,up,down}`
//                           — the AR twins. Block index decimal and UNPADDED.
//
// SHAPES. safetensors shapes are PyTorch order, outermost first; ggml's `ne` is
// the reverse. A ggml A of [in, rank] IS torch [rank, in]; a ggml B of
// [rank, out] IS torch [out, rank]. The BYTES need no transposition, only the
// shape array reversed — and yue2-adapter.h's preflight checks exactly that
// (A.shape[1] == base ne0, B.shape[0] == base ne1, B.shape[1] == A.shape[0]),
// reading the ACTUAL base shape, which is what makes it correct for attn_k/v at
// [2048, 1024] under GQA.
//
// ── THE LOADER SIDE, AND WHY `format` IS NOT DECORATION ────────────────────
//
// `yue2_lora_site_family` (yue2-adapter.h) accepted `nar_*` sites only until
// phase 7 (contract §7.3) extended it with the seven AR spellings, added
// `yue2-ar-lora.safetensors` to the directory probe, and GATED THE FAMILY ON
// `__metadata__.format`. That gate is the replacement for the guard the
// extension removed: both halves' sites are now legal keys, so a single
// mis-spelled `nar_` would otherwise land on whichever half the trainer did not
// touch, silently. The loader refuses a key whose family contradicts `format`,
// and refuses an unlabelled file that names both halves.
//
// So `format: "yue2-ar-lora-v1"` below is load-bearing, not a label. Write it,
// and do not write a NAR site from this exporter.
// `cursor_md` is the metadata value for `cursor`: "off", or
// "on:w=<weight>,bound=<n>/<m>". The cursor head itself is NOT exported: it has
// no inference role (nothing at generation time scores frames against lyric
// tokens), and the loader would refuse an unknown key.
static bool yue2_at_export(const Yue2AtAdapters & ad, const Yue2ArTrainArgs & a, Yue2AtTarget target,
                           int64_t steps_done, int64_t song_frames, const std::string & base_id,
                           bool minted_present, const std::string & path, std::string * err,
                           const std::string & cursor_md = "off") {
    struct Ent {
        std::string         name;
        const ggml_tensor * t;
    };
    std::vector<Ent> ents;
    auto             add_site = [&](const Yue2AtLora & lo, const std::string & mod) {
        if (!lo.on()) {
            return;
        }
        ents.push_back({ "yue2." + mod + ".lora_A.weight", lo.a });
        ents.push_back({ "yue2." + mod + ".lora_B.weight", lo.b });
    };
    for (size_t i = 0; i < ad.blk.size(); i++) {
        const std::string p = "blk." + std::to_string((long long) i) + ".";
        add_site(ad.blk[i].q, p + "attn_q");
        add_site(ad.blk[i].k, p + "attn_k");
        add_site(ad.blk[i].v, p + "attn_v");
        add_site(ad.blk[i].o, p + "attn_output");
        add_site(ad.blk[i].gate, p + "ffn_gate");
        add_site(ad.blk[i].up, p + "ffn_up");
        add_site(ad.blk[i].down, p + "ffn_down");
    }
    if (ents.empty()) {
        *err = "nothing to export: no LoRA site is active";
        return false;
    }

    // The blobs must outlive st_write_file, and STWTensor holds a bare pointer
    // into them — reserve so no push_back can reallocate the storage out from
    // under an already-recorded pointer.
    std::vector<std::vector<float>> blobs;
    blobs.reserve(ents.size());
    std::vector<STWTensor> tens;
    tens.reserve(ents.size());
    for (const Ent & e : ents) {
        const int64_t n0 = e.t->ne[0], n1 = e.t->ne[1];
        blobs.emplace_back((size_t) (n0 * n1), 0.0f);
        ggml_backend_tensor_get(const_cast<ggml_tensor *>(e.t), blobs.back().data(), 0,
                                (size_t) (n0 * n1) * sizeof(float));
        STWTensor t;
        t.name  = e.name;
        t.shape = { n1, n0 };  // torch order = ggml ne reversed
        t.data  = blobs.back().data();
        tens.push_back(std::move(t));
    }

    char                                             buf[64];
    std::vector<std::pair<std::string, std::string>> md;
    // DIFFERENT from the NAR exporter's "yue2-nar-lora-v1", and that difference
    // is what contract §7.3's format gate keys on.
    md.emplace_back("format", "yue2-ar-lora-v1");
    snprintf(buf, sizeof(buf), "%lld", (long long) a.rank);
    md.emplace_back("rank", buf);
    snprintf(buf, sizeof(buf), "%.6f", (double) a.alpha);
    md.emplace_back("alpha", buf);
    md.emplace_back("targets", yue2_at_target_name(target));
    snprintf(buf, sizeof(buf), "%lld", (long long) steps_done);
    md.emplace_back("steps", buf);
    snprintf(buf, sizeof(buf), "%lld", (long long) song_frames);
    md.emplace_back("song_frames", buf);
    md.emplace_back("trigger", a.trigger);
    md.emplace_back("style_template", a.style_template);
    // No sha is computed anywhere in this tree, so `base_sha` carries the
    // identity we actually have: the base LM file it trained against. An
    // invented hash would be worse than none.
    md.emplace_back("base_sha", base_id);
    md.emplace_back("cot", "off");
    md.emplace_back("minted", minted_present ? "present" : "absent");
    md.emplace_back("cursor", cursor_md);
    // Recorded because it is a real recipe knob and adapters outlive their logs.
    snprintf(buf, sizeof(buf), "%.4g,%.4g", (double) a.adam_beta1, (double) a.adam_beta2);
    md.emplace_back("adam_betas", buf);
    if (a.steps > 1500) {
        md.emplace_back("overtrain", "acknowledged");
    }

    if (!st_write_file(path.c_str(), tens, md, STW_F32)) {
        *err = "safetensors write failed for " + path;
        return false;
    }
    fprintf(stderr, "[yue2-ar-train] exported %zu tensors -> %s\n", tens.size(), path.c_str());
    return true;
}

// ── The banner (contract §5.3) ─────────────────────────────────────────────

static void yue2_at_no_minted_banner() {
    fprintf(stderr,
            "\n"
            "[yue2-ar-train] ================= REGULARIZER ABSENT =================\n"
            "[yue2-ar-train] Training 100%% on encoder-predicted codes, whose adjacent-repeat rate is\n"
            "[yue2-ar-train] 3.3-6.6%% against YuE2's own 0.02-0.17%%\n"
            "[yue2-ar-train] (docs/plans/yue2/14-tokenizer-truth-gate.md §4). This is out of\n"
            "[yue2-ar-train] distribution in the direction that produces LOOPING. There is no\n"
            "[yue2-ar-train] minted_val loss to watch. EXPECT LOOPING; do not read a falling artist\n"
            "[yue2-ar-train] loss as evidence against it.\n"
            "[yue2-ar-train] The pack is Mothersuperior/yue2-minted-corpus ->\n"
            "[yue2-ar-train] minted_regularizer_pack.pt (~100 MB, 4,732 songs), converted to a\n"
            "[yue2-ar-train] sources[] manifest and passed as --minted.\n"
            "[yue2-ar-train] ======================================================\n\n");
}

// ── The FD gate (contract §6) ──────────────────────────────────────────────

// ── THE STEP FLOOR, AND WHY IT IS NOT A FRACTION OF THE LOSS ───────────────
//
// The floor answers one question: how big must the loss CHANGE be before a
// central difference of it means anything? Call the forward's numerical
// resolution `res` — the size of the jitter between two evaluations at nearby
// weights. The measured derivative is
//
//     num = (L(+h) - L(-h)) / (2h),        error ~ res*sqrt(2) / (2h)
//
// and the gate scores `rel = |num - ||g||| / ||g||`, so the resolution enters
// the SCORE as
//
//     rel_res ~ res / (2h*||g||) = res / dL.
//
// The floor is therefore a demand on `dL / res`, a signal-to-resolution ratio,
// and nothing else. That ratio is what YUE2_AT_FD_SNR names.
//
// The NAR gate (and this one, until now) wrote the floor as
// `max(1e-4, 1e-3*|L0|)`, inherited from mm3-lm-train-run.h. It is not wrong
// there by luck: the NAR loss sits near 2.67, so 1e-3*|L0| = 2.7e-3, which
// happens to be ~2000 res at that magnitude. It is wrong HERE, and the reason
// is that |L0| is not a property of the arithmetic at all. A zero-init adapter
// over a sliced head of V rows starts at ln(V): this trainer's slice is
// 32769 rows, so L0 ~ 11.24, four times the NAR's — and the floor scaled with
// it, demanding a loss change four times larger out of a forward whose
// resolution did not change one bit. The result was measured: 6 of 7 probes
// hit the 0.05*||w|| linearity cap and came back INCONCLUSIVE, and the gate
// announced "PASS, 1/1 measurable probes". Widen the vocabulary slice again
// and the same correct backward would report zero measurable probes.
//
// So: measure `res` and multiply. `res` is estimated below by evaluating the
// forward at +/-h_res along the gradient direction with h_res = 1e-6*||w||,
// chosen so the TRUE loss change it induces (2*h_res*||g||, ~6e-8 here) sits
// well under one f32 ulp of L0 while the per-entry weight motion is still
// thousands of ulps — so whatever difference comes back is jitter, not signal.
// FLT_EPSILON*|L0| (1.3e-6 at L0 = 11.24) is kept as a hard floor under the
// estimate: the loss is reported and differenced at f32-sourced magnitudes, and
// claiming to resolve better than one ulp of the value itself would be a claim
// this gate cannot support.
//
// (The gate also prints an in-graph-vs-host-double delta, 1.07e-06 on the
// reference run. It is the same order, which is a useful corroboration — but it
// is a cross-IMPLEMENTATION delta, and the numeric arm of the FD never touches
// the in-graph kernel, so it is reported as a cross-check and not used as the
// estimate. The direct measurement below is on exactly the path the FD uses.)
//
// YUE2_AT_FD_SNR = 1000 holds rel_res at ~1e-3, one twentieth of the 2e-2
// isolated bar, so the resolution can never be what decides a verdict. It is
// NOT a tolerance and must not be traded against the bar: raising it makes
// probes cap (fewer measurements, same bar), lowering it makes measurements
// noisier (same count, looser measurements). The bar stays at 2e-2 either way.
static const double YUE2_AT_FD_SNR = 1000.0;

// Relative size of the weight perturbation used to MEASURE the forward's
// resolution: ||w||*1e-6 along g/||g||. Large enough that every entry moves by
// thousands of f32 ulps (so the forward really is re-evaluated), small enough
// that the true loss change it causes is ~20x below one ulp of L0 (so the
// difference that comes back is jitter). The gate prints both numbers and says
// so, rather than asking anyone to take that on trust.
static const double YUE2_AT_FD_RES_REL = 1e-6;

static int yue2_ar_fdcheck_main(const Yue2ArTrainArgs & a) {
    // TF32 turns an F32 matmul into a ~1e-3-accurate one, which is the same
    // order as the defect the gate looks for.
#ifdef _WIN32
    _putenv_s("NVIDIA_TF32_OVERRIDE", "0");
#else
    setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
#endif

    std::string err;
    static Yue2Model m;
    if (!yue2_at_open_model(&m, a, "yue2-ar-fd", &err)) {
        fprintf(stderr, "[yue2-ar-fd] %s\n", err.c_str());
        return 1;
    }
    const Yue2LmConfig & c = m.lm_cfg;

    Yue2AtTarget target = YUE2_AT_T_ATTN_MLP;
    std::string  why;
    if (!yue2_at_parse_target(a.target, &target, &why)) {
        fprintf(stderr, "[yue2-ar-fd] %s\n", why.c_str());
        return 2;
    }
    Yue2AtAttnMode attn = YUE2_AT_FA_EXACT;
    if (!yue2_at_parse_attn(a.attn, &attn)) {
        fprintf(stderr, "[yue2-ar-fd] --attn must be exact, flash or flash-f32\n");
        return 2;
    }

    fprintf(stderr, "[yue2-ar-fd] the gate uses a SYNTHETIC codec stream; it tests the GRAPH, not the data, "
                    "so no manifest is read here\n");

    // ── conditioning: the cot=off prefix, exactly as inference builds it ──
    BPETokenizer tok;
    if (!yue2_tokenizer_load_from_gguf(&tok, m.lm_file.path)) {
        fprintf(stderr, "[yue2-ar-fd] tokenizer: cannot read tokenizer.ggml.* from %s\n",
                m.lm_file.path.c_str());
        return 1;
    }
    std::vector<int32_t> prefix;
    try {
        const std::vector<int> pre = yue2_token_prefixes(&tok, a.style, a.lyrics, YUE2_COT_OFF, nullptr);
        prefix.assign(pre.begin(), pre.end());
    } catch (const std::exception & e) {
        fprintf(stderr, "[yue2-ar-fd] prefix assembly failed: %s\n", e.what());
        return 1;
    }

    // A synthetic RAW code stream from the same seeded generator the adapters
    // use, so --seed 42 means the same gate on every build.
    std::vector<int32_t> codec((size_t) std::max<int64_t>(8, a.fd_frames), 0);
    {
        Yue2NtRng rng(a.seed ^ 0x5Aull);
        for (size_t i = 0; i < codec.size(); i++) {
            codec[i] = (int32_t) (rng.u01() * (double) YUE2_CODEC_SIZE) % (int32_t) YUE2_CODEC_SIZE;
        }
    }
    Yue2AtSeq seq;
    if (!yue2_at_build_sequence(prefix, codec, a.max_len, &seq, &err)) {
        fprintf(stderr, "[yue2-ar-fd] %s\n", err.c_str());
        return 1;
    }
    const int64_t S = (int64_t) seq.ids.size();
    fprintf(stderr, "[yue2-ar-fd] prefix %lld ids + %zu codec + MUSIC_END -> S %lld, supervised %lld\n",
            (long long) seq.prefix, codec.size(), (long long) S, (long long) seq.n_sup);

    // ── F32 isolation ──
    Yue2AtF32Slice iso;
    bool           isolated = false;
    const int      n_layers = a.ar_layers > 0 ? std::min(a.ar_layers, (int) c.block_count) : (int) c.block_count;
    if (a.ar_layers > 0) {
        if (!yue2_at_f32_isolate(&m, n_layers, &iso, &err)) {
            fprintf(stderr, "[yue2-ar-fd] %s\n", err.c_str());
            yue2_at_f32_free(&iso);
            return 1;
        }
        isolated = true;
    } else {
        fprintf(stderr, "[yue2-ar-fd] --ar-layers 0: no F32 isolation and the full 28-layer stack, so this "
                        "run REPORTS, it does not gate\n");
    }
    if (a.rank < 64) {
        // Contract §6.4, measured and documented at 08-nar-lora-trainer.md:14-18
        // and fixed into an honest verdict by 44ef43f5: a B initialised at
        // sigma 1e-2 has ||w|| ~ 1e-2*sqrt(n), so at low rank the 5%-of-||w||
        // step cap lands below the step needed to clear the loss floor and
        // every probe comes back INCONCLUSIVE.
        fprintf(stderr,
                "[yue2-ar-fd] WARNING: --rank %lld. The gate's default is 64 for a measured reason — at low "
                "rank the step cap (0.05*||w||) sits BELOW the step needed to move the loss, and every "
                "probe returns INCONCLUSIVE rather than a verdict.\n",
                (long long) a.rank);
    }

    // ── adapters, optimizer, scheduler ──
    //
    // B NON-ZERO for the gate (b_sigma 1e-2). With B == 0 (the production init)
    // dL/dA is identically zero and every .A probe passes while measuring
    // nothing.
    //
    // YUE2_FD_LOSSGRAD is the negative control: seed dL/dloss at 2.0 and every
    // probe must land at rel ~= 0.5.
    float        lg = 1.0f;
    const char * ev = std::getenv("YUE2_FD_LOSSGRAD");
    if (ev && ev[0]) {
        lg = (float) atof(ev);
        fprintf(stderr, "[yue2-ar-fd] NEGATIVE CONTROL: dL/dloss seeded at %.3f — expect rel ~= %.3f on "
                        "every probe, and a FAIL\n",
                (double) lg, (double) std::fabs(1.0f - lg) / (double) std::max(1e-6f, std::fabs(lg)));
    }
    // ── the cursor term in the gate: SYNTHETIC spans over the REAL prefix ──
    //
    // The gate tests the graph, not the data, so the word spans are made up:
    // six words spread evenly over the synthetic stream's duration and over
    // the lyric sheet's characters. That still exercises the real target
    // builder (tokenisation, codepoint offsets, carry-forward) and, more to the
    // point, puts a non-trivial [L, nF] distribution behind four hand-derived
    // transposes so `cursor_head.W` can be probed like any LoRA factor.
    const bool   cursor_on = a.cursor_weight > 0.0 && !a.lyrics.empty();
    Yue2AtCursor fd_cur;
    if (cursor_on) {
        const int64_t      nW = 6, nch = yue2_at_utf8_len(a.lyrics);
        const double       dur = (double) codec.size() / 25.0;
        std::vector<float> w5((size_t) (nW * 5), 0.0f);
        for (int64_t w = 0; w < nW; w++) {
            float * r5 = w5.data() + (size_t) w * 5;
            r5[0] = (float) (dur * (double) w / (double) nW);
            r5[1] = r5[0] + 0.3f;
            r5[2] = 0.5f;
            r5[3] = (float) (nch * w / nW);
            r5[4] = (float) (nch * (w + 1) / nW);
        }
        fd_cur = yue2_at_cursor_build(&tok, a.style, a.lyrics, prefix, (int64_t) codec.size(), w5, seq.n_sup);
        if (!fd_cur.bound) {
            fprintf(stderr, "[yue2-ar-fd] cursor: could not bind synthetic spans (%s) — the term is NOT "
                            "under test\n",
                    fd_cur.why.c_str());
        } else {
            fprintf(stderr, "[yue2-ar-fd] cursor: weight %.3f, %lld lyric tokens at [%lld, %lld), %lld "
                            "frames — cursor_head.W joins the probes\n",
                    a.cursor_weight, (long long) fd_cur.L, (long long) fd_cur.j0,
                    (long long) (fd_cur.j0 + fd_cur.L), (long long) fd_cur.nF);
        }
    } else if (a.cursor_weight > 0.0) {
        fprintf(stderr, "[yue2-ar-fd] cursor: --lyrics is empty, so there is no lyric span to point at — "
                        "the term is NOT under test\n");
    }
    const bool cursor_live = cursor_on && fd_cur.bound;

    Yue2AtTrainCtx C;
    if (!yue2_at_train_ctx_init(m, n_layers, a.rank, a.alpha, target, a.seed, /*b_sigma=*/1e-2f,
                                /*lossgrad=*/lg, /*grad_clip=*/1.0f, "yue2-ar-fd", &C, &err,
                                /*cursor_head=*/cursor_live)) {
        fprintf(stderr, "[yue2-ar-fd] %s\n", err.c_str());
        return 1;
    }

    if (attn != YUE2_AT_FA_EXACT && !yue2_at_flash_probe(m.backend, c, S, &err)) {
        fprintf(stderr, "[yue2-ar-fd] %s\n", err.c_str());
        return 1;
    }

    Yue2AtState st;
    if (!yue2_at_state_alloc(&st, &m, S, a.chunk, C.sched, &err, cursor_live ? fd_cur.L : 0)) {
        fprintf(stderr, "[yue2-ar-fd] %s\n", err.c_str());
        return 1;
    }

    Yue2AtRun r;
    r.m                = &m;
    r.opt              = &C.opt;
    r.sched            = C.sched;
    r.st               = &st;
    r.ad               = &C.ad;
    r.opts.attn        = attn;
    r.opts.weights_f32 = a.weights_f32;
    r.grad_accum       = 1;
    r.n_layers         = n_layers;
    r.cur              = cursor_live ? &fd_cur : nullptr;
    r.cur_w            = C.cursor_w;
    r.cursor_weight    = (float) a.cursor_weight;
    // The negative control's ONLY effective statement. Without this line the
    // seed reaches a device scalar the backward never reads, every probe comes
    // back byte-identical to the positive run, and the gate cannot fail.
    r.lossgrad         = C.lossgrad;
    // Contract §6.5's other control. The gate's stream is SYNTHETIC, so the
    // shifted loss here is a weak signal by construction — random codes carry
    // no information for a neighbouring hidden state to lose. `--forward-check`
    // runs the same control over a real song, which is where it has teeth.
    r.sup_shift        = yue2_at_sup_shift_env();

    // ── numeric arm: forward only, loss re-aggregated ON THE HOST IN DOUBLE ──
    //
    // The in-graph ggml_cross_entropy_loss is deterministic to the bit but only
    // ~1e-4 accurate, and differencing two such values is catastrophic
    // cancellation (mm3-lm-train-run.h:1445-1487 measured it). Aggregating in
    // double removes that floor and, as a bonus, makes the numeric arm an
    // INDEPENDENT implementation of the loss rather than the same kernel twice.
    // The total the gate differences is upstream's `lm + CUR_W * cl` — both
    // terms on the host in double, the cursor's via Yue2AtRun::host_cur.
    double host_ce = 0.0, host_cur = 0.0;
    auto   forward_loss = [&]() -> double {
        r.forward_only = true;
        r.host_ce      = &host_ce;
        r.host_cur     = cursor_live ? &host_cur : nullptr;
        std::string e2;
        if (!yue2_at_micro_step(r, seq, /*count_loss=*/false, nullptr, &e2)) {
            fprintf(stderr, "[yue2-ar-fd] forward failed: %s\n", e2.c_str());
            return std::nan("");
        }
        r.host_cur = nullptr;
        return host_ce + (cursor_live ? a.cursor_weight * host_cur : 0.0);
    };

    const double l0 = forward_loss();
    const double l1 = forward_loss();
    if (std::isnan(l0) || std::isnan(l1)) {
        return 1;
    }
    fprintf(stderr, "[yue2-ar-fd] base loss %.6f (repeat %.6f, |delta| %.2e); ln(%lld) = %.4f is what a "
                    "zero-init adapter should sit near\n",
            l0, l1, std::fabs(l1 - l0), (long long) YUE2_AT_SLICE_ROWS,
            std::log((double) YUE2_AT_SLICE_ROWS));

    // ── analytic arm: one forward + backward ──
    ggml_backend_buffer_clear(C.opt.buf_grad, 0);
    r.forward_only = false;
    r.host_ce      = nullptr;
    double ce_graph = 0.0, cur_graph = std::nan("");
    if (!yue2_at_micro_step(r, seq, /*count_loss=*/true, &ce_graph, &err, &cur_graph)) {
        fprintf(stderr, "[yue2-ar-fd] backward failed: %s\n", err.c_str());
        return 1;
    }
    // Compare like with like: the host-double figure is upstream's total,
    // lm + CUR_W * cursor, so the in-graph one must be too.
    const double total_graph = ce_graph + (cursor_live && cur_graph == cur_graph ? a.cursor_weight * cur_graph : 0.0);
    fprintf(stderr, "[yue2-ar-fd] in-graph loss %.6f (lm %.6f%s) vs host-double %.6f (|delta| %.2e)\n",
            total_graph, ce_graph, cursor_live ? " + w*cursor" : "", l0, std::fabs(total_graph - l0));

    auto find_param = [&](const char * name) -> ggml_tensor * {
        for (ggml_tensor * t : C.params) {
            const char * nm = ggml_get_name(t);
            if (nm && strcmp(nm, name) == 0) {
                return t;
            }
        }
        return nullptr;
    };
    auto grad_vec = [&](ggml_tensor * par) -> std::vector<float> {
        auto it = C.opt.param_slot.find(par);
        GGML_ASSERT(it != C.opt.param_slot.end());
        ggml_tensor *      acc = C.opt.acc[(size_t) it->second];
        std::vector<float> g((size_t) ggml_nelements(acc));
        ggml_backend_tensor_get(acc, g.data(), 0, g.size() * sizeof(float));
        return g;
    };

    // Contract §6.3's seven, in order. Blocks 0 and 1 rather than 0/13/27
    // because --ar-layers 2 is what makes the check a verdict. The .A/.B mix is
    // deliberate: they exercise different arms of the LoRA branch's backward.
    // The cursor head goes SECOND when it is live: first stays a LoRA factor so
    // the resolution measurement below is the usual one, and second guarantees
    // the head is probed at any --fd-check >= 2 rather than falling off the
    // end of a six-probe default.
    std::vector<const char *> probe_names = { "blk.0.attn_q.A" };
    if (cursor_live) {
        probe_names.push_back("cursor_head.W");
    }
    for (const char * nm : { "blk.0.attn_q.B", "blk.0.ffn_down.B", "blk.1.attn_v.B", "blk.1.ffn_gate.A",
                             "blk.1.attn_output.B", "blk.0.ffn_up.A" }) {
        probe_names.push_back(nm);
    }
    const int n_candidates = (int) probe_names.size();
    const int n_want       = a.fd_check > 0 ? a.fd_check : (cursor_live ? 7 : 6);
    struct Probe {
        const char *  name;
        ggml_tensor * par;
    };
    std::vector<Probe> probes;
    for (int i = 0; i < n_candidates && (int) probes.size() < n_want; i++) {
        ggml_tensor * p = find_param(probe_names[i]);
        if (!p) {
            fprintf(stderr, "[yue2-ar-fd] probe '%s' skipped: no such site under --target %s / --ar-layers "
                            "%d\n",
                    probe_names[i], yue2_at_target_name(target), n_layers);
            continue;
        }
        probes.push_back({ probe_names[i], p });
    }
    if (probes.empty()) {
        fprintf(stderr, "[yue2-ar-fd] no probe has a tensor to address — nothing was checked\n");
        return 1;
    }

    // THE STEP IS PER PROBE and --fd-eps is only its FLOOR. What must clear the
    // forward's own resolution is the LOSS CHANGE, which along the unit
    // gradient direction is exactly 2*h*||g||; a fixed eps measures every probe
    // at a different signal-to-noise ratio, in direct proportion to ||g||.
    //
    // The floor itself is derived from the forward's MEASURED resolution, not
    // from |l0| — the full derivation is on YUE2_AT_FD_SNR above, and the
    // measurement it needs is taken here.
    double res_meas   = 0.0;   // observed |L(+h_res) - L(-h_res)|
    double res_signal = 0.0;   // the true loss change inside it, 2*h_res*||g||
    {
        const Probe &            pr = probes[0];
        const std::vector<float> g  = grad_vec(pr.par);
        std::vector<float>       w0((size_t) ggml_nelements(pr.par)), wtmp(w0.size());
        ggml_backend_tensor_get(pr.par, w0.data(), 0, w0.size() * sizeof(float));
        double gn2 = 0.0, wn2 = 0.0;
        for (float x : g) {
            gn2 += (double) x * (double) x;
        }
        for (float x : w0) {
            wn2 += (double) x * (double) x;
        }
        const double gnorm = std::sqrt(gn2);
        const double wnorm = std::sqrt(wn2);
        const double h_res = YUE2_AT_FD_RES_REL * wnorm;
        if (gnorm > 0.0 && h_res > 0.0) {
            for (size_t k = 0; k < w0.size(); k++) {
                wtmp[k] = (float) ((double) w0[k] + h_res * (double) g[k] / gnorm);
            }
            ggml_backend_tensor_set(pr.par, wtmp.data(), 0, wtmp.size() * sizeof(float));
            const double lp = forward_loss();
            for (size_t k = 0; k < w0.size(); k++) {
                wtmp[k] = (float) ((double) w0[k] - h_res * (double) g[k] / gnorm);
            }
            ggml_backend_tensor_set(pr.par, wtmp.data(), 0, wtmp.size() * sizeof(float));
            const double lm_ = forward_loss();
            ggml_backend_tensor_set(pr.par, w0.data(), 0, w0.size() * sizeof(float));
            if (!std::isnan(lp) && !std::isnan(lm_)) {
                res_meas = std::fabs(lp - lm_);
            }
            res_signal = 2.0 * h_res * gnorm;
        }
    }
    // One f32 ulp of the loss magnitude is the hard floor: the loss is
    // aggregated in double, but out of f32 logits and at an f32-sourced
    // magnitude, so resolving better than that is not a claim this gate can
    // support. NOT a fraction of |l0| chosen for taste — it is FLT_EPSILON,
    // and it moves with |l0| only because an ulp does.
    const double res_ulp = (double) FLT_EPSILON * std::fabs(l0);
    const double res     = std::max(res_meas, res_ulp);
    const double dl_min  = YUE2_AT_FD_SNR * res;
    fprintf(stderr,
            "[yue2-ar-fd] forward resolution: |dL| %.2e measured at a perturbation whose TRUE loss change "
            "is %.2e; one f32 ulp of l0 is %.2e; taking %.2e\n",
            res_meas, res_signal, res_ulp, res);
    if (res_meas > 0.0 && res_signal > 0.25 * res_meas) {
        // Honest about which way the estimate errs. If the deliberately tiny
        // perturbation's own signal is a real part of what came back, the
        // forward resolves BETTER than this number and the floor is
        // conservative — probes may cap that did not need to.
        fprintf(stderr, "[yue2-ar-fd]   NOTE: that measurement is signal-dominated, so the true resolution "
                        "is BELOW it and the floor below is conservative.\n");
    }
    fprintf(stderr,
            "[yue2-ar-fd] step floor: dL >= %.0f * resolution = %.2e (NOT a fraction of l0 = %.3f — see "
            "YUE2_AT_FD_SNR); a probe whose 2*eps*||g|| clears it keeps eps %.3g, and a raised step is "
            "capped at 0.05*||w||\n",
            YUE2_AT_FD_SNR, dl_min, l0, a.fd_eps);

    fprintf(stderr, "\n[yue2-ar-fd] %-24s %10s %13s %9s %9s %13s %8s\n", "probe (whole tensor)", "n", "||g||",
            "step", "h/||w||", "numeric", "rel");
    double              worst    = 0.0;
    int                 n_raised = 0;
    std::vector<bool>   fd_clamped;
    std::vector<double> fd_rel;
    for (const Probe & pr : probes) {
        const std::vector<float> g     = grad_vec(pr.par);
        double                   norm2 = 0.0;
        for (float x : g) {
            norm2 += (double) x * (double) x;
        }
        const double gnorm = std::sqrt(norm2);

        // PERTURB A DIRECTION, NOT A SINGLE ENTRY. Along v = g/||g|| the
        // directional derivative is exactly ||g|| and the loss change is
        // ~2*h*||g||, thousands of times the f32 floor; a per-entry difference
        // measures rounding only.
        std::vector<float> w0((size_t) ggml_nelements(pr.par)), wtmp(w0.size());
        ggml_backend_tensor_get(pr.par, w0.data(), 0, w0.size() * sizeof(float));
        double wnorm2 = 0.0;
        for (float x : w0) {
            wnorm2 += (double) x * (double) x;
        }
        const double wnorm = std::sqrt(wnorm2);

        double h       = a.fd_eps;
        bool   clamped = false;
        if (gnorm > 0.0 && 2.0 * a.fd_eps * gnorm < dl_min) {
            h = dl_min / (2.0 * gnorm);
            n_raised++;
            const double h_max = 0.05 * wnorm;
            if (wnorm > 0.0 && h > h_max) {
                h       = h_max;
                clamped = true;
            }
        }
        const double h_over_w = wnorm > 0.0 ? h / wnorm : std::nan("");

        double num = std::nan("");
        if (gnorm > 0.0) {
            for (size_t k = 0; k < w0.size(); k++) {
                wtmp[k] = (float) ((double) w0[k] + h * (double) g[k] / gnorm);
            }
            ggml_backend_tensor_set(pr.par, wtmp.data(), 0, wtmp.size() * sizeof(float));
            const double lp = forward_loss();
            for (size_t k = 0; k < w0.size(); k++) {
                wtmp[k] = (float) ((double) w0[k] - h * (double) g[k] / gnorm);
            }
            ggml_backend_tensor_set(pr.par, wtmp.data(), 0, wtmp.size() * sizeof(float));
            const double lm_ = forward_loss();
            num              = (lp - lm_) / (2.0 * h);
        }
        ggml_backend_tensor_set(pr.par, w0.data(), 0, w0.size() * sizeof(float));

        const double rel = std::fabs(num - gnorm) / std::max(1e-12, gnorm);
        fprintf(stderr, "[yue2-ar-fd] %-24s %10zu %13.6e %9.3g %9.3g %13.6e %8.3f%s\n", pr.name, g.size(),
                gnorm, h, h_over_w, num, rel,
                clamped ? "  <- step CAPPED at 0.05*||w||: could not reach the signal floor" : "");
        fd_rel.push_back(rel);
        fd_clamped.push_back(clamped);
        if (!clamped) {
            worst = std::max(worst, rel);
        }
    }

    // The bar is MM3's, unchanged, and it is a VERDICT only under isolation.
    // Central differencing carries a genuine O(h^2 * f''') truncation error
    // that no amount of precision removes; 2e-2 leaves room for probe-to-probe
    // variation without admitting a real scale error — a wrong gradient scale
    // misses by a FACTOR, not by a percent.
    const double bar            = isolated ? 2e-2 : 0.15;
    int          n_bad          = 0;
    int          n_inconclusive = 0;
    for (size_t i = 0; i < fd_rel.size(); i++) {
        // A CAPPED PROBE IS INCONCLUSIVE, NOT A FAILURE. Counting it as bad
        // would print "do NOT train past it" over a measurement that says
        // nothing. NaN fails: `!(r < bar)` rather than `r >= bar`.
        if (fd_clamped[i]) {
            n_inconclusive++;
            continue;
        }
        if (!(fd_rel[i] < bar)) {
            n_bad++;
        }
    }
    const int n_checked = (int) fd_rel.size() - n_inconclusive;

    if (n_inconclusive) {
        fprintf(stderr,
                "\n[yue2-ar-fd] %d/%zu probes INCONCLUSIVE: the step needed to move the loss by %.2e would\n"
                "[yue2-ar-fd]   have exceeded 5%% of the tensor's own norm, which is outside the linear\n"
                "[yue2-ar-fd]   regime a central difference assumes. Raise --rank (64 is the default for\n"
                "[yue2-ar-fd]   this reason) or probe another site — do NOT read a capped probe as a wrong\n"
                "[yue2-ar-fd]   gradient.\n",
                n_inconclusive, probes.size(), dl_min);
    }

    if (isolated) {
        const bool verdict_pass = (n_bad == 0 && n_checked > 0);
        // "%d/%d within the bar" and "%d measurable", not "%d/%d measurable
        // probes" — those are two different counts and conflating them made the
        // FAIL line read "0/7 measurable probes, capped on 0", which is a
        // contradiction. It was invisible while the negative control was a
        // no-op and no FAIL could be reached.
        fprintf(stderr,
                "\n[yue2-ar-fd] GATE %s: finite differences, F32-isolated to %d AR layers, --attn %s, bar "
                "%.0e, %d/%d measurable probes within the bar (%d of %zu measurable), worst %.4f (step "
                "raised on %d, capped on %d)\n",
                n_checked == 0 ? "NO VERDICT" : (verdict_pass ? "PASS" : "FAIL"), n_layers,
                yue2_at_attn_name(attn), bar, n_checked - n_bad, n_checked, n_checked, probes.size(), worst,
                n_raised, n_inconclusive);
        if (n_checked == 0) {
            fprintf(stderr, "[yue2-ar-fd]   Every probe was capped, so NOTHING was measured. This is not a "
                            "pass.\n");
        } else if (n_bad) {
            fprintf(stderr,
                    "[yue2-ar-fd]   The analytic gradient disagrees with the measured loss change. This is\n"
                    "[yue2-ar-fd]   the only gradient gate this trainer has — do NOT train past it.\n");
        }
    } else {
        fprintf(stderr,
                "\n[yue2-ar-fd] %d/%d measurable probes within %.0f%% (worst %.4f) — INDICATIVE ONLY (no "
                "F32 isolation; a probe that capped is not counted here at all).\n"
                "[yue2-ar-fd]   Add --ar-layers 2 to turn this into a verdict.\n",
                n_checked - n_bad, n_checked, bar * 100.0, worst);
    }
    if (attn != YUE2_AT_FA_EXACT) {
        fprintf(stderr, "[yue2-ar-fd]   resolved fused-attention precision: %s (the REQUEST was --attn %s)\n",
                dit_flash_prec_label(m.backend).c_str(), yue2_at_attn_name(attn));
    }

    yue2_at_state_free(&st);
    yue2_at_train_ctx_free(&C);
    yue2_at_f32_free(&iso);
    // No measurable probe is not a pass: exit non-zero so a script cannot read
    // "every probe was capped" as a green gate.
    return (isolated && (n_bad || n_checked == 0)) ? 1 : 0;
}

// ── --forward-check: the two checks FD cannot see (contract §§6.5-6.6) ─────
//
// FINITE DIFFERENCES VERIFY THE BACKWARD AGAINST THE FORWARD. A wrong FORWARD
// therefore passes the gate happily: both arms move together, and the gate is
// measuring their agreement, not their correctness. These are the two cheap
// checks that cover that hole, and both want a REAL prefix and REAL codes —
// the gate's synthetic stream is correct for a gradient check and useless here,
// because random codes carry no information for a neighbouring hidden state to
// lose.
//
//   CHECK 1, zero-init identity (§6.6). With B == 0 the LoRA branch is an exact
//   no-op, so the trainer's forward must reproduce `yue2_ar_forward`'s logits
//   at the supervised rows. NOT bit-for-bit: the trainer dropped the KV cache
//   (SET_ROWS has no backward) and inference keeps K/V in F16, so the expected
//   answer is "an F16-sized delta", stated as a tolerance rather than asserted
//   as equality (§1.2). Two further named divergences, both in the trainer's
//   favour: the frozen projections are widened to F32 in-graph
//   (Yue2AtOpts::weights_f32), and the head is an F32 copy of the slice rather
//   than the base's native type. `YUE2_LM_NO_FLASH=1` puts the reference on the
//   same manual soft_max path the trainer's `--attn exact` uses, which is what
//   separates the F16-KV delta from the fused kernel's.
//
//   CHECK 2, the supervised-slice off-by-one (§6.5). Run twice with
//   YUE2_AT_SUP_SHIFT=0 and =-1 and compare the step-0 losses. If shifting the
//   window by one row does NOT move the loss, the supervision window is not
//   where the contract says it is, and that is a finding, not a formality.
static int yue2_ar_forwardcheck_main(const Yue2ArTrainArgs & a) {
#ifdef _WIN32
    _putenv_s("NVIDIA_TF32_OVERRIDE", "0");
#else
    setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
#endif
    std::string err;

    if (a.manifest.empty()) {
        fprintf(stderr, "ace-train yue2-ar-train --forward-check: --manifest <yue2_preprocess.json> is "
                        "required. These checks are about REAL data: the gate's synthetic stream cannot "
                        "show either fault.\n");
        return 2;
    }
    Yue2AtTarget target = YUE2_AT_T_ATTN_MLP;
    std::string  why;
    if (!yue2_at_parse_target(a.target, &target, &why)) {
        fprintf(stderr, "[yue2-ar-fwd] %s\n", why.c_str());
        return 2;
    }
    Yue2AtAttnMode attn = YUE2_AT_FA_EXACT;
    if (!yue2_at_parse_attn(a.attn, &attn)) {
        fprintf(stderr, "[yue2-ar-fwd] --attn must be exact, flash or flash-f32\n");
        return 2;
    }

    Yue2ArSet artist;
    if (!yue2_at_load_manifest(a.manifest, /*want_minted=*/false, a, &artist, &err)) {
        fprintf(stderr, "[yue2-ar-fwd] %s\n", err.c_str());
        return 1;
    }
    if (artist.songs.empty()) {
        fprintf(stderr, "[yue2-ar-fwd] the manifest has no sources\n");
        return 1;
    }
    if (a.fc_song < 0 || a.fc_song >= (int64_t) artist.songs.size()) {
        fprintf(stderr, "[yue2-ar-fwd] --fc-song %lld is outside [0, %zu)\n", (long long) a.fc_song,
                artist.songs.size());
        return 2;
    }
    Yue2ArSong & s = artist.songs[(size_t) a.fc_song];

    static Yue2Model m;
    if (!yue2_at_open_model(&m, a, "yue2-ar-fwd", &err)) {
        fprintf(stderr, "[yue2-ar-fwd] %s\n", err.c_str());
        return 1;
    }
    const Yue2LmConfig & c = m.lm_cfg;

    BPETokenizer tok;
    if (!yue2_tokenizer_load_from_gguf(&tok, m.lm_file.path)) {
        fprintf(stderr, "[yue2-ar-fwd] tokenizer: cannot read tokenizer.ggml.* from %s\n",
                m.lm_file.path.c_str());
        return 1;
    }
    try {
        const std::vector<int> pre = yue2_token_prefixes(&tok, s.style, s.lyrics, YUE2_COT_OFF, nullptr);
        s.prefix.assign(pre.begin(), pre.end());
    } catch (const std::exception & e) {
        fprintf(stderr, "[yue2-ar-fwd] prefix assembly failed for \"%s\": %s\n", s.name.c_str(), e.what());
        return 1;
    }
    if (!yue2_at_song_codes(&s, &err)) {
        fprintf(stderr, "[yue2-ar-fwd] %s\n", err.c_str());
        return 1;
    }
    Yue2AtSeq seq;
    if (!yue2_at_build_sequence(s.prefix, s.codec, a.max_len, &seq, &err)) {
        fprintf(stderr, "[yue2-ar-fwd] %s\n", err.c_str());
        return 1;
    }
    const int64_t S = (int64_t) seq.ids.size();
    fprintf(stderr,
            "[yue2-ar-fwd] song %lld/%zu \"%s\": prefix %lld (style %zu ch, lyrics %zu ch) + %lld codec%s "
            "-> S %lld, supervised %lld\n",
            (long long) a.fc_song, artist.songs.size(), s.name.c_str(), (long long) seq.prefix,
            s.style.size(), s.lyrics.size(), (long long) (S - seq.prefix - (seq.has_end ? 1 : 0)),
            seq.has_end ? " + MUSIC_END" : " (TRUNCATED, no MUSIC_END)", (long long) S,
            (long long) seq.n_sup);

    // ── the trainer's forward, with the adapter at an EXACT no-op ──
    //
    // b_sigma 0 is the production init and the whole point of check 1: with
    // B == 0 the LoRA branch contributes nothing and any difference against
    // yue2_ar_forward belongs to the graph, not to the adapter.
    Yue2AtTrainCtx C;
    if (!yue2_at_train_ctx_init(m, (int) c.block_count, a.rank, a.alpha, target, a.seed, /*b_sigma=*/0.0f,
                                /*lossgrad=*/1.0f, /*grad_clip=*/1.0f, "yue2-ar-fwd", &C, &err)) {
        fprintf(stderr, "[yue2-ar-fwd] %s\n", err.c_str());
        return 1;
    }
    if (attn != YUE2_AT_FA_EXACT && !yue2_at_flash_probe(m.backend, c, S, &err)) {
        fprintf(stderr, "[yue2-ar-fwd] %s\n", err.c_str());
        return 1;
    }
    Yue2AtState st;
    if (!yue2_at_state_alloc(&st, &m, S, a.chunk, C.sched, &err)) {
        fprintf(stderr, "[yue2-ar-fwd] %s\n", err.c_str());
        return 1;
    }

    Yue2AtRun r;
    r.m                = &m;
    r.opt              = &C.opt;
    r.sched            = C.sched;
    r.st               = &st;
    r.ad               = &C.ad;
    r.opts.attn        = attn;
    r.opts.weights_f32 = a.weights_f32;
    r.grad_accum       = 1;
    r.n_layers         = (int) c.block_count;
    r.lossgrad         = 1.0f;
    r.sup_shift        = yue2_at_sup_shift_env();
    r.forward_only     = true;

    double host_ce = 0.0;
    r.host_ce      = &host_ce;
    if (!yue2_at_micro_step(r, seq, /*count_loss=*/false, nullptr, &err)) {
        fprintf(stderr, "[yue2-ar-fwd] forward failed: %s\n", err.c_str());
        return 1;
    }
    const double l0 = host_ce;
    fprintf(stderr,
            "\n[yue2-ar-fwd] ==== STEP-0 LOSS (contract §6.5) ====\n"
            "[yue2-ar-fwd] sup_shift %d: sliced CE %.6f over %lld supervised rows "
            "(ln(%lld) = %.4f is the uniform reference)\n",
            r.sup_shift, l0, (long long) seq.n_sup, (long long) YUE2_AT_SLICE_ROWS,
            std::log((double) YUE2_AT_SLICE_ROWS));

    // ── check 1: the sampled supervised rows ──
    const int n_cmp = (int) (a.fwd_check > 0 ? std::min<int64_t>(a.fwd_check, seq.n_sup) : 32);
    std::vector<int32_t> cols((size_t) n_cmp, 0);
    std::vector<int64_t> abs_rows((size_t) n_cmp, 0);
    for (int k = 0; k < n_cmp; k++) {
        const int64_t i = (n_cmp == 1) ? 0 : (int64_t) k * (seq.n_sup - 1) / (int64_t) (n_cmp - 1);
        abs_rows[(size_t) k] = seq.prefix - 1 + i;
        cols[(size_t) k]     = (int32_t) (seq.prefix - 1 + (int64_t) r.sup_shift + i);
    }

    // The trainer's logits at those rows, off the SAME t_H and the SAME sliced
    // F32 head the loss uses. st.t_ids is reusable here: the forward is done
    // with it, and get_rows wants an I32 index vector of exactly this shape.
    std::vector<float> tr_logits((size_t) n_cmp * (size_t) YUE2_AT_SLICE_ROWS, 0.0f);
    {
        const int64_t H = (int64_t) c.embedding_length;
        ggml_backend_tensor_set(st.t_ids, cols.data(), 0, (size_t) n_cmp * sizeof(int32_t));
        ggml_init_params ip  = { st.arena.size(), st.arena.data(), /*no_alloc*/ true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 64, /*grads=*/false);
        ggml_tensor *    idx = ggml_view_1d(ctx, st.t_ids, n_cmp, 0);
        ggml_tensor *    hd  = ggml_get_rows(ctx, st.t_H, idx);          // [H, n_cmp]
        ggml_tensor *    lg  = ggml_mul_mat(ctx, st.t_head, hd);         // [SL, n_cmp]
        ggml_set_output(lg);
        ggml_build_forward_expand(gf, lg);
        ggml_backend_sched_reset(C.sched);
        const bool ok = ggml_backend_sched_graph_compute(C.sched, gf) == GGML_STATUS_SUCCESS;
        if (ok) {
            ggml_backend_tensor_get(lg, tr_logits.data(), 0, tr_logits.size() * sizeof(float));
        }
        ggml_free(ctx);
        (void) H;
        if (!ok) {
            fprintf(stderr, "[yue2-ar-fwd] gathering the trainer's logits failed\n");
            return 1;
        }
    }

    // Free the trainer's buffers BEFORE the reference forward: yue2_ar_forward
    // allocates its own T-row KV cache and graph, and there is no reason for
    // both to be resident at once.
    yue2_at_state_free(&st);
    yue2_at_train_ctx_free(&C);

    Yue2ArForwardRequest req;
    req.ids             = seq.ids;
    req.logit_positions = abs_rows;
    Yue2ArForwardResult res;
    if (!yue2_ar_forward(m, req, &res, &err)) {
        fprintf(stderr, "[yue2-ar-fwd] yue2_ar_forward failed: %s\n", err.c_str());
        return 1;
    }
    const int64_t V = res.V;
    if ((int64_t) res.logits.size() != (int64_t) n_cmp * V) {
        fprintf(stderr, "[yue2-ar-fwd] reference returned %zu logits, expected %lld\n", res.logits.size(),
                (long long) ((int64_t) n_cmp * V));
        return 1;
    }

    // rel-L2 over the whole compared block, the worst single row, argmax
    // agreement, and the CE the two heads disagree by at those rows.
    double num2 = 0.0, den2 = 0.0, worst_row = 0.0, max_abs = 0.0, ref_absmax = 0.0;
    int    agree = 0, agree_top5 = 0;
    double ce_tr = 0.0, ce_ref = 0.0;
    int    worst_k = -1;
    for (int k = 0; k < n_cmp; k++) {
        const float * tr  = tr_logits.data() + (size_t) k * (size_t) YUE2_AT_SLICE_ROWS;
        const float * ref = res.logits.data() + (size_t) k * (size_t) V + (size_t) YUE2_AT_SLICE_ROW0;
        double rn = 0.0, rd = 0.0;
        int64_t am_tr = 0, am_ref = 0;
        double  mx_tr = (double) tr[0], mx_ref = (double) ref[0];
        for (int64_t j = 0; j < YUE2_AT_SLICE_ROWS; j++) {
            const double d = (double) tr[j] - (double) ref[j];
            rn += d * d;
            rd += (double) ref[j] * (double) ref[j];
            max_abs    = std::max(max_abs, std::fabs(d));
            ref_absmax = std::max(ref_absmax, std::fabs((double) ref[j]));
            if ((double) tr[j] > mx_tr) {
                mx_tr = (double) tr[j];
                am_tr = j;
            }
            if ((double) ref[j] > mx_ref) {
                mx_ref = (double) ref[j];
                am_ref = j;
            }
        }
        num2 += rn;
        den2 += rd;
        const double rel_row = rd > 0.0 ? std::sqrt(rn / rd) : 0.0;
        if (rel_row > worst_row) {
            worst_row = rel_row;
            worst_k   = k;
        }
        if (am_tr == am_ref) {
            agree++;
        }
        // top-5 agreement on the REFERENCE's argmax: a tie between two nearly
        // equal logits is not the same defect as a different distribution.
        {
            int better = 0;
            for (int64_t j = 0; j < YUE2_AT_SLICE_ROWS; j++) {
                if ((double) tr[j] > (double) tr[am_ref]) {
                    better++;
                    if (better >= 5) {
                        break;
                    }
                }
            }
            if (better < 5) {
                agree_top5++;
            }
        }
        // the per-row CE at the true target, both heads
        const int64_t tgt = (int64_t) seq.rows[(size_t) (abs_rows[(size_t) k] - (seq.prefix - 1))];
        double sum_tr = 0.0, sum_ref = 0.0;
        for (int64_t j = 0; j < YUE2_AT_SLICE_ROWS; j++) {
            sum_tr += std::exp((double) tr[j] - mx_tr);
            sum_ref += std::exp((double) ref[j] - mx_ref);
        }
        ce_tr += mx_tr + std::log(sum_tr) - (double) tr[tgt];
        ce_ref += mx_ref + std::log(sum_ref) - (double) ref[tgt];
    }
    const double rel_l2 = den2 > 0.0 ? std::sqrt(num2 / den2) : 0.0;

    fprintf(stderr,
            "\n[yue2-ar-fwd] ==== ZERO-INIT IDENTITY vs yue2_ar_forward (contract §6.6) ====\n"
            "[yue2-ar-fwd] %d supervised rows compared over the %lld-row scored slice, trainer --attn %s, "
            "weights %s\n"
            "[yue2-ar-fwd]   rel-L2 (all rows)   %.3e\n"
            "[yue2-ar-fwd]   worst row rel-L2    %.3e (row %d of %d, abs position %lld)\n"
            "[yue2-ar-fwd]   max |delta| logit   %.3e   (reference |logit| max %.3f)\n"
            "[yue2-ar-fwd]   argmax agreement    %d/%d exact, %d/%d within the trainer's top-5\n"
            "[yue2-ar-fwd]   CE at those rows    trainer %.6f vs reference %.6f (delta %.2e)\n",
            n_cmp, (long long) YUE2_AT_SLICE_ROWS, yue2_at_attn_name(attn),
            a.weights_f32 ? "f32-widened" : "native", rel_l2, worst_row, worst_k, n_cmp,
            worst_k >= 0 ? (long long) abs_rows[(size_t) worst_k] : -1LL, max_abs, ref_absmax, agree, n_cmp,
            agree_top5, n_cmp, ce_tr / (double) n_cmp, ce_ref / (double) n_cmp,
            std::fabs(ce_tr - ce_ref) / (double) n_cmp);
    fprintf(stderr,
            "[yue2-ar-fwd] EXPECTED, and this is a TOLERANCE not an equality: inference keeps K and V in "
            "F16 (yue2-lm-graph.h:265-266) and the trainer dropped the cache entirely, so an F16-sized "
            "delta is the correct answer. F16 carries ~3 decimal digits, so a rel-L2 in the 1e-3 band is "
            "the cache round trip and nothing else; 1e-1 or a broken argmax would be a different forward.\n");
    if (r.sup_shift != 0) {
        fprintf(stderr, "[yue2-ar-fwd] NOTE: YUE2_AT_SUP_SHIFT=%d was set, so the rows compared are the "
                        "SHIFTED ones — the identity number above is only meaningful at shift 0.\n",
                r.sup_shift);
    }
    return 0;
}

// ── The loop ───────────────────────────────────────────────────────────────

static int yue2_ar_train_loop(const Yue2ArTrainArgs & a) {
    std::string err;

    // ── arguments ──
    if (a.manifest.empty()) {
        fprintf(stderr,
                "ace-train yue2-ar-train: --manifest <yue2_preprocess.json> is required for training.\n"
                "  Build one with `ace-train yue2-preprocess`, fill its codec_ids with\n"
                "  `ace-train yue2-tokenize`, or run the gradient gate instead:\n"
                "      ace-train yue2-ar-train --lm <yue2-lm-*.gguf> --fd-check 6 --ar-layers 2\n");
        return 2;
    }
    if (a.out_dir.empty()) {
        fprintf(stderr, "ace-train yue2-ar-train: --out <dir> is required (checkpoints + the adapter)\n");
        return 2;
    }
    if (a.steps <= 0 || a.grad_accum <= 0 || a.rank <= 0 || a.max_len <= 0 || a.chunk <= 0) {
        fprintf(stderr, "ace-train yue2-ar-train: --steps, --grad-accum, --rank, --max-len and --chunk must "
                        "all be > 0\n");
        return 2;
    }
    if (a.artist_frac < 0.0 || a.artist_frac > 1.0) {
        fprintf(stderr, "ace-train yue2-ar-train: --artist-frac must be in [0, 1]\n");
        return 2;
    }
    if (a.lr_scheduler != "cosine" && a.lr_scheduler != "constant") {
        fprintf(stderr, "ace-train yue2-ar-train: --lr-scheduler must be cosine or constant\n");
        return 2;
    }
    if (a.cursor_weight < 0.0 || !(a.cursor_weight == a.cursor_weight)) {
        fprintf(stderr, "ace-train yue2-ar-train: --cursor-weight must be >= 0 (0 switches the term off)\n");
        return 2;
    }
    // The README, verbatim: "Do not train longer: past ~1,500 steps the model
    // memorises the songs." Upstream's own pick from the ladder was step 800.
    if (a.steps > 1500 && !a.allow_overtrain) {
        fprintf(stderr,
                "ace-train yue2-ar-train: --steps %lld exceeds the recipe's ceiling. Upstream's README is "
                "emphatic: past ~1500 steps the model MEMORISES the songs, and its own pick from the "
                "checkpoint ladder was step 800. Pass --allow-overtrain to proceed (it is recorded in the "
                "adapter's metadata), or train to 1500 and audition the ladder.\n",
                (long long) a.steps);
        return 2;
    }
    Yue2AtTarget target = YUE2_AT_T_ATTN_MLP;
    std::string  why;
    if (!yue2_at_parse_target(a.target, &target, &why)) {
        fprintf(stderr, "ace-train yue2-ar-train: %s\n", why.c_str());
        return 2;
    }
    Yue2AtAttnMode attn = YUE2_AT_FA_EXACT;
    if (!yue2_at_parse_attn(a.attn, &attn)) {
        fprintf(stderr, "ace-train yue2-ar-train: --attn must be exact, flash or flash-f32\n");
        return 2;
    }

    // ── the minted regularizer, or an explicit refusal ──
    const bool minted_present = !a.minted.empty();
    if (minted_present) {
        if (a.minted.size() > 3 && a.minted.compare(a.minted.size() - 3, 3, ".pt") == 0) {
            fprintf(stderr,
                    "ace-train yue2-ar-train: --minted %s is a torch pickle. This trainer reads a JSON "
                    "manifest in the yue2-preprocess sources[] shape (style / lyrics / codec_ids path per "
                    "song, plus \"src\": \"minted\"|\"minted_val\"). Convert the pack once with\n"
                    "  python engine/tools/convert-yue2-minted.py <pack.pt> <out_dir>\n"
                    "and pass the minted_manifest.json it writes (spec: docs/plans/yue2/15-minted-pack.md). "
                    "Nothing here will half-read a pickle.\n",
                    a.minted.c_str());
            return 2;
        }
        if (!pm_file_exists(a.minted)) {
            fprintf(stderr, "ace-train yue2-ar-train: --minted %s does not exist\n", a.minted.c_str());
            return 1;
        }
    } else if (!a.allow_no_minted) {
        yue2_at_no_minted_banner();
        fprintf(stderr,
                "ace-train yue2-ar-train: REFUSING TO START without --minted. The 50/50 artist/minted split "
                "is the recipe's counterweight to a MEASURED defect in our encoder's output, not optional "
                "hygiene. Supply --minted <converted manifest>, or pass --allow-no-minted to train without "
                "it with your eyes open.\n");
        return 2;
    }

    // ── dataset ──
    Yue2ArSet artist;
    if (!yue2_at_load_manifest(a.manifest, /*want_minted=*/false, a, &artist, &err)) {
        fprintf(stderr, "[yue2-ar-train] %s\n", err.c_str());
        return 1;
    }
    Yue2ArSet minted;
    if (minted_present && !yue2_at_load_manifest(a.minted, /*want_minted=*/true, a, &minted, &err)) {
        fprintf(stderr, "[yue2-ar-train] minted: %s\n", err.c_str());
        return 1;
    }

    static Yue2Model m;
    if (!yue2_at_open_model(&m, a, "yue2-ar-train", &err)) {
        fprintf(stderr, "[yue2-ar-train] %s\n", err.c_str());
        return 1;
    }
    const Yue2LmConfig & c = m.lm_cfg;

    if (a.max_len > (int64_t) c.context_length) {
        fprintf(stderr, "[yue2-ar-train] --max-len %lld exceeds the model's context_length %u\n",
                (long long) a.max_len, c.context_length);
        return 2;
    }

    // ── tokenizer + prefixes ──
    BPETokenizer tok;
    if (!yue2_tokenizer_load_from_gguf(&tok, m.lm_file.path)) {
        fprintf(stderr, "[yue2-ar-train] tokenizer: cannot read tokenizer.ggml.* from %s\n",
                m.lm_file.path.c_str());
        return 1;
    }
    auto tokenize_prefix = [&](Yue2ArSong & s) -> bool {
        try {
            const std::vector<int> pre = yue2_token_prefixes(&tok, s.style, s.lyrics, YUE2_COT_OFF, nullptr);
            s.prefix.assign(pre.begin(), pre.end());
        } catch (const std::exception & e) {
            fprintf(stderr, "[yue2-ar-train] prefix assembly failed for \"%s\": %s\n", s.name.c_str(),
                    e.what());
            return false;
        }
        return true;
    };

    // Artist prefixes and codes up front: the corpus is small, and a manifest
    // whose codes are missing should be a startup error, not a failure 137
    // steps in.
    int64_t frames_sum = 0, longest = 0, n_lyricless = 0;
    for (Yue2ArSong & s : artist.songs) {
        if (!tokenize_prefix(s) || !yue2_at_song_codes(&s, &err)) {
            if (!err.empty()) {
                fprintf(stderr, "[yue2-ar-train] %s\n", err.c_str());
            }
            return 1;
        }
        frames_sum += (int64_t) s.codec.size();
        longest = std::max(longest, (int64_t) s.prefix.size() + (int64_t) s.codec.size() + 1);
        if (s.lyrics.empty()) {
            n_lyricless++;
        }
    }
    const int64_t mean_frames = artist.songs.empty() ? 0 : frames_sum / (int64_t) artist.songs.size();

    // ── the lyric-cursor targets, bound up front and reported per song ──
    //
    // Binding can fail per song for upstream's own reason (:39 — the prefix
    // does not reconstruct from head + lyrics, usually a BPE merge across the
    // [Lyrics] boundary) or because the manifest names no spans. Either is
    // reported by name; a run with the term on and NOTHING bound is refused.
    int64_t cursor_l_max = 0, n_bound = 0, n_with_spans = 0;
    if (a.cursor_weight > 0.0) {
        for (Yue2ArSong & s : artist.songs) {
            if (s.cursor_words.empty()) {
                fprintf(stderr, "[yue2-ar-train] cursor: \"%s\" names no cursor_words — no targets\n",
                        s.name.c_str());
                continue;
            }
            n_with_spans++;
            std::string e2;
            if (!yue2_at_read_words5(s.cursor_words, yue2_at_utf8_len(s.lyrics), &s.words5, &e2)) {
                fprintf(stderr, "[yue2-ar-train] cursor: %s\n", e2.c_str());
                return 1;
            }
            // nF = min(frames, n_sup): the sequence this song actually trains as.
            Yue2AtSeq   probe;
            if (!yue2_at_build_sequence(s.prefix, s.codec, a.max_len, &probe, &e2)) {
                continue;  // the training loop names the skip itself
            }
            s.cursor = yue2_at_cursor_build(&tok, s.style, s.lyrics, s.prefix, (int64_t) s.codec.size(),
                                            s.words5, probe.n_sup);
            if (!s.cursor.bound) {
                fprintf(stderr, "[yue2-ar-train] cursor: \"%s\" UNBOUND — %s\n", s.name.c_str(),
                        s.cursor.why.c_str());
                continue;
            }
            n_bound++;
            cursor_l_max = std::max(cursor_l_max, s.cursor.L);
            fprintf(stderr, "[yue2-ar-train] cursor: \"%s\" bound — %lld lyric tokens at [%lld, %lld), %lld "
                            "frames, %zu words\n",
                    s.name.c_str(), (long long) s.cursor.L, (long long) s.cursor.j0,
                    (long long) (s.cursor.j0 + s.cursor.L), (long long) s.cursor.nF, s.words5.size() / 5);
        }
        if (n_bound == 0) {
            fprintf(stderr,
                    "\n[yue2-ar-train] --cursor-weight %.3f with NO artist song bound (%lld of %zu named "
                    "spans). A run like this prints `cursor nan` on every step and finishes — the term it "
                    "claims to train is not there. Produce the spans (the cursor bridge, or the aligner) "
                    "and point the manifest's `cursor_words` at them, or pass --cursor-weight 0 to train "
                    "without the term on purpose.\n",
                    a.cursor_weight, (long long) n_with_spans, artist.songs.size());
            return 2;
        }
        fprintf(stderr, "[yue2-ar-train] cursor: weight %.3f, %lld of %zu artist songs bound, longest lyric "
                        "span %lld tokens\n",
                a.cursor_weight, (long long) n_bound, artist.songs.size(), (long long) cursor_l_max);
    } else {
        fprintf(stderr, "[yue2-ar-train] cursor: OFF (--cursor-weight 0). Upstream trains with 0.08; without "
                        "it the frame-to-lyric alignment degrades as the songs are memorised.\n");
    }
    if (n_lyricless) {
        fprintf(stderr,
                "\n[yue2-ar-train] ---- %lld of %zu songs have NO LYRICS in their prefix ----\n"
                "[yue2-ar-train] The AR prefix is instruction + [Tags] style + [Lyrics] lyrics. Without the\n"
                "[yue2-ar-train] lyric sheet the model is being taught to write a song from a style string\n"
                "[yue2-ar-train] alone, and upstream's README is emphatic that truncated lyrics ruin\n"
                "[yue2-ar-train] structure. Fixes, in order: add a `lyrics` field per source to the\n"
                "[yue2-ar-train] manifest (yue2-preprocess --caption-mode ace, contract §5.2); put an ACE\n"
                "[yue2-ar-train] sidecar <stem>.txt beside each source audio file; or pass --lyrics.\n\n",
                (long long) n_lyricless, artist.songs.size());
    }
    fprintf(stderr,
            "[yue2-ar-train] artist: %zu song(s), mean %lld codec frames (%.0f s at 25 Hz), longest "
            "sequence %lld tokens\n",
            artist.songs.size(), (long long) mean_frames, (double) mean_frames / 25.0, (long long) longest);
    if (minted_present) {
        int64_t n_val = 0;
        for (Yue2ArSong & s : minted.songs) {
            if (!tokenize_prefix(s)) {
                return 1;
            }
            if (s.minted_val) {
                n_val++;
            }
        }
        fprintf(stderr, "[yue2-ar-train] minted: %zu song(s), %lld held out as minted_val\n",
                minted.songs.size(), (long long) n_val);
    } else {
        yue2_at_no_minted_banner();
    }

    // Whole songs, no crop policy (contract §4.4). The buffers below are sized
    // for the WORST CASE --max-len allows rather than for the corpus, so VRAM
    // is predictable at step 0 instead of at step 900 — and lowering --max-len
    // is the lever.
    const int64_t s_max = std::min<int64_t>(a.max_len, (int64_t) c.context_length);

    // ── trainable state ──
    const int n_layers = (int) c.block_count;
    // b_sigma 0: B stays EXACTLY zero, so the adapter starts as an exact no-op.
    // lossgrad 1.0, NOT 1/grad_accum: the grad-accum scaling lives in the
    // chunked head's `gs` (lm-ckpt.h's D9), and seeding it twice would square
    // it.
    Yue2AtTrainCtx C;
    if (!yue2_at_train_ctx_init(m, n_layers, a.rank, a.alpha, target, a.seed, /*b_sigma=*/0.0f,
                                /*lossgrad=*/1.0f, a.max_grad_norm, "yue2-ar-train", &C, &err,
                                /*cursor_head=*/n_bound > 0)) {
        fprintf(stderr, "[yue2-ar-train] %s\n", err.c_str());
        return 1;
    }
    C.opt.weight_decay = a.weight_decay;
    // upstream's betas, not lm-optim.h's default (ar_lora_cursor.py:31:
    // `betas=(0.9,0.95)`). This used to be a known, deliberate divergence, and
    // it stopped being defensible once the two trainers were run side by side
    // on the same twelve songs: from an identical step-0 artist loss (5.3658
    // against upstream's 5.370) ours descended visibly faster, reaching 0.032
    // by step 800 where upstream was still at 0.165. beta2 is the length of
    // AdamW's second-moment memory, so it directly sets how hard a tiny dataset
    // is memorised.
    C.opt.adam_beta1   = a.adam_beta1;
    C.opt.adam_beta2   = a.adam_beta2;
    // Neutralise LmOptim's own cosine — yue2_at_lr_at carries the schedule.
    C.opt.lr_floor     = 1.0f;
    C.opt.total_steps  = 1;
    C.opt.warmup_steps = 0;

    if (attn != YUE2_AT_FA_EXACT && !yue2_at_flash_probe(m.backend, c, s_max, &err)) {
        fprintf(stderr, "[yue2-ar-train] %s\n", err.c_str());
        return 1;
    }

    {
        long long bytes = 0, mtime = 0;
        pm_stat_file(m.lm_file.path, &bytes, &mtime);
        const Yue2AtVram v =
            yue2_at_vram(c, s_max, a.rank, target, attn, a.chunk, (double) std::max<long long>(0, bytes));
        yue2_at_vram_report("yue2-ar-train", v, s_max, attn);
    }

    Yue2AtState st;
    if (!yue2_at_state_alloc(&st, &m, s_max, a.chunk, C.sched, &err, cursor_l_max)) {
        fprintf(stderr, "[yue2-ar-train] %s\n", err.c_str());
        return 1;
    }
    fprintf(stderr, "[yue2-ar-train] persistent training buffers: %.2f GiB measured (the VRAM model above "
                    "is a counted UPPER bound — correct it from this number, not the other way round)\n",
            (double) st.fixed_bytes() / (1024.0 * 1024.0 * 1024.0));

    Yue2AtRun r;
    r.m                = &m;
    r.opt              = &C.opt;
    r.sched            = C.sched;
    r.st               = &st;
    r.ad               = &C.ad;
    r.opts.attn        = attn;
    r.opts.weights_f32 = a.weights_f32;
    r.grad_accum       = a.grad_accum;
    r.lossgrad         = C.lossgrad;  // 1.0; the 1/grad_accum lives in `gs` (D9)
    r.sup_shift        = yue2_at_sup_shift_env();  // 0 unless the §6.5 control is set
    r.cur_w            = C.cursor_w;
    r.cursor_weight    = (float) a.cursor_weight;
    std::string cursor_md = "off";
    if (C.cursor_w) {
        char cb[96];
        snprintf(cb, sizeof(cb), "on:w=%.4g,bound=%lld/%zu", a.cursor_weight, (long long) n_bound,
                 artist.songs.size());
        cursor_md = cb;
    }

    // ── resume ──
    Yue2AtCkptState want;
    want.rank          = (int32_t) a.rank;
    want.n_params      = (int32_t) C.params.size();
    want.n_layers      = (int32_t) n_layers;
    want.target        = (int32_t) target;
    want.grad_accum    = (int32_t) a.grad_accum;
    want.total_steps   = (int32_t) a.steps;
    want.alpha         = a.alpha;
    want.lr            = a.lr;
    want.seed          = a.seed;
    want.cond_hash     = yue2_at_cond_hash(a);
    want.warmup        = (int32_t) a.warmup;
    want.lr_sched      = (a.lr_scheduler == "constant") ? 1 : 0;
    want.sched_steps   = (int32_t) a.sched_steps;
    want.weight_decay  = a.weight_decay;
    want.max_grad_norm = a.max_grad_norm;
    want.adam_beta1    = a.adam_beta1;
    want.adam_beta2    = a.adam_beta2;

    int64_t           step0    = 0;
    double            loss_sum = 0.0;
    int64_t           n_micro  = 0;
    const std::string ckpt_path = yue2_at_ckpt_path(a.out_dir);
    if (a.resume) {
        Yue2AtCkptState got;
        if (!yue2_at_ckpt_load(ckpt_path, want, &got, &C, &err)) {
            fprintf(stderr, "[yue2-ar-train] resume: %s\n", err.c_str());
            return 1;
        }
        step0    = got.steps_done;
        loss_sum = got.loss_sum;
        n_micro  = got.n_micro;
        fprintf(stderr, "[yue2-ar-train] resumed from %s at step %lld/%lld (AdamW iter %d)\n",
                ckpt_path.c_str(), (long long) step0, (long long) a.steps, C.opt.opt_iter);
        if (step0 >= a.steps) {
            fprintf(stderr, "[yue2-ar-train] that run is already finished; nothing to do\n");
            yue2_at_state_free(&st);
            yue2_at_train_ctx_free(&C);
            return 0;
        }
    } else if (pm_file_exists(ckpt_path)) {
        fprintf(stderr,
                "[yue2-ar-train] %s already exists. Pass --resume to continue that run, or point --out "
                "somewhere else to start a new one.\n",
                ckpt_path.c_str());
        return 1;
    }
    if (!pm_mkdir_p(a.out_dir)) {
        fprintf(stderr, "[yue2-ar-train] cannot create %s\n", a.out_dir.c_str());
        return 1;
    }

    fprintf(stderr,
            "[yue2-ar-train] lr %.3g (%s over %lld, warmup %lld), %lld steps x %lld micro, clip %.2f, wd "
            "%.3g, artist-frac %.2f, --attn %s, --max-len %lld, chunk %lld, trigger \"%s\"\n",
            (double) a.lr, a.lr_scheduler.c_str(), (long long) a.sched_steps, (long long) a.warmup,
            (long long) a.steps, (long long) a.grad_accum, (double) a.max_grad_norm, (double) a.weight_decay,
            a.artist_frac, yue2_at_attn_name(attn), (long long) a.max_len, (long long) a.chunk,
            a.trigger.c_str());
    // Was a reported divergence, now a matched one. lm_optim_step used to
    // hard-code (0.9, 0.999); LmOptim carries the betas as fields defaulting to
    // exactly that, so the four other trainers sharing the header are
    // unaffected, and this one sets upstream's (0.9, 0.95).
    fprintf(stderr, "[yue2-ar-train] AdamW betas (%.3g, %.3g)%s\n",
            (double) a.adam_beta1, (double) a.adam_beta2,
            (a.adam_beta1 == 0.9f && a.adam_beta2 == 0.95f)
                ? " — upstream's (ar_lora_cursor.py:31)"
                : " — NOT upstream's (0.9, 0.95)");
    if (a.trigger.empty()) {
        fprintf(stderr, "[yue2-ar-train] WARNING: no --trigger. The adapter will have no word to address it "
                        "by at generation time, which is upstream's whole mechanism for reaching the "
                        "style.\n");
    }

    // ── the loop ──
    // The training pools are POINTERS, and the minted one excludes minted_val.
    // Upstream splits the same way (`minted=[x for x in data if x["src"]=="minted"]`,
    // ar_lora_cursor.py) and it is the whole point of the hold-out: a minted_val
    // song that also appears in training makes the one number this run is
    // supposed to watch — "has YuE2's token grammar been damaged" — a training
    // loss wearing a hold-out's name, and it would stay flat for the wrong
    // reason.
    std::vector<Yue2ArSong *> train_artist, train_minted;
    for (Yue2ArSong & s : artist.songs) {
        train_artist.push_back(&s);
    }
    for (Yue2ArSong & s : minted.songs) {
        if (!s.minted_val) {
            train_minted.push_back(&s);
        }
    }
    if (minted_present) {
        fprintf(stderr, "[yue2-ar-train] minted training pool: %zu song(s) (%zu minted_val held OUT of "
                        "training)\n",
                train_minted.size(), minted.songs.size() - train_minted.size());
    }
    auto pick_song = [&](uint64_t k, bool * was_artist) -> Yue2ArSong * {
        bool use_artist = true;
        if (minted_present && !train_minted.empty()) {
            Yue2NtRng rmix(yue2_at_seed_mix(a.seed, k, YUE2_AT_TAG_MIX));
            use_artist = rmix.u01() < a.artist_frac;
        }
        std::vector<Yue2ArSong *> & pool = use_artist ? train_artist : train_minted;
        Yue2NtRng                   rsong(yue2_at_seed_mix(a.seed, k, YUE2_AT_TAG_SONG));
        size_t                      idx = (size_t) (rsong.u01() * (double) pool.size());
        idx                             = std::min(idx, pool.size() - 1);
        *was_artist                     = use_artist;
        return pool[idx];
    };

    // The held-out evals. minted_val answers "has YuE2's token grammar been
    // damaged"; the artist sample answers "is the artist loss still falling".
    // They are DIFFERENT QUESTIONS and are never substituted for one another.
    std::vector<Yue2ArSong *> eval_val, eval_artist;
    if (minted_present) {
        for (Yue2ArSong & s : minted.songs) {
            if (s.minted_val && eval_val.size() < 6) {
                eval_val.push_back(&s);
            }
        }
    }
    for (size_t i = 0; i < artist.songs.size() && eval_artist.size() < 6; i++) {
        eval_artist.push_back(&artist.songs[i]);
    }

    Yue2AtSeq seq;
    auto      build_seq = [&](Yue2ArSong * s, Yue2AtSeq * out) -> bool {
        std::string e2;
        if (!yue2_at_song_codes(s, &e2)) {
            fprintf(stderr, "[yue2-ar-train] %s\n", e2.c_str());
            return false;
        }
        if (!yue2_at_build_sequence(s->prefix, s->codec, a.max_len, out, &e2)) {
            // SKIP, with the song named — never crop (contract §4.4).
            fprintf(stderr, "[yue2-ar-train] SKIPPING \"%s\": %s\n", s->name.c_str(), e2.c_str());
            return false;
        }
        return true;
    };

    auto run_eval = [&](int64_t step) {
        auto score = [&](std::vector<Yue2ArSong *> & pool, double * sliced, double * full) -> bool {
            double  acc_s = 0.0, acc_f = 0.0;
            int64_t n = 0;
            for (Yue2ArSong * s : pool) {
                Yue2AtSeq es;
                if (!build_seq(s, &es)) {
                    continue;
                }
                r.forward_only = true;
                r.host_ce      = nullptr;
                double      ce = 0.0;
                std::string e2;
                if (!yue2_at_micro_step(r, es, /*count_loss=*/true, &ce, &e2)) {
                    fprintf(stderr, "[yue2-ar-train] eval: %s\n", e2.c_str());
                    r.forward_only = false;
                    return false;
                }
                acc_s += ce;
                if (a.eval_full_vocab) {
                    // Contract §3.3: our sliced CE omits the text-token mass, so
                    // our numbers are NOT comparable to upstream's README —
                    // which matters, because minted_val loss is the thing to
                    // watch. Evals are infrequent, so they pay the 5.6x for a
                    // quotable number.
                    double fce = 0.0;
                    if (yue2_at_head_fullvocab_ce(r, es, 128, &fce)) {
                        acc_f += fce;
                    }
                }
                n++;
            }
            r.forward_only = false;
            if (!n) {
                return false;
            }
            *sliced = acc_s / (double) n;
            *full   = acc_f / (double) n;
            return true;
        };
        double     vs = 0.0, vf = 0.0, as = 0.0, af = 0.0;
        const bool have_val = !eval_val.empty() && score(eval_val, &vs, &vf);
        const bool have_art = !eval_artist.empty() && score(eval_artist, &as, &af);
        if (have_art) {
            fprintf(stderr, "[yue2-ar-train] EVAL step %lld  artist %.4f (full-vocab %.4f)\n",
                    (long long) step, as, af);
        }
        if (have_val) {
            fprintf(stderr, "[yue2-ar-train] EVAL step %lld  minted_val %.4f (full-vocab %.4f) <- THE "
                            "NUMBER TO WATCH: it should stay FLAT\n",
                    (long long) step, vs, vf);
        } else {
            // Not substituted with the artist hold-out, ever: that answers "has
            // the artist been memorised", which is a different question from
            // "has YuE2's token grammar been damaged" (contract §5.3 item 4).
            fprintf(stderr, "[yue2-ar-train] EVAL step %lld  minted_val ABSENT — no minted pack, so this "
                            "run has NO token-grammar check\n",
                    (long long) step);
        }
    };

    double        window_sum = 0.0;
    int64_t       window_n   = 0;
    const int64_t t0_ms      = ggml_time_ms();
    if (a.eval_every > 0) {
        run_eval(step0);
    }

    for (int64_t step = step0 + 1; step <= a.steps; step++) {
        const double lr_now = yue2_at_lr_at(a, step);
        C.opt.base_lr       = (float) lr_now;
        lm_optim_zero_grad(&C.opt);

        double      step_loss  = 0.0;
        double      step_cur   = 0.0;  // upstream prints the LAST micro-step's cursor loss; we print the mean
        int64_t     n_ok       = 0, n_cur = 0;
        int64_t     last_S     = 0;
        const char * last_id   = "";
        const char * last_pool = "";
        for (int64_t g = 0; g < a.grad_accum; g++) {
            const uint64_t k = (uint64_t) ((step - 1) * a.grad_accum + g);
            bool           was_artist = true;
            Yue2ArSong *   s          = pick_song(k, &was_artist);
            if (!build_seq(s, &seq)) {
                continue;  // the skip has already been named
            }
            last_id   = s->name.c_str();
            last_pool = was_artist ? "artist" : "minted";
            last_S    = (int64_t) seq.ids.size();

            r.forward_only = false;
            r.host_ce      = nullptr;
            // Artist songs carry targets; minted songs never do (ar_lora_cursor.py:84-86).
            r.cur          = (was_artist && s->cursor.bound) ? &s->cursor : nullptr;
            double ce = 0.0, cv = std::nan("");
            if (!yue2_at_micro_step(r, seq, /*count_loss=*/true, &ce, &err, &cv)) {
                fprintf(stderr, "[yue2-ar-train] step %lld: %s\n", (long long) step, err.c_str());
                return 1;
            }
            r.cur = nullptr;
            step_loss += ce;
            loss_sum += ce;
            n_micro++;
            n_ok++;
            if (cv == cv) {
                step_cur += cv;
                n_cur++;
            }
        }
        if (!n_ok) {
            fprintf(stderr, "[yue2-ar-train] step %lld: every drawn song was skipped\n", (long long) step);
            return 1;
        }

        LmStepStats sstat{};
        if (!lm_optim_step(&C.opt, C.osched, &sstat)) {
            fprintf(stderr, "[yue2-ar-train] optimizer step failed at step %lld\n", (long long) step);
            return 1;
        }

        const double mean = step_loss / (double) n_ok;
        window_sum += mean;
        window_n++;
        if (a.log_every > 0 && (step % a.log_every == 0 || step == step0 + 1 || step == a.steps)) {
            const double    elapsed = (double) (ggml_time_ms() - t0_ms) / 1000.0;
            const DitGpuMem gm      = dit_gpu_mem_query(m.backend);
            char curbuf[32];
            if (n_cur > 0) {
                snprintf(curbuf, sizeof(curbuf), "%.3f", step_cur / (double) n_cur);
            } else {
                snprintf(curbuf, sizeof(curbuf), "%s", C.cursor_w ? "nan" : "off");
            }
            fprintf(stderr,
                    "[yue2-ar-train] step %5lld/%lld  loss %.5f (win %.5f, run %.5f)  cursor %s  |g| %.4f  "
                    "lr %.2e  S %lld  %.2fs/it  vram %zu/%zu MB (%s)  [%s %s]\n",
                    (long long) step, (long long) a.steps, mean, window_sum / (double) window_n,
                    n_micro ? loss_sum / (double) n_micro : 0.0, curbuf, (double) sstat.grad_norm,
                    (double) sstat.lr, (long long) last_S, elapsed / (double) std::max<int64_t>(1, step - step0),
                    gm.used_mb(), gm.total_mb(), gm.source(), last_pool, last_id);
            window_sum = 0.0;
            window_n   = 0;
        }
        if (a.eval_every > 0 && step % a.eval_every == 0) {
            run_eval(step);
        }

        // Upstream's ladder: checkpoints from CK_FROM every CK_EVERY, which is
        // what the ear test picks from (theirs was step 800).
        if (a.save_every > 0 && step >= a.ckpt_from && (step - a.ckpt_from) % a.save_every == 0 &&
            step < a.steps) {
            Yue2AtCkptState st_save = want;
            st_save.steps_done      = (int32_t) step;
            st_save.loss_sum        = loss_sum;
            st_save.n_micro         = n_micro;
            st_save.opt_step        = C.opt.opt_step;
            st_save.opt_iter        = C.opt.opt_iter;
            if (!yue2_at_ckpt_save(ckpt_path, st_save, C, &err)) {
                fprintf(stderr, "[yue2-ar-train] checkpoint: %s\n", err.c_str());
                return 1;
            }
            const std::string snap =
                a.out_dir + "/" + a.name + "_step" + std::to_string((long long) step) + ".safetensors";
            if (!yue2_at_export(C.ad, a, target, step, mean_frames, m.lm_file.path, minted_present, snap,
                                &err, cursor_md)) {
                fprintf(stderr, "[yue2-ar-train] snapshot export: %s\n", err.c_str());
                return 1;
            }
            fprintf(stderr, "[yue2-ar-train] checkpoint at step %lld -> %s (+ snapshot)\n", (long long) step,
                    ckpt_path.c_str());
        }
    }

    if (a.eval_every > 0) {
        run_eval(a.steps);
    }

    const std::string out = a.out_dir + "/" + a.name + ".safetensors";
    if (!yue2_at_export(C.ad, a, target, a.steps, mean_frames, m.lm_file.path, minted_present, out, &err,
                        cursor_md)) {
        fprintf(stderr, "[yue2-ar-train] export: %s\n", err.c_str());
        return 1;
    }
    fprintf(stderr, "[yue2-ar-train] done: %lld steps, mean loss %.5f\n", (long long) a.steps,
            n_micro ? loss_sum / (double) n_micro : 0.0);
    if (attn != YUE2_AT_FA_EXACT) {
        fprintf(stderr, "[yue2-ar-train] resolved fused-attention precision: %s (the REQUEST was --attn %s)\n",
                dit_flash_prec_label(m.backend).c_str(), yue2_at_attn_name(attn));
    }
    if (!minted_present) {
        yue2_at_no_minted_banner();
    }
    fprintf(stderr, "[yue2-ar-train] the checkpoint ladder is what the ear test picks from — audition "
                    "600/800/1000/1200/1400/1600 against the base before shipping one.\n");

    // The resume state is the run's MIDDLE, not its result.
    hs_remove(ckpt_path);
    yue2_at_state_free(&st);
    yue2_at_train_ctx_free(&C);
    return 0;
}

// ── Entry point ────────────────────────────────────────────────────────────

static int yue2_ar_train_run(const Yue2ArTrainArgs & a) {
    if (a.fd_check > 0) {
        return yue2_ar_fdcheck_main(a);
    }
    if (a.fwd_check > 0) {
        return yue2_ar_forwardcheck_main(a);
    }
    return yue2_ar_train_loop(a);
}
