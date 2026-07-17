// ace-synth.cpp: ACE-Step synthesis CLI
// Thin wrapper: parses args, scans the model registry, calls pipeline-synth,
// writes output files. Model selection (synth_model, adapter, output_format)
// comes from the request JSON. The registry resolves names to GGUF paths
// under --models <dir> and --adapters <dir>.

#include "audio-io.h"
#include "backend.h"
#include "ggml.h"
#include "model-registry.h"
#include "model-store.h"
#include "pipeline-synth.h"
#include "request.h"
#include "synth-batch-runner.h"
#include "task-types.h"
#include "version.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ─── Per-section mask broadcast self-test (HOTSTEP_BCAST_TEST) ───────────────
// Verifies the exact tensor ops the per-section masking relies on, on the real
// backend, with no models. Test 1: ggml_mul([out,S,N] f32, [1,S,1] f32) — the
// per-frame mask broadcast over the feature (ne0) and batch (ne2) dims. Test 2:
// the real chain mul_mat(BF16 delta, f32 x) -> mul(mask). Run: set the env var
// and invoke ace-synth (it runs the test and exits).
static int run_bcast_selftest() {
    BackendPair    bp      = backend_init("BcastTest");
    ggml_backend_t backend = bp.backend;
    fprintf(stderr, "[BcastTest] backend=%s\n", ggml_backend_name(backend));

    const int64_t out = 8, S = 6, N = 2, in = 4;
    int           fails = 0;

    // Test 1: pure elementwise broadcast mul.
    {
        struct ggml_init_params p   = { ggml_tensor_overhead() * 16 + ggml_graph_overhead() + 4096, NULL, true };
        struct ggml_context *   ctx = ggml_init(p);
        struct ggml_tensor *    a    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, out, S, N);
        struct ggml_tensor *    mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, S, 1);
        ggml_set_input(a);
        ggml_set_input(mask);
        struct ggml_tensor * y  = ggml_mul(ctx, a, mask);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        std::vector<float>    ad((size_t) (out * S * N), 1.0f), md((size_t) S);
        for (int s = 0; s < S; s++) md[s] = (float) (s + 1) * 0.25f;  // distinct per frame
        ggml_backend_tensor_set(a, ad.data(), 0, ad.size() * sizeof(float));
        ggml_backend_tensor_set(mask, md.data(), 0, md.size() * sizeof(float));
        ggml_backend_graph_compute(backend, gf);
        std::vector<float> yd((size_t) (out * S * N));
        ggml_backend_tensor_get(y, yd.data(), 0, yd.size() * sizeof(float));
        int bad = 0;
        for (int n = 0; n < N; n++)
            for (int s = 0; s < S; s++)
                for (int f = 0; f < out; f++) {
                    float got = yd[(size_t) n * S * out + (size_t) s * out + f];
                    if (fabsf(got - md[s]) > 1e-4f) {
                        if (bad < 6) fprintf(stderr, "[BcastTest] T1 MISMATCH f=%d s=%d n=%d exp=%.3f got=%.3f\n", f, s, n, md[s], got);
                        bad++;
                    }
                }
        fprintf(stderr, "[BcastTest] T1 ggml_mul[out,S,N]x[1,S,1]: %s (%d/%lld bad)\n",
                bad ? "FAIL" : "PASS", bad, (long long) (out * S * N));
        fails += bad ? 1 : 0;
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    // Test 2: real chain — mul_mat(BF16 delta[in,out], f32 x[in,S,N]) then mask.
    {
        struct ggml_init_params p   = { ggml_tensor_overhead() * 32 + ggml_graph_overhead() + 4096, NULL, true };
        struct ggml_context *   ctx = ggml_init(p);
        struct ggml_tensor *    d    = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, in, out);
        struct ggml_tensor *    x    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in, S, N);
        struct ggml_tensor *    mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, S, 1);
        ggml_set_input(d);
        ggml_set_input(x);
        ggml_set_input(mask);
        struct ggml_tensor * dy   = ggml_mul_mat(ctx, d, x);      // [out,S,N]
        struct ggml_tensor * dym  = ggml_mul(ctx, dy, mask);
        ggml_set_output(dy);
        ggml_set_output(dym);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, dym);
        ggml_build_forward_expand(gf, dy);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        std::vector<ggml_bf16_t> dd((size_t) (in * out));
        for (int i = 0; i < in * out; i++) dd[i] = ggml_fp32_to_bf16(0.5f);  // all 0.5
        std::vector<float> xd((size_t) (in * S * N), 1.0f), md((size_t) S);
        for (int s = 0; s < S; s++) md[s] = (float) (s + 1) * 0.25f;
        ggml_backend_tensor_set(d, dd.data(), 0, dd.size() * sizeof(ggml_bf16_t));
        ggml_backend_tensor_set(x, xd.data(), 0, xd.size() * sizeof(float));
        ggml_backend_tensor_set(mask, md.data(), 0, md.size() * sizeof(float));
        ggml_backend_graph_compute(backend, gf);
        std::vector<float> dyd((size_t) (out * S * N)), dymd((size_t) (out * S * N));
        ggml_backend_tensor_get(dy, dyd.data(), 0, dyd.size() * sizeof(float));
        ggml_backend_tensor_get(dym, dymd.data(), 0, dymd.size() * sizeof(float));
        // Each dy element = sum_in(0.5 * 1.0) = in*0.5 = 2.0; masked = 2.0*md[s].
        int bad = 0;
        for (int n = 0; n < N; n++)
            for (int s = 0; s < S; s++)
                for (int f = 0; f < out; f++) {
                    size_t idx = (size_t) n * S * out + (size_t) s * out + f;
                    float  exp = dyd[idx] * md[s];
                    if (fabsf(dymd[idx] - exp) > 1e-3f) {
                        if (bad < 6) fprintf(stderr, "[BcastTest] T2 MISMATCH f=%d s=%d n=%d dy=%.3f exp=%.3f got=%.3f\n", f, s, n, dyd[idx], exp, dymd[idx]);
                        bad++;
                    }
                }
        fprintf(stderr, "[BcastTest] T2 dy=%.3f (expect 2.0); mask chain: %s (%d/%lld bad)\n",
                dyd[0], bad ? "FAIL" : "PASS", bad, (long long) (out * S * N));
        fails += bad ? 1 : 0;
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    fprintf(stderr, "[BcastTest] RESULT: %s\n", fails ? "FAIL — broadcast is the bug" : "PASS — broadcast is fine, look elsewhere");
    backend_release(bp.backend, bp.cpu_backend);
    return fails ? 1 : 0;
}

// ─── LoKr Kronecker-apply self-test (HOTSTEP_KRON_TEST) ──────────────────────
// Phase-2 prototype for low-rank runtime adapters (docs/plans/lowrank-runtime-
// adapters.md): verifies that (w1 ⊗ w2)@x can be computed on the real backend
// from the factors alone — no materialized Kronecker delta — via
//   y[oa·c+oc, s] = Σ_ib w1[oa,ib] · ( Σ_id w2[oc,id] · x[ib·d+id, s] )
// i.e. mul_mat(w2) → permute/cont → mul_mat(w1) → permute/cont. Pass 0 runs
// F32 factors (validates the choreography exactly), pass 1 runs BF16 factors
// (the production storage type). Compared against a host-side dense-kron
// reference. Batched inputs [in,S,N] flatten to [in,S·N] first, so 2D covers
// them. Run: set the env var and invoke ace-synth (runs the test and exits).
static int run_kron_selftest() {
    BackendPair    bp      = backend_init("KronTest");
    ggml_backend_t backend = bp.backend;
    fprintf(stderr, "[KronTest] backend=%s\n", ggml_backend_name(backend));

    // PyTorch shapes: w1 [a,b], w2 [c,d]; delta = kron(w1,w2) [out=a·c, in=b·d]
    const int64_t a = 3, b = 4, c = 5, d = 6, S = 7;
    const int64_t out = a * c, in = b * d;
    int           fails = 0;

    // deterministic fill (LCG) — same values every run and backend
    auto fill = [](std::vector<float> & v, uint32_t seed) {
        uint32_t s = seed;
        for (auto & f : v) {
            s = s * 1664525u + 1013904223u;
            f = ((float) (s >> 8) / (float) (1u << 24)) - 0.5f;  // [-0.5, 0.5)
        }
    };
    std::vector<float> w1d((size_t) (a * b)), w2d((size_t) (c * d)), xd((size_t) (in * S));
    fill(w1d, 1);
    fill(w2d, 2);
    fill(xd, 3);

    // Host reference: dense kron, then y = kron(w1,w2) @ x.
    // w1d/w2d are row-major PyTorch [out, in]: w1[oa,ib] = w1d[oa·b+ib].
    std::vector<float> yref((size_t) (out * S), 0.0f);
    for (int64_t oa = 0; oa < a; oa++)
        for (int64_t oc = 0; oc < c; oc++)
            for (int64_t s = 0; s < S; s++) {
                float acc = 0.0f;
                for (int64_t ib = 0; ib < b; ib++)
                    for (int64_t id = 0; id < d; id++)
                        acc += w1d[(size_t) (oa * b + ib)] * w2d[(size_t) (oc * d + id)]
                             * xd[(size_t) ((ib * d + id) + in * s)];
                yref[(size_t) ((oa * c + oc) + out * s)] = acc;
            }

    for (int pass = 0; pass < 2; pass++) {
        const bool      bf16  = (pass == 1);
        const ggml_type ftype = bf16 ? GGML_TYPE_BF16 : GGML_TYPE_F32;
        // F32 tol allows CUDA's TF32-accumulated cuBLAS matmul (~1e-3 rel), which
        // reorders/rounds vs the naive host reference; a wrong permute would be
        // off by whole values on most elements, not 1e-4 on a few.
        const float     tol   = bf16 ? 2e-2f : 1e-3f;

        struct ggml_init_params p   = { ggml_tensor_overhead() * 32 + ggml_graph_overhead() + 4096, NULL, true };
        struct ggml_context *   ctx = ggml_init(p);
        // Row-major PyTorch [rows, cols] uploads directly as ggml [cols, rows]:
        // w1g [b, a] element (ib, oa) == w1[oa, ib]; same for w2g [d, c].
        struct ggml_tensor * w1g = ggml_new_tensor_2d(ctx, ftype, b, a);
        struct ggml_tensor * w2g = ggml_new_tensor_2d(ctx, ftype, d, c);
        struct ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in, S);
        ggml_set_input(w1g);
        ggml_set_input(w2g);
        ggml_set_input(x);

        // x [in=b·d, S] viewed as [d, b·S]: column ib·d+id has id fastest — matches
        // the kron column convention, so the reshape is a free view.
        struct ggml_tensor * X2 = ggml_reshape_2d(ctx, x, d, b * S);
        struct ggml_tensor * T  = ggml_mul_mat(ctx, w2g, X2);                         // [c, b·S] = T(oc; ib,s)
        struct ggml_tensor * T3 = ggml_reshape_3d(ctx, T, c, b, S);
        struct ggml_tensor * P  = ggml_cont(ctx, ggml_permute(ctx, T3, 1, 0, 2, 3));  // [b, c, S]
        struct ggml_tensor * P2 = ggml_reshape_2d(ctx, P, b, c * S);
        struct ggml_tensor * Y  = ggml_mul_mat(ctx, w1g, P2);                         // [a, c·S] = y(oa; oc,s)
        struct ggml_tensor * Y3 = ggml_reshape_3d(ctx, Y, a, c, S);
        struct ggml_tensor * YP = ggml_cont(ctx, ggml_permute(ctx, Y3, 1, 0, 2, 3));  // [c, a, S] → flat out=oa·c+oc
        ggml_set_output(YP);

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, YP);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

        if (bf16) {
            std::vector<ggml_bf16_t> w1b((size_t) (a * b)), w2b((size_t) (c * d));
            ggml_fp32_to_bf16_row(w1d.data(), w1b.data(), a * b);
            ggml_fp32_to_bf16_row(w2d.data(), w2b.data(), c * d);
            ggml_backend_tensor_set(w1g, w1b.data(), 0, w1b.size() * sizeof(ggml_bf16_t));
            ggml_backend_tensor_set(w2g, w2b.data(), 0, w2b.size() * sizeof(ggml_bf16_t));
        } else {
            ggml_backend_tensor_set(w1g, w1d.data(), 0, w1d.size() * sizeof(float));
            ggml_backend_tensor_set(w2g, w2d.data(), 0, w2d.size() * sizeof(float));
        }
        ggml_backend_tensor_set(x, xd.data(), 0, xd.size() * sizeof(float));
        ggml_backend_graph_compute(backend, gf);

        std::vector<float> yd((size_t) (out * S));
        ggml_backend_tensor_get(YP, yd.data(), 0, yd.size() * sizeof(float));

        int   bad     = 0;
        float max_err = 0.0f;
        for (int64_t o = 0; o < out; o++)
            for (int64_t s = 0; s < S; s++) {
                float exp = yref[(size_t) (o + out * s)];
                float got = yd[(size_t) (o + out * s)];
                float err = fabsf(got - exp);
                if (err > max_err) max_err = err;
                if (err > tol) {
                    if (bad < 6) fprintf(stderr, "[KronTest] %s MISMATCH o=%lld s=%lld exp=%.5f got=%.5f\n",
                                         bf16 ? "BF16" : "F32", (long long) o, (long long) s, exp, got);
                    bad++;
                }
            }
        fprintf(stderr, "[KronTest] %s factors: %s (%d/%lld bad, max_err=%.3g, tol=%.3g)\n",
                bf16 ? "BF16" : "F32", bad ? "FAIL" : "PASS", bad, (long long) (out * S), max_err, tol);
        fails += bad ? 1 : 0;
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    fprintf(stderr, "[KronTest] RESULT: %s\n",
            fails ? "FAIL — Kronecker apply choreography is wrong" : "PASS — LoKr factor apply is viable on this backend");
    backend_release(bp.backend, bp.cpu_backend);
    return fails ? 1 : 0;
}

static void usage(const char * prog) {
    AceSynthParams d;
    ace_synth_default_params(&d);

    fprintf(stderr, "acestep.cpp %s\n\n", ACE_VERSION);
    fprintf(stderr,
            "Usage: %s --models <dir> --request <json...> [options]\n\n"
            "Required:\n"
            "  --models <dir>          Directory of GGUF model files\n"
            "  --request <json...>     One or more request JSONs (from ace-lm --request)\n\n"
            "Optional:\n"
            "  --adapters <dir>        Directory of adapter files (enables JSON adapter field)\n"
            "  --src-audio <file>      Source audio (WAV or MP3)\n"
            "  --ref-audio <file>      Timbre reference audio (WAV or MP3)\n\n"
            "Model selection comes from the request JSON: synth_model picks the DiT,\n"
            "adapter picks an adapter from --adapters, output_format picks the output\n"
            "extension. When synth_model is empty the first DiT in the registry is used;\n"
            "text-encoder and VAE are always the first in their registry bucket.\n\n"
            "Audio encoding:\n"
            "  --mp3-bitrate <kbps>    MP3 bitrate (default: 128)\n\n"
            "Memory control:\n"
            "  --vae-chunk <N>         Latent frames per tile (default: %d)\n"
            "  --vae-overlap <N>       Overlap frames per side (default: %d)\n\n"
            "Debug:\n"
            "  --no-fa                 Disable flash attention\n"
            "  --no-batch-cfg          Split DiT CFG into two separate forwards\n"
            "  --clamp-fp16            Clamp hidden states to FP16 range\n"
            "  --dump <dir>            Dump intermediate tensors\n",
            prog, d.vae_chunk, d.vae_overlap);
}

int main(int argc, char ** argv) {
    if (std::getenv("HOTSTEP_BCAST_TEST")) {
        return run_bcast_selftest();
    }
    if (std::getenv("HOTSTEP_KRON_TEST")) {
        return run_kron_selftest();
    }
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    // Defaults live in ace_synth_default_params. CLI locals read from params
    // so there is exactly one place in the codebase that picks the numbers.
    AceSynthParams params;
    ace_synth_default_params(&params);

    std::vector<const char *> request_paths;
    const char *              models_dir     = NULL;
    const char *              adapters_dir   = NULL;
    const char *              src_audio_path = NULL;
    const char *              ref_audio_path = NULL;
    const char *              dump_dir       = NULL;
    bool                      use_fa         = true;
    bool                      use_batch_cfg  = true;
    bool                      clamp_fp16     = false;
    int                       vae_chunk      = params.vae_chunk;
    int                       vae_overlap    = params.vae_overlap;
    int                       mp3_kbps       = 128;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--request")) {
            // Collect all following non-option args
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                request_paths.push_back(argv[++i]);
            }
        } else if (!strcmp(argv[i], "--models") && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (!strcmp(argv[i], "--adapters") && i + 1 < argc) {
            adapters_dir = argv[++i];
        } else if (!strcmp(argv[i], "--src-audio") && i + 1 < argc) {
            src_audio_path = argv[++i];
        } else if (!strcmp(argv[i], "--ref-audio") && i + 1 < argc) {
            ref_audio_path = argv[++i];
        } else if (!strcmp(argv[i], "--dump") && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (!strcmp(argv[i], "--no-fa")) {
            use_fa = false;
        } else if (!strcmp(argv[i], "--no-batch-cfg")) {
            use_batch_cfg = false;
        } else if (!strcmp(argv[i], "--clamp-fp16")) {
            clamp_fp16 = true;
        } else if (!strcmp(argv[i], "--vae-chunk") && i + 1 < argc) {
            vae_chunk = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--vae-overlap") && i + 1 < argc) {
            vae_overlap = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--mp3-bitrate") && i + 1 < argc) {
            mp3_kbps = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!models_dir) {
        fprintf(stderr, "[CLI] ERROR: --models required\n");
        usage(argv[0]);
        return 1;
    }
    if (request_paths.empty()) {
        fprintf(stderr, "[CLI] ERROR: --request required\n");
        usage(argv[0]);
        return 1;
    }

    // Parse all requests first: the first request drives model selection.
    int                      batch_n = (int) request_paths.size();
    std::vector<AceRequest>  reqs(batch_n);
    std::vector<std::string> basenames(batch_n);
    for (int ri = 0; ri < batch_n; ri++) {
        const char * rpath = request_paths[ri];
        if (!request_parse(&reqs[ri], rpath)) {
            fprintf(stderr, "[Ace-Synth] FATAL: failed to parse %s\n", rpath);
            return 1;
        }
        request_dump(&reqs[ri], stderr);
        if (reqs[ri].caption.empty() && reqs[ri].task_type != TASK_LEGO && reqs[ri].task_type != TASK_EXTRACT &&
            reqs[ri].task_type != TASK_COMPLETE) {
            fprintf(stderr, "[Ace-Synth] FATAL: caption is empty in %s\n", rpath);
            return 1;
        }
        // output basename: strip .json suffix
        basenames[ri] = rpath;
        size_t dot    = basenames[ri].rfind(".json");
        if (dot != std::string::npos) {
            basenames[ri] = basenames[ri].substr(0, dot);
        }
    }
    fprintf(stderr, "[Ace-Synth] Batch: %d request(s)\n", batch_n);

    // Scan the registry and resolve model paths from the first request.
    ModelRegistry registry;
    if (!registry_scan(&registry, models_dir)) {
        fprintf(stderr, "[Ace-Synth] FATAL: cannot scan --models %s\n", models_dir);
        return 1;
    }
    if (adapters_dir) {
        registry_scan_adapters(&registry, adapters_dir);
    }
    if (registry.dit.empty() || registry.text_enc.empty() || registry.vae.empty()) {
        fprintf(stderr, "[Ace-Synth] FATAL: registry needs DiT, text-encoder and VAE models\n");
        return 1;
    }
    const ModelEntry * dit_entry =
        reqs[0].synth_model.empty() ? &registry.dit[0] : registry_find(registry.dit, reqs[0].synth_model.c_str());
    if (!dit_entry) {
        fprintf(stderr, "[Ace-Synth] FATAL: synth_model '%s' not found in registry\n", reqs[0].synth_model.c_str());
        return 1;
    }
    const AdapterEntry * adapter_entry = NULL;
    if (!reqs[0].adapter.empty()) {
        adapter_entry = registry_find_adapter(registry, reqs[0].adapter.c_str());
        if (!adapter_entry) {
            fprintf(stderr, "[Ace-Synth] FATAL: adapter '%s' not found (use --adapters <dir>)\n",
                    reqs[0].adapter.c_str());
            return 1;
        }
    }

    // Multi-adapter stack: the `adapters` array supersedes the single `adapter`
    // field. Fold the single field into a one-element stack so the load path is
    // uniform. The resolved stack drives merge/runtime loading via the sideband.
    g_hotstep_params.adapters.clear();
    {
        std::vector<AceAdapterRef> stack = reqs[0].adapters;
        if (stack.empty() && adapter_entry) {
            stack.push_back({ reqs[0].adapter, reqs[0].adapter_scale });
        }
        for (const auto & ar : stack) {
            const AdapterEntry * e = registry_find_adapter(registry, ar.name.c_str());
            std::string          path;
            if (e) {
                path = e->path;
            } else {
                FILE * t = fopen(ar.name.c_str(), "rb");
                if (t) { fclose(t); path = ar.name; }
            }
            if (path.empty()) {
                fprintf(stderr, "[Ace-Synth] FATAL: adapter '%s' not found (use --adapters <dir>)\n",
                        ar.name.c_str());
                return 1;
            }
            g_hotstep_params.adapters.push_back({ path, ar.scale });
        }
    }

    // Resolve output_format to (is_mp3, wav_fmt).
    bool      is_mp3  = true;
    WavFormat wav_fmt = WAV_S16;
    if (!audio_parse_format(reqs[0].output_format.c_str(), is_mp3, wav_fmt)) {
        fprintf(stderr, "[Ace-Synth] FATAL: invalid output_format '%s' (use: mp3, wav16, wav24, wav32)\n",
                reqs[0].output_format.c_str());
        return 1;
    }

    // Fill params from registry lookups and CLI flags.
    params.text_encoder_path = registry.text_enc[0].path.c_str();
    params.dit_path          = dit_entry->path.c_str();
    params.vae_path          = registry.vae[0].path.c_str();
    params.adapter_path      = g_hotstep_params.adapters.empty() ? NULL
                                                                  : g_hotstep_params.adapters[0].path.c_str();
    params.adapter_scale     = g_hotstep_params.adapters.empty() ? 1.0f
                                                                 : g_hotstep_params.adapters[0].scale;
    params.use_fa            = use_fa;
    params.use_batch_cfg     = use_batch_cfg;
    params.clamp_fp16        = clamp_fp16;
    params.vae_chunk         = vae_chunk;
    params.vae_overlap       = vae_overlap;
    params.dump_dir          = dump_dir;

    // Local store with the default STRICT policy: at most one GPU module
    // resident at a time for this one-shot CLI. No module sharing across runs,
    // so EVICT_STRICT frees the DiT before the VAE loads, and so on.
    ModelStore * store = store_create(EVICT_STRICT);
    AceSynth *   ctx   = ace_synth_load(store, &params);
    if (!ctx) {
        store_free(store);
        return 1;
    }

    // Read source audio (cover/lego mode)
    float * src_interleaved = NULL;
    int     src_len         = 0;
    if (src_audio_path) {
        int     T_audio = 0;
        float * planar  = audio_read_48k(src_audio_path, &T_audio);
        if (!planar) {
            fprintf(stderr, "[Ace-Synth] FATAL: cannot read --src-audio %s\n", src_audio_path);
            ace_synth_free(ctx);
            store_free(store);
            return 1;
        }
        fprintf(stderr, "[Ace-Synth] Source audio: %.2fs @ 48kHz\n", (float) T_audio / 48000.0f);

        src_interleaved = audio_planar_to_interleaved(planar, T_audio);
        free(planar);
        src_len = T_audio;
    }

    // Read reference audio (timbre conditioning)
    float * ref_interleaved = NULL;
    int     ref_len         = 0;
    if (ref_audio_path) {
        int     T_audio = 0;
        float * planar  = audio_read_48k(ref_audio_path, &T_audio);
        if (!planar) {
            fprintf(stderr, "[Ace-Synth] FATAL: cannot read --ref-audio %s\n", ref_audio_path);
            free(src_interleaved);
            ace_synth_free(ctx);
            store_free(store);
            return 1;
        }
        fprintf(stderr, "[Ace-Synth] Reference audio: %.2fs @ 48kHz\n", (float) T_audio / 48000.0f);
        ref_interleaved = audio_planar_to_interleaved(planar, T_audio);
        free(planar);
        ref_len = T_audio;
    }

    // Generate every request in one DiT batch. synth_batch_size expands each
    // request into per-seed variants in groups[0]. Total clamped to DiT max 9.
    int total_alloc = 0;
    for (int ri = 0; ri < batch_n; ri++) {
        int sbs = reqs[ri].synth_batch_size;
        total_alloc += sbs < 1 ? 1 : (sbs > 9 ? 9 : sbs);
    }
    if (total_alloc > 9) {
        fprintf(stderr, "[Ace-Synth] Batch %d exceeds DiT max 9, clamping\n", total_alloc);
        total_alloc = 9;
    }
    std::vector<AceAudio>                all_audio(total_alloc);
    std::vector<std::string>             all_basenames(total_alloc);
    std::vector<int>                     all_synth_indices(total_alloc);
    std::vector<std::vector<AceRequest>> groups(1);
    groups[0].reserve(total_alloc);

    int off = 0;
    for (int ri = 0; ri < batch_n && off < total_alloc; ri++) {
        int sbs = reqs[ri].synth_batch_size;
        if (sbs < 1) {
            sbs = 1;
        }
        if (sbs > 9) {
            sbs = 9;
        }
        if (off + sbs > total_alloc) {
            sbs = total_alloc - off;
        }

        // resolve seed once per original request
        request_resolve_seed(&reqs[ri]);
        const long long base_seed = reqs[ri].seed;

        for (int i = 0; i < sbs; i++) {
            AceRequest r = reqs[ri];
            r.seed       = base_seed + i;
            groups[0].push_back(r);
            all_basenames[off + i]     = basenames[ri];
            all_synth_indices[off + i] = i;
        }
        off += sbs;
    }

    if (total_alloc > 1) {
        fprintf(stderr, "[Ace-Synth] Batch: %d track(s) from %d request(s)\n", total_alloc, batch_n);
    }

    // Two-phase run: DiT resident for all groups, then VAE for all jobs.
    const int rc = synth_batch_run(ctx, groups, src_interleaved, src_len,
                                   nullptr, 0,  // src_latents
                                   ref_interleaved, ref_len,
                                   nullptr, 0,  // ref_latents
                                   all_audio.data());
    if (rc != 0) {
        fprintf(stderr, "[Ace-Synth] ERROR: batch run failed\n");
        for (auto & a : all_audio) {
            ace_audio_free(&a);
        }
        free(src_interleaved);
        free(ref_interleaved);
        ace_synth_free(ctx);
        store_free(store);
        return 1;
    }

    // Write output files
    for (int b = 0; b < (int) all_audio.size(); b++) {
        if (!all_audio[b].samples) {
            continue;
        }
        const char * ext = is_mp3 ? ".mp3" : ".wav";
        char         out_path[1024];
        snprintf(out_path, sizeof(out_path), "%s%d%s", all_basenames[b].c_str(), all_synth_indices[b], ext);
        if (!audio_write(out_path, all_audio[b].samples, all_audio[b].n_samples, 48000, mp3_kbps, wav_fmt)) {
            fprintf(stderr, "[Ace-Synth Batch%d] FATAL: failed to write %s\n", b, out_path);
        }
        ace_audio_free(&all_audio[b]);
    }

    free(src_interleaved);
    free(ref_interleaved);
    ace_synth_free(ctx);
    store_free(store);
    fprintf(stderr, "[Ace-Synth] All done\n");
    return 0;
}
