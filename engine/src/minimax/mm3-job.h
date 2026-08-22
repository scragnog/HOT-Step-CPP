#pragma once
// minimax/mm3-job.h — POST /mm3/synth on the SHARED GPU worker queue.
//
// HOT-Step file (does not exist upstream). This is the SECOND hook into
// engine/tools/hot-step-server.cpp, and unlike minimax/mm3-server.h it is a
// MID-FILE include, deliberately:
//
//     hot-step-server.cpp:86    #include "minimax/mm3-server.h"   (top: routes)
//     hot-step-server.cpp:~410  #include "minimax/mm3-job.h"      (after the job system)
//                               mm3_register_job_routes(svr);     (next to mm3_register_routes)
//
// It has to come after `Job`, `job_create`, `job_set_phase`, `work_push` and
// `g_store` exist, and all of those are defined in the middle of that file.
// Both halves are checked by engine/verify-hooks.ps1.
//
// ── Why this exists ─────────────────────────────────────────────────────────
//
// POST /mm3/synth-e2e (mm3-server.h) does minutes of GPU work on an httplib
// thread, racing the ACE worker for the device. This endpoint does the same
// generation as a first-class engine job: same FIFO queue, same single worker
// thread, same GET /job?id=&result=1 retrieval, same POST /job?id=&cancel=1.
// MM3 and ACE therefore serialise on the GPU by construction, exactly as two
// ACE jobs do.
//
// ── Job-system integration points ───────────────────────────────────────────
//
//   job_create()                  allocates the Job + its id, evicts old ones
//   work_push(fn)                 hands `fn` to the ONE worker thread (FIFO)
//   job_set_phase(job, p, s, t)   phase + sub-progress, read by GET /job
//   job->cancel                   set by POST /job?id=&cancel=1
//   job->result_body/result_mime  read by GET /job?id=&result=1
//   job->status                   0 running, 1 done, 2 failed, 3 cancelled
//
// ── Phase mapping ───────────────────────────────────────────────────────────
//
// `JobPhase` is the ACE pipeline's vocabulary and lives in hot-step-server.cpp;
// adding MM3-specific members would be an edit to an upstream-derived file that
// a sync could silently revert. Instead the MM3 stages are mapped onto the
// existing phases by ROLE, so every existing poller keeps working:
//
//   VRAM arbitration + MM3 warm  -> loading_dit        (0/0)
//   AR planning stage            -> encoding_cond      (frame / max_frames)
//   condition encoder            -> encoding_cond      (window / n_windows)
//   flow DiT (Euler loop)        -> dit_inference      (k*steps + i / NW*steps)
//   vocoder                      -> vae_decode         (window / n_windows)
//   stitch + WAV encode          -> encoding_output    (0/0)
//
// The mapping is honest rather than merely convenient: the AR stage's whole
// output IS the conditioning block the condition encoder consumes, the flow
// stage IS a DiT, and the vocoder IS the latent->audio decoder. `dit_inference`
// counts across ALL windows so its step is monotonic for the whole run.
//
// Callers that want the real MM3 vocabulary poll GET /mm3/job?id=... instead,
// which reports the raw stage name plus the run's shape, timings and seed.
//
// ── Streaming (GET /mm3/stream?id=) ─────────────────────────────────────────
//
// Opt in with `"stream": true` on the request. Each 200-frame window's audio is
// final the moment it is vocoded and cropped, so it is pushed to a per-job
// queue and served as a chunked body of concatenated WAVs while the render
// continues. It is an ADDITIONAL output: the complete WAV is still built,
// encoded onto Job::result_body and fetched the usual way, so the song row,
// the library entry, mastering and stems are untouched.

#include "mm3-ar-cache.h"
#include "mm3-lm-merge.h"
#include "mm3-request.h"
#include "mm3-server.h"

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>

// Headroom on top of the two GGUFs' tensor bytes when deciding whether ACE
// models have to go. The AR stage's KV cache is sized from the request and
// computed exactly below; this covers the ggml compute buffers of the five
// graphs (LM prefill/decode, depth, cond, DiT, vocoder), which are allocated
// lazily and not knowable before the first run.
#define MM3_VRAM_COMPUTE_HEADROOM ((size_t) 3 * 1024 * 1024 * 1024)

// ── Streaming: the per-job chunk queue ──────────────────────────────────────
//
// The GPU worker pushes finished windows in (mm3-pipeline.h MM3ChunkCb); the
// GET /mm3/stream handler, on an httplib thread, pops them out.
//
// It has its OWN mutex and that is load-bearing twice over. It must never take
// g_mm3_mutex — an in-flight generation holds that for its entire run, which is
// exactly why GET /mm3/props blocks and GET /mm3/job does not, and a stream
// that blocked on it could not deliver the audio of the very job holding it.
// And it must not share MM3JobState::mtx either, which the worker locks on
// every Euler step to publish progress.
//
// Each entry is a COMPLETE, self-contained WAV — same transport shape as
// POST /api/generate/storm/stream, so ui/src/hooks/useStreamAudio.ts's
// WAV-extraction helper reads this byte stream unmodified.
struct MM3StreamQueue {
    std::mutex              mtx;
    std::condition_variable cv;
    std::deque<std::string> chunks;
    bool                    finished  = false;  // the run ended (any outcome)
    bool                    abandoned = false;  // buffered past the cap with nobody reading
    std::string             error;              // set when the run ended badly
    int64_t                 bytes_queued = 0;   // sitting in the deque right now (under mtx)

    // ATOMIC, deliberately not guarded by `mtx`. GET /mm3/job reads these while
    // already holding MM3JobState::mtx — the lock the worker takes on every
    // Euler step to publish progress. Reaching for `mtx` from inside that would
    // nest two locks the other paths take independently, and would put the
    // generation behind an HTTP handler. Nothing here needs a consistent
    // snapshot; they are counters for a progress line.
    std::atomic<bool>    attached{ false };  // a reader currently holds the stream
    std::atomic<int64_t> bytes_total{ 0 };
    std::atomic<int64_t> n_chunks{ 0 };
    std::atomic<int>     sample_rate{ 0 };
};

// Safety valve, not a policy: the server attaches its reader immediately after
// POST /mm3/synth returns, long before the first window exists, so in practice
// nothing accumulates. This bounds the damage when nobody ever reads — a
// submitted-and-forgotten stream, or a browser tab that went away mid-render.
//
// 128 MB, chosen to clear the longest render MM3 accepts: 360 s of 16-bit
// 44.1 kHz stereo is 63.5 MB, so a user who submits and only presses Listen at
// the end still gets the whole song. Sized to the format, not to a round
// number — 32-bit float output is 4x, and that one CAN hit the cap.
#define MM3_STREAM_MAX_UNREAD_BYTES ((int64_t) 128 * 1024 * 1024)

/** Hand one finished window to whoever is reading. Never blocks the worker. */
static void mm3_stream_push(MM3StreamQueue & q, std::string wav, int sample_rate) {
    std::lock_guard<std::mutex> lock(q.mtx);
    if (q.abandoned || q.finished) {
        return;
    }
    q.sample_rate.store(sample_rate);
    if (!q.attached.load() && q.bytes_queued + (int64_t) wav.size() > MM3_STREAM_MAX_UNREAD_BYTES) {
        q.abandoned = true;
        q.chunks.clear();
        q.bytes_queued = 0;
        fprintf(stderr, "[MM3-Stream] no reader after %.0f MB buffered - dropping the stream (the render continues)\n",
                (double) q.bytes_total.load() / 1048576.0);
        q.cv.notify_all();
        return;
    }
    q.bytes_queued += (int64_t) wav.size();
    q.bytes_total.fetch_add((int64_t) wav.size());
    q.n_chunks.fetch_add(1);
    q.chunks.push_back(std::move(wav));
    q.cv.notify_all();
}

/** End the stream. MUST be called on every exit path of the worker — a reader
 *  blocked on the condition variable has no other way to learn the run is over.
 *  Queued chunks are left in place so a reader still drains what did render. */
static void mm3_stream_close(MM3StreamQueue & q, const std::string & error) {
    std::lock_guard<std::mutex> lock(q.mtx);
    if (!q.finished) {
        q.finished = true;
        q.error    = error;
    }
    q.cv.notify_all();
}

// ── Per-job MM3 state ───────────────────────────────────────────────────────

// Live stage + finished-run detail for GET /mm3/job. Kept beside the engine's
// own Job rather than inside it so hot-step-server.cpp's Job struct is
// untouched.
struct MM3JobState {
    std::mutex  mtx;
    std::string stage = "queued";  // "queued" | "arbitrating" | "warming" | mm3_generate's stages | "encoding" | terminal
    int64_t     window    = -1;
    int64_t     n_windows = 0;
    int64_t     step      = 0;
    int64_t     n_steps   = 0;
    std::string error;

    // Echoed straight back so a caller that sent seed = -1 learns the seed it
    // actually got, and can reproduce the take.
    uint64_t seed       = 0;
    int64_t  max_frames = 0;
    int64_t  n_tokens   = 0;
    bool     instrumental = false;
    /** True when stage 1 was served from the AR cache (no LM, no AR compute). */
    bool     ar_cached  = false;
    /** True when windows were dispatched WHILE the planner ran (co-resident).
     *  False on a streamed run means the audio starts after planning, which is
     *  the difference a user actually feels. */
    bool     interleaved = false;

    bool    have_result = false;
    int64_t frames      = 0;
    int64_t n_samples   = 0;
    int     sample_rate = 0;
    double  duration_s  = 0.0;
    double  rms         = 0.0;
    double  peak        = 0.0;
    bool    eos         = false;
    bool    has_nan     = false;
    double  ar_ms = 0.0, cond_ms = 0.0, flow_ms = 0.0, voc_ms = 0.0, total_ms = 0.0;
    double  arbitrate_ms = 0.0, warm_ms = 0.0;
    size_t  evicted_bytes = 0;
    int     evicted_modules = 0;

    /** Live chunk queue, allocated only when the request asked for streaming.
     *  Null means this job produces no stream and GET /mm3/stream will say so.
     *  shared_ptr because the reader outlives nothing in particular: the job
     *  state is evicted on a 32-entry LRU while a slow client may still be
     *  draining. */
    std::shared_ptr<MM3StreamQueue> stream;
};

static std::mutex                                                   g_mm3_jobs_mtx;
static std::map<std::string, std::shared_ptr<MM3JobState>>          g_mm3_jobs;
static std::deque<std::string>                                      g_mm3_job_order;

// hot-step-server.cpp keeps at most MAX_JOBS = 32 jobs; mirror that so this
// side table cannot outgrow it.
static void mm3_job_state_put(const std::string & id, std::shared_ptr<MM3JobState> st) {
    std::lock_guard<std::mutex> lock(g_mm3_jobs_mtx);
    g_mm3_jobs[id] = std::move(st);
    g_mm3_job_order.push_back(id);
    while (g_mm3_job_order.size() > 32) {
        g_mm3_jobs.erase(g_mm3_job_order.front());
        g_mm3_job_order.pop_front();
    }
}

static std::shared_ptr<MM3JobState> mm3_job_state_get(const std::string & id) {
    std::lock_guard<std::mutex> lock(g_mm3_jobs_mtx);
    auto                        it = g_mm3_jobs.find(id);
    return it != g_mm3_jobs.end() ? it->second : nullptr;
}

// ── VRAM arbitration ────────────────────────────────────────────────────────

// Exact KV-cache cost for this run: one [key_length, n_ctx, head_count_kv, 2]
// F16 tensor per block (mm3-lm-graph.h:190-196), n_ctx bucketed the same way
// mm3_lm_prepare() buckets it.
static size_t mm3_kv_bytes_estimate(const MM3Model & m, int64_t n_ctx_needed) {
    const MM3LmConfig & c = m.lm_cfg;
    const int64_t       b = MM3_LM_KV_BUCKET;
    const int64_t       n = ((n_ctx_needed + b - 1) / b) * b;
    return (size_t) c.block_count * (size_t) c.key_length * (size_t) c.head_count_kv * 2 /*K+V*/ * 2 /*f16*/ *
           (size_t) n;
}

// Weights + KV + compute headroom. Distinct files only — in bundle mode the
// four role files alias one GGUF and must not be counted four times.
//
// `stage2_only` is the AR-cache-hit shape: the LM is never loaded and there is
// no KV cache, so asking for the full figure would evict the ACE side for
// VRAM this run will not touch.
static size_t mm3_vram_need(const MM3Model & m, int64_t n_ctx_needed, bool stage2_only = false) {
    uint64_t weights = mm3_total_tensor_bytes(m);
    if (stage2_only) {
        weights -= m.lm_file.found ? m.lm_file.tensor_bytes : 0;  // the LM is never a role file
        return (size_t) weights + MM3_VRAM_COMPUTE_HEADROOM;
    }
    return (size_t) weights + mm3_kv_bytes_estimate(m, n_ctx_needed) + MM3_VRAM_COMPUTE_HEADROOM;
}

static bool mm3_cuda_free_bytes(size_t * free_b, size_t * total_b) {
#ifdef GGML_USE_CUDA
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) {
        return false;
    }
    *free_b  = f;
    *total_b = t;
    return true;
#else
    (void) free_b;
    (void) total_b;
    return false;
#endif
}

// If MM3 will not fit alongside whatever the ACE pipeline has resident, evict
// the ACE side.
//
// MECHANISM: store_evict_all(g_store, &still) — the ModelStore's own
// force-eviction pass (engine/src/model-store.h:174-179). It drops EVERY
// unreferenced GPU module (LM, DiT, VAE-Enc/Dec, text encoder, ...) regardless
// of the eviction policy, so it works even under --keep-loaded / EVICT_NEVER,
// which is exactly the case that would otherwise OOM. It refuses to touch
// modules with refcount > 0, and reports how many stayed.
//
// This runs ON the GPU worker thread, so by construction no ACE job is
// mid-flight and no module should be referenced: `still > 0` means something
// leaked a handle and is worth a loud line in the log.
//
// Non-CUDA builds cannot measure free VRAM; there the eviction is
// unconditional whenever ACE modules are resident and MM3 is cold, which is
// the conservative choice.
static void mm3_arbitrate_vram(const MM3Model & m, int64_t n_ctx_needed, MM3JobState * st,
                               bool stage2_only = false) {
    if (m.loaded) {
        return;  // already resident: whatever fit before still fits
    }
    const size_t need = mm3_vram_need(m, n_ctx_needed, stage2_only);

    size_t free_b = 0, total_b = 0;
    const bool have_cuda = mm3_cuda_free_bytes(&free_b, &total_b);

    const size_t ace_bytes = g_store ? store_vram_bytes(g_store) : 0;
    const int    ace_mods  = g_store ? store_gpu_module_count(g_store) : 0;

    if (ace_mods == 0) {
        fprintf(stderr, "[MM3-Job] VRAM: need %.2f GB, ACE store empty, free %.2f GB - no arbitration\n",
                (double) need / 1073741824.0, (double) free_b / 1073741824.0);
        return;
    }
    if (have_cuda && free_b >= need) {
        fprintf(stderr, "[MM3-Job] VRAM: need %.2f GB, free %.2f GB with %d ACE module(s) resident - they can stay\n",
                (double) need / 1073741824.0, (double) free_b / 1073741824.0, ace_mods);
        return;
    }

    fprintf(stderr,
            "[MM3-Job] VRAM: need %.2f GB, free %.2f GB, %d ACE module(s) holding %.2f GB - evicting the ACE side\n",
            (double) need / 1073741824.0, (double) free_b / 1073741824.0, ace_mods, (double) ace_bytes / 1073741824.0);

    int       still = 0;
    const int freed = store_evict_all(g_store, &still);
    if (still > 0) {
        fprintf(stderr, "[MM3-Job] VRAM: %d ACE module(s) still referenced after eviction - MM3 may not fit\n", still);
    }
    if (st) {
        std::lock_guard<std::mutex> lock(st->mtx);
        st->evicted_modules = freed;
        st->evicted_bytes   = ace_bytes - store_vram_bytes(g_store);
    }

    if (mm3_cuda_free_bytes(&free_b, &total_b)) {
        fprintf(stderr, "[MM3-Job] VRAM: freed %d module(s), free now %.2f GB of %.2f GB\n", freed,
                (double) free_b / 1073741824.0, (double) total_b / 1073741824.0);
    }
}

// ── AR cache key ────────────────────────────────────────────────────────────
//
// The slot itself lives in mm3-ar-cache.h (it has to be visible to
// mm3-server.h's unload/select-model handlers too). What lives here is the
// KEY, because building it needs the parsed request.
//
// THE KEY IS THE WHOLE CORRECTNESS ARGUMENT. Anything that changes the emitted
// hiddens must appear in it, or the feature becomes "my knob does nothing" —
// the exact failure the ACE-side lmCache learned the hard way (a repetition
// penalty missing from computeLmCacheKey looked like a dead slider). Stored as
// a plain string and compared exactly: no hashing, no collisions.

/** Every input that can change the frame-hidden block, in one exact string.
 *
 *  Deliberately NOT in here, because stage 2 consumes them and re-runs anyway:
 *  steps, cfg_flow, the solver/scheduler/guidance plugin selection and their
 *  params, flow_shift, forced_noise, wav_bits, and the cond/dit/voc model
 *  choices. Those are precisely the knobs this cache exists to let you tweak. */
static std::string mm3_ar_cache_key(const MM3Model & m, const MM3SynthRequest & req) {
    std::string k;
    k.reserve(req.prompt.size() + 1024);

    auto add_s = [&](const char * n, const std::string & v) { k += n; k += '='; k += v; k += '\n'; };
    auto add_i = [&](const char * n, long long v) { k += n; k += '='; k += std::to_string(v); k += '\n'; };
    auto add_u = [&](const char * n, unsigned long long v) { k += n; k += '='; k += std::to_string(v); k += '\n'; };
    auto add_f = [&](const char * n, double v) {
        char b[64];
        snprintf(b, sizeof(b), "%.9g", v);
        k += n; k += '='; k += b; k += '\n';
    };
    // FNV-1a over a code array — the plank blobs are the one input too big to
    // inline, and a digest of them is enough to tell two planks apart.
    auto add_codes = [&](const char * n, const std::vector<int32_t> & v) {
        uint64_t h = 1469598103934665603ULL;
        for (const int32_t x : v) {
            const uint32_t u = (uint32_t) x;
            for (int b = 0; b < 4; b++) {
                h = (h ^ (uint64_t) ((u >> (b * 8)) & 0xFF)) * 1099511628211ULL;
            }
        }
        k += n; k += '='; k += std::to_string(v.size()); k += ':'; k += std::to_string(h); k += '\n';
    };

    // Bump when the AR path itself changes shape (new sampler behaviour, a
    // different hidden layout) so stale slots from an older binary can't hit.
    add_i("v", 1);

    // The prompt carries caption + lyrics + the instrumental substitution, and
    // the tokenizer is a pure function of it plus the LM GGUF (below).
    add_s("prompt", req.prompt);
    add_i("max_frames", req.max_frames);
    add_u("ar_seed", req.gen.ar_seed_set ? req.gen.ar_seed : req.gen.seed);
    // The alignment heads run a manual-attention path on 3 of 36 layers, so
    // asking for LRC can move the emitted codes — and the LRC text itself is
    // cached alongside, so a hit must not be able to hand back a missing one.
    add_i("lrc", (req.want_lrc && !req.instrumental) ? 1 : 0);

    // LM adapter: identity (path + mtime, so a retrained checkpoint under the
    // same name misses), how it is applied, and every dial.
    if (!req.lm_adapter.empty()) {
        struct stat sb {};
        const bool  ok = stat(req.lm_adapter.c_str(), &sb) == 0;
        add_s("ad", req.lm_adapter);
        add_i("ad_mtime", ok ? (long long) sb.st_mtime : -1);
        add_s("ad_mode", req.lm_adapter_mode);
        const MM3LmAdapterScales & s = req.lm_adapter_scales;
        add_f("ad_g", s.global); add_f("ad_a", s.attn);  add_f("ad_m", s.mlp);
        add_f("ad_e", s.early);  add_f("ad_i", s.mid);   add_f("ad_l", s.late);
    }

    // Plank replay pins the codes, which pins the hiddens with them.
    if (!req.forced_semantic.empty()) {
        add_codes("fs", req.forced_semantic);
        add_codes("fa", req.forced_acoustic);
    }

    // The two models that produce the block. Quant flips borderline codes, so
    // a role swap must miss — path alone is not enough when a file is replaced
    // in place, hence the byte counts too.
    add_s("lm", m.lm_file.path);
    add_u("lm_b", (unsigned long long) m.lm_file.tensor_bytes);
    add_u("lm_t", m.lm_file.file_type);
    add_s("dp", m.role_file[MM3_R_DEPTH].path);
    add_u("dp_b", (unsigned long long) m.role_file[MM3_R_DEPTH].tensor_bytes);
    add_u("dp_t", m.role_file[MM3_R_DEPTH].file_type);
    return k;
}

// Base64 for the x-lrc-text response header. Same encoding the ACE /synth path
// uses (hot-step-server.cpp) — an LRC body is UTF-8 with newlines, neither of
// which survives a raw HTTP header.
static std::string mm3_b64_encode(const std::string & in) {
    static const char T[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string       out;
    out.reserve((in.size() + 2) / 3 * 4);
    for (size_t i = 0; i < in.size(); i += 3) {
        uint32_t v = (uint32_t) (uint8_t) in[i] << 16;
        if (i + 1 < in.size()) v |= (uint32_t) (uint8_t) in[i + 1] << 8;
        if (i + 2 < in.size()) v |= (uint32_t) (uint8_t) in[i + 2];
        out += T[(v >> 18) & 0x3F];
        out += T[(v >> 12) & 0x3F];
        out += (i + 1 < in.size()) ? T[(v >> 6) & 0x3F] : '=';
        out += (i + 2 < in.size()) ? T[v & 0x3F] : '=';
    }
    return out;
}

// ── The worker ──────────────────────────────────────────────────────────────

static void mm3_synth_worker(std::shared_ptr<Job> job, std::shared_ptr<MM3JobState> st, MM3SynthRequest req) {
    auto set_stage = [&](const char * stage, int64_t window, int64_t n_windows, int64_t step, int64_t n_steps) {
        std::lock_guard<std::mutex> lock(st->mtx);
        st->stage     = stage;
        st->window    = window;
        st->n_windows = n_windows;
        st->step      = step;
        st->n_steps   = n_steps;
    };
    // ── Streaming: close the queue however this function exits ─────────────
    // Every `return` below — early cancel, load failure, adapter failure,
    // generation failure, success — has to release a reader parked on the
    // condition variable. A destructor is the only spelling that cannot be
    // forgotten when a new failure path is added later.
    struct StreamCloser {
        std::shared_ptr<MM3StreamQueue> q;
        std::string                     why = "the run ended before it finished";
        ~StreamCloser() {
            if (q) {
                mm3_stream_close(*q, why);
            }
        }
    } stream_closer{ st->stream };

    // Post-run residency, mirroring the ACE side's EVICT_STRICT default.
    //
    // MM3 keeps its weights in its own buffers rather than in the ModelStore, so
    // the store's eviction policy never reaches them — without this an MM3 job
    // left ~12-23 GB resident for the rest of the process even with "keep models
    // loaded" switched OFF, which is exactly what that setting exists to avoid.
    // `g_keep_loaded` is the same flag ACE's store is built from (and that the
    // audition flow latches), so the two backends now agree on the policy.
    //
    // Guarded by `holds_mm3_lock`: the early-cancel path below runs BEFORE the
    // worker takes g_mm3_mutex, and touching g_mm3 unlocked would race the
    // bring-up endpoints. Nothing of ours is resident that early anyway.
    // Graph state points into the model's buffers and holds backend references,
    // so it goes first — same order as POST /mm3/unload.
    bool holds_mm3_lock       = false;
    auto release_if_transient = [&]() {
        // Tested on resident BYTES, not on `loaded`: after a staged run the LM
        // has already been dropped, so `loaded` is false while the depth and
        // flow buffers are still holding gigabytes. Keying off the flag would
        // silently skip exactly the case this exists for.
        if (!holds_mm3_lock || g_keep_loaded || mm3_vram_bytes(g_mm3) == 0) {
            return;
        }
        const size_t freed = mm3_vram_bytes(g_mm3) + g_mm3_lm.kv_bytes;
        mm3_lm_adapter_drop();  // borrowed pointer cleared while the graph still exists
        mm3_vocoder_free(&g_mm3_voc);
        mm3_dit_free(&g_mm3_dit);
        mm3_depth_free(&g_mm3_depth);
        mm3_cond_free(&g_mm3_cond);
        mm3_lm_free(&g_mm3_lm);
        mm3_unload(&g_mm3);
        fprintf(stderr, "[MM3-Job] %s: released %.2f GB (keep-loaded off)\n", job->id.c_str(),
                (double) freed / 1073741824.0);
    };

    auto fail = [&](int status, const char * stage, const std::string & msg) {
        {
            std::lock_guard<std::mutex> lock(st->mtx);
            st->stage = stage;
            st->error = msg;
        }
        stream_closer.why = msg;  // a reader learns WHY the audio stopped
        job_set_phase(*job, status == 3 ? JobPhase::CANCELLED : JobPhase::FAILED);
        job->status.store(status);
        // A failed or cancelled run must not leave the weights parked either —
        // a cancel is the most likely moment for a user to be reclaiming VRAM.
        release_if_transient();
    };

    // MM3 Plank replay. Worth saying out loud what this does and does not buy:
    // the AR loop still runs every per-frame forward pass, so the planning
    // progress bar will still crawl to 100%. Forcing pins which TOKENS come out,
    // not how much compute runs — the win is an identical semantic bed to A/B
    // the flow stage against, not latency.
    if (!req.forced_semantic.empty()) {
        fprintf(stderr,
                "[MM3-Job] %s: AR replay — %zu forced iterations (codes pinned; AR compute still runs)\n",
                job->id.c_str(), req.forced_semantic.size());
    }

    if (job->cancel.load()) {
        fail(3, "cancelled", "cancelled before it started");
        return;
    }

    // Serialise against the bring-up /mm3/* endpoints, which still run on
    // httplib threads. Two MM3 jobs can never contend here — they are both on
    // the one worker thread.
    std::lock_guard<std::mutex> mm3_lock(g_mm3_mutex);
    holds_mm3_lock = true;

    // ── AR cache lookup ─────────────────────────────────────────────────────
    //
    // Decided BEFORE arbitration and loading, because a hit changes both: no
    // LM weights, no KV cache, and the flow stack comes in up front instead of
    // through the mid-run handover.
    const std::string ar_key = mm3_ar_cache_key(g_mm3, req);
    const bool        ar_hit = req.reuse_ar && !g_mm3_ar_cache.key.empty() && g_mm3_ar_cache.key == ar_key &&
                        g_mm3_ar_cache.frames > 0 &&
                        (int64_t) g_mm3_ar_cache.hiddens.size() ==
                            g_mm3_ar_cache.frames * (int64_t) g_mm3.lm_cfg.num_codebooks *
                                (int64_t) g_mm3.lm_cfg.embedding_length;
    if (ar_hit) {
        g_mm3_ar_cache.hits++;
        req.gen.cached_hiddens = g_mm3_ar_cache.hiddens.data();
        req.gen.cached_frames  = g_mm3_ar_cache.frames;
        fprintf(stderr, "[MM3-Job] %s: AR cache HIT - %lld frames, %.0f MB reused, stage 1 skipped (hit #%lld)\n",
                job->id.c_str(), (long long) g_mm3_ar_cache.frames, (double) mm3_ar_cache_bytes() / 1048576.0,
                (long long) g_mm3_ar_cache.hits);
    } else if (req.reuse_ar) {
        fprintf(stderr, "[MM3-Job] %s: AR cache MISS (%s) - planning fresh\n", job->id.c_str(),
                g_mm3_ar_cache.key.empty() ? "slot empty" : "an AR-affecting input changed");
        // Free the stale block NOW rather than after the run: it is hundreds of
        // MB that this render has no use for, and holding it while the next one
        // is built would double the host-RAM peak for no reason.
        mm3_ar_cache_clear("superseded");
    }
    {
        std::lock_guard<std::mutex> lock(st->mtx);
        st->ar_cached = ar_hit;
    }

    // ── VRAM arbitration + warm ──
    job_set_phase(*job, JobPhase::LOADING_DIT);
    set_stage("arbitrating", -1, 0, 0, 0);

    // n_ctx the AR stage will want: prompt + every frame it may emit, plus the
    // slack mm3_lm_prepare() adds for the feedback position.
    const int64_t n_ctx_needed = req.n_tokens + req.max_frames + 8;

    const auto t_arb = std::chrono::steady_clock::now();
    mm3_arbitrate_vram(g_mm3, n_ctx_needed, st.get(), /*stage2_only=*/ar_hit);
    const double arb_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_arb).count();

    if (job->cancel.load()) {
        fail(3, "cancelled", "cancelled during VRAM arbitration");
        return;
    }

    // Staged residency, mirroring how the ACE pipeline swaps DiT and VAE.
    //
    // Stage 1 (AR) needs the LM and the depth decoder; stage 2 (cond/flow/
    // vocode) needs neither. Loading only what stage 1 uses drops the peak by
    // the size of the flow stack — the difference between "needs a 24 GB card"
    // and "runs on 12". The cost is one mid-run load of cond/dit/voc, the same
    // trade the ACE side already makes.
    //
    // Under keep-loaded the staging is skipped entirely: the whole point of
    // that setting is to pay VRAM to avoid reload latency.
    // On an AR cache hit stage 1 never runs, so the LM and the depth decoder
    // are not loaded AT ALL — that saves the 8.5-16 GB load as well as the AR
    // compute, and in staged mode it also saves the mid-run free/reload, since
    // the flow stack comes in here instead of at the handover.
    // ── Interleaved streaming: can the LM and the flow stack co-reside? ────
    //
    // Streaming after the AR stage is cheap and always available. Streaming
    // WHILE it plans is what makes the feature worth having on a fresh plan —
    // and it is incompatible with the staged handover below, which exists
    // precisely so the two stacks never have to be resident at once. So this is
    // a per-run VRAM decision, exactly like mm3_arbitrate_vram's.
    //
    // Declining is not a failure: the render proceeds, still streams, and the
    // first window simply arrives when the planner is done. It is reported so
    // the caller can say which it got rather than guess from the timing.
    //
    // An AR cache hit is excluded because there is nothing to interleave with:
    // stage 1 never runs, so the serial sweep starts emitting immediately.
    bool interleave = false;
    if (req.stream && !ar_hit) {
        if (g_keep_loaded) {
            // Keep-loaded already means co-resident by policy — that is what
            // the setting buys, and it is the same reason `staged` is false.
            interleave = true;
            fprintf(stderr, "[MM3-Job] %s: interleaved streaming (keep-loaded: both stacks stay resident)\n",
                    job->id.c_str());
        } else {
            // The co-resident figure is what mm3_vram_need already computes with
            // stage2_only=false: every role file + this run's KV + compute
            // headroom. Whatever is already ours does not need finding again.
            const size_t need     = mm3_vram_need(g_mm3, n_ctx_needed, /*stage2_only=*/false);
            const size_t resident = mm3_vram_bytes(g_mm3);
            const size_t want     = need > resident ? need - resident : 0;
            size_t       free_b = 0, total_b = 0;
            if (!mm3_cuda_free_bytes(&free_b, &total_b)) {
                // No way to measure (CPU/Vulkan build). Do not gamble a
                // several-GB over-commit on an unmeasured guess.
                fprintf(stderr, "[MM3-Job] %s: streaming stays serial - free VRAM is not measurable on this build\n",
                        job->id.c_str());
            } else if (free_b >= want) {
                interleave = true;
                fprintf(stderr,
                        "[MM3-Job] %s: interleaved streaming - need %.2f GB more, %.2f GB free of %.2f GB\n",
                        job->id.c_str(), (double) want / 1073741824.0, (double) free_b / 1073741824.0,
                        (double) total_b / 1073741824.0);
            } else {
                fprintf(stderr,
                        "[MM3-Job] %s: streaming stays serial - co-residency needs %.2f GB more, only %.2f GB free "
                        "(the render is unaffected; audio starts once planning finishes)\n",
                        job->id.c_str(), (double) want / 1073741824.0, (double) free_b / 1073741824.0);
            }
        }
    }
    req.gen.stream_interleave = interleave;
    {
        std::lock_guard<std::mutex> lock(st->mtx);
        st->interleaved = interleave;
    }

    // Interleaving means no handover: the LM has to still be there when window
    // k is conditioned, and the flow stack has to already be there when the
    // planner is only a fifth of the way through.
    const bool staged   = !g_keep_loaded && !ar_hit && !interleave;
    const bool was_warm = ar_hit ? g_mm3.rest_resident : g_mm3.loaded;
    set_stage(was_warm ? "warm" : "warming", -1, 0, 0, 0);
    const auto  t_warm = std::chrono::steady_clock::now();
    std::string err;
    if (!mm3_load_parts(&g_mm3, /*lm=*/!ar_hit, /*depth=*/!ar_hit, /*rest=*/!staged, &err)) {
        fail(2, "failed", err.empty() ? "MiniMax-Music3 load failed" : err);
        fprintf(stderr, "[MM3-Job] %s: load failed: %s\n", job->id.c_str(), err.c_str());
        return;
    }
    const double warm_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_warm).count();
    fprintf(stderr, "[MM3-Job] %s: %s (%.0f ms arbitration, %.0f ms %s), %.2f GB resident\n", job->id.c_str(),
            was_warm ? "already warm" : "warmed", arb_ms, warm_ms, was_warm ? "check" : "load",
            (double) mm3_vram_bytes(g_mm3) / 1073741824.0);
    {
        std::lock_guard<std::mutex> lock(st->mtx);
        st->arbitrate_ms = arb_ms;
        st->warm_ms      = warm_ms;
    }

    // ── progress -> job phase ──
    // `flow` counts across ALL windows so dit_inference's step never goes
    // backwards between windows.
    const int flow_total_hint = req.gen.steps;
    MM3ProgressCb progress = [&](const MM3GenProgress & p) {
        const std::string stage = p.stage;
        if (stage == "ar") {
            job_set_phase(*job, JobPhase::ENCODING_COND, (int) p.step, (int) p.n_steps);
        } else if (stage == "cond") {
            job_set_phase(*job, JobPhase::ENCODING_COND, (int) (p.window + 1), (int) p.n_windows);
        } else if (stage == "flow") {
            const int64_t done  = p.window * flow_total_hint + p.step;
            const int64_t total = p.n_windows * flow_total_hint;
            job_set_phase(*job, JobPhase::DIT_INFERENCE, (int) done, (int) total);
        } else if (stage == "stream") {
            // Interleaved: the planner and the flow stage take turns several
            // times a second, so their own bands would make the bar oscillate.
            // One axis instead — audio actually produced, with partial credit
            // for the window currently being planned. See mm3-pipeline.h.
            const int64_t done  = p.window * flow_total_hint + (p.n_steps > 0 ? p.step * flow_total_hint / p.n_steps : 0);
            const int64_t total = (p.n_windows > 0 ? p.n_windows : 1) * flow_total_hint;
            job_set_phase(*job, JobPhase::DIT_INFERENCE, (int) done, (int) total);
        } else if (stage == "vocode") {
            job_set_phase(*job, JobPhase::VAE_DECODE, (int) (p.window + 1), (int) p.n_windows);
        } else if (stage == "stitch") {
            job_set_phase(*job, JobPhase::ENCODING_OUTPUT);
        }
        set_stage(p.stage, p.window, p.n_windows, p.step, p.n_steps);
    };

    // Lyric timestamps: only meaningful when there are lyrics to align, so an
    // instrumental never pays the capture cost.
    req.gen.want_lrc      = req.want_lrc && !req.instrumental;
    req.gen.should_cancel = [&job]() { return job->cancel.load(); };

    // ── LM adapter (worker thread, g_mm3_mutex held) ───────────────────────
    // Load fails the JOB, never falls back silently — an adapter the user
    // asked for that quietly didn't load would be indistinguishable from "the
    // adapter does nothing" (the exact failure mode the ablation work exists
    // to study). A cached adapter for a DIFFERENT path is dropped first; one
    // for the same path is revalidated by mtime so a retrained checkpoint
    // under the same name is picked up.
    //
    // Two application modes (req.lm_adapter_mode):
    //   runtime — low-rank deltas in-graph, live dials, +28 %/step at r256.
    //   merge   — scale·B·A folded into the resident weights once
    //             (mm3-lm-merge.h); the AR loop then runs the plain base graph
    //             at full speed. MM3Model::lm_merge_tag tracks what is baked
    //             in; any mismatch (different adapter, different dials, or a
    //             base render on a merged LM under keep-loaded) forces a
    //             pristine LM reload first. In staged mode the LM reloads
    //             from disk every generation anyway, so the reload branch
    //             below only ever fires under keep-loaded.
    //
    // Skipped wholesale on an AR cache hit: there is no LM to adapt, and the
    // adapter's identity and dials are already baked into the cache key, so a
    // hit means "this exact adapter produced this exact block".
    const bool merge_mode = !req.lm_adapter.empty() && req.lm_adapter_mode == "merge";
    if (!ar_hit) {
        struct stat asb {};

        const bool  stat_ok  = !req.lm_adapter.empty() && stat(req.lm_adapter.c_str(), &asb) == 0;
        std::string want_tag;  // "" = this render wants a pristine base
        if (merge_mode) {
            want_tag = mm3_lm_merge_make_tag(req.lm_adapter, stat_ok ? (int64_t) asb.st_mtime : 0,
                                             req.gen.lm_adapter_scales);
        }
        if (g_mm3.lm_merge_tag != want_tag && !g_mm3.lm_merge_tag.empty()) {
            // Something else is baked into the resident LM — reload pristine.
            fprintf(stderr, "[MM3-Job] %s: merged LM is stale — reloading pristine base\n", job->id.c_str());
            set_stage("swapping", -1, 0, 0, 0);
            mm3_lm_free(&g_mm3_lm);
            mm3_free_lm(&g_mm3);
            std::string lerr;
            if (!mm3_load_parts(&g_mm3, /*lm=*/true, /*depth=*/true, /*rest=*/!staged, &lerr)) {
                fail(2, "failed", lerr.empty() ? "pristine LM reload failed" : lerr);
                return;
            }
        }

        if (!req.lm_adapter.empty()) {
            const bool cached = g_mm3_lm_adapter && g_mm3_lm_adapter->path == req.lm_adapter && stat_ok &&
                                (int64_t) asb.st_mtime == g_mm3_lm_adapter->mtime;
            if (!cached) {
                mm3_lm_adapter_drop();
                std::string aerr;
                g_mm3_lm_adapter = mm3_lm_adapter_load(req.lm_adapter.c_str(), &aerr);
                if (!g_mm3_lm_adapter) {
                    fail(2, "lm_adapter", aerr);  // fail() handles transient release
                    return;
                }
            }
            if (merge_mode) {
                if (g_mm3.lm_merge_tag != want_tag) {
                    set_stage("merging", -1, 0, 0, 0);
                    std::string merr;
                    if (!mm3_lm_merge_apply(&g_mm3, g_mm3_lm_adapter, req.gen.lm_adapter_scales, &merr)) {
                        // A part-way merge leaves mixed weights — drop the LM so
                        // the next generation reloads a pristine base.
                        mm3_lm_free(&g_mm3_lm);
                        mm3_free_lm(&g_mm3);
                        fail(2, "lm_adapter", merr.empty() ? "LM adapter merge failed" : merr);
                        return;
                    }
                    g_mm3.lm_merge_tag = want_tag;
                }
                req.gen.lm_adapter = nullptr;  // baked in; no runtime deltas on top
            } else {
                req.gen.lm_adapter = g_mm3_lm_adapter;
            }
        } else {
            req.gen.lm_adapter = nullptr;  // cached adapter stays resident but inert
        }
    }

    // The stage-1 -> stage-2 handover (see mm3-pipeline.h). Runs on the worker
    // thread with g_mm3_mutex still held.
    if (staged) {
        req.gen.after_ar = [&](std::string * e) {
            // The LM graph state holds the KV cache and pointers into the LM
            // weight buffer, so it has to go before the buffer does. The LM
            // adapter goes too: staged mode exists because VRAM is tight, and
            // the adapter is dead weight once the AR stage is done.
            const size_t kv = g_mm3_lm.kv_bytes;
            mm3_lm_adapter_drop();
            mm3_lm_free(&g_mm3_lm);
            const size_t freed = mm3_free_lm(&g_mm3) + kv;
            set_stage("swapping", -1, 0, 0, 0);
            if (!mm3_load_parts(&g_mm3, /*lm=*/false, /*depth=*/true, /*rest=*/true, e)) {
                return false;
            }
            fprintf(stderr, "[MM3-Job] %s: staged handover - freed %.2f GB (LM+KV), flow stack in, %.2f GB resident\n",
                    job->id.c_str(), (double) freed / 1073741824.0,
                    (double) mm3_vram_bytes(g_mm3) / 1073741824.0);
            return true;
        };
    }

    // ── Streaming sink ─────────────────────────────────────────────────────
    //
    // One self-contained WAV per window, pushed the moment that window's PCM
    // is final. The samples handed over are already clamped exactly as the
    // saved file's are (mm3-pipeline.h mm3_clamp_sample), and `wav_bits` is
    // the SAME format the finished file uses — so concatenating every chunk's
    // sample data reproduces the saved WAV byte for byte. That identity is the
    // validation bar for this feature, and it only holds while both sides
    // encode through audio_encode_wav with the same format.
    if (st->stream) {
        const WavFormat sfmt = req.wav_bits == 32 ? WAV_F32 : (req.wav_bits == 24 ? WAV_S24 : WAV_S16);
        auto            q    = st->stream;
        const std::string jid = job->id;
        // The take index arrives but is not routed yet: this job layer still
        // owns exactly one stream queue, so it only ever runs one-take renders
        // (mm3_generate_takes with K = 1). Multi-take streaming needs a queue
        // per take, which is the next increment.
        req.gen.on_chunk = [q, sfmt, jid](int take, int64_t seq, int64_t off, int64_t n, const float * planar,
                                          int sr) {
            (void) take;
            // A degenerate final window can crop to nothing. Emitting a
            // header-only WAV would be harmless here but `DataSink::write`
            // treats a zero-length write as end-of-stream, so skip it.
            if (n <= 0 || !planar) {
                return;
            }
            std::string wav = audio_encode_wav(planar, (int) n, sr, sfmt);
            if (seq == 0) {
                fprintf(stderr, "[MM3-Stream] %s: first chunk at %.2f s of audio (%zu bytes)\n", jid.c_str(),
                        (double) n / (double) (sr > 0 ? sr : 1), wav.size());
            }
            (void) off;  // carried by the concatenation order, not the WAV header
            mm3_stream_push(*q, std::move(wav), sr);
        };
    }

    MM3GenResult r;
    const bool   ok = mm3_generate(g_mm3, req.gen, &g_mm3_tokenizer, progress, &r, &err);
    if (!ok) {
        if (err == MM3_ERR_CANCELLED) {
            fprintf(stderr, "[MM3-Job] %s: cancelled\n", job->id.c_str());
            fail(3, "cancelled", "cancelled");
        } else {
            fprintf(stderr, "[MM3-Job] %s: generation failed: %s\n", job->id.c_str(), err.c_str());
            fail(2, "failed", err.empty() ? "MM3 generation failed" : err);
        }
        return;
    }

    // ── AR cache: restore on a hit, fill on a miss ─────────────────────────
    //
    // The pipeline only ever borrowed the hidden block, so on a hit `r.ar` is
    // otherwise empty. The stage-1 BYPRODUCTS — the codes, the LRC, the EOS
    // flag — are cached alongside it and put back here, so a cache hit and a
    // fresh plan are indistinguishable to everything downstream (plank capture,
    // x-lrc-text, the job detail).
    if (ar_hit) {
        r.ar.semantic_all = g_mm3_ar_cache.semantic_all;
        r.ar.acoustic_all = g_mm3_ar_cache.acoustic_all;
        r.ar.lrc          = g_mm3_ar_cache.lrc;
        r.ar.n_frames     = g_mm3_ar_cache.frames;
        r.ar.eos_hit      = g_mm3_ar_cache.eos_hit;
    } else if (req.reuse_ar && r.frames > 0 && !r.ar.frame_hiddens.empty()) {
        // MOVED, not copied: the run is finished with the block and this is the
        // only reason host RAM never holds two of them.
        g_mm3_ar_cache.key          = ar_key;
        g_mm3_ar_cache.frames       = r.frames;
        g_mm3_ar_cache.hiddens      = std::move(r.ar.frame_hiddens);
        g_mm3_ar_cache.semantic_all = r.ar.semantic_all;
        g_mm3_ar_cache.acoustic_all = r.ar.acoustic_all;
        g_mm3_ar_cache.lrc          = r.ar.lrc;
        g_mm3_ar_cache.eos_hit      = r.ar.eos_hit;
        g_mm3_ar_cache.hits         = 0;
        fprintf(stderr, "[MM3-Job] %s: AR cache filled - %lld frames, %.0f MB held for the next render\n",
                job->id.c_str(), (long long) r.frames, (double) mm3_ar_cache_bytes() / 1048576.0);
    }

    // ── encode ──
    job_set_phase(*job, JobPhase::ENCODING_OUTPUT);
    set_stage("encoding", -1, r.n_windows, 0, 0);
    const WavFormat fmt = req.wav_bits == 32 ? WAV_F32 : (req.wav_bits == 24 ? WAV_S24 : WAV_S16);
    job->result_body = audio_encode_wav(r.audio.data(), (int) r.n_samples, r.sample_rate, fmt);
    job->result_mime = "audio/wav";
    // LRC rides the SAME transport ACE uses: Job::result_lrc is base64 and
    // the shared /job?result=1 handler emits it as x-lrc-text, so the Node
    // side needs no MM3-specific parsing.
    if (!r.ar.lrc.empty()) {
        job->result_lrc = mm3_b64_encode(r.ar.lrc);
        fprintf(stderr, "[MM3-Job] %s: LRC %zu bytes\n", job->id.c_str(), r.ar.lrc.size());
    }

    // ── MM3 Plank: serialise the AR stage's codes ──────────────────────────
    // Flat little-endian i32 blob, retrieved via GET /mm3/job?id=<id>&ar=1:
    //   [i32] n_semantic
    //   [i32 * n_semantic] semantic_all
    //   [i32] n_acoustic
    //   [i32 * n_acoustic] acoustic_all
    // Iteration-indexed, entry 0 un-emitted — the exact shape mm3-ar-loop.h
    // wants back as forced_semantic/forced_acoustic, so a replay is a
    // byte-for-byte round trip with no reindexing at either end.
    if (req.get_ar_codes && !r.ar.semantic_all.empty()) {
        const int32_t n_sem = (int32_t) r.ar.semantic_all.size();
        const int32_t n_ac  = (int32_t) r.ar.acoustic_all.size();
        job->result_ar_codes.resize(sizeof(int32_t) * (2 + (size_t) n_sem + (size_t) n_ac));
        char * wp = &job->result_ar_codes[0];
        memcpy(wp, &n_sem, sizeof(int32_t));
        wp += sizeof(int32_t);
        memcpy(wp, r.ar.semantic_all.data(), (size_t) n_sem * sizeof(int32_t));
        wp += (size_t) n_sem * sizeof(int32_t);
        memcpy(wp, &n_ac, sizeof(int32_t));
        wp += sizeof(int32_t);
        memcpy(wp, r.ar.acoustic_all.data(), (size_t) n_ac * sizeof(int32_t));
        fprintf(stderr, "[MM3-Job] %s: AR codes captured — %d semantic + %d acoustic i32 (%.1f KB)\n",
                job->id.c_str(), n_sem, n_ac, (double) job->result_ar_codes.size() / 1024.0);
    }

    {
        std::lock_guard<std::mutex> lock(st->mtx);
        st->have_result = true;
        st->stage       = "done";
        st->frames      = r.frames;
        st->n_windows   = r.n_windows;
        st->n_samples   = r.n_samples;
        st->sample_rate = r.sample_rate;
        st->duration_s  = r.sample_rate > 0 ? (double) r.n_samples / (double) r.sample_rate : 0.0;
        st->rms         = r.rms;
        st->peak        = (double) r.peak;
        st->eos         = r.ar.eos_hit;
        st->has_nan     = r.has_nan;
        st->ar_ms       = r.ar_ms;
        st->cond_ms     = r.cond_ms;
        st->flow_ms     = r.flow_ms;
        st->voc_ms      = r.voc_ms;
        st->total_ms    = r.total_ms;
    }

    // A cancel that lands between the last progress poll and here still counts.
    if (job->cancel.load()) {
        fail(3, "cancelled", "cancelled after generation, before delivery");
        return;
    }

    stream_closer.why.clear();  // clean end — a reader sees EOF, not an error
    if (st->stream) {
        fprintf(stderr, "[MM3-Stream] %s: %lld chunk(s), %.1f MB streamed\n", job->id.c_str(),
                (long long) st->stream->n_chunks.load(), (double) st->stream->bytes_total.load() / 1048576.0);
    }

    job_set_phase(*job, JobPhase::DONE);
    job->status.store(1);
    fprintf(stderr, "[MM3-Job] %s: done - %lld frames, %lld samples/ch (%.2f s @ %d Hz), rms %.4f, %.0f ms\n",
            job->id.c_str(), (long long) r.frames, (long long) r.n_samples,
            r.sample_rate > 0 ? (double) r.n_samples / (double) r.sample_rate : 0.0, r.sample_rate, r.rms, r.total_ms);

    // Result is already encoded into job->result_body, so dropping the weights
    // here cannot affect delivery.
    release_if_transient();
}

// ── POST /mm3/synth ─────────────────────────────────────────────────────────
//
// The PRODUCTION generation endpoint. Body is JSON; see mm3-request.h for the
// full field contract. Returns {"id":"<job id>"} immediately — poll
// GET /job?id=<id> (or GET /mm3/job?id=<id> for MM3 stage detail), fetch the
// WAV with GET /job?id=<id>&result=1, cancel with POST /job?id=<id>&cancel=1.
//
// Everything that can be rejected synchronously IS: a bad field type, a blank
// caption, a missing duration, an over-long prompt. Only work that needs the
// GPU is deferred to the job, so a 200 here means "this will actually run".
static void mm3_handle_synth(const httplib::Request & hreq, httplib::Response & res) {
    if (!mm3_available(g_mm3)) {
        mm3_json_error(res, 501, "MiniMax-Music3 model files were not found (see GET /mm3/props)");
        return;
    }

    yyjson_doc * doc  = hreq.body.empty() ? nullptr : yyjson_read(hreq.body.data(), hreq.body.size(), 0);
    yyjson_val * root = doc ? yyjson_doc_get_root(doc) : nullptr;
    if (!root || !yyjson_is_obj(root)) {
        if (doc) {
            yyjson_doc_free(doc);
        }
        mm3_json_error(res, 400, "body must be a JSON object");
        return;
    }
    struct DocGuard {
        yyjson_doc * d;
        ~DocGuard() {
            if (d) {
                yyjson_doc_free(d);
            }
        }
    } guard{ doc };

    MM3SynthRequest req;
    std::string     err;
    if (!mm3_parse_synth_request(g_mm3, root, &req, &err)) {
        mm3_json_error(res, 400, err);
        return;
    }

    // Tokenise + enforce the 5,000-token limit here, on the httplib thread: it
    // is a header-only GGUF read, it works cold, and a caller deserves the
    // rejection now rather than as a failed job later.
    {
        std::lock_guard<std::mutex> lock(g_mm3_mutex);
        if (!mm3_request_tokenize(g_mm3, &g_mm3_tokenizer, &req, &err)) {
            mm3_json_error(res, 400, err);
            return;
        }
    }

    auto job = job_create();
    auto st  = std::make_shared<MM3JobState>();
    // Allocated here, not in the worker: the caller opens GET /mm3/stream as
    // soon as this response lands, which is long before the job reaches the
    // front of the FIFO. The queue has to exist for that reader to attach to,
    // or the stream would 409 on every render that had to wait its turn.
    if (req.stream) {
        st->stream = std::make_shared<MM3StreamQueue>();
    }
    st->seed         = req.gen.seed;
    st->max_frames   = req.max_frames;
    st->n_tokens     = req.n_tokens;
    st->instrumental = req.instrumental;
    mm3_job_state_put(job->id, st);

    fprintf(stderr, "[MM3-Job] %s created - %lld prompt tokens, %lld frames (%.1f s), seed %llu, %d steps, cfg %.2f%s\n",
            job->id.c_str(), (long long) req.n_tokens, (long long) req.max_frames, req.duration,
            (unsigned long long) req.gen.seed, req.gen.steps, (double) req.gen.cfg_flow,
            req.instrumental ? ", instrumental" : "");

    // Caption echo. The ACE side prints its CoT block ([LM-Phase2] CoT[0]) so an
    // operator can see what the model was actually handed; without an equivalent
    // here, a caption that arrived mangled — or never arrived — is invisible
    // until you listen to the result. What is printed is the CLEANED caption,
    // i.e. the text as it sits inside the assembled template after
    // mm3_clean_caption stripped markdown/bullets/rules, NOT the raw box
    // contents: the difference is exactly what a Structured Caption pasted from
    // a markdown-emitting tool loses. MM3_LOG_PROMPT=1 dumps the whole assembled
    // template (caption + lyric block + control tags) instead.
    if (std::getenv("MM3_LOG_PROMPT")) {
        fprintf(stderr, "[MM3-Job] %s prompt (%zu bytes, assembled):\n%s\n", job->id.c_str(), req.prompt.size(),
                req.prompt.c_str());
    } else {
        const std::string cap = mm3_clean_caption(req.caption);
        fprintf(stderr, "[MM3-Job] %s caption (%zu bytes in, %zu cleaned), lyrics %zu bytes:\n%s\n", job->id.c_str(),
                req.caption.size(), cap.size(), req.lyrics.size(), cap.c_str());
    }

    work_push([job, st, req]() mutable { mm3_synth_worker(job, st, std::move(req)); });

    yyjson_mut_doc * o    = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * orot = yyjson_mut_obj(o);
    yyjson_mut_doc_set_root(o, orot);
    yyjson_mut_obj_add_strcpy(o, orot, "job_id", job->id.c_str());
    yyjson_mut_obj_add_strcpy(o, orot, "id", job->id.c_str());  // the ACE /synth spelling
    yyjson_mut_obj_add_uint(o, orot, "seed", req.gen.seed);
    // Echoed so a caller can prove the AR-side fields landed: ar_seed is the
    // RESOLVED value (equal to seed unless it was split), reuse_ar is what the
    // job will actually do a lookup with.
    yyjson_mut_obj_add_uint(o, orot, "ar_seed", req.gen.ar_seed_set ? req.gen.ar_seed : req.gen.seed);
    yyjson_mut_obj_add_bool(o, orot, "reuse_ar", req.reuse_ar);
    // Whether GET /mm3/stream will actually serve this job. Today it is simply
    // what was asked for — the serial pipeline can always emit windows as they
    // finish. It is echoed rather than assumed because the co-residency work
    // (sliding AR/flow pipeline) has a VRAM gate that CAN decline, and when it
    // does the caller must be told here rather than discovering it as a
    // silent absence of audio. Never a reason to fail a render.
    yyjson_mut_obj_add_bool(o, orot, "streaming", req.stream);
    yyjson_mut_obj_add_uint(o, orot, "max_frames", req.max_frames);
    yyjson_mut_obj_add_real(o, orot, "duration", req.duration);
    yyjson_mut_obj_add_uint(o, orot, "prompt_tokens", req.n_tokens);
    yyjson_mut_obj_add_uint(o, orot, "prompt_token_limit", MM3_MAX_PROMPT_TOKENS);
    yyjson_mut_obj_add_bool(o, orot, "instrumental", req.instrumental);
    yyjson_mut_obj_add_uint(o, orot, "steps", req.gen.steps);
    yyjson_mut_obj_add_real(o, orot, "cfg_flow", (double) req.gen.cfg_flow);
    yyjson_mut_obj_add_uint(o, orot, "wav_bits", req.wav_bits);
    // Echo the sampler-plugin selection back. Absent from the response == the
    // native flow loop ran, which is the answer to "did my picks actually
    // reach the engine?" without reading the engine log.
    if (req.gen.plugins.any()) {
        yyjson_mut_val * pl = yyjson_mut_obj(o);
        yyjson_mut_obj_add_strcpy(o, pl, "solver", req.gen.plugins.solver.c_str());
        yyjson_mut_obj_add_strcpy(o, pl, "scheduler", req.gen.plugins.scheduler.c_str());
        yyjson_mut_obj_add_strcpy(o, pl, "guidance", req.gen.plugins.guidance.c_str());
        yyjson_mut_obj_add_real(o, pl, "shift", (double) req.gen.plugins.shift);
        yyjson_mut_obj_add_uint(o, pl, "n_params", req.gen.plugins.params.size());
        yyjson_mut_obj_add_val(o, orot, "sampler_plugins", pl);
    }
    char * json = yyjson_mut_write(o, 0, NULL);
    yyjson_mut_doc_free(o);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
}

// ── GET /mm3/job?id=... ─────────────────────────────────────────────────────
//
// MM3-native progress. GET /job?id=... already gives coarse status plus the
// mapped ACE phase; this adds the real stage name ("ar" / "cond" / "flow" /
// "vocode" / "stitch"), the window index, the resolved seed, and — once the
// job is done — the run's shape and per-stage timings. Never takes
// g_mm3_mutex, so it answers instantly while a generation is running.
static void mm3_handle_job(const httplib::Request & hreq, httplib::Response & res) {
    if (!hreq.has_param("id")) {
        mm3_json_error(res, 400, "missing ?id=<job id>");
        return;
    }
    const std::string id = hreq.get_param_value("id");
    auto              st = mm3_job_state_get(id);
    if (!st) {
        mm3_json_error(res, 404, "no MM3 job with that id (it may have been evicted)");
        return;
    }
    auto job = job_find(id);

    // ── MM3 Plank retrieval: ?ar=1 returns the raw code blob ───────────────
    // Deliberately ahead of the state lock — this path touches only the job's
    // completed result, never the progress record, and the blob can be several
    // hundred kB. Layout is documented at the capture site in mm3_synth_worker.
    if (hreq.has_param("ar") && hreq.get_param_value("ar") == "1") {
        if (!job || job->status.load() != 1 || job->result_ar_codes.empty()) {
            mm3_json_error(res, 404,
                           "AR codes not available (job unfinished, get_ar_codes was not set, or none captured)");
            return;
        }
        res.set_content(job->result_ar_codes, "application/octet-stream");
        return;
    }

    std::lock_guard<std::mutex> lock(st->mtx);

    yyjson_mut_doc * o    = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * orot = yyjson_mut_obj(o);
    yyjson_mut_doc_set_root(o, orot);
    yyjson_mut_obj_add_strcpy(o, orot, "id", id.c_str());
    yyjson_mut_obj_add_strcpy(o, orot, "status", job ? job_status_str(job->status.load()) : "unknown");
    yyjson_mut_obj_add_strcpy(o, orot, "phase", job ? job_phase_str(job->phase.load()) : "unknown");
    yyjson_mut_obj_add_strcpy(o, orot, "stage", st->stage.c_str());
    yyjson_mut_obj_add_int(o, orot, "window", st->window);
    yyjson_mut_obj_add_int(o, orot, "n_windows", st->n_windows);
    yyjson_mut_obj_add_int(o, orot, "step", st->step);
    yyjson_mut_obj_add_int(o, orot, "n_steps", st->n_steps);
    yyjson_mut_obj_add_uint(o, orot, "seed", st->seed);
    yyjson_mut_obj_add_int(o, orot, "max_frames", st->max_frames);
    yyjson_mut_obj_add_int(o, orot, "prompt_tokens", st->n_tokens);
    yyjson_mut_obj_add_bool(o, orot, "instrumental", st->instrumental);
    // AR cache: true means stage 1 was skipped entirely for this job. Reported
    // live (not only on the finished result) so a poller can say so while the
    // flow stage is still running.
    yyjson_mut_obj_add_bool(o, orot, "ar_cached", st->ar_cached);
    // Streaming: whether GET /mm3/stream can serve this job, and how much has
    // gone out so far. Reported live so a poller can distinguish "streaming is
    // on but the AR stage has not produced a window yet" from "not streaming".
    yyjson_mut_obj_add_bool(o, orot, "streaming", (bool) st->stream);
    // Whether windows were dispatched DURING planning. Decided on the worker
    // thread after the VRAM check, so it cannot be echoed at submit time.
    yyjson_mut_obj_add_bool(o, orot, "stream_interleaved", st->interleaved);
    if (st->stream) {
        yyjson_mut_obj_add_int(o, orot, "stream_chunks", st->stream->n_chunks.load());
        yyjson_mut_obj_add_real(o, orot, "stream_mb", (double) st->stream->bytes_total.load() / 1048576.0);
        yyjson_mut_obj_add_bool(o, orot, "stream_reader", st->stream->attached.load());
    }
    yyjson_mut_obj_add_int(o, orot, "evicted_modules", st->evicted_modules);
    yyjson_mut_obj_add_real(o, orot, "evicted_mb", (double) st->evicted_bytes / 1048576.0);
    yyjson_mut_obj_add_real(o, orot, "arbitrate_ms", st->arbitrate_ms);
    yyjson_mut_obj_add_real(o, orot, "warm_ms", st->warm_ms);
    if (!st->error.empty()) {
        yyjson_mut_obj_add_strcpy(o, orot, "error", st->error.c_str());
    }
    if (st->have_result) {
        yyjson_mut_val * rr = yyjson_mut_obj(o);
        yyjson_mut_obj_add_val(o, orot, "result", rr);
        yyjson_mut_obj_add_int(o, rr, "frames", st->frames);
        yyjson_mut_obj_add_int(o, rr, "n_samples", st->n_samples);
        yyjson_mut_obj_add_int(o, rr, "sample_rate", st->sample_rate);
        yyjson_mut_obj_add_int(o, rr, "channels", 2);
        yyjson_mut_obj_add_real(o, rr, "duration_sec", st->duration_s);
        yyjson_mut_obj_add_real(o, rr, "rms", st->rms);
        yyjson_mut_obj_add_real(o, rr, "peak", st->peak);
        yyjson_mut_obj_add_bool(o, rr, "eos", st->eos);
        yyjson_mut_obj_add_bool(o, rr, "has_nan", st->has_nan);
        // MM3 Plank: whether GET /mm3/job?id=<id>&ar=1 will return a blob.
        yyjson_mut_obj_add_bool(o, rr, "ar_codes_available", job && !job->result_ar_codes.empty());
        yyjson_mut_val * ms = yyjson_mut_obj(o);
        yyjson_mut_obj_add_val(o, rr, "ms", ms);
        yyjson_mut_obj_add_real(o, ms, "ar", st->ar_ms);
        yyjson_mut_obj_add_real(o, ms, "cond", st->cond_ms);
        yyjson_mut_obj_add_real(o, ms, "flow", st->flow_ms);
        yyjson_mut_obj_add_real(o, ms, "voc", st->voc_ms);
        yyjson_mut_obj_add_real(o, ms, "total", st->total_ms);
    }
    char * json = yyjson_mut_write(o, 0, NULL);
    yyjson_mut_doc_free(o);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
}

// ── GET /mm3/stream?id=... ──────────────────────────────────────────────────
//
// The audio of a still-running job, as a chunked HTTP body of concatenated
// self-contained WAVs — the SAME transport POST /api/generate/storm/stream
// already uses, so the browser-side WAV extractor is shared rather than
// reinvented.
//
// Requires the job to have been submitted with `"stream": true`. Answers 409
// rather than 404 when it was not, because "this job exists but produces no
// stream" and "no such job" are different problems for a caller to handle.
//
// NEVER takes g_mm3_mutex. The generation it is streaming holds that lock for
// its whole run, so acquiring it here would deadlock the feature against
// itself — the same rule GET /mm3/job follows.
//
// One reader at a time. Chunks are CONSUMED as they are written, so a second
// reader would silently steal half the song from the first; a reconnect after
// a drop cannot resume for the same reason, and gets what is still queued.
static void mm3_handle_stream(const httplib::Request & hreq, httplib::Response & res) {
    if (!hreq.has_param("id")) {
        mm3_json_error(res, 400, "missing ?id=<job id>");
        return;
    }
    const std::string id = hreq.get_param_value("id");
    auto              st = mm3_job_state_get(id);
    if (!st) {
        mm3_json_error(res, 404, "no MM3 job with that id (it may have been evicted)");
        return;
    }
    auto q = st->stream;
    if (!q) {
        mm3_json_error(res, 409, "this job was not submitted with \"stream\": true");
        return;
    }
    {
        std::lock_guard<std::mutex> lock(q->mtx);
        if (q->abandoned) {
            mm3_json_error(res, 409, "the stream was dropped (nothing was reading it) - the render is unaffected");
            return;
        }
        if (q->attached.load()) {
            mm3_json_error(res, 409, "this job's stream already has a reader");
            return;
        }
        if (q->finished && q->chunks.empty()) {
            mm3_json_error(res, 409, "this job has already finished - fetch the WAV from GET /job?id=&result=1");
            return;
        }
        q->attached.store(true);
    }

    fprintf(stderr, "[MM3-Stream] %s: reader attached\n", id.c_str());
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");  // stop any proxy from buffering the point of the feature
    res.set_chunked_content_provider(
        "audio/wav",
        [q](size_t, httplib::DataSink & sink) -> bool {
            std::unique_lock<std::mutex> lock(q->mtx);
            // A 1 s ceiling rather than an open wait: httplib re-enters the
            // provider in a loop and checks peer liveness between calls, so
            // returning empty-handed is how a dropped client gets noticed.
            q->cv.wait_for(lock, std::chrono::seconds(1),
                           [&] { return !q->chunks.empty() || q->finished || q->abandoned; });
            while (!q->chunks.empty()) {
                std::string c = std::move(q->chunks.front());
                q->chunks.pop_front();
                q->bytes_queued -= (int64_t) c.size();
                lock.unlock();
                // Never a zero-length write: DataSink reads that as done().
                if (c.empty() || !sink.write(c.data(), c.size())) {
                    return false;
                }
                lock.lock();
            }
            if (q->finished || q->abandoned) {
                lock.unlock();
                sink.done();
                return true;
            }
            return true;
        },
        [q](bool success) {
            q->attached.store(false);
            fprintf(stderr, "[MM3-Stream] reader detached (%s), %lld chunk(s) sent\n",
                    success ? "complete" : "disconnected", (long long) q->n_chunks.load());
        });
}

// The second half of the hook. Called next to mm3_register_routes().
static void mm3_register_job_routes(httplib::Server & svr) {
    svr.Post("/mm3/synth", mm3_handle_synth);
    svr.Get("/mm3/job", mm3_handle_job);
    svr.Get("/mm3/stream", mm3_handle_stream);
}
