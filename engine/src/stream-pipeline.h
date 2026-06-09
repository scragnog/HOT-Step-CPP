#pragma once
// stream-pipeline.h — DEMON-style ring buffer streaming pipeline for TRT
//
// Port of DEMON's StreamPipeline (acestep/engine/stream.py) to C++.
// Maintains a ring buffer of in-flight generation slots at different
// denoising stages. Each tick() runs one batched TRT forward pass
// advancing all active slots. After warmup, every few ticks produce
// a completed slot whose latent is decoded via windowed VAE.
//
// Usage:
//   StreamPipeline pipeline(trt, vae_ort, config);
//   pipeline.submit(request);
//   while (pipeline.active_slots() > 0) {
//       auto preview = pipeline.tick();
//       if (preview) emit_preview(*preview);
//   }
//
// The existing dit_trt_generate() / GGML pipeline is completely untouched.

#ifdef HOT_STEP_TRT

#include "dit-trt.h"
#include "hot-step-sampler-trt.h"
#include "vae-ort.h"
#include "philox.h"
#include "sampler-schedule.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// ── Configuration ───────────────────────────────────────────────────────

struct StreamConfig {
    int   depth       = 8;      // ring buffer depth (user-configurable, default 8)
    int   num_steps   = 50;     // denoising steps (adapts to user's step setting)
    float shift       = 3.0f;   // timestep shift
    float denoise     = 1.0f;   // global denoise strength
    int   vae_chunk   = 64;     // VAE tiled decode chunk size
    int   vae_overlap = 4;      // VAE tiled decode overlap
    std::string chunk_dir;      // directory for preview WAV files
};

// ── Slot request (what the caller submits) ──────────────────────────────

struct StreamSlotRequest {
    // Conditioning
    std::vector<float> enc_hidden;       // [S, 2048] flattened
    std::vector<float> context_latents;  // [T, 128] flattened
    int enc_S = 0;
    int T     = 0;

    // Generation
    int64_t seed    = 0;
    float   denoise = 1.0f;

    // Source latents for cover mode (optional)
    std::vector<float> source_latents;   // [T, 64] or empty

    // Null cond for RCFG-self (optional)
    std::vector<float> null_cond;        // [2048] single null cond vector
};

// ── Preview result from a completed slot ────────────────────────────────

struct StreamPreview {
    std::string wav_path;   // path to preview WAV file
    float  t_start;         // window start in seconds
    float  duration;        // window duration in seconds
    int    tick_idx;        // which tick produced this
    int    slot_idx;        // which slot completed
    bool   is_final;        // true if this is the last preview
};

// ── Internal slot state ─────────────────────────────────────────────────

struct StreamSlot {
    StreamSlotRequest request;
    std::vector<float> xt;         // [T, 64] current noisy latent
    std::vector<float> schedule;   // timestep schedule for this slot
    int step_idx      = 0;         // current denoising step (0-indexed)
    bool completed    = false;
    int slot_id       = 0;         // unique ID for this slot instance

    // Pre-packed input_latents [T, 192] = context[128] + xt[64]
    // Updated each tick before packing into the batch
    std::vector<float> input_packed;
};

// ── StreamPipeline ──────────────────────────────────────────────────────

class StreamPipeline {
public:
    StreamPipeline(DitTrt* trt, VaeOrt* vae, const StreamConfig& config)
        : m_trt(trt)
        , m_vae(vae)
        , m_config(config)
        , m_tick_count(0)
        , m_last_tick_ms(0)
        , m_next_slot_id(0)
    {
        m_slots.resize(config.depth, nullptr);
        fprintf(stderr, "[StreamPipeline] Created: depth=%d steps=%d shift=%.1f\n",
                config.depth, config.num_steps, config.shift);
    }

    ~StreamPipeline() {
        for (auto& slot : m_slots) {
            delete slot;
            slot = nullptr;
        }
        dit_trt_stream_free(&m_bufs);
    }

    // Submit a new generation request (queued until a slot is free)
    void submit(const StreamSlotRequest& req) {
        m_queue.push_back(req);
        fprintf(stderr, "[StreamPipeline] Queued request (queue_size=%zu)\n",
                m_queue.size());
    }

    // Run one batched forward pass. Returns a preview if a slot completed.
    std::optional<StreamPreview> tick() {
        auto t0 = std::chrono::steady_clock::now();
        std::optional<StreamPreview> result;

        // 1. Check for completed slots → harvest result
        for (int i = 0; i < (int)m_slots.size(); i++) {
            if (m_slots[i] && m_slots[i]->completed) {
                // Already harvested, free the slot
                delete m_slots[i];
                m_slots[i] = nullptr;
            }
        }

        // 2. Fill empty slots from queue (one per tick for staggered startup)
        for (int i = 0; i < (int)m_slots.size(); i++) {
            if (!m_slots[i] && !m_queue.empty()) {
                m_slots[i] = init_slot(m_queue.front());
                m_queue.pop_front();
                fprintf(stderr, "[StreamPipeline] Filled slot %d (step 0/%d, id=%d)\n",
                        i, (int)m_slots[i]->schedule.size() - 1, m_slots[i]->slot_id);
                break;  // One slot per tick for staggered warmup
            }
        }

        // 3. Count active (non-null, non-completed) slots
        std::vector<int> active_indices;
        for (int i = 0; i < (int)m_slots.size(); i++) {
            if (m_slots[i] && !m_slots[i]->completed) {
                active_indices.push_back(i);
            }
        }

        if (active_indices.empty()) {
            m_tick_count++;
            return std::nullopt;
        }

        int B = (int)active_indices.size();
        int T = m_slots[active_indices[0]]->request.T;
        int S = m_slots[active_indices[0]]->request.enc_S;

        // 4. Ensure stream buffers are allocated
        if (!m_bufs.allocated ||
            B > m_bufs.max_B || T > m_bufs.T || S > m_bufs.S) {
            dit_trt_stream_alloc(&m_bufs, m_config.depth, T, S, m_trt->io_dtype);
        }

        // 5. Pack all active slots into batched tensors
        const int in_ch  = 192;
        const int ctx_ch = 128;
        const int Oc     = 64;
        const int H_enc  = 2048;

        // Pack input_latents: [B, T, 192]
        std::vector<float> input_packed(B * T * in_ch);
        std::vector<float> enc_packed(B * S * H_enc);
        std::vector<float> t_packed(B);

        for (int bi = 0; bi < B; bi++) {
            StreamSlot* slot = m_slots[active_indices[bi]];
            int step = slot->step_idx;
            float t_curr = slot->schedule[step];

            // Pack timestep (per-row!)
            t_packed[bi] = t_curr;

            // Pack input: context[128] + xt[64] → [T, 192]
            float* dst = input_packed.data() + bi * T * in_ch;
            const float* ctx_src = slot->request.context_latents.data();
            const float* xt_src  = slot->xt.data();
            for (int t = 0; t < T; t++) {
                memcpy(dst + t * in_ch,          ctx_src + t * ctx_ch, ctx_ch * sizeof(float));
                memcpy(dst + t * in_ch + ctx_ch,  xt_src + t * Oc,      Oc * sizeof(float));
            }

            // Pack encoder hidden states
            memcpy(enc_packed.data() + bi * S * H_enc,
                   slot->request.enc_hidden.data(),
                   S * H_enc * sizeof(float));
        }

        // 6. Batched TRT forward
        std::vector<float> vel_packed(B * T * Oc);
        bool ok = dit_trt_step(m_trt, &m_bufs,
                               input_packed.data(),
                               enc_packed.data(),
                               t_packed.data(),
                               B, T, S,
                               vel_packed.data());
        if (!ok) {
            fprintf(stderr, "[StreamPipeline] ERROR: dit_trt_step failed at tick %d\n",
                    m_tick_count);
            m_tick_count++;
            return std::nullopt;
        }

        // 7. Unpack velocity and integrate each slot (Euler ODE step)
        for (int bi = 0; bi < B; bi++) {
            StreamSlot* slot = m_slots[active_indices[bi]];
            int step = slot->step_idx;
            float t_curr = slot->schedule[step];
            float t_next = (step + 1 < (int)slot->schedule.size())
                           ? slot->schedule[step + 1] : 0.0f;

            // Extract this slot's velocity
            const float* vt = vel_packed.data() + bi * T * Oc;

            // Euler ODE step: xt += (t_next - t_curr) * vt
            dit_trt_euler_step(slot->xt.data(), vt, t_curr, t_next, T * Oc);

            // Advance step
            slot->step_idx++;

            // Check completion (reached last step)
            if (slot->step_idx >= (int)slot->schedule.size()) {
                slot->completed = true;

                // Windowed VAE decode of completed slot
                result = decode_completed(slot, active_indices[bi]);

                fprintf(stderr, "[StreamPipeline] Slot %d completed (id=%d, tick=%d)\n",
                        active_indices[bi], slot->slot_id, m_tick_count);
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        m_last_tick_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        m_tick_count++;

        if (m_tick_count % 10 == 0 || result.has_value()) {
            fprintf(stderr, "[StreamPipeline] Tick %d: %d active, %.1f ms%s\n",
                    m_tick_count, B, m_last_tick_ms,
                    result ? " [COMPLETED]" : "");
        }

        return result;
    }

    // Drain: tick until all slots complete and queue is empty
    std::vector<StreamPreview> drain() {
        std::vector<StreamPreview> previews;
        while (active_slots() > 0 || !m_queue.empty()) {
            auto p = tick();
            if (p) previews.push_back(std::move(*p));
        }
        return previews;
    }

    // How many slots are currently in-flight
    int active_slots() const {
        int n = 0;
        for (auto* s : m_slots) {
            if (s && !s->completed) n++;
        }
        return n;
    }

    // How many are completed but not yet harvested
    int completed_slots() const {
        int n = 0;
        for (auto* s : m_slots) {
            if (s && s->completed) n++;
        }
        return n;
    }

    float last_tick_ms() const { return m_last_tick_ms; }
    int   total_ticks()  const { return m_tick_count; }
    bool  queue_empty()  const { return m_queue.empty(); }

private:
    DitTrt*              m_trt;
    VaeOrt*              m_vae;
    StreamConfig         m_config;
    DitTrtStreamBuffers  m_bufs = {};
    std::vector<StreamSlot*> m_slots;
    std::deque<StreamSlotRequest> m_queue;
    int   m_tick_count;
    float m_last_tick_ms;
    int   m_next_slot_id;

    // ── Build timestep schedule ─────────────────────────────────────────
    // Same as DEMON's _build_schedule: shift-weighted schedule from 1→0
    // then optionally truncated by denoise.
    std::vector<float> build_schedule(float denoise, int num_steps, float shift) {
        std::vector<float> sched(num_steps + 1);
        for (int i = 0; i <= num_steps; i++) {
            float t = 1.0f - (float)i / (float)num_steps;
            sched[i] = shift * t / (1.0f + (shift - 1.0f) * t);
        }

        // Truncate by denoise (like DEMON's per-slot denoise)
        if (denoise < 1.0f && denoise > 0.0f) {
            // Find the start index where schedule <= denoise threshold
            float threshold = sched[0] * denoise;
            int start = 0;
            for (int i = 0; i < (int)sched.size(); i++) {
                if (sched[i] <= threshold) {
                    start = i;
                    break;
                }
            }
            if (start > 0) {
                sched.erase(sched.begin(), sched.begin() + start);
            }
        }
        return sched;
    }

    // ── Initialize a slot from a request ────────────────────────────────
    StreamSlot* init_slot(const StreamSlotRequest& req) {
        auto* slot = new StreamSlot();
        slot->request = req;
        slot->slot_id = m_next_slot_id++;
        slot->step_idx = 0;
        slot->completed = false;

        // Build per-slot schedule
        slot->schedule = build_schedule(
            req.denoise, m_config.num_steps, m_config.shift);

        // Initialize xt: pure noise (or blended with source for covers)
        int T = req.T;
        int Oc = 64;
        slot->xt.resize(T * Oc);

        // Generate Philox noise (matches torch.randn on CUDA with bf16)
        philox_randn(req.seed, slot->xt.data(), T * Oc, /*bf16_round=*/true);

        // Cover mode: blend noise with source latents
        if (!req.source_latents.empty() && req.denoise < 1.0f) {
            float sigma = slot->schedule[0];  // starting sigma
            for (int i = 0; i < T * Oc; i++) {
                slot->xt[i] = sigma * slot->xt[i]
                            + (1.0f - sigma) * req.source_latents[i];
            }
        }

        return slot;
    }

    // ── Decode completed slot via windowed VAE ──────────────────────────
    std::optional<StreamPreview> decode_completed(StreamSlot* slot, int slot_idx) {
        if (!m_vae) {
            fprintf(stderr, "[StreamPipeline] WARNING: no VAE available, skipping decode\n");
            return std::nullopt;
        }

        int T = slot->request.T;
        int T_audio_max = T * 1920;
        std::vector<float> audio(2 * T_audio_max);

        auto t0 = std::chrono::steady_clock::now();

        int T_audio = vae_ort_decode_tiled(
            m_vae, slot->xt.data(), T, audio.data(), T_audio_max,
            m_config.vae_chunk, m_config.vae_overlap);

        auto t1 = std::chrono::steady_clock::now();
        float vae_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        if (T_audio <= 0) {
            fprintf(stderr, "[StreamPipeline] WARNING: VAE decode failed for slot %d\n",
                    slot_idx);
            return std::nullopt;
        }

        fprintf(stderr, "[StreamPipeline] VAE decode: slot=%d T=%d T_audio=%d %.1f ms\n",
                slot_idx, T, T_audio, vae_ms);

        // Write preview WAV
        std::string wav_path;
        if (!m_config.chunk_dir.empty()) {
            char fname[128];
            snprintf(fname, sizeof(fname), "stream_%04d_slot%d.wav",
                     m_tick_count, slot_idx);
            wav_path = m_config.chunk_dir + "/" + fname;

            // Write 48kHz stereo WAV (interleaved from planar)
            write_wav_planar(wav_path.c_str(), audio.data(), T_audio, 2, 48000);
        }

        float duration = (float)T_audio / 48000.0f;

        StreamPreview preview;
        preview.wav_path  = wav_path;
        preview.t_start   = 0.0f;  // full decode for now
        preview.duration  = duration;
        preview.tick_idx  = m_tick_count;
        preview.slot_idx  = slot_idx;
        preview.is_final  = m_queue.empty() && active_slots() <= 1;

        return preview;
    }

    // ── Minimal WAV writer (planar float → interleaved 16-bit) ──────────
    static bool write_wav_planar(const char* path, const float* planar,
                                  int samples_per_ch, int channels, int sr) {
        FILE* f = fopen(path, "wb");
        if (!f) return false;

        int total_samples = samples_per_ch * channels;
        int data_bytes = total_samples * 2;  // 16-bit PCM
        int file_size = 36 + data_bytes;

        // WAV header
        fwrite("RIFF", 1, 4, f);
        uint32_t chunk_size = file_size;
        fwrite(&chunk_size, 4, 1, f);
        fwrite("WAVE", 1, 4, f);

        // fmt sub-chunk
        fwrite("fmt ", 1, 4, f);
        uint32_t fmt_size = 16;
        fwrite(&fmt_size, 4, 1, f);
        uint16_t audio_format = 1;  // PCM
        fwrite(&audio_format, 2, 1, f);
        uint16_t num_channels = (uint16_t)channels;
        fwrite(&num_channels, 2, 1, f);
        uint32_t sample_rate = (uint32_t)sr;
        fwrite(&sample_rate, 4, 1, f);
        uint32_t byte_rate = sample_rate * num_channels * 2;
        fwrite(&byte_rate, 4, 1, f);
        uint16_t block_align = num_channels * 2;
        fwrite(&block_align, 2, 1, f);
        uint16_t bits_per_sample = 16;
        fwrite(&bits_per_sample, 2, 1, f);

        // data sub-chunk
        fwrite("data", 1, 4, f);
        uint32_t data_size = (uint32_t)data_bytes;
        fwrite(&data_size, 4, 1, f);

        // Convert planar float → interleaved 16-bit PCM
        // Input: [ch0: s0,s1,...,sN, ch1: s0,s1,...,sN]
        // Output: [s0_ch0, s0_ch1, s1_ch0, s1_ch1, ...]
        std::vector<int16_t> pcm(total_samples);
        for (int s = 0; s < samples_per_ch; s++) {
            for (int c = 0; c < channels; c++) {
                float v = planar[c * samples_per_ch + s];
                v = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
                pcm[s * channels + c] = (int16_t)(v * 32767.0f);
            }
        }
        fwrite(pcm.data(), 2, total_samples, f);
        fclose(f);
        return true;
    }
};

#endif // HOT_STEP_TRT
