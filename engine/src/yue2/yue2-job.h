#pragma once
// yue2/yue2-job.h — wires yue2-pipeline.h into the shared Job/work_push
// system defined earlier in engine/tools/hot-step-server.cpp. Milestones
// M7/M8, docs/plans/yue2/06-engine-port-plan.md §7.
//
// HOT-Step file (no acestep.cpp analog). MID-FILE include (same convention as
// minimax/mm3-job.h): needs Job, job_create(), job_set_phase(), work_push()
// and g_store, all defined earlier in hot-step-server.cpp. Registers NO
// routes of its own — yue2-server.h (which includes this file) is what calls
// svr.Post("/yue2/synth", ...); this file only supplies the worker function
// and the process-wide YuE2 model/tokenizer state.
//
// Unlike MM3, YuE2 registers no separate job route at all (no /yue2/job) —
// progress/result/cancel all go through the SHARED /job routes
// (hot-step-server.cpp's own GET/POST /job, already serving ACE and MM3).
// This file's only obligation is to keep Job's generic phase/result fields
// (plus the two additive result_end_reason/result_stage_end_reasons fields)
// correctly filled while the pipeline runs.
//
// `arbitratesResidencyInEngine = false` for YuE2 (docs/plans/yue2/
// 01-seam-design.md §2.5, same value as ACE's own) — the Node-side eviction
// runner clears other families' VRAM before calling this backend; this file
// only handles YuE2's OWN "keep loaded" toggle (g_keep_loaded), mirroring
// mm3-job.h's own release_if_transient posture but far simpler (no AR
// cache/Plank/ensemble state to consider).

#include "yue2-model.h"
#include "yue2-pipeline.h"
#include "yue2-request.h"
#include "yue2-tokenizer.h"

#include "audio-io.h"
#include "yyjson.h"

#include <cstdio>
#include <memory>
#include <mutex>
#include <string>

static std::mutex   g_yue2_mutex;
static Yue2Model     g_yue2;
static BPETokenizer g_yue2_tok;
static bool           g_yue2_tok_loaded = false;

// Loads the BPE tokenizer from the LM GGUF the first time it's needed.
// Caller must hold g_yue2_mutex and must already know g_yue2.lm_file is
// present/probed (yue2_available(g_yue2)).
static bool yue2_ensure_tokenizer(std::string * err) {
    if (g_yue2_tok_loaded) {
        return true;
    }
    if (!yue2_available(g_yue2)) {
        if (err) {
            *err = "YuE2 LM GGUF not found or metadata probe failed";
        }
        return false;
    }
    if (!yue2_tokenizer_load_from_gguf(&g_yue2_tok, g_yue2.lm_file.path)) {
        if (err) {
            *err = "failed to load the YuE2 tokenizer from " + g_yue2.lm_file.path;
        }
        return false;
    }
    g_yue2_tok_loaded = true;
    return true;
}

static const char * yue2_stage_name(Yue2Stage s) {
    switch (s) {
        case YUE2_STAGE_PLAN:
            return "plan";
        case YUE2_STAGE_SEMANTIC:
            return "semantic";
        case YUE2_STAGE_NAR:
            return "nar";
        case YUE2_STAGE_VAE:
            return "vae";
    }
    return "?";
}

static JobPhase yue2_job_phase_for_stage(Yue2Stage s) {
    switch (s) {
        case YUE2_STAGE_PLAN:
            return JobPhase::YUE2_PLAN;
        case YUE2_STAGE_SEMANTIC:
            return JobPhase::YUE2_SEMANTIC;
        case YUE2_STAGE_NAR:
            return JobPhase::YUE2_NAR;
        case YUE2_STAGE_VAE:
            return JobPhase::VAE_DECODE;  // same concept ACE's own VAE_DECODE phase names
    }
    return JobPhase::DIT_INFERENCE;
}

// The one worker function, run on the shared work_push() GPU-serializing
// thread. Single-take, non-streaming (v1 scope) — no ensemble takes, no
// AR-cache replay, no Play-While-Rendering (docs/plans/yue2/
// 06-engine-port-plan.md §7's explicit v1 exclusions).
static void yue2_synth_worker(std::shared_ptr<Job> job, Yue2Request req) {
    if (job->cancel.load()) {
        job_set_phase(*job, JobPhase::CANCELLED);
        job->status.store(3);
        return;
    }

    std::lock_guard<std::mutex> lock(g_yue2_mutex);

    job_set_phase(*job, JobPhase::LOADING_DIT);
    std::string err;
    if (!yue2_load_parts(&g_yue2, /*want_lm=*/true, /*want_vae=*/true, req.vae_variant, /*want_encoder=*/false,
                         &err)) {
        job->result_body = err.empty() ? "YuE2 load failed" : err;
        job->result_mime  = "text/plain";
        job_set_phase(*job, JobPhase::FAILED);
        job->status.store(2);
        return;
    }
    if (!yue2_ensure_tokenizer(&err)) {
        job->result_body = err;
        job->result_mime  = "text/plain";
        job_set_phase(*job, JobPhase::FAILED);
        job->status.store(2);
        return;
    }

    Yue2ProgressFn progress = [&job](const Yue2Progress & p) {
        job_set_phase(*job, yue2_job_phase_for_stage(p.stage), (int) p.step, (int) (p.total > 0 ? p.total : 0));
    };

    Yue2PipelineResult result;
    const bool ok = yue2_pipeline_run(g_yue2, g_yue2_tok, req, progress, &job->cancel, &result, &err);

    // Post-run residency: mirrors ACE's own EVICT_STRICT default / mm3-job.h's
    // release_if_transient posture. arbitratesResidencyInEngine=false means
    // the Node-side runner is what evicts OTHER families before a YuE2 call;
    // this only handles YuE2's own "keep models loaded" toggle.
    if (!g_keep_loaded) {
        yue2_unload(&g_yue2);
    }

    if (!ok) {
        const bool cancelled = job->cancel.load();
        job->result_body = err.empty() ? "YuE2 generation failed" : err;
        job->result_mime  = "text/plain";
        job_set_phase(*job, cancelled ? JobPhase::CANCELLED : JobPhase::FAILED);
        job->status.store(cancelled ? 3 : 2);
        return;
    }

    job->result_body        = audio_encode_wav_s16(result.audio_planar.data(), (int) result.samples,
                                                    result.sample_rate);
    job->result_mime         = "audio/wav";
    job->result_end_reason  = result.end_reason;
    {
        yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
        yyjson_mut_val * root = yyjson_mut_obj(doc);
        yyjson_mut_doc_set_root(doc, root);
        // Only plan/semantic carry a meaningful per-stage terminator/limit
        // value (docs/plans/yue2/06-engine-port-plan.md §7) — nar/vae stay
        // absent (omitted), never a fabricated "completed" value.
        for (int s = 0; s < 2; s++) {
            if (!result.stage_end_reason[s].empty()) {
                yyjson_mut_obj_add_strcpy(doc, root, yue2_stage_name((Yue2Stage) s),
                                          result.stage_end_reason[s].c_str());
            }
        }
        char * json = yyjson_mut_write(doc, 0, NULL);
        job->result_stage_end_reasons = json ? json : "{}";
        yyjson_mut_doc_free(doc);
        if (json) {
            free(json);
        }
    }

    job_set_phase(*job, JobPhase::DONE);
    job->status.store(1);
    fprintf(stderr, "[YuE2-Job] %s: stage_ms plan=%.1f semantic=%.1f nar=%.1f vae=%.1f\n",
            job->id.c_str(), result.stage_ms[YUE2_STAGE_PLAN], result.stage_ms[YUE2_STAGE_SEMANTIC],
            result.stage_ms[YUE2_STAGE_NAR], result.stage_ms[YUE2_STAGE_VAE]);
    fprintf(stderr, "[YuE2-Job] %s: done (%lld frames, %lld samples, end_reason=%s)\n", job->id.c_str(),
            (long long) result.total_frames, (long long) result.samples, result.end_reason.c_str());
}
