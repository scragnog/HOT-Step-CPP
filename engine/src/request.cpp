// request.cpp: AceStep request JSON read/write (yyjson)

#include "request.h"

#include "task-types.h"
#include "yyjson.h"

#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const yyjson_write_flag WRITE_FLAGS =
    YYJSON_WRITE_PRETTY | YYJSON_WRITE_PRETTY_TWO_SPACES | YYJSON_WRITE_FP_TO_FIXED(2);

// Defaults (aligned with Python GenerationParams)
void request_init(AceRequest * r) {
    r->caption          = "";
    r->lyrics           = "";
    r->negative_prompt  = "";

    r->bpm                  = 0;
    r->duration             = 0.0f;
    r->keyscale             = "";
    r->timesignature        = "";
    r->vocal_language       = "";
    r->lm_batch_size        = 1;
    r->synth_batch_size     = 1;
    r->seed                 = -1;
    r->lm_temperature       = 0.85f;
    r->lm_cfg_scale         = 2.0f;
    r->lm_cfg_cutoff_ratio  = 1.0f;
    r->lm_top_p             = 0.9f;
    r->lm_top_k             = 0;
    r->lm_negative_prompt   = "";
    r->lm_seed              = -1;
    r->use_cot_caption      = true;
    r->audio_codes          = "";
    r->inference_steps      = 0;     // 0 = auto (turbo: 8, base/sft: 50)
    r->guidance_scale       = 0.0f;  // 0 = auto (1.0 for all models)
    r->shift                = 0.0f;  // 0 = auto (turbo: 3.0, base/sft: 1.0)
    r->dcw_scaler           = 0.0f;
    r->dcw_high_scaler      = 0.0f;
    r->dcw_mode             = DCW_MODE_LOW;
    r->audio_cover_strength = 1.0f;
    r->cover_noise_strength = 0.0f;
    r->cover_noise_method   = "";
    r->repainting_start     = 0.0f;
    r->repainting_end       = -1.0f;
    r->latent_shift         = 0.0f;
    r->latent_rescale       = 1.0f;
    r->lss_strength         = 0.0f;
    r->lss_var_thresh       = 0.15f;
    r->lss_dc_remove        = true;
    r->custom_timesteps     = "";
    r->task_type            = TASK_TEXT2MUSIC;
    r->track                = "";
    r->solver               = SOLVER_EULER;
    r->stork_substeps       = STORK_SUBSTEPS_DEFAULT;
    r->lm_mode              = LM_MODE_NAME_GENERATE;
    r->output_format        = OUTPUT_FORMAT_MP3;
    r->synth_model          = "";
    r->lm_model             = "";
    r->adapter              = "";
    r->adapter_scale        = 1.0f;
    r->vae                  = "";
    r->peak_clip            = 10;
    r->mp3_bitrate          = 128;
    r->pp_vae_reencode      = false;
    r->postprocess_plugin   = "";
    r->get_lrc              = false;
    r->use_ort_vae          = false;
    r->stream_mode           = false;
    r->stream_depth          = 8;
    r->stream_chunk_dir      = "";
}

// helper: get yyjson string as std::string
static inline std::string yy_str(yyjson_val * v) {
    return std::string(yyjson_get_str(v), yyjson_get_len(v));
}

// populate AceRequest fields from a yyjson object (must be pre-initialized)
static void request_parse_obj(yyjson_val * obj, AceRequest * r) {
    yyjson_val * v;

    // strings
    if ((v = yyjson_obj_get(obj, "caption")) && yyjson_is_str(v)) {
        r->caption = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "lyrics")) && yyjson_is_str(v)) {
        r->lyrics = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "keyscale")) && yyjson_is_str(v)) {
        r->keyscale = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "timesignature")) && yyjson_is_str(v)) {
        r->timesignature = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "vocal_language")) && yyjson_is_str(v)) {
        r->vocal_language = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "audio_codes")) && yyjson_is_str(v)) {
        r->audio_codes = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "negative_prompt")) && yyjson_is_str(v)) {
        r->negative_prompt = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "lm_negative_prompt")) && yyjson_is_str(v)) {
        r->lm_negative_prompt = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "task_type")) && yyjson_is_str(v) && yyjson_get_len(v) > 0) {
        r->task_type = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "track")) && yyjson_is_str(v)) {
        r->track = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "solver")) && yyjson_is_str(v) && yyjson_get_len(v) > 0) {
        r->solver = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "stork_substeps")) && yyjson_is_int(v)) {
        r->stork_substeps = (int) yyjson_get_int(v);
    }
    if ((v = yyjson_obj_get(obj, "custom_timesteps")) && yyjson_is_str(v)) {
        r->custom_timesteps = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "lm_mode")) && yyjson_is_str(v) && yyjson_get_len(v) > 0) {
        r->lm_mode = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "output_format")) && yyjson_is_str(v) && yyjson_get_len(v) > 0) {
        r->output_format = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "synth_model")) && yyjson_is_str(v)) {
        r->synth_model = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "lm_model")) && yyjson_is_str(v)) {
        r->lm_model = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "adapter")) && yyjson_is_str(v)) {
        r->adapter = yy_str(v);
    }
    // Multi-adapter stack: array of {name, scale} objects (or bare name strings).
    // Supersedes the single `adapter`/`adapter_scale` when present and non-empty.
    if ((v = yyjson_obj_get(obj, "adapters")) && yyjson_is_arr(v)) {
        size_t       a_idx, a_max;
        yyjson_val * a_val;
        yyjson_arr_foreach(v, a_idx, a_max, a_val) {
            AceAdapterRef ref;
            if (yyjson_is_str(a_val)) {
                ref.name = yy_str(a_val);
            } else if (yyjson_is_obj(a_val)) {
                yyjson_val * nv = yyjson_obj_get(a_val, "name");
                yyjson_val * sv = yyjson_obj_get(a_val, "scale");
                if (nv && yyjson_is_str(nv)) ref.name = yy_str(nv);
                if (sv && yyjson_is_num(sv)) ref.scale = (float) yyjson_get_num(sv);
                // timestep gain curve: array of g(x) samples over x ∈ [0,1]
                yyjson_val * gv = yyjson_obj_get(a_val, "gain_curve");
                if (gv && yyjson_is_arr(gv)) {
                    size_t       g_idx, g_max;
                    yyjson_val * g_val;
                    yyjson_arr_foreach(gv, g_idx, g_max, g_val) {
                        if (yyjson_is_num(g_val)) ref.gain_curve.push_back((float) yyjson_get_num(g_val));
                    }
                }
                // curve x-axis domain: "steps" (remaining-steps fraction) or "t"
                yyjson_val * gd = yyjson_obj_get(a_val, "gain_domain");
                if (gd && yyjson_is_str(gd)) ref.gain_in_steps = (yy_str(gd) == "steps");
            }
            if (!ref.name.empty()) r->adapters.push_back(ref);
        }
    }
    // Per-section adapter masking: array of { weights: [..], size: n }.
    if ((v = yyjson_obj_get(obj, "adapter_sections")) && yyjson_is_arr(v)) {
        size_t       s_idx, s_max;
        yyjson_val * s_val;
        yyjson_arr_foreach(v, s_idx, s_max, s_val) {
            if (!yyjson_is_obj(s_val)) continue;
            AceAdapterSection sec;
            yyjson_val * wv = yyjson_obj_get(s_val, "weights");
            if (wv && yyjson_is_arr(wv)) {
                size_t       w_idx, w_max;
                yyjson_val * w_val;
                yyjson_arr_foreach(wv, w_idx, w_max, w_val) {
                    if (yyjson_is_num(w_val)) sec.weights.push_back((float) yyjson_get_num(w_val));
                }
            }
            yyjson_val * sz = yyjson_obj_get(s_val, "size");
            if (sz && yyjson_is_num(sz)) sec.size = (float) yyjson_get_num(sz);
            r->adapter_sections.push_back(sec);
        }
    }
    if ((v = yyjson_obj_get(obj, "vae")) && yyjson_is_str(v)) {
        r->vae = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "postprocess_plugin")) && yyjson_is_str(v)) {
        r->postprocess_plugin = yy_str(v);
    }

    // ints
    if ((v = yyjson_obj_get(obj, "bpm")) && yyjson_is_num(v)) {
        r->bpm = (int) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "lm_batch_size")) && yyjson_is_num(v)) {
        r->lm_batch_size = (int) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "synth_batch_size")) && yyjson_is_num(v)) {
        r->synth_batch_size = (int) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "seed")) && yyjson_is_num(v)) {
        r->seed = (int64_t) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "lm_seed")) && yyjson_is_num(v)) {
        r->lm_seed = (int64_t) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "lm_top_k")) && yyjson_is_num(v)) {
        r->lm_top_k = (int) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "inference_steps")) && yyjson_is_num(v)) {
        r->inference_steps = (int) yyjson_get_num(v);
    }

    // floats
    if ((v = yyjson_obj_get(obj, "duration")) && yyjson_is_num(v)) {
        r->duration = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "lm_temperature")) && yyjson_is_num(v)) {
        r->lm_temperature = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "lm_cfg_scale")) && yyjson_is_num(v)) {
        r->lm_cfg_scale = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "lm_cfg_cutoff_ratio")) && yyjson_is_num(v)) {
        r->lm_cfg_cutoff_ratio = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "lm_top_p")) && yyjson_is_num(v)) {
        r->lm_top_p = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "guidance_scale")) && yyjson_is_num(v)) {
        r->guidance_scale = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "shift")) && yyjson_is_num(v)) {
        r->shift = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "dcw_scaler")) && yyjson_is_num(v)) {
        r->dcw_scaler = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "dcw_high_scaler")) && yyjson_is_num(v)) {
        r->dcw_high_scaler = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "dcw_mode")) && yyjson_is_str(v)) {
        r->dcw_mode = yyjson_get_str(v);
    }
    if ((v = yyjson_obj_get(obj, "audio_cover_strength")) && yyjson_is_num(v)) {
        r->audio_cover_strength = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "cover_noise_strength")) && yyjson_is_num(v)) {
        r->cover_noise_strength = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "cover_noise_method")) && yyjson_is_str(v)) {
        r->cover_noise_method = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "repainting_start")) && yyjson_is_num(v)) {
        r->repainting_start = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "repainting_end")) && yyjson_is_num(v)) {
        r->repainting_end = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "latent_shift")) && yyjson_is_num(v)) {
        r->latent_shift = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "latent_rescale")) && yyjson_is_num(v)) {
        r->latent_rescale = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "lss_strength")) && yyjson_is_num(v)) {
        r->lss_strength = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "lss_var_thresh")) && yyjson_is_num(v)) {
        r->lss_var_thresh = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "lss_dc_remove")) && yyjson_is_bool(v)) {
        r->lss_dc_remove = yyjson_get_bool(v);
    }
    if ((v = yyjson_obj_get(obj, "peak_clip")) && yyjson_is_num(v)) {
        r->peak_clip = (int) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "mp3_bitrate")) && yyjson_is_num(v)) {
        r->mp3_bitrate = (int) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "adapter_scale")) && yyjson_is_num(v)) {
        r->adapter_scale = (float) yyjson_get_num(v);
    }

    // bools
    if ((v = yyjson_obj_get(obj, "use_cot_caption"))) {
        if (yyjson_is_bool(v)) {
            r->use_cot_caption = yyjson_get_bool(v);
        } else if (yyjson_is_str(v)) {
            const char * s     = yyjson_get_str(v);
            r->use_cot_caption = (strcmp(s, "true") == 0 || strcmp(s, "1") == 0);
        }
    }
    if ((v = yyjson_obj_get(obj, "pp_vae_reencode"))) {
        if (yyjson_is_bool(v)) {
            r->pp_vae_reencode = yyjson_get_bool(v);
        } else if (yyjson_is_str(v)) {
            const char * s       = yyjson_get_str(v);
            r->pp_vae_reencode   = (strcmp(s, "true") == 0 || strcmp(s, "1") == 0);
        }
    }
    if ((v = yyjson_obj_get(obj, "get_lrc"))) {
        if (yyjson_is_bool(v)) {
            r->get_lrc = yyjson_get_bool(v);
        } else if (yyjson_is_str(v)) {
            const char * s = yyjson_get_str(v);
            r->get_lrc     = (strcmp(s, "true") == 0 || strcmp(s, "1") == 0);
        }
    }
    if ((v = yyjson_obj_get(obj, "use_ort_vae"))) {
        if (yyjson_is_bool(v)) {
            r->use_ort_vae = yyjson_get_bool(v);
        } else if (yyjson_is_str(v)) {
            const char * s = yyjson_get_str(v);
            r->use_ort_vae = (strcmp(s, "true") == 0 || strcmp(s, "1") == 0);
        }
    }

    // streaming
    if ((v = yyjson_obj_get(obj, "stream_mode"))) {
        if (yyjson_is_bool(v)) {
            r->stream_mode = yyjson_get_bool(v);
        } else if (yyjson_is_str(v)) {
            const char * s = yyjson_get_str(v);
            r->stream_mode = (strcmp(s, "true") == 0 || strcmp(s, "1") == 0);
        }
    }
    if ((v = yyjson_obj_get(obj, "stream_depth")) && yyjson_is_num(v)) {
        r->stream_depth = (int) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "stream_chunk_dir")) && yyjson_is_str(v)) {
        r->stream_chunk_dir = yy_str(v);
    }

    // Lyrics is the source of truth for instrumental mode.
    // The DiT was trained with lyrics="[Instrumental]" and language="unknown".
    // Force vocal_language to "unknown" to match the training distribution.
    if (r->lyrics == "[Instrumental]" && r->vocal_language != "unknown") {
        r->vocal_language = "unknown";
    }
}

// Core parser: takes a raw JSON string. Used by the server directly.
bool request_parse_json(AceRequest * r, const char * json) {
    request_init(r);

    yyjson_doc * doc = yyjson_read(json, strlen(json), 0);
    if (!doc) {
        return false;
    }

    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) {
        yyjson_doc_free(doc);
        return false;
    }

    request_parse_obj(root, r);
    yyjson_doc_free(doc);
    return true;
}

// Parse JSON: single object {} or array [{}, ...] into a vector.
bool request_parse_json_array(const char * json, std::vector<AceRequest> * out) {
    yyjson_doc * doc = yyjson_read(json, strlen(json), 0);
    if (!doc) {
        return false;
    }

    yyjson_val * root = yyjson_doc_get_root(doc);

    if (yyjson_is_obj(root)) {
        AceRequest r;
        request_init(&r);
        request_parse_obj(root, &r);
        out->push_back(r);
    } else if (yyjson_is_arr(root)) {
        size_t       idx, max;
        yyjson_val * val;
        yyjson_arr_foreach(root, idx, max, val) {
            if (!yyjson_is_obj(val)) {
                yyjson_doc_free(doc);
                return false;
            }
            AceRequest r;
            request_init(&r);
            request_parse_obj(val, &r);
            out->push_back(r);
        }
    } else {
        yyjson_doc_free(doc);
        return false;
    }

    yyjson_doc_free(doc);
    return !out->empty();
}

// File I/O
static std::string read_file(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return "";
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string buf((size_t) sz, '\0');
    size_t      nr = fread(&buf[0], 1, (size_t) sz, f);
    fclose(f);
    if ((long) nr != sz) {
        buf.resize(nr);
    }
    return buf;
}

// File wrapper: reads JSON from disk, then delegates to request_parse_json.
bool request_parse(AceRequest * r, const char * path) {
    std::string json = read_file(path);
    if (json.empty()) {
        fprintf(stderr, "[Request] ERROR: cannot read %s\n", path);
        return false;
    }

    if (!request_parse_json(r, json.c_str())) {
        fprintf(stderr, "[Request] ERROR: malformed JSON in %s\n", path);
        return false;
    }

    fprintf(stderr, "[Request] Parsed %s\n", path);
    return true;
}

// build a yyjson mutable document from an AceRequest.
// sparse=true: omit fields at their request_init default.
// sparse=false: serialize everything (for /props documentation).
static yyjson_mut_doc * request_build_doc(const AceRequest * r, bool sparse) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    // single source of truth: compare against request_init defaults
    AceRequest def;
    request_init(&def);
    bool all = !sparse;

    // always present
    yyjson_mut_obj_add_str(doc, root, "caption", r->caption.c_str());
    yyjson_mut_obj_add_sint(doc, root, "seed", r->seed);

    // text
    if (all || r->lyrics != def.lyrics) {
        yyjson_mut_obj_add_str(doc, root, "lyrics", r->lyrics.c_str());
    }
    if (all || r->audio_codes != def.audio_codes) {
        yyjson_mut_obj_add_str(doc, root, "audio_codes", r->audio_codes.c_str());
    }

    // metadata
    if (all || r->bpm != def.bpm) {
        yyjson_mut_obj_add_int(doc, root, "bpm", r->bpm);
    }
    if (all || r->duration != def.duration) {
        yyjson_mut_obj_add_real(doc, root, "duration", r->duration);
    }
    if (all || r->keyscale != def.keyscale) {
        yyjson_mut_obj_add_str(doc, root, "keyscale", r->keyscale.c_str());
    }
    if (all || r->timesignature != def.timesignature) {
        yyjson_mut_obj_add_str(doc, root, "timesignature", r->timesignature.c_str());
    }
    if (all || r->vocal_language != def.vocal_language) {
        yyjson_mut_obj_add_str(doc, root, "vocal_language", r->vocal_language.c_str());
    }

    // LM control
    if (all || r->lm_batch_size != def.lm_batch_size) {
        yyjson_mut_obj_add_int(doc, root, "lm_batch_size", r->lm_batch_size);
    }
    if (all || r->lm_temperature != def.lm_temperature) {
        yyjson_mut_obj_add_real(doc, root, "lm_temperature", r->lm_temperature);
    }
    if (all || r->lm_cfg_scale != def.lm_cfg_scale) {
        yyjson_mut_obj_add_real(doc, root, "lm_cfg_scale", r->lm_cfg_scale);
    }
    if (all || r->lm_cfg_cutoff_ratio != def.lm_cfg_cutoff_ratio) {
        yyjson_mut_obj_add_real(doc, root, "lm_cfg_cutoff_ratio", r->lm_cfg_cutoff_ratio);
    }
    if (all || r->lm_top_p != def.lm_top_p) {
        yyjson_mut_obj_add_real(doc, root, "lm_top_p", r->lm_top_p);
    }
    if (all || r->lm_top_k != def.lm_top_k) {
        yyjson_mut_obj_add_int(doc, root, "lm_top_k", r->lm_top_k);
    }
    if (all || r->negative_prompt != def.negative_prompt) {
        yyjson_mut_obj_add_str(doc, root, "negative_prompt", r->negative_prompt.c_str());
    }
    if (all || r->lm_negative_prompt != def.lm_negative_prompt) {
        yyjson_mut_obj_add_str(doc, root, "lm_negative_prompt", r->lm_negative_prompt.c_str());
    }
    if (all || r->lm_seed != def.lm_seed) {
        yyjson_mut_obj_add_sint(doc, root, "lm_seed", r->lm_seed);
    }
    if (all || r->use_cot_caption != def.use_cot_caption) {
        yyjson_mut_obj_add_bool(doc, root, "use_cot_caption", r->use_cot_caption);
    }

    // DiT control (0 = auto-detect from model)
    if (all || r->inference_steps != def.inference_steps) {
        yyjson_mut_obj_add_int(doc, root, "inference_steps", r->inference_steps);
    }
    if (all || r->guidance_scale != def.guidance_scale) {
        yyjson_mut_obj_add_real(doc, root, "guidance_scale", r->guidance_scale);
    }
    if (all || r->shift != def.shift) {
        yyjson_mut_obj_add_real(doc, root, "shift", r->shift);
    }
    if (all || r->dcw_scaler != def.dcw_scaler) {
        yyjson_mut_obj_add_real(doc, root, "dcw_scaler", r->dcw_scaler);
    }
    if (all || r->dcw_high_scaler != def.dcw_high_scaler) {
        yyjson_mut_obj_add_real(doc, root, "dcw_high_scaler", r->dcw_high_scaler);
    }
    if (all || r->dcw_mode != def.dcw_mode) {
        yyjson_mut_obj_add_str(doc, root, "dcw_mode", r->dcw_mode.c_str());
    }
    // solver is always emitted for the same reason as task_type: the
    // request is explicit about its solver choice in any round trip.
    yyjson_mut_obj_add_str(doc, root, "solver", r->solver.c_str());
    if (all || r->stork_substeps != def.stork_substeps) {
        yyjson_mut_obj_add_int(doc, root, "stork_substeps", r->stork_substeps);
    }
    // lm_mode and output_format follow the same rule: enumerations with a
    // guaranteed non-empty value, always explicit in serialized output.
    yyjson_mut_obj_add_str(doc, root, "lm_mode", r->lm_mode.c_str());
    yyjson_mut_obj_add_str(doc, root, "output_format", r->output_format.c_str());

    // batch
    if (all || r->synth_batch_size != def.synth_batch_size) {
        yyjson_mut_obj_add_int(doc, root, "synth_batch_size", r->synth_batch_size);
    }

    // cover/repaint
    if (all || r->audio_cover_strength != def.audio_cover_strength) {
        yyjson_mut_obj_add_real(doc, root, "audio_cover_strength", r->audio_cover_strength);
    }
    if (all || r->cover_noise_strength != def.cover_noise_strength) {
        yyjson_mut_obj_add_real(doc, root, "cover_noise_strength", r->cover_noise_strength);
    }
    if (all || !r->cover_noise_method.empty()) {
        yyjson_mut_obj_add_strcpy(doc, root, "cover_noise_method", r->cover_noise_method.c_str());
    }
    if (all || r->repainting_start != def.repainting_start) {
        yyjson_mut_obj_add_real(doc, root, "repainting_start", r->repainting_start);
    }
    if (all || r->repainting_end != def.repainting_end) {
        yyjson_mut_obj_add_real(doc, root, "repainting_end", r->repainting_end);
    }
    if (all || r->latent_shift != def.latent_shift) {
        yyjson_mut_obj_add_real(doc, root, "latent_shift", r->latent_shift);
    }
    if (all || r->latent_rescale != def.latent_rescale) {
        yyjson_mut_obj_add_real(doc, root, "latent_rescale", r->latent_rescale);
    }
    if (all || r->lss_strength != def.lss_strength) {
        yyjson_mut_obj_add_real(doc, root, "lss_strength", r->lss_strength);
    }
    if (all || r->lss_var_thresh != def.lss_var_thresh) {
        yyjson_mut_obj_add_real(doc, root, "lss_var_thresh", r->lss_var_thresh);
    }
    if (all || r->lss_dc_remove != def.lss_dc_remove) {
        yyjson_mut_obj_add_bool(doc, root, "lss_dc_remove", r->lss_dc_remove);
    }
    if (all || r->custom_timesteps != def.custom_timesteps) {
        yyjson_mut_obj_add_str(doc, root, "custom_timesteps", r->custom_timesteps.c_str());
    }
    // task_type is always emitted: it is the single source of truth for the
    // request and must be explicit in any round trip.
    yyjson_mut_obj_add_str(doc, root, "task_type", r->task_type.c_str());
    if (all || r->track != def.track) {
        yyjson_mut_obj_add_str(doc, root, "track", r->track.c_str());
    }
    if (all || r->peak_clip != def.peak_clip) {
        yyjson_mut_obj_add_int(doc, root, "peak_clip", r->peak_clip);
    }
    if (all || r->mp3_bitrate != def.mp3_bitrate) {
        yyjson_mut_obj_add_int(doc, root, "mp3_bitrate", r->mp3_bitrate);
    }
    if (all || r->synth_model != def.synth_model) {
        yyjson_mut_obj_add_str(doc, root, "synth_model", r->synth_model.c_str());
    }
    if (all || r->lm_model != def.lm_model) {
        yyjson_mut_obj_add_str(doc, root, "lm_model", r->lm_model.c_str());
    }
    if (all || r->adapter != def.adapter) {
        yyjson_mut_obj_add_str(doc, root, "adapter", r->adapter.c_str());
    }
    if (all || r->adapter_scale != def.adapter_scale) {
        yyjson_mut_obj_add_real(doc, root, "adapter_scale", r->adapter_scale);
    }
    if (all || !r->adapters.empty()) {
        yyjson_mut_val * arr = yyjson_mut_arr(doc);
        for (const auto & a : r->adapters) {
            yyjson_mut_val * o = yyjson_mut_obj(doc);
            yyjson_mut_obj_add_strcpy(doc, o, "name", a.name.c_str());
            yyjson_mut_obj_add_real(doc, o, "scale", a.scale);
            if (!a.gain_curve.empty()) {
                yyjson_mut_val * ga = yyjson_mut_arr(doc);
                for (float g : a.gain_curve) yyjson_mut_arr_add_real(doc, ga, g);
                yyjson_mut_obj_add_val(doc, o, "gain_curve", ga);
                yyjson_mut_obj_add_str(doc, o, "gain_domain", a.gain_in_steps ? "steps" : "t");
            }
            yyjson_mut_arr_append(arr, o);
        }
        yyjson_mut_obj_add_val(doc, root, "adapters", arr);
    }
    if (all || !r->adapter_sections.empty()) {
        yyjson_mut_val * arr = yyjson_mut_arr(doc);
        for (const auto & sec : r->adapter_sections) {
            yyjson_mut_val * o  = yyjson_mut_obj(doc);
            yyjson_mut_val * wa = yyjson_mut_arr(doc);
            for (float w : sec.weights) yyjson_mut_arr_add_real(doc, wa, w);
            yyjson_mut_obj_add_val(doc, o, "weights", wa);
            yyjson_mut_obj_add_real(doc, o, "size", sec.size);
            yyjson_mut_arr_append(arr, o);
        }
        yyjson_mut_obj_add_val(doc, root, "adapter_sections", arr);
    }
    if (all || r->vae != def.vae) {
        yyjson_mut_obj_add_str(doc, root, "vae", r->vae.c_str());
    }
    if (all || r->pp_vae_reencode != def.pp_vae_reencode) {
        yyjson_mut_obj_add_bool(doc, root, "pp_vae_reencode", r->pp_vae_reencode);
    }
    if (all || r->postprocess_plugin != def.postprocess_plugin) {
        yyjson_mut_obj_add_str(doc, root, "postprocess_plugin", r->postprocess_plugin.c_str());
    }
    if (all || r->get_lrc != def.get_lrc) {
        yyjson_mut_obj_add_bool(doc, root, "get_lrc", r->get_lrc);
    }
    if (all || r->use_ort_vae != def.use_ort_vae) {
        yyjson_mut_obj_add_bool(doc, root, "use_ort_vae", r->use_ort_vae);
    }
    // streaming
    if (all || r->stream_mode != def.stream_mode) {
        yyjson_mut_obj_add_bool(doc, root, "stream_mode", r->stream_mode);
    }
    if (all || r->stream_depth != def.stream_depth) {
        yyjson_mut_obj_add_int(doc, root, "stream_depth", r->stream_depth);
    }
    if (all || r->stream_chunk_dir != def.stream_chunk_dir) {
        yyjson_mut_obj_add_str(doc, root, "stream_chunk_dir", r->stream_chunk_dir.c_str());
    }

    return doc;
}

bool request_write(const AceRequest * r, const char * path) {
    FILE * f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "[Request] ERROR: cannot write %s\n", path);
        return false;
    }

    yyjson_mut_doc * doc = request_build_doc(r, true);
    size_t           len;
    char *           json = yyjson_mut_write(doc, WRITE_FLAGS | YYJSON_WRITE_NEWLINE_AT_END, &len);
    yyjson_mut_doc_free(doc);

    fwrite(json, 1, len, f);
    free(json);
    fclose(f);

    fprintf(stderr, "[Request] Wrote %s\n", path);
    return true;
}

std::string request_to_json(const AceRequest * r, bool sparse) {
    yyjson_mut_doc * doc = request_build_doc(r, sparse);
    size_t           len;
    char *           json = yyjson_mut_write(doc, WRITE_FLAGS, &len);
    yyjson_mut_doc_free(doc);

    std::string result(json, len);
    free(json);
    return result;
}

void request_dump(const AceRequest * r, FILE * f) {
    fprintf(f, "[Request] seed=%lld lm_batch=%d synth_batch=%d\n", (long long) r->seed, r->lm_batch_size,
            r->synth_batch_size);
    fprintf(f, "[Request] caption: %.60s%s\n", r->caption.c_str(), r->caption.size() > 60 ? "..." : "");
    fprintf(f, "[Request] lyrics: %zu bytes\n", r->lyrics.size());
    fprintf(f, "[Request] bpm=%d dur=%.0f key=%s ts=%s lang=%s\n", r->bpm, r->duration, r->keyscale.c_str(),
            r->timesignature.c_str(), r->vocal_language.c_str());
    fprintf(f, "[Request] lm: temp=%.2f cfg=%.1f top_p=%.2f top_k=%d\n", r->lm_temperature, r->lm_cfg_scale,
            r->lm_top_p, r->lm_top_k);
    if (r->lm_cfg_cutoff_ratio != 1.0f) {
        fprintf(f, "[Request] lm_cfg_cutoff_ratio: %.2f\n", r->lm_cfg_cutoff_ratio);
    }
    fprintf(f, "[Request] dit: steps=%d guidance=%.1f shift=%.1f\n", r->inference_steps, r->guidance_scale, r->shift);
    if (r->dcw_scaler > 0.0f || r->dcw_high_scaler > 0.0f) {
        fprintf(f, "[Request] dit: dcw_mode=%s scaler=%.3f high_scaler=%.3f\n", r->dcw_mode.c_str(), r->dcw_scaler,
                r->dcw_high_scaler);
    }
    if (r->audio_cover_strength != 1.0f || r->cover_noise_strength != 0.0f) {
        fprintf(f, "[Request] cover: strength=%.2f noise_strength=%.2f\n", r->audio_cover_strength,
                r->cover_noise_strength);
    }
    if (r->repainting_start != 0.0f || r->repainting_end >= 0.0f) {
        fprintf(f, "[Request] repaint: start=%.1f end=%.1f\n", r->repainting_start, r->repainting_end);
    }
    if (r->latent_shift != 0.0f || r->latent_rescale != 1.0f) {
        fprintf(f, "[Request] latent post: shift=%.3f rescale=%.3f\n", r->latent_shift, r->latent_rescale);
    }
    if (r->lss_strength > 0.0f) {
        fprintf(f, "[Request] LSS: strength=%.2f var_thresh=%.2f dc_remove=%d\n",
                r->lss_strength, r->lss_var_thresh, r->lss_dc_remove ? 1 : 0);
    }
    if (!r->custom_timesteps.empty()) {
        fprintf(f, "[Request] custom_timesteps: %s\n", r->custom_timesteps.c_str());
    }
    fprintf(f, "[Request] task_type: %s\n", r->task_type.c_str());
    if (!r->track.empty()) {
        fprintf(f, "[Request] track: %s\n", r->track.c_str());
    }
    fprintf(f, "[Request] solver: %s (stork_substeps=%d)\n", r->solver.c_str(), r->stork_substeps);
    fprintf(f, "[Request] lm_mode: %s\n", r->lm_mode.c_str());
    fprintf(f, "[Request] output_format: %s\n", r->output_format.c_str());
    if (r->peak_clip != 10) {
        fprintf(f, "[Request] peak_clip: %d\n", r->peak_clip);
    }
    if (r->output_format == "mp3" && r->mp3_bitrate != 128) {
        fprintf(f, "[Request] mp3_bitrate: %d kbps\n", r->mp3_bitrate);
    }
    if (!r->synth_model.empty()) {
        fprintf(f, "[Request] synth_model: %s\n", r->synth_model.c_str());
    }
    if (!r->lm_model.empty()) {
        fprintf(f, "[Request] lm_model: %s\n", r->lm_model.c_str());
    }
    if (!r->adapter.empty()) {
        fprintf(f, "[Request] adapter: %s (scale=%.2f)\n", r->adapter.c_str(), r->adapter_scale);
    }
    if (!r->adapters.empty()) {
        fprintf(f, "[Request] adapter stack (%zu):\n", r->adapters.size());
        for (const auto & a : r->adapters) {
            fprintf(f, "[Request]   - %s (scale=%.2f)\n", a.name.c_str(), a.scale);
        }
    }
    if (!r->vae.empty()) {
        fprintf(f, "[Request] vae: %s\n", r->vae.c_str());
    }
    if (!r->postprocess_plugin.empty()) {
        fprintf(f, "[Request] postprocess_plugin: %s\n", r->postprocess_plugin.c_str());
    }
    fprintf(f, "[Request] audio_codes: %s\n", r->audio_codes.empty() ? "(none)" : "(present)");
}

void request_resolve_seed(AceRequest * r) {
    if (r->seed < 0) {
        std::random_device rd;
        r->seed = (int64_t) rd();
    }
}

void request_resolve_lm_seed(AceRequest * r) {
    if (r->lm_seed < 0) {
        std::random_device rd;
        r->lm_seed = (int64_t) rd();
    }
}
