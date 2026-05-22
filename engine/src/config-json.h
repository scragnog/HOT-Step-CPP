#pragma once
// config-json.h: read HuggingFace config.json sidecar for model configuration
//
// Replaces GGUF KV metadata when loading from safetensors.
// Uses yyjson (already linked in the project).

#include "yyjson.h"

#include <cstdio>
#include <cstring>
#include <string>

// ─── DiT config from config.json ─────────────────────────────────────

struct DiTGGMLConfig;  // forward decl from dit.h

// Read DiT config values from config.json.
// Maps HuggingFace config keys to DiTGGMLConfig fields.
// Returns true on success.
static bool config_json_load_dit(DiTGGMLConfig * cfg, const char * json_path) {
    yyjson_doc * doc = yyjson_read_file(json_path, 0, NULL, NULL);
    if (!doc) {
        fprintf(stderr, "[Config] Cannot read %s\n", json_path);
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!root || !yyjson_is_obj(root)) {
        yyjson_doc_free(doc);
        return false;
    }

    yyjson_val * v;

    // Direct mappings from HuggingFace AceStep config.json
    if ((v = yyjson_obj_get(root, "num_hidden_layers")) && yyjson_is_int(v))
        cfg->n_layers = (int) yyjson_get_int(v);
    // Also accept num_audio_decoder_hidden_layers (same field in some configs)
    if (!cfg->n_layers) {
        if ((v = yyjson_obj_get(root, "num_audio_decoder_hidden_layers")) && yyjson_is_int(v))
            cfg->n_layers = (int) yyjson_get_int(v);
    }
    if ((v = yyjson_obj_get(root, "hidden_size")) && yyjson_is_int(v))
        cfg->hidden_size = (int) yyjson_get_int(v);
    if ((v = yyjson_obj_get(root, "intermediate_size")) && yyjson_is_int(v))
        cfg->intermediate_size = (int) yyjson_get_int(v);
    if ((v = yyjson_obj_get(root, "num_attention_heads")) && yyjson_is_int(v))
        cfg->n_heads = (int) yyjson_get_int(v);
    if ((v = yyjson_obj_get(root, "num_key_value_heads")) && yyjson_is_int(v))
        cfg->n_kv_heads = (int) yyjson_get_int(v);
    if ((v = yyjson_obj_get(root, "head_dim")) && yyjson_is_int(v))
        cfg->head_dim = (int) yyjson_get_int(v);
    if ((v = yyjson_obj_get(root, "in_channels")) && yyjson_is_int(v))
        cfg->in_channels = (int) yyjson_get_int(v);
    if ((v = yyjson_obj_get(root, "audio_acoustic_hidden_dim")) && yyjson_is_int(v))
        cfg->out_channels = (int) yyjson_get_int(v);
    if ((v = yyjson_obj_get(root, "patch_size")) && yyjson_is_int(v))
        cfg->patch_size = (int) yyjson_get_int(v);
    if ((v = yyjson_obj_get(root, "sliding_window")) && yyjson_is_int(v))
        cfg->sliding_window = (int) yyjson_get_int(v);

    // Float fields
    if ((v = yyjson_obj_get(root, "rope_theta")) && yyjson_is_num(v))
        cfg->rope_theta = (float) yyjson_get_num(v);
    if ((v = yyjson_obj_get(root, "rms_norm_eps")) && yyjson_is_num(v))
        cfg->rms_norm_eps = (float) yyjson_get_num(v);

    yyjson_doc_free(doc);
    return true;
}

// ─── Qwen3 config from config.json ───────────────────────────────────

struct Qwen3Config;  // forward decl from qwen3-enc.h

static bool config_json_load_qwen3(Qwen3Config * cfg, const char * json_path) {
    yyjson_doc * doc = yyjson_read_file(json_path, 0, NULL, NULL);
    if (!doc) {
        fprintf(stderr, "[Config] Cannot read %s\n", json_path);
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!root || !yyjson_is_obj(root)) {
        yyjson_doc_free(doc);
        return false;
    }

    yyjson_val * v;

    if ((v = yyjson_obj_get(root, "hidden_size")) && yyjson_is_int(v))
        cfg->hidden_size = (int) yyjson_get_int(v);
    if ((v = yyjson_obj_get(root, "intermediate_size")) && yyjson_is_int(v))
        cfg->intermediate_size = (int) yyjson_get_int(v);
    if ((v = yyjson_obj_get(root, "num_attention_heads")) && yyjson_is_int(v))
        cfg->n_heads = (int) yyjson_get_int(v);
    if ((v = yyjson_obj_get(root, "num_key_value_heads")) && yyjson_is_int(v))
        cfg->n_kv_heads = (int) yyjson_get_int(v);
    if ((v = yyjson_obj_get(root, "head_dim")) && yyjson_is_int(v))
        cfg->head_dim = (int) yyjson_get_int(v);
    if ((v = yyjson_obj_get(root, "num_hidden_layers")) && yyjson_is_int(v))
        cfg->n_layers = (int) yyjson_get_int(v);
    if ((v = yyjson_obj_get(root, "rope_theta")) && yyjson_is_num(v))
        cfg->rope_theta = (float) yyjson_get_num(v);
    if ((v = yyjson_obj_get(root, "rms_norm_eps")) && yyjson_is_num(v))
        cfg->rms_norm_eps = (float) yyjson_get_num(v);

    yyjson_doc_free(doc);
    return true;
}

// ─── Model classification from config.json ───────────────────────────

// Classify a model directory by reading its config.json.
// Returns: "DiT", "Text-Enc", "LM", or "" if unrecognized.
static std::string config_json_classify(const char * json_path) {
    yyjson_doc * doc = yyjson_read_file(json_path, 0, NULL, NULL);
    if (!doc) return "";
    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!root || !yyjson_is_obj(root)) {
        yyjson_doc_free(doc);
        return "";
    }

    std::string result;

    // Check model_type
    yyjson_val * mt = yyjson_obj_get(root, "model_type");
    std::string model_type;
    if (mt && yyjson_is_str(mt)) {
        model_type = yyjson_get_str(mt);
    }

    // Check architectures array
    std::string arch;
    yyjson_val * archs = yyjson_obj_get(root, "architectures");
    if (archs && yyjson_is_arr(archs)) {
        yyjson_val * first = yyjson_arr_get_first(archs);
        if (first && yyjson_is_str(first)) {
            arch = yyjson_get_str(first);
        }
    }

    if (model_type == "acestep") {
        result = "DiT";
    } else if (model_type == "qwen3") {
        if (arch == "Qwen3ForCausalLM") {
            result = "LM";
        } else if (arch == "Qwen3Model") {
            // Qwen3Model is used by both Text Encoder AND some LM variants
            // (e.g. 1.7B LM). Discriminate by vocab_size:
            //   Text Encoder: 151669 (standard Qwen3 vocab)
            //   LM:           217204 (extended with music tokens)
            int vocab_size = 0;
            yyjson_val * vs = yyjson_obj_get(root, "vocab_size");
            if (vs && yyjson_is_int(vs)) {
                vocab_size = (int) yyjson_get_int(vs);
            }
            result = (vocab_size > 200000) ? "LM" : "Text-Enc";
        }
    }

    yyjson_doc_free(doc);
    return result;
}

// ─── Extra DiT config fields for ModelStore ──────────────────────────

// Read is_turbo flag from config.json (DiT models)
static bool config_json_get_is_turbo(const char * json_path) {
    yyjson_doc * doc = yyjson_read_file(json_path, 0, NULL, NULL);
    if (!doc) return false;
    yyjson_val * root = yyjson_doc_get_root(doc);
    bool result = false;
    if (root && yyjson_is_obj(root)) {
        yyjson_val * v = yyjson_obj_get(root, "is_turbo");
        if (v && yyjson_is_bool(v)) {
            result = yyjson_get_bool(v);
        }
    }
    yyjson_doc_free(doc);
    return result;
}

// Check if model_version contains "merge" (base/turbo blend)
static bool config_json_get_is_merge(const char * json_path) {
    yyjson_doc * doc = yyjson_read_file(json_path, 0, NULL, NULL);
    if (!doc) return false;
    yyjson_val * root = yyjson_doc_get_root(doc);
    bool result = false;
    if (root && yyjson_is_obj(root)) {
        yyjson_val * v = yyjson_obj_get(root, "model_version");
        if (v && yyjson_is_str(v)) {
            const char * ver = yyjson_get_str(v);
            result = (strstr(ver, "merge") != nullptr);
        }
    }
    yyjson_doc_free(doc);
    return result;
}

// ─── Derive directory path from a file path or dir path ──────────────

// If path ends with .safetensors or .gguf, return its parent directory.
// If path is already a directory, return it as-is.
// Used to locate sidecar files (config.json, silence_latent.pt, etc.)
static std::string ws_dir_from_path(const std::string & path) {
    // Check if it looks like a file path (has a known extension)
    size_t dot = path.rfind('.');
    if (dot != std::string::npos) {
        std::string ext = path.substr(dot);
        // Lowercase compare
        for (auto & c : ext) c = (char) tolower(c);
        if (ext == ".safetensors" || ext == ".gguf" || ext == ".json" || ext == ".pt") {
            size_t sep = path.find_last_of("/\\");
            if (sep != std::string::npos) {
                return path.substr(0, sep);
            }
        }
    }
    // Already a directory path or no recognizable extension
    return path;
}
