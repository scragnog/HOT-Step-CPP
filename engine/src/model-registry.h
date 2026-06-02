#pragma once
// model-registry.h: scan directories for GGUF, SafeTensors, and ONNX models.
//
// Reads GGUF headers, SafeTensors config.json, and ONNX filenames to classify
// each model into lm/dit/text-enc/vae buckets.
// Adapter entries are .safetensors files or PEFT directories.
//
// Usage:
//   ModelRegistry reg;
//   registry_scan(&reg, "./models");
//   registry_scan_adapters(&reg, "./adapters");
//   const ModelEntry * dit = registry_find(reg.dit, "acestep-v15-turbo-Q8_0.gguf");

#include "config-json.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#else
#    include <dirent.h>
#    include <sys/stat.h>
#endif

struct ModelEntry {
    std::string name;  // filename (e.g. "acestep-v15-turbo-Q8_0.gguf")
    std::string path;  // full path
};

struct AdapterEntry {
    std::string name;  // filename or directory name (e.g. "singer.safetensors" or "my-adapter")
    std::string path;  // full path (file or PEFT directory)
};

struct ModelRegistry {
    std::vector<ModelEntry>   lm;
    std::vector<ModelEntry>   dit;
    std::vector<ModelEntry>   text_enc;
    std::vector<ModelEntry>   vae;
    std::vector<ModelEntry>   pp_vae;  // PP-VAE (post-processing VAE, auto-detected)
    std::vector<AdapterEntry> adapters;
};

// find an entry by name in a bucket. returns NULL if not found.
static const ModelEntry * registry_find(const std::vector<ModelEntry> & bucket, const char * name) {
    for (const auto & e : bucket) {
        if (e.name == name) {
            return &e;
        }
    }
    return nullptr;
}

// Find the first non-ONNX entry in a bucket, or a specific named entry if it's non-ONNX.
// Use this when an ONNX model cannot serve the purpose (e.g., VAE encode — ONNX VAE files
// are decoder-only and don't contain encoder weights).
static const ModelEntry * registry_find_non_onnx(const std::vector<ModelEntry> & bucket, const char * name = nullptr) {
    auto is_onnx = [](const std::string & n) {
        return n.size() >= 5 && n.substr(n.size() - 5) == ".onnx";
    };
    if (name && name[0]) {
        // Specific name requested — return it only if it's not ONNX
        for (const auto & e : bucket) {
            if (e.name == name && !is_onnx(e.name)) {
                return &e;
            }
        }
        return nullptr;
    }
    // No specific name — return first non-ONNX entry
    for (const auto & e : bucket) {
        if (!is_onnx(e.name)) {
            return &e;
        }
    }
    return nullptr;
}

// find an adapter entry by name. returns NULL if not found.
static const AdapterEntry * registry_find_adapter(const ModelRegistry & reg, const char * name) {
    for (const auto & e : reg.adapters) {
        if (e.name == name) {
            return &e;
        }
    }
    return nullptr;
}

// classify a GGUF file by reading its header.
// returns: "lm", "dit", "text-enc", "vae", or "" if unrecognized.
static std::string registry_classify_gguf(const char * path) {
    struct gguf_init_params params = { true, nullptr };
    struct gguf_context *   ctx    = gguf_init_from_file(path, params);
    if (!ctx) {
        return "";
    }

    std::string arch;
    int64_t     idx = gguf_find_key(ctx, "general.architecture");
    if (idx >= 0) {
        arch = gguf_get_val_str(ctx, idx);
    }
    gguf_free(ctx);

    // map GGUF architecture string to bucket name
    if (arch == "acestep-lm") {
        return "LM";
    }
    if (arch == "acestep-dit") {
        return "DiT";
    }
    if (arch == "acestep-text-enc") {
        return "Text-Enc";
    }
    if (arch == "acestep-vae") {
        return "VAE";
    }
    if (arch == "pp-vae") {
        return "PP-VAE";
    }
    return "";
}

// check if a string ends with a suffix
static bool str_ends_with(const std::string & s, const char * suffix) {
    size_t slen = strlen(suffix);
    return s.size() >= slen && s.compare(s.size() - slen, slen, suffix) == 0;
}

#ifdef _WIN32

// scan a directory for files matching a pattern (Windows)
static void registry_list_dir(const char * dir, std::vector<std::string> * names) {
    std::string      pattern = std::string(dir) + "\\*";
    WIN32_FIND_DATAA fd;
    HANDLE           h = FindFirstFileA(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        return;
    }
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            names->push_back(fd.cFileName);
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
}

// list subdirectories (Windows)
static void registry_list_subdirs(const char * dir, std::vector<std::string> * names) {
    std::string      pattern = std::string(dir) + "\\*";
    WIN32_FIND_DATAA fd;
    HANDLE           h = FindFirstFileA(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        return;
    }
    do {
        if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && strcmp(fd.cFileName, ".") != 0 &&
            strcmp(fd.cFileName, "..") != 0) {
            names->push_back(fd.cFileName);
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
}

static bool registry_is_file(const char * path) {
    DWORD attr = GetFileAttributesA(path);
    return attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
}

#else

// scan a directory for files (POSIX)
static void registry_list_dir(const char * dir, std::vector<std::string> * names) {
    DIR * d = opendir(dir);
    if (!d) {
        return;
    }
    struct dirent * entry;
    while ((entry = readdir(d)) != nullptr) {
        // skip directories
        std::string full = std::string(dir) + "/" + entry->d_name;
        struct stat sb;
        if (stat(full.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)) {
            names->push_back(entry->d_name);
        }
    }
    closedir(d);
}

// list subdirectories (POSIX)
static void registry_list_subdirs(const char * dir, std::vector<std::string> * names) {
    DIR * d = opendir(dir);
    if (!d) {
        return;
    }
    struct dirent * entry;
    while ((entry = readdir(d)) != nullptr) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        std::string full = std::string(dir) + "/" + entry->d_name;
        struct stat sb;
        if (stat(full.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
            names->push_back(entry->d_name);
        }
    }
    closedir(d);
}

static bool registry_is_file(const char * path) {
    struct stat sb;
    return stat(path, &sb) == 0 && S_ISREG(sb.st_mode);
}

#endif

// path separator
#ifdef _WIN32
#    define REGISTRY_SEP "\\"
#else
#    define REGISTRY_SEP "/"
#endif

// scan a directory for .gguf files, classify each by architecture.
// returns true if at least one model was found.
static bool registry_scan(ModelRegistry * reg, const char * models_dir) {
    std::vector<std::string> files;
    registry_list_dir(models_dir, &files);
    std::sort(files.begin(), files.end());

    int count = 0;
    for (const auto & fname : files) {
        if (!str_ends_with(fname, ".gguf")) {
            continue;
        }

        std::string full = std::string(models_dir) + REGISTRY_SEP + fname;
        std::string type = registry_classify_gguf(full.c_str());
        if (type.empty()) {
            fprintf(stderr, "[Registry] WARNING: skipping %s (unknown architecture)\n", fname.c_str());
            continue;
        }

        ModelEntry entry = { fname, full };
        if (type == "LM") {
            reg->lm.push_back(entry);
        } else if (type == "DiT") {
            reg->dit.push_back(entry);
        } else if (type == "Text-Enc") {
            reg->text_enc.push_back(entry);
        } else if (type == "VAE") {
            reg->vae.push_back(entry);
        } else if (type == "PP-VAE") {
            reg->pp_vae.push_back(entry);
        }

        fprintf(stderr, "[Registry] %s -> %s\n", fname.c_str(), type.c_str());
        count++;
    }

    // Scan for safetensors VAE files (classified by filename prefix)
    for (const auto & fname : files) {
        if (!str_ends_with(fname, ".safetensors")) {
            continue;
        }

        // Classify by lowercase filename prefix
        std::string lower = fname;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        std::string full = std::string(models_dir) + REGISTRY_SEP + fname;
        ModelEntry  entry = { fname, full };

        if (lower.find("pp-vae") == 0 || lower.find("pp_vae") == 0) {
            reg->pp_vae.push_back(entry);
            fprintf(stderr, "[Registry] %s -> PP-VAE (safetensors)\n", fname.c_str());
            count++;
        } else if (lower.find("vae") == 0 || lower.find("scragvae") == 0) {
            reg->vae.push_back(entry);
            fprintf(stderr, "[Registry] %s -> VAE (safetensors)\n", fname.c_str());
            count++;
        }
    }

    // Scan for ONNX files (pre-exported models for TRT acceleration)
    // Classified by filename prefix: dit_* -> DiT, vae_* -> VAE, etc.
    for (const auto & fname : files) {
        if (!str_ends_with(fname, ".onnx")) {
            continue;
        }

        std::string lower = fname;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        std::string full = std::string(models_dir) + REGISTRY_SEP + fname;
        ModelEntry  entry = { fname, full };

        if (lower.find("dit_") == 0 || lower.find("dit-") == 0) {
            reg->dit.push_back(entry);
            fprintf(stderr, "[Registry] %s -> DiT (ONNX)\n", fname.c_str());
            count++;
        } else if (lower.find("vae_") == 0 || lower.find("vae-") == 0 ||
                   lower.find("scragvae_") == 0 || lower.find("scragvae-") == 0) {
            reg->vae.push_back(entry);
            fprintf(stderr, "[Registry] %s -> VAE (ONNX)\n", fname.c_str());
            count++;
        } else if (lower.find("lm_") == 0 || lower.find("lm-") == 0) {
            reg->lm.push_back(entry);
            fprintf(stderr, "[Registry] %s -> LM (ONNX)\n", fname.c_str());
            count++;
        } else if (lower.find("text_enc") == 0 || lower.find("text-enc") == 0) {
            reg->text_enc.push_back(entry);
            fprintf(stderr, "[Registry] %s -> Text-Enc (ONNX)\n", fname.c_str());
            count++;
        } else {
            fprintf(stderr, "[Registry] WARNING: skipping %s (unrecognized ONNX prefix)\n", fname.c_str());
        }
    }

    // Scan subdirectories for safetensors model checkpoints (HuggingFace format).
    // Each subdirectory should contain model.safetensors (or sharded index) + config.json.
    // Classification uses config.json content, not directory name.
    std::vector<std::string> subdirs;
    registry_list_subdirs(models_dir, &subdirs);
    std::sort(subdirs.begin(), subdirs.end());

    for (const auto & dname : subdirs) {
        std::string dir_path = std::string(models_dir) + REGISTRY_SEP + dname;

        // Must have config.json
        std::string cfg_path = dir_path + REGISTRY_SEP + "config.json";
        if (!registry_is_file(cfg_path.c_str())) {
            continue;
        }

        // Must have model.safetensors or model.safetensors.index.json (sharded)
        std::string st_path = dir_path + REGISTRY_SEP + "model.safetensors";
        std::string st_index = dir_path + REGISTRY_SEP + "model.safetensors.index.json";
        if (!registry_is_file(st_path.c_str()) && !registry_is_file(st_index.c_str())) {
            continue;
        }

        // Classify by config.json content
        std::string type = config_json_classify(cfg_path.c_str());
        if (type.empty()) {
            fprintf(stderr, "[Registry] WARNING: skipping %s/ (unrecognized config.json)\n", dname.c_str());
            continue;
        }

        // ModelEntry.path is the directory path for safetensors models.
        // The loader will find model.safetensors + sidecar files inside.
        ModelEntry entry = { dname, dir_path };
        if (type == "LM") {
            reg->lm.push_back(entry);
        } else if (type == "DiT") {
            reg->dit.push_back(entry);
        } else if (type == "Text-Enc") {
            reg->text_enc.push_back(entry);
        } else if (type == "VAE") {
            reg->vae.push_back(entry);
        }

        bool is_sharded = registry_is_file(st_index.c_str());
        fprintf(stderr, "[Registry] %s/ -> %s (safetensors%s)\n", dname.c_str(), type.c_str(),
                is_sharded ? ", sharded" : "");
        count++;
    }

    // Scan subdirectories for ONNX model directories (TRT acceleration).
    // Each subdirectory should contain at least one .onnx file.
    // Classification by .onnx filename prefix (lm_* -> LM, dit_* -> DiT, etc.)
    // or by directory name prefix if ONNX filename is ambiguous.
    for (const auto & dname : subdirs) {
        std::string dir_path = std::string(models_dir) + REGISTRY_SEP + dname;

        // Skip directories already registered (safetensors path above)
        // by checking for model.safetensors — if it has that, it was already handled
        std::string st_check = dir_path + REGISTRY_SEP + "model.safetensors";
        std::string st_idx   = dir_path + REGISTRY_SEP + "model.safetensors.index.json";
        if (registry_is_file(st_check.c_str()) || registry_is_file(st_idx.c_str())) {
            continue;
        }

        // Look for .onnx files in this subdirectory
        std::vector<std::string> sub_files;
        registry_list_dir(dir_path.c_str(), &sub_files);
        std::string onnx_name;
        for (const auto & f : sub_files) {
            if (str_ends_with(f, ".onnx")) {
                onnx_name = f;
                break;
            }
        }
        if (onnx_name.empty()) {
            continue;
        }

        // Classify: try ONNX filename first, then directory name
        std::string lower_onnx = onnx_name;
        std::transform(lower_onnx.begin(), lower_onnx.end(), lower_onnx.begin(), ::tolower);
        std::string lower_dir = dname;
        std::transform(lower_dir.begin(), lower_dir.end(), lower_dir.begin(), ::tolower);

        std::string type;
        // Check ONNX filename prefix
        if (lower_onnx.find("lm_") == 0 || lower_onnx.find("lm-") == 0) {
            type = "LM";
        } else if (lower_onnx.find("dit_") == 0 || lower_onnx.find("dit-") == 0) {
            type = "DiT";
        } else if (lower_onnx.find("vae_") == 0 || lower_onnx.find("vae-") == 0) {
            type = "VAE";
        } else if (lower_onnx.find("text_enc") == 0 || lower_onnx.find("text-enc") == 0) {
            type = "Text-Enc";
        }
        // Fallback: check directory name prefix
        if (type.empty()) {
            if (lower_dir.find("lm") == 0) {
                type = "LM";
            } else if (lower_dir.find("dit") == 0) {
                type = "DiT";
            } else if (lower_dir.find("vae") == 0) {
                type = "VAE";
            } else if (lower_dir.find("text") == 0 || lower_dir.find("enc") == 0) {
                type = "Text-Enc";
            }
        }

        if (type.empty()) {
            fprintf(stderr, "[Registry] WARNING: skipping %s/ (unrecognized ONNX directory)\n", dname.c_str());
            continue;
        }

        // ModelEntry.path is the directory path. The TRT runtime finds the .onnx inside.
        ModelEntry entry = { dname, dir_path };
        if (type == "LM") {
            reg->lm.push_back(entry);
        } else if (type == "DiT") {
            reg->dit.push_back(entry);
        } else if (type == "Text-Enc") {
            reg->text_enc.push_back(entry);
        } else if (type == "VAE") {
            reg->vae.push_back(entry);
        }

        fprintf(stderr, "[Registry] %s/ -> %s (ONNX: %s)\n", dname.c_str(), type.c_str(), onnx_name.c_str());
        count++;
    }

    return count > 0;
}

// scan a directory for adapters.
// - .safetensors files: ComfyUI single-file format (alpha baked in)
// - subdirectories containing adapter_model.safetensors: PEFT format
// returns true if at least one adapter was found.
static bool registry_scan_adapters(ModelRegistry * reg, const char * adapters_dir) {
    int count = 0;

    // single .safetensors files
    std::vector<std::string> files;
    registry_list_dir(adapters_dir, &files);
    std::sort(files.begin(), files.end());
    for (const auto & fname : files) {
        if (!str_ends_with(fname, ".safetensors")) {
            continue;
        }
        std::string full = std::string(adapters_dir) + REGISTRY_SEP + fname;
        reg->adapters.push_back({ fname, full });
        fprintf(stderr, "[Registry] Adapter: %s (ComfyUI)\n", fname.c_str());
        count++;
    }

    // PEFT directories (contain adapter_model.safetensors)
    std::vector<std::string> subdirs;
    registry_list_subdirs(adapters_dir, &subdirs);
    std::sort(subdirs.begin(), subdirs.end());
    for (const auto & dname : subdirs) {
        std::string adapter =
            std::string(adapters_dir) + REGISTRY_SEP + dname + REGISTRY_SEP + "adapter_model.safetensors";
        if (registry_is_file(adapter.c_str())) {
            std::string full = std::string(adapters_dir) + REGISTRY_SEP + dname;
            reg->adapters.push_back({ dname, full });
            fprintf(stderr, "[Registry] Adapter: %s (PEFT)\n", dname.c_str());
            count++;
        }
    }

    return count > 0;
}
