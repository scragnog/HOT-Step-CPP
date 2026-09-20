#pragma once

// Safe core of the native -> AI Toolkit dataset bridge.  It accepts only
// caller-verified native records; it does not parse or bless legacy manifests.

#include "yue2-aitk-dataset.h"
#include "yue2-aitk-sha256.h"
#include "yyjson.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <cstdlib>
#include <string>
#include <unordered_set>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <unistd.h>
#endif

namespace yue2_aitk {

struct VerifiedNativeItem {
    std::string id;
    std::filesystem::path latent_file;   // verified native frame-major F32 clip
    std::filesystem::path semantic_file; // verified little-endian int32 raw IDs
    size_t frames = 0;
    PromptInput prompt;                   // produced by installed tokenizer API
    std::string prompt_style;
    std::string prompt_style_nocap;       // the trigger alone; what --caption-dropout trains on
    std::string prompt_lyrics;
    bool instrumental = false;
};

struct ModelProvenance {
    std::string role;
    std::filesystem::path path;
};

struct PrepareRequest {
    std::filesystem::path output_dir;          // must not exist
    std::filesystem::path source_manifest;     // native manifest used by verifier
    std::filesystem::path base_checkpoint;     // exact checkpoint used by producer
    std::string verified_contract;             // explicit caller proof, nonempty
    std::string trigger;                       // exact trigger included in each prepared style
    std::vector<ModelProvenance> models;       // tokenizer/VAE/model files actually used
    std::vector<VerifiedNativeItem> items;
};

namespace prepare_detail {
inline bool prep_fail(std::string * e, const char * msg) { if (e) *e = msg; return false; }
inline bool no_nul(const std::string & value) { return value.find('\0') == std::string::npos; }

inline bool read_file(const std::filesystem::path & path, std::vector<uint8_t> * out, std::string * e) {
    std::error_code ec;
    const auto n = std::filesystem::file_size(path, ec);
    if (ec || n > 512u * 1024u * 1024u) return prep_fail(e, "native payload missing or exceeds 512 MiB");
    std::ifstream f(path, std::ios::binary);
    if (!f) return prep_fail(e, "native payload cannot be opened");
    out->resize(static_cast<size_t>(n));
    if (!out->empty() && !f.read(reinterpret_cast<char *>(out->data()), static_cast<std::streamsize>(out->size())))
        return prep_fail(e, "native payload read failed");
    return true;
}

inline bool token_ids(const PromptInput & p, size_t frames, std::string * e) {
    if (p.retained_prefix_ids.empty() || p.dropped_prefix_ids.empty()) return prep_fail(e, "verified prefixes cannot be empty");
    for (const auto & ids : {std::cref(p.retained_prefix_ids), std::cref(p.dropped_prefix_ids), std::cref(p.abc_ids),
                             std::cref(p.retained_nocap_prefix_ids), std::cref(p.dropped_nocap_prefix_ids)})
        for (int32_t id : ids.get()) if (id < 0 || id >= kVocabSize) return prep_fail(e, "prompt token outside vocabulary");
    if (p.retained_nocap_prefix_ids.empty() != p.dropped_nocap_prefix_ids.empty()) return prep_fail(e, "trigger-only prefixes must come as a pair");
    const size_t crop = std::min(frames, size_t(1500));
    if (p.retained_prefix_ids.size() + p.abc_ids.size() + 2 * crop + 5 > 24576 ||
        p.dropped_prefix_ids.size() + p.abc_ids.size() + 2 * crop + 5 > 24576)
        return prep_fail(e, "prompt context exceeds YuE2 sequence limit");
    return true;
}

inline bool finite_latents(const std::vector<uint8_t> & bytes, std::string * e) {
    for (size_t i = 0; i < bytes.size() / sizeof(float); ++i) {
        float value; std::memcpy(&value, bytes.data() + i * sizeof(value), sizeof(value));
        if (!std::isfinite(value)) return prep_fail(e, "native latent contains non-finite value");
    }
    return true;
}

inline bool add_ids(yyjson_mut_doc * doc, yyjson_mut_val * obj, const char * key, const std::vector<int32_t> & ids) {
    yyjson_mut_val * arr = yyjson_mut_arr(doc);
    if (!arr) return false;
    for (int32_t id : ids) if (!yyjson_mut_arr_add_int(doc, arr, id)) return false;
    return yyjson_mut_obj_add_val(doc, obj, key, arr);
}
inline bool add_f32s(yyjson_mut_doc * doc, yyjson_mut_val * obj, const char * key, const std::vector<float> & values) {
    yyjson_mut_val * arr = yyjson_mut_arr(doc); if (!arr) return false;
    for (float value : values) if (!std::isfinite(value) || !yyjson_mut_arr_add_real(doc, arr, value)) return false;
    return yyjson_mut_obj_add_val(doc, obj, key, arr);
}
inline bool add_i64s(yyjson_mut_doc * doc, yyjson_mut_val * obj, const char * key, const std::vector<int64_t> & values) {
    yyjson_mut_val * arr = yyjson_mut_arr(doc); if (!arr) return false;
    for (int64_t value : values) if (value < 0 || !yyjson_mut_arr_add_sint(doc, arr, value)) return false;
    return yyjson_mut_obj_add_val(doc, obj, key, arr);
}
inline bool add_cursor(yyjson_mut_doc * doc, yyjson_mut_val * obj, const CursorMetadata & cursor) {
    if (cursor.present && cursor.instrumental) {
        yyjson_mut_val * root = yyjson_mut_obj(doc);
        return root && yyjson_mut_obj_add_bool(doc, root, "enabled", false) &&
               yyjson_mut_obj_add_bool(doc, root, "instrumental", true) &&
               yyjson_mut_obj_add_val(doc, obj, "cursor", root);
    }
    if (!cursor.present || cursor.lyric_codepoints <= 0 || cursor.words5.empty() ||
        cursor.full_lyric_token_end_codepoints.empty() || cursor.off_lyric_token_end_codepoints.empty() ||
        !cursor.full.bound || !cursor.off.bound || cursor.lyrics_sha256.size() != 64 ||
        cursor.tokenizer_sha256.size() != 64) return false;
    yyjson_mut_val * root = yyjson_mut_obj(doc), * full = yyjson_mut_obj(doc), * off = yyjson_mut_obj(doc);
    if (!root || !full || !off || !yyjson_mut_obj_add_strcpy(doc, root, "lyrics_sha256", cursor.lyrics_sha256.c_str()) ||
        !yyjson_mut_obj_add_strcpy(doc, root, "tokenizer_sha256", cursor.tokenizer_sha256.c_str()) ||
        !yyjson_mut_obj_add_bool(doc, root, "enabled", cursor.enabled) ||
        !yyjson_mut_obj_add_bool(doc, root, "instrumental", cursor.instrumental) ||
        !yyjson_mut_obj_add_sint(doc, root, "lyric_codepoints", cursor.lyric_codepoints) ||
        !add_f32s(doc, root, "words5", cursor.words5) ||
        !yyjson_mut_obj_add_sint(doc, full, "head_tokens", cursor.full_head_tokens) ||
        !add_i64s(doc, full, "token_end_codepoints", cursor.full_lyric_token_end_codepoints) ||
        !yyjson_mut_obj_add_sint(doc, off, "head_tokens", cursor.off_head_tokens) ||
        !add_i64s(doc, off, "token_end_codepoints", cursor.off_lyric_token_end_codepoints) ||
        !yyjson_mut_obj_add_val(doc, root, "full", full) || !yyjson_mut_obj_add_val(doc, root, "off", off)) return false;
    if (cursor.has_nocap()) {
        yyjson_mut_val * full_nocap = yyjson_mut_obj(doc), * off_nocap = yyjson_mut_obj(doc);
        if (!full_nocap || !off_nocap ||
            !yyjson_mut_obj_add_sint(doc, full_nocap, "head_tokens", cursor.full_nocap_head_tokens) ||
            !add_i64s(doc, full_nocap, "token_end_codepoints", cursor.full_nocap_lyric_token_end_codepoints) ||
            !yyjson_mut_obj_add_sint(doc, off_nocap, "head_tokens", cursor.off_nocap_head_tokens) ||
            !add_i64s(doc, off_nocap, "token_end_codepoints", cursor.off_nocap_lyric_token_end_codepoints) ||
            !yyjson_mut_obj_add_val(doc, root, "full_nocap", full_nocap) || !yyjson_mut_obj_add_val(doc, root, "off_nocap", off_nocap)) return false;
    }
    if (!yyjson_mut_obj_add_val(doc, obj, "cursor", root)) return false;
    return true;
}

inline std::string fnv_hex(const std::vector<uint8_t> & bytes) {
    uint64_t h = 14695981039346656037ull;
    for (uint8_t byte : bytes) { h ^= byte; h *= 1099511628211ull; }
    std::ostringstream out; out << std::hex << std::setfill('0') << std::setw(16) << h;
    return out.str();
}

inline bool publish_no_replace(const std::filesystem::path & temp, const std::filesystem::path & final_path, std::string * e) {
#ifdef _WIN32
    if (!MoveFileExW(temp.wstring().c_str(), final_path.wstring().c_str(), MOVEFILE_WRITE_THROUGH))
        return prep_fail(e, "cannot publish dataset manifest without replacement");
    return true;
#else
    if (::link(temp.c_str(), final_path.c_str()) != 0) return prep_fail(e, "cannot publish dataset manifest without replacement");
    if (::unlink(temp.c_str()) != 0) return prep_fail(e, "cannot remove temporary dataset manifest");
    return true;
#endif
}
} // namespace prepare_detail

inline bool prepare_dataset(const PrepareRequest & request, std::string * error = nullptr) {
    using namespace prepare_detail;
    if (request.output_dir.empty() || request.source_manifest.empty() || request.base_checkpoint.empty() ||
        request.verified_contract.empty() || !no_nul(request.verified_contract) || request.models.empty() || request.items.empty() || request.items.size() > 10000)
        return prep_fail(error, "dataset preparation request is incomplete");
    const uint16_t endian_probe = 1;
    if (*reinterpret_cast<const uint8_t *>(&endian_probe) != 1) return prep_fail(error, "native preparation requires little-endian host");
    std::error_code ec;
    if (std::filesystem::exists(request.output_dir, ec) || ec ||
        !std::filesystem::create_directory(request.output_dir, ec) || ec)
        return prep_fail(error, "output directory must be fresh and creatable");
    sha256::digest source_hash, base_hash;
    std::vector<uint8_t> source_bytes;
    if (!read_file(request.source_manifest, &source_bytes, error)) return false;
    source_hash = sha256::bytes(source_bytes.data(), source_bytes.size());
    if (!sha256::file(request.base_checkpoint, base_hash, error)) return false;
    std::vector<std::pair<std::string, sha256::digest>> model_hashes;
    for (const auto & model : request.models) {
        if (model.role.empty() || !no_nul(model.role) || model.path.empty()) return prep_fail(error, "model provenance role/path is invalid");
        sha256::digest hash;
        if (!sha256::file(model.path, hash, error)) return false;
        model_hashes.emplace_back(model.role, hash);
    }
    std::ofstream source_copy(request.output_dir / "source-manifest.json", std::ios::binary);
    if (!source_copy || !source_copy.write(reinterpret_cast<const char *>(source_bytes.data()), static_cast<std::streamsize>(source_bytes.size())) || !source_copy.flush())
        return prep_fail(error, "cannot copy verified source manifest");
    source_copy.close();
    if (source_copy.fail()) return prep_fail(error, "cannot close copied source manifest");
    sha256::digest copied_source_hash;
    if (!sha256::file(request.output_dir / "source-manifest.json", copied_source_hash, error) || copied_source_hash.hex() != source_hash.hex())
        return prep_fail(error, "copied source manifest hash mismatch");

    yyjson_mut_doc * doc = yyjson_mut_doc_new(nullptr);
    if (!doc) return prep_fail(error, "cannot allocate manifest document");
    struct Guard { yyjson_mut_doc * d; ~Guard() { yyjson_mut_doc_free(d); } } guard{doc};
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_val * items = yyjson_mut_arr(doc);
    if (!root || !items ||
        !yyjson_mut_obj_add_int(doc, root, "schema_version", 1) ||
        !yyjson_mut_obj_add_strcpy(doc, root, "recipe_version", "aitk-yue2-2026-09-16") ||
        !yyjson_mut_obj_add_strcpy(doc, root, "cot", "full") ||
        !yyjson_mut_obj_add_strcpy(doc, root, "base_sha256", base_hash.hex().c_str()) ||
        !yyjson_mut_obj_add_strcpy(doc, root, "source_manifest_sha256", source_hash.hex().c_str()) ||
        !yyjson_mut_obj_add_strcpy(doc, root, "verified_contract", request.verified_contract.c_str()))
        return prep_fail(error, "cannot construct manifest metadata");
    if (!request.trigger.empty() && (!prepare_detail::no_nul(request.trigger) ||
        !yyjson_mut_obj_add_strcpy(doc, root, "trigger", request.trigger.c_str()) ||
        !yyjson_mut_obj_add_strcpy(doc, root, "style_template", "upstream")))
        return prep_fail(error, "cannot record trigger metadata");
    yyjson_mut_val * provenance = yyjson_mut_obj(doc);
    yyjson_mut_val * model_array = yyjson_mut_arr(doc);
    if (!provenance || !model_array || !yyjson_mut_obj_add_strcpy(doc, provenance, "source_manifest_file", "source-manifest.json") ||
        !yyjson_mut_obj_add_strcpy(doc, provenance, "source_manifest_sha256", source_hash.hex().c_str()) ||
        !yyjson_mut_obj_add_strcpy(doc, provenance, "base_checkpoint_file", request.base_checkpoint.u8string().c_str()) ||
        !yyjson_mut_obj_add_strcpy(doc, provenance, "base_checkpoint_sha256", base_hash.hex().c_str()))
        return prep_fail(error, "cannot construct provenance metadata");
    for (const auto & model : model_hashes) {
        yyjson_mut_val * model_obj = yyjson_mut_obj(doc);
        const auto & model_request = request.models[&model - &model_hashes[0]];
        if (!model_obj || !yyjson_mut_obj_add_strcpy(doc, model_obj, "role", model.first.c_str()) ||
            !yyjson_mut_obj_add_strcpy(doc, model_obj, "file", model_request.path.u8string().c_str()) ||
            !yyjson_mut_obj_add_strcpy(doc, model_obj, "sha256", model.second.hex().c_str()) ||
            !yyjson_mut_arr_add_val(model_array, model_obj)) return prep_fail(error, "cannot construct model provenance");
    }
    if (!yyjson_mut_obj_add_val(doc, provenance, "models", model_array) || !yyjson_mut_obj_add_val(doc, root, "provenance", provenance))
        return prep_fail(error, "cannot attach provenance metadata");
    yyjson_mut_doc_set_root(doc, root);
    std::unordered_set<std::string> ids;
    size_t aggregate_payload = 0;
    for (size_t index = 0; index < request.items.size(); ++index) {
        const auto & item = request.items[index];
        if (item.id.empty() || !no_nul(item.id) || !ids.insert(item.id).second || item.frames == 0 || item.frames > 24576 || !token_ids(item.prompt, item.frames, error))
            return prep_fail(error, "native item metadata is invalid");
        if (item.prompt.cursor.present &&
            (!no_nul(item.prompt.cursor.lyrics_sha256) || !no_nul(item.prompt.cursor.tokenizer_sha256)))
            return prep_fail(error, "cursor provenance contains embedded NUL");
        if (item.frames > std::numeric_limits<size_t>::max() / (kLatentChannels * sizeof(float)))
            return prep_fail(error, "latent size overflow");
        const size_t latent_bytes = item.frames * kLatentChannels * sizeof(float);
        if (aggregate_payload > 512u * 1024u * 1024u - latent_bytes)
            return prep_fail(error, "prepared latent payload exceeds 512 MiB");
        aggregate_payload += latent_bytes;
        std::vector<uint8_t> latent;
        if (!read_file(item.latent_file, &latent, error) || latent.size() != latent_bytes)
            return prep_fail(error, "native latent size does not match frames");
        if (!finite_latents(latent, error)) return false;
        sha256::digest latent_hash;
        if (!sha256::file(item.latent_file, latent_hash, error)) return false;
        std::vector<uint8_t> codes;
        if (!read_file(item.semantic_file, &codes, error) || codes.size() != item.frames * sizeof(int32_t))
            return prep_fail(error, "native semantic code size does not match frames");
        sha256::digest semantic_hash;
        if (!sha256::file(item.semantic_file, semantic_hash, error)) return false;
        std::vector<int32_t> semantic(item.frames);
        for (size_t j = 0; j < item.frames; ++j) {
            int32_t value; std::memcpy(&value, codes.data() + j * sizeof(value), sizeof(value));
            if (value < 0 || value >= kCodecSize) return prep_fail(error, "native semantic code outside raw codec range");
            semantic[j] = value;
        }
        const std::string payload_name = "payload/item_" + std::to_string(index) + ".f32";
        const auto payload_path = request.output_dir / payload_name;
        if (!std::filesystem::create_directory(request.output_dir / "payload", ec) &&
            !std::filesystem::is_directory(request.output_dir / "payload", ec)) return prep_fail(error, "cannot create payload directory");
        std::ofstream out(payload_path, std::ios::binary);
        if (!out || (!latent.empty() && !out.write(reinterpret_cast<const char *>(latent.data()), static_cast<std::streamsize>(latent.size()))))
            return prep_fail(error, "cannot write prepared latent payload");
        if (!out.flush()) return prep_fail(error, "cannot flush prepared latent payload");
        out.close();
        if (out.fail()) return prep_fail(error, "cannot close prepared latent payload");
        yyjson_mut_val * obj = yyjson_mut_obj(doc);
        const std::string latent_fnv = fnv_hex(latent);
        if (!obj || !yyjson_mut_obj_add_strcpy(doc, obj, "id", item.id.c_str()) ||
            !yyjson_mut_obj_add_uint(doc, obj, "frames", item.frames) ||
            !yyjson_mut_obj_add_strcpy(doc, obj, "style", item.prompt_style.c_str()) ||
            !yyjson_mut_obj_add_strcpy(doc, obj, "lyrics", item.prompt_lyrics.c_str()) ||
            !yyjson_mut_obj_add_bool(doc, obj, "instrumental", item.instrumental) ||
            !yyjson_mut_obj_add_strcpy(doc, obj, "latent_file", payload_name.c_str()) ||
            !yyjson_mut_obj_add_strcpy(doc, obj, "latent_fnv1a64", latent_fnv.c_str()) ||
            !yyjson_mut_obj_add_strcpy(doc, obj, "native_latent_sha256", latent_hash.hex().c_str()) ||
            !yyjson_mut_obj_add_strcpy(doc, obj, "native_semantic_sha256", semantic_hash.hex().c_str()) ||
            !add_ids(doc, obj, "semantic_tokens", semantic) || !add_ids(doc, obj, "prefix_full_ids", item.prompt.retained_prefix_ids) ||
            !add_ids(doc, obj, "prefix_off_ids", item.prompt.dropped_prefix_ids) || !add_ids(doc, obj, "abc_ids", item.prompt.abc_ids) ||
            !yyjson_mut_arr_add_val(items, obj)) return prep_fail(error, "cannot construct manifest item");
        if (item.prompt.has_nocap() &&
            (!yyjson_mut_obj_add_strcpy(doc, obj, "style_nocap", item.prompt_style_nocap.c_str()) ||
             !add_ids(doc, obj, "prefix_full_nocap_ids", item.prompt.retained_nocap_prefix_ids) ||
             !add_ids(doc, obj, "prefix_off_nocap_ids", item.prompt.dropped_nocap_prefix_ids)))
            return prep_fail(error, "cannot construct manifest caption-dropout prefixes");
        if (item.prompt.cursor.present && !add_cursor(doc, obj, item.prompt.cursor))
            return prep_fail(error, "cannot serialize cursor metadata");
    }
    yyjson_mut_obj_add_val(doc, root, "items", items);
    size_t json_len = 0; char * json = yyjson_mut_write(doc, 0, &json_len);
    if (!json) return prep_fail(error, "cannot serialize dataset manifest");
    const auto temp_manifest = request.output_dir / "dataset.json.tmp";
    const auto final_manifest = request.output_dir / "dataset.json";
    if (std::filesystem::exists(final_manifest, ec)) { std::free(json); return prep_fail(error, "dataset manifest already exists"); }
    std::ofstream manifest(temp_manifest, std::ios::binary);
    const bool ok = manifest && manifest.write(json, static_cast<std::streamsize>(json_len)) && manifest.flush();
    manifest.close();
    std::free(json);
    if (!ok) return prep_fail(error, "cannot write dataset manifest");
    Dataset verified;
    std::string verify_error;
    if (!read_dataset(temp_manifest.u8string(), &verified, &verify_error)) return prep_fail(error, "prepared manifest failed reader roundtrip");
    if (std::filesystem::exists(final_manifest, ec)) return prep_fail(error, "dataset manifest appeared during publish");
    if (!publish_no_replace(temp_manifest, final_manifest, error)) return false;
    return true;
}

} // namespace yue2_aitk

