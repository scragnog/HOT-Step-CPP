#pragma once

// Immutable YuE2 cache-manifest reader.  This is deliberately a staging seam:
// it has no legacy-cache fallback and never mutates the destination on error.

#include "yue2-aitk-batch.h"
#include "yue2-aitk-sha256.h"
#include "../hot-step-fsutf8.h"
#include "yyjson.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

namespace yue2_aitk {

struct DatasetItem {
    std::string id;
    SongInput song;
    PromptInput prompt;
};

struct Dataset {
    std::string recipe_version;
    std::string base_sha256;
    std::string source_manifest_sha256;
    std::vector<DatasetItem> items;
};

namespace dataset_detail {
constexpr size_t kMaxManifestBytes = 16u * 1024u * 1024u;
constexpr size_t kMaxItems = 10000;
constexpr size_t kMaxFrames = 24576;
constexpr size_t kMaxPayloadBytes = 512u * 1024u * 1024u;
constexpr const char * kRecipe = "aitk-yue2-2026-09-16";

inline bool bad(std::string * e, const char * msg) { if (e) *e = msg; return false; }

inline bool string_value(yyjson_val * v, std::string * out, bool nonempty = false) {
    if (!yyjson_is_str(v) || yyjson_get_len(v) != std::strlen(yyjson_get_str(v))) return false;
    out->assign(yyjson_get_str(v), yyjson_get_len(v));
    return !nonempty || !out->empty();
}

inline bool unique_keys(yyjson_val * value, size_t depth = 0) {
    if (depth > 64) return false;
    if (yyjson_is_str(value) && yyjson_get_len(value) != std::strlen(yyjson_get_str(value))) return false;
    if (yyjson_is_obj(value)) {
        std::unordered_set<std::string> keys;
        size_t i = 0, max = 0; yyjson_val * key = nullptr; yyjson_val * child = nullptr;
        yyjson_obj_foreach(value, i, max, key, child) {
            if (yyjson_get_len(key) != std::strlen(yyjson_get_str(key))) return false;
            const std::string name(yyjson_get_str(key), yyjson_get_len(key));
            if (!keys.insert(name).second || !unique_keys(child, depth + 1)) return false;
        }
    } else if (yyjson_is_arr(value)) {
        size_t i = 0, max = 0; yyjson_val * child = nullptr;
        yyjson_arr_foreach(value, i, max, child) if (!unique_keys(child, depth + 1)) return false;
    }
    return true;
}

inline bool hex64(const std::string & s, bool lower_only) {
    if (s.size() != 16) return false;
    for (char c : s) {
        if ((c >= '0' && c <= '9') || (!lower_only && c >= 'A' && c <= 'F') ||
            (c >= 'a' && c <= 'f')) continue;
        return false;
    }
    return true;
}

inline bool hex256(const std::string & s) {
    if (s.size() != 64) return false;
    for (char c : s) if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
                           (c >= 'A' && c <= 'F'))) return false;
    return true;
}

inline uint64_t fnv1a(const std::vector<uint8_t> & bytes) {
    uint64_t h = 14695981039346656037ull;
    for (uint8_t b : bytes) { h ^= b; h *= 1099511628211ull; }
    return h;
}

inline bool read_bytes(const std::filesystem::path & path, size_t max_bytes,
                       std::vector<uint8_t> * out, std::string * e) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec || size > max_bytes) return bad(e, "dataset file missing or too large");
    std::ifstream f(path, std::ios::binary);
    if (!f) return bad(e, "dataset file cannot be opened");
    out->resize(static_cast<size_t>(size));
    if (!out->empty() && !f.read(reinterpret_cast<char *>(out->data()), static_cast<std::streamsize>(out->size())))
        return bad(e, "dataset file read failed");
    return true;
}

inline bool within(const std::filesystem::path & base, const std::filesystem::path & file) {
    auto b = base.begin(), f = file.begin();
    for (; b != base.end() && f != file.end() && *b == *f; ++b, ++f) {}
    return b == base.end();
}

inline bool relative_payload(const std::filesystem::path & dir, const std::string & name,
                             std::filesystem::path * result, std::string * e) {
    if (name.empty() || name.find('\\') != std::string::npos) return bad(e, "latent_file must be UTF-8 slash-relative");
    const auto rel = std::filesystem::u8path(name);
    if (rel.empty() || rel.is_absolute() || rel.has_root_name() || rel.has_root_directory())
        return bad(e, "latent_file must be relative");
    for (const auto & part : rel) if (part == std::filesystem::path("..")) return bad(e, "latent_file escapes manifest directory");
    std::error_code ec1, ec2;
    const auto base = std::filesystem::weakly_canonical(dir, ec1);
    const auto file = std::filesystem::weakly_canonical(dir / rel, ec2);
    if (ec1 || ec2 || !within(base, file) || !std::filesystem::is_regular_file(file, ec2))
        return bad(e, "latent_file is outside manifest directory or not a regular file");
    *result = file;
    return true;
}

inline bool ids(yyjson_val * v, std::vector<int32_t> * out, bool allow_empty, std::string * e) {
    if (!yyjson_is_arr(v)) return bad(e, "token field must be an array");
    const size_t n = yyjson_arr_size(v);
    if (n > kMaxFrames) return bad(e, "token array exceeds sequence limit");
    if (!allow_empty && n == 0) return bad(e, "prefix context is empty");
    out->reserve(n);
    size_t i = 0, max = 0; yyjson_val * x = nullptr;
    yyjson_arr_foreach(v, i, max, x) {
        if (!yyjson_is_int(x)) return bad(e, "token value must be an integer");
        const int64_t value = yyjson_get_sint(x);
        if (value < 0 || value >= kVocabSize) return bad(e, "token value out of vocabulary range");
        out->push_back(static_cast<int32_t>(value));
    }
    return true;
}

inline bool f32s(yyjson_val * v, std::vector<float> * out, std::string * e) {
    if (!yyjson_is_arr(v) || yyjson_arr_size(v) == 0 || yyjson_arr_size(v) > 5u * 10000u)
        return bad(e, "cursor words must be a nonempty numeric array");
    size_t i = 0, max = 0; yyjson_val * x = nullptr;
    yyjson_arr_foreach(v, i, max, x) {
        if (!yyjson_is_num(x) || !std::isfinite(yyjson_get_num(x))) return bad(e, "cursor word value is not finite");
        out->push_back(static_cast<float>(yyjson_get_num(x)));
    }
    if (out->size() % 5 != 0) return bad(e, "cursor words must contain five values per word");
    return true;
}

inline bool i64s(yyjson_val * v, std::vector<int64_t> * out, std::string * e) {
    if (!yyjson_is_arr(v) || yyjson_arr_size(v) == 0 || yyjson_arr_size(v) > kMaxFrames)
        return bad(e, "cursor token offsets are invalid");
    size_t i = 0, max = 0; yyjson_val * x = nullptr;
    yyjson_arr_foreach(v, i, max, x) {
        if (!yyjson_is_int(x) || yyjson_get_sint(x) < 0) return bad(e, "cursor token offset is invalid");
        out->push_back(yyjson_get_sint(x));
    }
    return true;
}

inline bool cursor_side(yyjson_val * v, const std::vector<int32_t> & prefix, int64_t frames,
                        int64_t lyric_codepoints, const std::vector<float> & words,
                        CursorTargets * out, std::string * e) {
    if (!yyjson_is_obj(v)) return bad(e, "cursor prefix binding is missing");
    yyjson_val * h = yyjson_obj_get(v, "head_tokens");
    if (!yyjson_is_int(h) || yyjson_get_sint(h) < 0) return bad(e, "cursor head_tokens is invalid");
    std::vector<int64_t> ends;
    if (!i64s(yyjson_obj_get(v, "token_end_codepoints"), &ends, e)) return false;
    if (!bind_cursor_targets(prefix, yyjson_get_sint(h), ends, lyric_codepoints, frames, words, out))
        return bad(e, out->why.c_str());
    return true;
}
} // namespace dataset_detail

inline bool read_dataset(const std::string & manifest_path, Dataset * out, std::string * error = nullptr) {
    using namespace dataset_detail;
    if (!out) return bad(error, "dataset output is null");
    std::vector<uint8_t> json_bytes;
    const std::filesystem::path manifest = std::filesystem::u8path(manifest_path);
    if (!read_bytes(manifest, kMaxManifestBytes, &json_bytes, error)) return false;
    yyjson_doc * doc = yyjson_read(reinterpret_cast<const char *>(json_bytes.data()), json_bytes.size(), 0);
    if (!doc) return bad(error, "invalid dataset JSON");
    struct DocGuard { yyjson_doc * p; ~DocGuard() { yyjson_doc_free(p); } } guard{doc};
    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) return bad(error, "dataset root must be an object");
    if (!unique_keys(root)) return bad(error, "dataset contains duplicate keys or embedded NUL");
    yyjson_val * schema = yyjson_obj_get(root, "schema_version");
    yyjson_val * recipe = yyjson_obj_get(root, "recipe_version");
    yyjson_val * cot = yyjson_obj_get(root, "cot");
    yyjson_val * base = yyjson_obj_get(root, "base_sha256");
    yyjson_val * source = yyjson_obj_get(root, "source_manifest_sha256");
    yyjson_val * items = yyjson_obj_get(root, "items");
    std::string recipe_s, cot_s, base_s, source_s;
    if (!yyjson_is_int(schema) || yyjson_get_sint(schema) != 1 || !string_value(recipe, &recipe_s) ||
        recipe_s != kRecipe || !string_value(cot, &cot_s) || cot_s != "full" ||
        !string_value(base, &base_s) || !hex256(base_s) || !string_value(source, &source_s) || !hex256(source_s) ||
        !yyjson_is_arr(items) || yyjson_arr_size(items) == 0 || yyjson_arr_size(items) > kMaxItems)
        return bad(error, "dataset metadata is invalid");
    Dataset tmp;
    tmp.recipe_version = recipe_s;
    tmp.base_sha256 = base_s;
    tmp.source_manifest_sha256 = source_s;
    tmp.items.reserve(yyjson_arr_size(items));
    std::unordered_set<std::string> seen;
    const auto dir = manifest.parent_path();
    size_t aggregate_payload = 0;
    size_t index = 0, max_items = 0; yyjson_val * item = nullptr;
    yyjson_arr_foreach(items, index, max_items, item) {
        if (!yyjson_is_obj(item)) return bad(error, "dataset item must be an object");
        yyjson_val * id = yyjson_obj_get(item, "id"); yyjson_val * frames = yyjson_obj_get(item, "frames");
        yyjson_val * lf = yyjson_obj_get(item, "latent_file"); yyjson_val * fh = yyjson_obj_get(item, "latent_fnv1a64");
        yyjson_val * sem = yyjson_obj_get(item, "semantic_tokens"); yyjson_val * full = yyjson_obj_get(item, "prefix_full_ids");
        yyjson_val * off = yyjson_obj_get(item, "prefix_off_ids"); yyjson_val * abc = yyjson_obj_get(item, "abc_ids");
        std::string id_s, latent_file_s, fnv_s;
        if (!string_value(id, &id_s, true) || !seen.insert(id_s).second ||
            !yyjson_is_int(frames) || yyjson_get_sint(frames) <= 0 || size_t(yyjson_get_sint(frames)) > kMaxFrames ||
            !string_value(lf, &latent_file_s, true) || !string_value(fh, &fnv_s) || !hex64(fnv_s, true) || !yyjson_is_arr(sem))
            return bad(error, "dataset item metadata is invalid");
        const size_t n = static_cast<size_t>(yyjson_get_sint(frames));
        if (yyjson_arr_size(sem) != n) return bad(error, "semantic token count does not match frames");
        DatasetItem result; result.id = std::move(id_s); result.song.latent_channels = kLatentChannels;
        yyjson_val * style = yyjson_obj_get(item, "style"); yyjson_val * lyrics = yyjson_obj_get(item, "lyrics");
        if (style && !string_value(style, &result.song.style)) return bad(error, "dataset style must be a UTF-8 string");
        if (lyrics && !string_value(lyrics, &result.song.lyrics)) return bad(error, "dataset lyrics must be a UTF-8 string");
        result.song.instrumental = yyjson_obj_get(item, "instrumental") && yyjson_is_bool(yyjson_obj_get(item, "instrumental"))
            ? yyjson_get_bool(yyjson_obj_get(item, "instrumental")) : result.song.lyrics.empty();
        if (!ids(full, &result.prompt.retained_prefix_ids, false, error) ||
            !ids(off, &result.prompt.dropped_prefix_ids, false, error) ||
            !ids(abc, &result.prompt.abc_ids, true, error)) return false;
        result.prompt.retain_abc = !result.prompt.abc_ids.empty();
        yyjson_val * cursor = yyjson_obj_get(item, "cursor");
        if (cursor) {
            if (!yyjson_is_obj(cursor)) return bad(error, "cursor metadata must be an object");
            auto & cm = result.prompt.cursor;
            cm.present = true;
            yyjson_val * enabled = yyjson_obj_get(cursor, "enabled");
            yyjson_val * instrumental = yyjson_obj_get(cursor, "instrumental");
            if (enabled && !yyjson_is_bool(enabled)) return bad(error, "cursor enabled must be boolean");
            if (instrumental && !yyjson_is_bool(instrumental)) return bad(error, "cursor instrumental must be boolean");
            cm.enabled = enabled ? yyjson_get_bool(enabled) : true;
            cm.instrumental = instrumental && yyjson_get_bool(instrumental);
            if (cm.instrumental) { cm.enabled = false; }
            yyjson_val * lh = yyjson_obj_get(cursor, "lyrics_sha256");
            yyjson_val * th = yyjson_obj_get(cursor, "tokenizer_sha256");
            yyjson_val * cp = yyjson_obj_get(cursor, "lyric_codepoints");
            if (cm.instrumental) {
                if (!result.song.lyrics.empty()) return bad(error, "instrumental cursor metadata conflicts with nonempty lyrics");
            } else {
            if (!string_value(lh, &cm.lyrics_sha256) || !hex256(cm.lyrics_sha256) ||
                !string_value(th, &cm.tokenizer_sha256) || !hex256(cm.tokenizer_sha256) ||
                !yyjson_is_int(cp) || yyjson_get_sint(cp) <= 0 || yyjson_get_sint(cp) > 10000000 ||
                !f32s(yyjson_obj_get(cursor, "words5"), &cm.words5, error)) return bad(error, "cursor metadata is invalid");
            const int64_t codepoints = yyjson_get_sint(cp);
            cm.lyric_codepoints = codepoints;
            if (sha256::bytes(result.song.lyrics.data(), result.song.lyrics.size()).hex() != cm.lyrics_sha256)
                return bad(error, "cursor lyrics hash does not match normalized item lyrics");
            if (!cursor_side(yyjson_obj_get(cursor, "full"), result.prompt.retained_prefix_ids, n,
                              codepoints, cm.words5, &cm.full, error) ||
                !cursor_side(yyjson_obj_get(cursor, "off"), result.prompt.dropped_prefix_ids, n,
                              codepoints, cm.words5, &cm.off, error)) return false;
            cm.j0_full = cm.full.j0; cm.j0_off = cm.off.j0; cm.L = cm.full.L;
            if (cm.off.L != cm.L || cm.full.nF != cm.off.nF) return bad(error, "cursor full/off token geometry differs");
            cursor_ranges(cm.full, &cm.full_frame_ranges); cursor_ranges(cm.off, &cm.off_frame_ranges);
            cm.full.T.clear(); cm.off.T.clear(); // retain compact ranges, not a dataset-wide dense matrix
            cm.full_lyric_token_end_codepoints.clear(); cm.off_lyric_token_end_codepoints.clear();
            // The binding function has validated these; retain the arrays for
            // callers that need to rebind after choosing a shorter crop.
            yyjson_val * full_obj = yyjson_obj_get(cursor, "full");
            yyjson_val * off_obj = yyjson_obj_get(cursor, "off");
            if (!i64s(yyjson_obj_get(full_obj, "token_end_codepoints"), &cm.full_lyric_token_end_codepoints, error) ||
                !i64s(yyjson_obj_get(off_obj, "token_end_codepoints"), &cm.off_lyric_token_end_codepoints, error)) return false;
            cm.full_head_tokens = cm.full.j0 - 1; cm.off_head_tokens = cm.off.j0 - 1;
            }
        }
        size_t si = 0, max_sem = 0; yyjson_val * sv = nullptr; yyjson_arr_foreach(sem, si, max_sem, sv) {
            if (!yyjson_is_int(sv) || yyjson_get_sint(sv) < 0 || yyjson_get_sint(sv) >= kCodecSize)
                return bad(error, "semantic token out of codec range");
            result.song.semantic_tokens.push_back(static_cast<int32_t>(yyjson_get_sint(sv)));
        }
        std::filesystem::path payload;
        if (!relative_payload(dir, latent_file_s, &payload, error)) return false;
        if (n > std::numeric_limits<size_t>::max() / (kLatentChannels * sizeof(float))) return bad(error, "latent payload size overflow");
        const size_t payload_bytes = n * kLatentChannels * sizeof(float);
        if (aggregate_payload > kMaxPayloadBytes - payload_bytes) return bad(error, "aggregate latent payload exceeds 512 MiB");
        aggregate_payload += payload_bytes;
        std::vector<uint8_t> bytes;
        if (!read_bytes(payload, payload_bytes, &bytes, error) || bytes.size() != payload_bytes)
            return bad(error, "latent payload size mismatch");
        if (fnv1a(bytes) != std::stoull(fnv_s, nullptr, 16)) return bad(error, "latent FNV-1a corruption check failed");
        if (result.prompt.retained_prefix_ids.size() + result.prompt.abc_ids.size() + n + 3 > kMaxFrames ||
            result.prompt.dropped_prefix_ids.size() + result.prompt.abc_ids.size() + n + 3 > kMaxFrames ||
            result.prompt.retained_prefix_ids.size() + result.prompt.abc_ids.size() + 2 * std::min(n, size_t(1500)) + 5 > kMaxFrames ||
            result.prompt.dropped_prefix_ids.size() + 2 * std::min(n, size_t(1500)) + 5 > kMaxFrames)
            return bad(error, "prefix context exceeds YuE2 sequence limit");
        result.song.latents.resize(n * kLatentChannels);
        for (size_t p = 0; p < result.song.latents.size(); ++p) {
            float value; std::memcpy(&value, bytes.data() + p * sizeof(float), sizeof(value));
            if (!std::isfinite(value)) return bad(error, "latent payload contains non-finite value");
            result.song.latents[p] = value;
        }
        tmp.items.push_back(std::move(result));
    }
    *out = std::move(tmp);
    return true;
}

} // namespace yue2_aitk

