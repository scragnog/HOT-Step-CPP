#pragma once

// Staged importer for an AI Toolkit YuE2 latent cache.  It is deliberately
// separate from the raw-audio encoder path: Toolkit has already run the VAE,
// MERT semantic tokenizer, and SheetSage for this cache.  The importer only
// validates those products, converts BF16 latents to F32, builds tokenizer
// prefixes, and delegates schema-1 publication to yue2-aitk-prepare.h.

#include "yue2-aitk-prepare.h"
#include "../safetensors.h"
#include "../yue2/yue2-tokenizer.h"
#include "yyjson.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace yue2_aitk_import {

using namespace yue2_aitk;

struct ImportRequest {
    std::filesystem::path toolkit_dataset_dir; // copied dataset containing .txt and _latent_cache/
    std::filesystem::path source_manifest;     // Toolkit dataset-manifest.json
    std::filesystem::path base_checkpoint;     // exact raw ConvRot checkpoint used by the run
    std::filesystem::path tokenizer_dir;       // converted vocab.json + merges.txt
    std::filesystem::path output_dir;          // must not exist
    Yue2Cot cot = YUE2_COT_FULL;                // cache mode must match
};

namespace detail {

inline bool fail(std::string * e, const std::string & message) {
    if (e) *e = message;
    return false;
}

inline bool checked_mul(size_t a, size_t b, size_t * out) {
    if (a && b > (std::numeric_limits<size_t>::max)() / a) return false;
    *out = a * b;
    return true;
}

inline bool checked_add(size_t a, size_t b, size_t * out) {
    if (b > (std::numeric_limits<size_t>::max)() - a) return false;
    *out = a + b;
    return true;
}

inline const STEntry * entry(const STFile & st, const char * name) {
    return st_find(st, name);
}

inline bool entry_bytes(const STFile & st, const STEntry & e, size_t expected,
                        std::string * error) {
    if (e.data_end < e.data_start || e.data_end - e.data_start != expected)
        return fail(error, "Toolkit cache tensor byte size does not match its shape");
    if (e.data_end > st.file_size - st.data_offset)
        return fail(error, "Toolkit cache tensor payload exceeds file");
    return true;
}

inline bool validate_entries(const STFile & st, std::string * error) {
    std::unordered_set<std::string> names;
    std::vector<std::pair<size_t, size_t>> ranges;
    ranges.reserve(st.entries.size());
    if (st.data_offset > st.file_size) return fail(error, "Toolkit cache data offset exceeds file");
    for (const auto & e : st.entries) {
        if (!names.insert(e.name).second || e.data_end < e.data_start ||
            e.data_end > st.file_size - st.data_offset) return fail(error, "Toolkit cache has duplicate or out-of-range tensor entries");
        ranges.emplace_back(e.data_start, e.data_end);
    }
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i)
        if (ranges[i - 1].second > ranges[i].first) return fail(error, "Toolkit cache tensor payloads overlap");
    return true;
}

inline bool shape1(const STEntry & e, int64_t n) { return e.n_dims == 1 && e.shape[0] == n; }
inline bool shape2(const STEntry & e, int64_t n0, int64_t n1) {
    return e.n_dims == 2 && e.shape[0] == n0 && e.shape[1] == n1;
}

inline float bf16_to_float(const uint8_t * p) {
    uint16_t bits = 0; std::memcpy(&bits, p, sizeof(bits));
    uint32_t word = uint32_t(bits) << 16; float value = 0.0f;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

inline bool read_text(const std::filesystem::path & path, std::string * out, std::string * error) {
    std::error_code ec;
    const auto n = std::filesystem::file_size(path, ec);
    if (ec || n > 16u * 1024u * 1024u) return fail(error, "caption is missing or too large");
    std::ifstream in(path, std::ios::binary);
    if (!in) return fail(error, "caption cannot be opened");
    out->assign(static_cast<size_t>(n), '\0');
    if (n && !in.read(out->data(), static_cast<std::streamsize>(n)))
        return fail(error, "caption read failed");
    if (out->size() >= 3 && static_cast<uint8_t>((*out)[0]) == 0xef &&
        static_cast<uint8_t>((*out)[1]) == 0xbb && static_cast<uint8_t>((*out)[2]) == 0xbf)
        out->erase(0, 3);
    return true;
}

inline std::string normalize_section(const std::string & line) {
    static const std::regex section(R"(^\s*\[[^\]]+\]\s*$)");
    if (!std::regex_match(line, section)) return line;
    const size_t left = line.find('['), right = line.rfind(']');
    const std::string original = line.substr(left + 1, right - left - 1);
    std::string body;
    size_t start = 0;
    for (size_t i = 0; i <= original.size(); ++i) {
        if (i != original.size() && original[i] != ' ') continue;
        std::string word = original.substr(start, i - start);
        bool upper = word.size() > 1;
        for (char c : word) if (std::isalpha(static_cast<unsigned char>(c)) && std::islower(static_cast<unsigned char>(c))) upper = false;
        if (upper) {
            for (char & c : word) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (!word.empty()) word[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(word[0])));
        }
        if (start != 0) body.push_back(' ');
        body += word;
        start = i + 1;
    }
    if (!body.empty()) body[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(body[0])));
    for (size_t i = 1; i + 1 < body.size(); ++i)
        if (body[i] == '-' && std::isalpha(static_cast<unsigned char>(body[i + 1])))
            body[i + 1] = static_cast<char>(std::toupper(static_cast<unsigned char>(body[i + 1])));
    return line.substr(0, left + 1) + body + line.substr(right);
}

inline void trim(std::string * value) {
    size_t first = 0, last = value->size();
    while (first < last && std::isspace(static_cast<unsigned char>((*value)[first]))) ++first;
    while (last > first && std::isspace(static_cast<unsigned char>((*value)[last - 1]))) --last;
    *value = value->substr(first, last - first);
}

inline bool parse_caption(const std::string & text, std::string * style, std::string * lyrics,
                          std::string * error) {
    if (text.find('\0') != std::string::npos) return fail(error, "caption contains NUL");
    if (text.find("<CAPTION>") != std::string::npos || text.find("<LYRICS>") != std::string::npos)
        return fail(error, "convert legacy caption tags to [Tags]/[Lyrics] before import");
    // Prepared caption layout follows the pinned Toolkit parser.
    std::vector<std::string> tags, lyric_lines;
    enum class Section { tags, lyrics, duration } current = Section::tags;
    bool saw_lyrics_header = false;
    std::string line;
    std::istringstream in(text);
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::smatch m;
        if (std::regex_match(line, m, std::regex(R"(^\s*\[(Tags|Lyrics|Duration)\]\s*$)", std::regex::icase))) {
            std::string name = m[1].str();
            std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::tolower(c)); });
            if (name == "lyrics") { current = Section::lyrics; saw_lyrics_header = true; }
            else if (name == "duration") current = Section::duration;
            else current = Section::tags;
            continue;
        }
        if (current == Section::tags && !saw_lyrics_header && std::regex_match(line, std::regex(R"(^\s*\[[^\]]+\]\s*$)"))) { current = Section::lyrics; saw_lyrics_header = true; }
        if (current == Section::lyrics) lyric_lines.push_back(normalize_section(line));
        else if (current == Section::tags) tags.push_back(line);
        // Duration is parsed by the reference but does not enter the prefix.
    }
    *style = std::accumulate(tags.begin(), tags.end(), std::string(), [](std::string a, const std::string & b) { return a.empty() ? b : a + "\n" + b; });
    *lyrics = std::accumulate(lyric_lines.begin(), lyric_lines.end(), std::string(), [](std::string a, const std::string & b) { return a.empty() ? b : a + "\n" + b; });
    trim(style); trim(lyrics);
    return true;
}

inline bool cache_metadata_filename(const STFile & st, std::string * filename, std::string * error) {
    if (st.file_size < 8 || st.data_offset < 8) return fail(error, "invalid cache header bounds");
    yyjson_doc * doc = yyjson_read(reinterpret_cast<const char *>(st.mapping + 8), st.data_offset - 8, 0);
    if (!doc) return fail(error, "invalid Toolkit cache header JSON");
    yyjson_val * root = yyjson_doc_get_root(doc); yyjson_val * meta = root ? yyjson_obj_get(root, "__metadata__") : nullptr;
    yyjson_val * name = meta ? yyjson_obj_get(meta, "filename") : nullptr;
    const bool ok = name && yyjson_is_str(name) && yyjson_get_len(name) == std::strlen(yyjson_get_str(name));
    if (ok) filename->assign(yyjson_get_str(name), yyjson_get_len(name));
    yyjson_doc_free(doc);
    return ok ? true : fail(error, "Toolkit cache metadata has no valid filename");
}

inline bool source_manifest_files(const std::filesystem::path & path, std::unordered_set<std::string> * files,
                                  std::string * error) {
    std::string text;
    if (!read_text(path, &text, error)) return false;
    yyjson_doc * doc = yyjson_read(text.data(), text.size(), 0);
    if (!doc) return fail(error, "source manifest is invalid JSON");
    yyjson_val * root = yyjson_doc_get_root(doc); yyjson_val * tracks = root ? yyjson_obj_get(root, "tracks") : nullptr;
    if (!tracks || !yyjson_is_arr(tracks) || yyjson_arr_size(tracks) == 0) { yyjson_doc_free(doc); return fail(error, "source manifest has no tracks array"); }
    size_t i = 0, max = 0; yyjson_val * item = nullptr;
    yyjson_arr_foreach(tracks, i, max, item) {
        yyjson_val * file = yyjson_is_obj(item) ? yyjson_obj_get(item, "file") : nullptr;
        if (!file || !yyjson_is_str(file) || yyjson_get_len(file) != std::strlen(yyjson_get_str(file))) { yyjson_doc_free(doc); return fail(error, "source manifest track has no valid file"); }
        const std::string name(yyjson_get_str(file), yyjson_get_len(file));
        if (!files->insert(name).second) { yyjson_doc_free(doc); return fail(error, "source manifest contains duplicate track filename"); }
    }
    yyjson_doc_free(doc);
    return true;
}

inline bool import_one(const std::filesystem::path & cache, const std::filesystem::path & dataset_dir,
                       const std::filesystem::path & work_dir, size_t ordinal, BPETokenizer * tokenizer, VerifiedNativeItem * out,
                       std::vector<ModelProvenance> * provenance, std::string * error) {
    STFile st{};
    if (!st_open(&st, cache.u8string().c_str())) return fail(error, "cannot open Toolkit latent cache");
    struct Guard { STFile * p; ~Guard() { st_close(p); } } guard{&st};
    if (!validate_entries(st, error)) return false;
    std::string filename;
    if (!cache_metadata_filename(st, &filename, error)) return false;
    auto audio = dataset_dir / filename;
    auto caption = audio;
    caption.replace_extension(".txt");
    if (std::filesystem::path(filename).filename().u8string() != filename || filename.find('/') != std::string::npos || filename.find('\\') != std::string::npos || !std::filesystem::is_regular_file(audio) || !std::filesystem::is_regular_file(caption)) return fail(error, "cache metadata filename is not a matching basename/caption");
    const STEntry * lat = entry(st, "latent"), * tok = entry(st, "dto.tokens"), * abc = entry(st, "dto.abc_ids"), * mode = entry(st, "dto.abc_mode");
    if (!lat || !tok || !abc || !mode || st.entries.size() != 4) return fail(error, "Toolkit cache must contain exactly latent, tokens, abc_ids, and abc_mode");
    if (lat->dtype != "BF16" || tok->dtype != "I32" || abc->dtype != "I32" || mode->dtype != "I32" || lat->n_dims != 2 || lat->shape[1] != 64 || tok->n_dims != 1 || abc->n_dims != 1 || mode->n_dims != 0) return fail(error, "Toolkit cache stream dtype/rank/shape mismatch");
    const int64_t frames = lat->shape[0], abc_count = abc->shape[0];
    if (frames <= 0 || frames > 24576 || abc_count < 0 || abc_count > 24576 || tok->shape[0] != frames) return fail(error, "Toolkit cache frame/ABC limit or alignment mismatch");
    size_t latent_bytes = 0, token_bytes = 0, abc_bytes = 0;
    if (!checked_mul(static_cast<size_t>(frames), 64u * sizeof(uint16_t), &latent_bytes) || !checked_mul(static_cast<size_t>(frames), sizeof(int32_t), &token_bytes) || !checked_mul(static_cast<size_t>(abc_count), sizeof(int32_t), &abc_bytes) || !entry_bytes(st, *lat, latent_bytes, error) || !entry_bytes(st, *tok, token_bytes, error) || !entry_bytes(st, *abc, abc_bytes, error) || !entry_bytes(st, *mode, sizeof(int32_t), error)) return false;
    int32_t mode_value = 0; std::memcpy(&mode_value, st_data(st, *mode), sizeof(mode_value));
    if (mode_value != 2) return fail(error, "Toolkit cache is not a full SheetSage cache");
    std::string caption_text, style, lyrics;
    if (!read_text(caption, &caption_text, error) || !parse_caption(caption_text, &style, &lyrics, error)) return false;
    try {
        const auto full = yue2_token_prefixes(tokenizer, style, lyrics, YUE2_COT_FULL, nullptr);
        auto off = yue2_token_prefixes(tokenizer, style, lyrics, YUE2_COT_OFF, nullptr);
        if (off.size() < 2 || off[off.size() - 2] != YUE2_ABC_END || off.back() != YUE2_MUSIC_START) return fail(error, "unexpected native off-prefix tail");
        off.resize(off.size() - 2); // training PromptInput prefixes end at ABC_START in both modes
        out->id = filename; out->frames = static_cast<size_t>(frames);
        out->prompt.retained_prefix_ids.assign(full.begin(), full.end()); out->prompt.dropped_prefix_ids.assign(off.begin(), off.end()); out->prompt.retain_abc = true;
        out->prompt.abc_ids.resize(static_cast<size_t>(abc_count));
        const auto * abc_data = static_cast<const uint8_t *>(st_data(st, *abc));
        for (size_t i = 0; i < out->prompt.abc_ids.size(); ++i) { int32_t value = 0; std::memcpy(&value, abc_data + i * sizeof(value), sizeof(value)); if (value < 0 || value >= YUE2_EOD) return fail(error, "cached ABC ID outside ordinary text vocabulary"); out->prompt.abc_ids[i] = value; }
    } catch (const std::exception & ex) { return fail(error, std::string("native tokenizer rejected caption: ") + ex.what()); }
    const auto temp = work_dir / ("item_" + std::to_string(ordinal) + ".f32");
    std::ofstream payload(temp, std::ios::binary); if (!payload) return fail(error, "cannot create temporary converted latent payload");
    const auto * raw = static_cast<const uint8_t *>(st_data(st, *lat));
    std::vector<float> converted(static_cast<size_t>(frames) * 64u);
    for (size_t i = 0; i < converted.size(); ++i) { converted[i] = bf16_to_float(raw + i * 2); if (!std::isfinite(converted[i])) return fail(error, "cached BF16 latent is non-finite"); }
    payload.write(reinterpret_cast<const char *>(converted.data()), static_cast<std::streamsize>(converted.size() * sizeof(float))); payload.close();
    if (!payload) return fail(error, "cannot write converted latent payload");
    out->latent_file = temp;
    const auto * token_data = static_cast<const uint8_t *>(st_data(st, *tok));
    out->semantic_file = work_dir / ("item_" + std::to_string(ordinal) + ".tokens");
    std::ofstream token_out(out->semantic_file, std::ios::binary); if (!token_out) return fail(error, "cannot create temporary semantic payload");
    token_out.write(reinterpret_cast<const char *>(token_data), static_cast<std::streamsize>(token_bytes)); token_out.close();
    for (int32_t i = 0; i < frames; ++i) { int32_t value = 0; std::memcpy(&value, token_data + size_t(i) * sizeof(value), sizeof(value)); if (value < 0 || value >= kCodecSize) return fail(error, "cached semantic ID outside codec vocabulary"); }
    provenance->push_back({"toolkit_cache_" + out->id, cache}); provenance->push_back({"caption_" + out->id, caption});
    return true;
}

} // namespace detail

// CLI proposal (dispatch remains root-owned):
// yue2-joint-train --import-aitk-cache --toolkit-dataset <dir>
//   --source-manifest <dataset-manifest.json> --base-checkpoint <raw.safetensors>
//   --tokenizer-dir <converted-tokenizer> --output <fresh-dir> --cot full
// The command must validate and publish the fresh schema-1 directory before any
// CUDA/model allocation. It must report imported_toolkit_cache=true and must not
// invoke Python, MERT, SheetSage, VAE, or an unused encoder hash stage.
inline bool import_toolkit_cache(const ImportRequest & request, std::string * error = nullptr) {
    using namespace detail;
    if (std::filesystem::exists(request.output_dir)) return fail(error, "import output must be a new directory");
    if (request.toolkit_dataset_dir.empty() || request.source_manifest.empty() || request.base_checkpoint.empty() || request.tokenizer_dir.empty() || request.output_dir.empty()) return fail(error, "Toolkit cache import request is incomplete");
    if (request.cot != YUE2_COT_FULL) return fail(error, "this importer requires the pinned full-mode Toolkit cache; off is a training draw, not a cache mode");
    const auto cache_dir = request.toolkit_dataset_dir / "_latent_cache";
    if (!std::filesystem::is_directory(cache_dir) || !std::filesystem::is_regular_file(request.source_manifest)) return fail(error, "Toolkit dataset/cache or source manifest is missing");
    std::unordered_set<std::string> source_files;
    if (!source_manifest_files(request.source_manifest, &source_files, error)) return false;
    std::error_code ec;
    const auto work_dir = std::filesystem::u8path(request.output_dir.u8string() + ".import-work");
    if (std::filesystem::exists(work_dir, ec) || ec || !std::filesystem::create_directory(work_dir, ec) || ec) return fail(error, "import work directory must be fresh and creatable");
    BPETokenizer tokenizer{};
    if (!yue2_tokenizer_load_from_dir(&tokenizer, request.tokenizer_dir.u8string())) return fail(error, "cannot load native tokenizer sidecar");
    PrepareRequest prep; prep.output_dir = request.output_dir; prep.source_manifest = request.source_manifest; prep.base_checkpoint = request.base_checkpoint;
    prep.verified_contract = "imported_toolkit_cache_v1;encoders=already_cached;latent=BF16_to_F32;abc=cache;prefix=tokenizer_sidecar";
    for (const auto & name : {std::string("vocab.json"), std::string("merges.txt")}) prep.models.push_back({"tokenizer_" + name, request.tokenizer_dir / name});
    std::vector<std::filesystem::path> caches;
    for (const auto & file : std::filesystem::directory_iterator(cache_dir)) {
        if (!file.is_regular_file() || file.path().extension() != ".safetensors") continue;
        caches.push_back(file.path());
    }
    std::sort(caches.begin(), caches.end());
    if (caches.empty() || caches.size() > 10000) return fail(error, "Toolkit cache directory has no usable items or too many items");
    std::unordered_set<std::string> imported_files;
    for (size_t ordinal = 0; ordinal < caches.size(); ++ordinal) {
        VerifiedNativeItem item;
        if (!import_one(caches[ordinal], request.toolkit_dataset_dir, work_dir, ordinal, &tokenizer, &item, &prep.models, error)) return false;
        if (!source_files.count(item.id)) return fail(error, "cache item is absent from source manifest");
        imported_files.insert(item.id);
        prep.items.push_back(std::move(item));
    }
    if (imported_files.size() != source_files.size()) return fail(error, "source manifest and Toolkit cache coverage differ");
    // prepare_dataset hashes provenance and publishes a fresh schema-1 directory.
    return prepare_dataset(prep, error);
}

} // namespace yue2_aitk_import
