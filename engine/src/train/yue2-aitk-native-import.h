#pragma once

// Bridge from the existing HOT-Step YuE2 cache to AITK schema1.
// It consumes the full-source artifacts produced by yue2-preprocess,
// yue2-tokenize and yue2-sheet; it never treats the Legacy manifest itself as
// an AITK manifest and never mutates the source cache.

#include "yue2-aitk-prepare.h"
#include "yue2-aitk-import.h"
#include "../yue2/yue2-tokenizer.h"
#include "yyjson.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

namespace yue2_aitk_native_import {

using namespace yue2_aitk;

struct ModelInput {
    std::string role;
    std::filesystem::path path;
};

struct Request {
    std::filesystem::path legacy_manifest; // yue2_preprocess.json after all cache stages
    std::filesystem::path raw_convrot_checkpoint;
    std::filesystem::path tokenizer_gguf_or_dir; // GGUF or vocab.json/merges.txt directory
    std::filesystem::path output_dir;             // must not exist
    std::vector<ModelInput> models;               // VAE, semantic tokenizer, SheetSage, etc.
    bool lyric_timing = false;                    // explicit opt-in; requires cursor_words for lyrical sources
};

namespace detail {
inline bool fail(std::string * e, const std::string & s) { if (e) *e = s; return false; }

inline bool read_manifest(const std::filesystem::path & p, std::string * text, std::string * e) {
    std::error_code ec; const auto n = std::filesystem::file_size(p, ec);
    if (ec || n > 16u * 1024u * 1024u) return fail(e, "Legacy YuE2 manifest is missing or exceeds 16 MiB");
    std::ifstream f(p, std::ios::binary);
    if (!f) return fail(e, "cannot open Legacy YuE2 manifest");
    text->assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    if (f.bad())
        return fail(e, "cannot read Legacy YuE2 manifest");
    return true;
}

inline bool str(yyjson_val * v, std::string * out, bool nonempty = false) {
    if (!v || !yyjson_is_str(v) || yyjson_get_len(v) != std::strlen(yyjson_get_str(v))) return false;
    out->assign(yyjson_get_str(v), yyjson_get_len(v)); return !nonempty || !out->empty();
}

inline bool safe_rel(const std::filesystem::path & base, const std::string & name,
                     std::filesystem::path * out, std::string * e) {
    if (name.empty() || name.find('\0') != std::string::npos || name.find('\\') != std::string::npos)
        return fail(e, "Legacy cache path is not a UTF-8 slash-relative path");
    const auto rel = std::filesystem::u8path(name);
    if (rel.is_absolute() || rel.has_root_name() || rel.has_root_directory()) return fail(e, "Legacy cache path is absolute");
    for (const auto & part : rel) if (part == std::filesystem::path("..")) return fail(e, "Legacy cache path escapes its manifest");
    std::error_code a, b; const auto root = std::filesystem::weakly_canonical(base, a);
    const auto file = std::filesystem::weakly_canonical(base / rel, b);
    if (a || b || !std::filesystem::is_regular_file(file, b)) return fail(e, "Legacy cache payload is missing or not a file");
    auto i = root.begin(), j = file.begin(); for (; i != root.end() && j != file.end() && *i == *j; ++i, ++j) {}
    if (i != root.end()) return fail(e, "Legacy cache payload escapes its manifest");
    *out = file; return true;
}

inline bool read_words5(const std::filesystem::path & path, int64_t lyric_chars,
                        std::vector<float> * out, std::string * e) {
    std::error_code ec; const auto bytes = std::filesystem::file_size(path, ec);
    if (ec || bytes == 0 || bytes > 512u * 1024u * 1024u || bytes % 20 != 0)
        return fail(e, "cursor_words must be a nonempty f32 [words,5] file");
    std::ifstream f(path, std::ios::binary);
    if (!f) return fail(e, "cannot open cursor_words file");
    out->resize(static_cast<size_t>(bytes / sizeof(float)));
    if (!f.read(reinterpret_cast<char *>(out->data()), static_cast<std::streamsize>(bytes)))
        return fail(e, "cursor_words read failed");
    float previous = -1.0f;
    for (size_t i = 0; i < out->size() / 5; ++i) {
        const float * row = out->data() + i * 5;
        if (!std::isfinite(row[0]) || !std::isfinite(row[1]) || !std::isfinite(row[2]) ||
            !std::isfinite(row[3]) || !std::isfinite(row[4]) || row[0] < 0 || row[1] < row[0] ||
            row[3] < 0 || row[4] < row[3] || row[4] > lyric_chars || row[0] < previous)
            return fail(e, "cursor_words contains an invalid time or codepoint span");
        previous = row[0];
    }
    return true;
}

inline int64_t utf8_codepoints(const std::string & text) {
    int64_t count = 0; for (unsigned char c : text) if ((c & 0xC0) != 0x80) ++count; return count;
}

inline bool lyric_token_ends(const BPETokenizer * tok, const std::vector<int> & ids,
                             size_t begin, std::vector<int64_t> * ends, std::string * e) {
    std::string byte2str[256]; build_byte_encoder(byte2str);
    std::unordered_map<std::string, uint8_t> byte_of;
    for (int i = 0; i < 256; ++i) byte_of[byte2str[i]] = static_cast<uint8_t>(i);
    int64_t cp = 0; ends->clear();
    for (size_t i = begin; i < ids.size(); ++i) {
        if (ids[i] < 0 || static_cast<size_t>(ids[i]) >= tok->id_to_str.size()) return fail(e, "tokenizer produced an invalid lyric token");
        const std::string & token = tok->id_to_str[static_cast<size_t>(ids[i])];
        for (size_t p = 0; p < token.size();) {
            int adv = 1; utf8_codepoint(token.c_str() + p, &adv);
            auto it = byte_of.find(token.substr(p, static_cast<size_t>(adv)));
            if (it != byte_of.end() && ((it->second & 0xC0u) != 0x80u)) ++cp;
            p += static_cast<size_t>(adv);
        }
        ends->push_back(cp);
    }
    return true;
}

inline bool tokenizer_hash(const std::filesystem::path & path, sha256::digest * out, std::string * e) {
    if (std::filesystem::is_regular_file(path)) return sha256::file(path, *out, e);
    std::vector<uint8_t> vocab, merges;
    if (!prepare_detail::read_file(path / "vocab.json", &vocab, e) ||
        !prepare_detail::read_file(path / "merges.txt", &merges, e)) return false;
    std::vector<uint8_t> joined; joined.reserve(vocab.size() + merges.size());
    joined.insert(joined.end(), vocab.begin(), vocab.end()); joined.insert(joined.end(), merges.begin(), merges.end());
    *out = sha256::bytes(joined.data(), joined.size()); return true;
}

} // namespace detail

// Converts one complete existing Legacy cache. The manifest must already have
// source-level latents, codec_ids and successful abc fields; partial caches are
// rejected rather than silently training a different recipe.
inline bool prepare_from_legacy(const Request & request, std::string * error = nullptr) {
    using namespace detail;
    if (request.legacy_manifest.empty() || request.raw_convrot_checkpoint.empty() ||
        request.tokenizer_gguf_or_dir.empty() || request.output_dir.empty() || request.models.empty())
        return fail(error, "native import requires Legacy manifest, raw ConvRot checkpoint, tokenizer and model provenance");
    std::error_code ec;
    if (std::filesystem::exists(request.output_dir, ec) || ec ||
        !std::filesystem::is_regular_file(request.legacy_manifest) ||
        !std::filesystem::is_regular_file(request.raw_convrot_checkpoint))
        return fail(error, "Legacy manifest/checkpoint is missing or output directory already exists");

    std::string text; if (!read_manifest(request.legacy_manifest, &text, error)) return false;
    yyjson_doc * doc = yyjson_read(text.data(), text.size(), 0);
    if (!doc) return fail(error, "Legacy YuE2 manifest is invalid JSON");
    struct Guard { yyjson_doc * d; ~Guard() { yyjson_doc_free(d); } } guard{doc};
    yyjson_val * root = yyjson_doc_get_root(doc), * sources = root ? yyjson_obj_get(root, "sources") : nullptr;
    if (!yyjson_is_obj(root) || !yyjson_is_arr(sources) || yyjson_arr_size(sources) == 0 || yyjson_arr_size(sources) > 10000)
        return fail(error, "Legacy YuE2 manifest has no sources[] array");
    if (!dataset_detail::unique_keys(root)) return fail(error, "Legacy YuE2 manifest contains duplicate keys or embedded NUL");
    std::string format, layout, index, dtype;
    const auto format_ok = str(yyjson_obj_get(root, "format"), &format) && format == "yue2-preprocess-v1";
    const auto layout_ok = str(yyjson_obj_get(root, "latent_layout"), &layout) && layout == "frame_major";
    const auto index_ok = str(yyjson_obj_get(root, "latent_index"), &index) && index == "t * latent_dim + c";
    const auto dtype_ok = str(yyjson_obj_get(root, "latent_dtype"), &dtype) && dtype == "f32";
    std::string caption_format;
    const auto caption_root = yyjson_obj_get(root, "caption_format");
    if (caption_root && !str(caption_root, &caption_format)) return fail(error, "caption_format must be a string");
    if (caption_format.empty()) caption_format = "plain";
    const auto caption_ok = caption_format == "ace-sidecar" || caption_format == "plain" || caption_format == "none";
    const auto fps = yyjson_obj_get(root, "frame_rate");
    if (!format_ok || !caption_ok || !layout_ok || !index_ok || !dtype_ok || !yyjson_is_num(fps) || yyjson_get_num(fps) != 25.0 ||
        !yyjson_is_int(yyjson_obj_get(root, "latent_dim")) || yyjson_get_sint(yyjson_obj_get(root, "latent_dim")) != 64)
        return fail(error, "Legacy manifest is not the required 25 Hz frame-major F32 [frames,64] cache");

    BPETokenizer tokenizer{};
    const auto token_path = request.tokenizer_gguf_or_dir;
    const bool loaded = token_path.extension() == ".gguf"
        ? yue2_tokenizer_load_from_gguf(&tokenizer, token_path.u8string())
        : yue2_tokenizer_load_from_dir(&tokenizer, token_path.u8string());
    if (!loaded) return fail(error, "cannot load YuE2 BPE tokenizer from GGUF or vocab.json/merges.txt");

    const auto base = request.legacy_manifest.parent_path();
    sha256::digest tokenizer_digest;
    if (!tokenizer_hash(request.tokenizer_gguf_or_dir, &tokenizer_digest, error)) return false;
    std::vector<VerifiedNativeItem> items;
    size_t i = 0, max = 0; yyjson_val * source = nullptr;
    yyjson_arr_foreach(sources, i, max, source) {
        if (!yyjson_is_obj(source)) return fail(error, "Legacy sources[] contains a non-object");
        std::string id, latent_name, semantic_name, style, lyrics, abc, abc_error;
        yyjson_val * frames = yyjson_obj_get(source, "frames");
        if (!str(yyjson_obj_get(source, "name"), &id, true)) str(yyjson_obj_get(source, "source"), &id, true);
        if (id.empty() || !str(yyjson_obj_get(source, "latents"), &latent_name, true) ||
            !str(yyjson_obj_get(source, "codec_ids"), &semantic_name, true) ||
            !yyjson_is_int(frames) || yyjson_get_sint(frames) <= 0 || yyjson_get_sint(frames) > 24576)
            return fail(error, "source lacks full-song latents, codec_ids or valid frames; run all YuE2 cache stages");
        const auto caption_value = yyjson_obj_get(source, "caption");
        const auto lyrics_value = yyjson_obj_get(source, "lyrics");
        const auto abc_value = yyjson_obj_get(source, "abc");
        const auto abc_error_value = yyjson_obj_get(source, "abc_error");
        if ((caption_value && !str(caption_value, &style)) || (lyrics_value && !str(lyrics_value, &lyrics)) ||
            (abc_value && !str(abc_value, &abc)) || (abc_error_value && !str(abc_error_value, &abc_error)))
            return fail(error, "Legacy caption, lyrics, ABC and ABC error fields must be strings");
        if (caption_format == "plain" && style.find("lyrics:") != std::string::npos)
            return fail(error, "plain caption contains a lyrics: section; rerun preprocessing with caption_mode=ace");
        if (caption_format != "none") {
            std::string normalized_style, normalized_lyrics, parse_error;
            const std::string assembled = caption_format == "plain" && lyrics.empty() ? style : "[Tags]\n" + style + "\n[Lyrics]\n" + lyrics;
            if (!yue2_aitk_import::detail::parse_caption(assembled, &normalized_style, &normalized_lyrics, &parse_error))
                return fail(error, "ace-sidecar caption/lyrics failed YuE2 normalization: " + parse_error);
            style = std::move(normalized_style); lyrics = std::move(normalized_lyrics);
        } else if (caption_format == "none") {
            style.clear(); lyrics.clear();
        }
        if (!abc_error.empty() || abc.empty()) return fail(error, "AITK native import requires a successful full ABC sheet for every source");
        std::filesystem::path latent, semantic;
        if (!safe_rel(base, latent_name, &latent, error) || !safe_rel(base, semantic_name, &semantic, error)) return false;
        const size_t n = static_cast<size_t>(yyjson_get_sint(frames));
        std::vector<int> abc_ids = yue2_bpe_encode(&tokenizer, abc);
        auto full = yue2_token_prefixes(&tokenizer, style, lyrics, YUE2_COT_FULL, nullptr);
        auto off = yue2_token_prefixes(&tokenizer, style, lyrics, YUE2_COT_OFF, nullptr);
        if (full.empty() || full.back() != YUE2_ABC_START || off.size() < 3 ||
            off[off.size() - 2] != YUE2_ABC_END || off.back() != YUE2_MUSIC_START)
            return fail(error, "YuE2 tokenizer produced an unexpected prefix tail");
        for (const int id : abc_ids) if (id < 0 || id >= YUE2_EOD) return fail(error, "ABC BPE ID is outside the ordinary YuE2 vocabulary");
        off.resize(off.size() - 2); // PromptInput prefixes end at ABC_START.
        VerifiedNativeItem item; item.id = id; item.latent_file = latent; item.semantic_file = semantic; item.frames = n;
        item.prompt_style = style; item.prompt_lyrics = lyrics; item.instrumental = lyrics.empty();
        if (request.lyric_timing && lyrics.empty()) {
            item.prompt.cursor.present = true;
            item.prompt.cursor.enabled = false;
            item.prompt.cursor.instrumental = true;
        }
        item.prompt.retained_prefix_ids.assign(full.begin(), full.end());
        item.prompt.dropped_prefix_ids.assign(off.begin(), off.end());
        item.prompt.abc_ids.assign(abc_ids.begin(), abc_ids.end()); item.prompt.retain_abc = true;
        item.prompt_style = style; item.prompt_lyrics = lyrics; item.instrumental = lyrics.empty();
        std::string cursor_name;
        const yyjson_val * cursor_value = yyjson_obj_get(source, "cursor_words");
        if (cursor_value && !str(const_cast<yyjson_val *>(cursor_value), &cursor_name, true))
            return fail(error, "source cursor_words must be a manifest-relative path");
        if (request.lyric_timing && !lyrics.empty() && cursor_name.empty())
            return fail(error, "--lyric-timing requires cursor_words for every lyrical source");
        if (request.lyric_timing && !cursor_name.empty()) {
            if (lyrics.empty()) return fail(error, "cursor_words is invalid for an instrumental source");
            std::filesystem::path cursor_path;
            if (!safe_rel(base, cursor_name, &cursor_path, error)) return false;
            auto & cm = item.prompt.cursor;
            cm.present = true; cm.enabled = request.lyric_timing; cm.instrumental = false;
            cm.lyric_codepoints = utf8_codepoints(lyrics);
            if (!read_words5(cursor_path, cm.lyric_codepoints, &cm.words5, error)) return false;
            const std::string full_text = std::string(yue2_instruction(YUE2_COT_FULL)) + "\n[Tags]\n" + style + "\n[Lyrics]\n" + lyrics;
            const std::string off_text = std::string(yue2_instruction(YUE2_COT_OFF)) + "\n[Tags]\n" + style + "\n[Lyrics]\n" + lyrics;
            const auto full_head = yue2_bpe_encode(&tokenizer, std::string(yue2_instruction(YUE2_COT_FULL)) + "\n[Tags]\n" + style + "\n[Lyrics]\n");
            const auto off_head = yue2_bpe_encode(&tokenizer, std::string(yue2_instruction(YUE2_COT_OFF)) + "\n[Tags]\n" + style + "\n[Lyrics]\n");
            const auto full_ids = yue2_bpe_encode(&tokenizer, full_text);
            const auto off_ids = yue2_bpe_encode(&tokenizer, off_text);
            auto prefix_matches = [](const std::vector<int32_t> & prefix, const std::vector<int> & ids) {
                if (prefix.size() < ids.size() + 1) return false;
                for (size_t k = 0; k < ids.size(); ++k) if (prefix[k + 1] != ids[k]) return false;
                return true;
            };
            if (!prefix_matches(item.prompt.retained_prefix_ids, full_ids) ||
                !prefix_matches(item.prompt.dropped_prefix_ids, off_ids))
                return fail(error, "cursor prefix does not match normalized tokenizer text");
            if (!lyric_token_ends(&tokenizer, full_ids, full_head.size(), &cm.full_lyric_token_end_codepoints, error) ||
                !lyric_token_ends(&tokenizer, off_ids, off_head.size(), &cm.off_lyric_token_end_codepoints, error)) return false;
            if (!bind_cursor_targets(item.prompt.retained_prefix_ids, static_cast<int64_t>(full_head.size()),
                                     cm.full_lyric_token_end_codepoints, cm.lyric_codepoints, n, cm.words5, &cm.full) ||
                !bind_cursor_targets(item.prompt.dropped_prefix_ids, static_cast<int64_t>(off_head.size()),
                                     cm.off_lyric_token_end_codepoints, cm.lyric_codepoints, n, cm.words5, &cm.off))
                return fail(error, cm.full.why.empty() ? cm.off.why.c_str() : cm.full.why.c_str());
            cm.lyrics_sha256 = sha256::bytes(reinterpret_cast<const uint8_t *>(lyrics.data()), lyrics.size()).hex();
            cm.tokenizer_sha256 = tokenizer_digest.hex(); cm.full_head_tokens = static_cast<int64_t>(full_head.size());
            cm.off_head_tokens = static_cast<int64_t>(off_head.size()); cm.j0_full = cm.full.j0; cm.j0_off = cm.off.j0; cm.L = cm.full.L;
            if (cm.off.L != cm.L) return fail(error, "cursor full/off token geometry differs");
            for (size_t wi = 0; wi < cm.words5.size() / 5; ++wi)
                if (cm.words5[wi * 5 + 1] > static_cast<float>(n) / 25.0f + 1.0f / 25.0f)
                    return fail(error, "cursor_words extends beyond the source frame duration");
            cursor_ranges(cm.full, &cm.full_frame_ranges); cursor_ranges(cm.off, &cm.off_frame_ranges);
            cm.full.T.clear(); cm.off.T.clear();
        }
        items.push_back(std::move(item));
    }

    const auto output_parent = request.output_dir.parent_path().empty() ? std::filesystem::current_path() : request.output_dir.parent_path();
    std::unordered_set<std::string> roles;
    for (const auto & model : request.models) {
        if (model.role.empty() || !roles.insert(model.role).second || !std::filesystem::is_regular_file(model.path))
            return fail(error, "model provenance must contain unique regular files");
    }
    if (roles.size() != 3 || roles.count("vae") == 0 || roles.count("semantic") == 0 || roles.count("sheetsage") == 0)
        return fail(error, "model provenance must include vae, semantic and sheetsage roles");
    if (std::filesystem::exists(request.output_dir, ec) || ec || !std::filesystem::is_directory(output_parent))
        return fail(error, "native import output must be fresh and have an existing parent");
    PrepareRequest prep; prep.output_dir = request.output_dir; prep.source_manifest = request.legacy_manifest;
    prep.base_checkpoint = request.raw_convrot_checkpoint; prep.verified_contract = "legacy-yue2-cache-v1;fullsong-f32-latents;raw-codec-ids;abc-bpe";
    prep.items = std::move(items);
    if (std::filesystem::is_regular_file(request.tokenizer_gguf_or_dir)) {
        prep.models.push_back({"tokenizer", request.tokenizer_gguf_or_dir});
    } else {
        const auto vocab = request.tokenizer_gguf_or_dir / "vocab.json";
        const auto merges = request.tokenizer_gguf_or_dir / "merges.txt";
        if (!std::filesystem::is_regular_file(vocab) || !std::filesystem::is_regular_file(merges)) {
            return fail(error, "tokenizer directory must contain vocab.json and merges.txt");
        }
        prep.models.push_back({"tokenizer-vocab", vocab});
        prep.models.push_back({"tokenizer-merges", merges});
    }
    for (const auto & model : request.models) prep.models.push_back({model.role, model.path});
    return prepare_dataset(prep, error);
}

// CLI proposal:
// ace-train yue2-prepare-aitk --legacy-manifest yue2_preprocess.json
//   --checkpoint rawConvRot.safetensors --tokenizer vocab-or-gguf
//   --model role=path ... --output fresh-dir [--lyric-timing 0|1]

} // namespace yue2_aitk_native_import
