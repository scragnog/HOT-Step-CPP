#pragma once

// Read-only manifest validator for the AI Toolkit Comfy YuE2 ConvRot base.
// This owns only the safetensors mmap and header records. It never creates a
// GGML tensor and never reads a complete tensor payload.
// It intentionally has no embedded-table dequantization stage: callers must
// handle the special ConvRot embedding policy explicitly after validation.

#include "../safetensors.h"
#include "../hot-step-fsutf8.h"
#include "../../vendor/yyjson/yyjson.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

struct Yue2AitkCheckpointRecord {
    std::string name;
    uint32_t rows = 0;
    uint32_t cols = 0;
    const int8_t * weight_i8 = nullptr;
    const float * row_scales_f32 = nullptr;
    const uint8_t * marker_json = nullptr;
    size_t weight_bytes = 0;
    size_t scale_bytes = 0;
    size_t marker_bytes = 0;
};

class Yue2AitkCheckpoint {
public:
    Yue2AitkCheckpoint() = default;
    ~Yue2AitkCheckpoint() { close_mapped(); }
    Yue2AitkCheckpoint(const Yue2AitkCheckpoint &) = delete;
    Yue2AitkCheckpoint & operator=(const Yue2AitkCheckpoint &) = delete;

    bool open(const char * path, std::string * error = nullptr) {
        if (!path || !*path) return set_error(error, "checkpoint path is empty");
        close_mapped();
        records_.clear();
        std::string why;
        if (!open_mapped(path, &why) || !validate_file(&why)) { close_mapped(); records_.clear(); return set_error(error, why); }
        return true;
    }

    const std::vector<Yue2AitkCheckpointRecord> & records() const { return records_; }
    const STFile & mapped_file() const { return file_; }

private:
    STFile file_{};
    std::vector<Yue2AitkCheckpointRecord> records_;
    bool resources_open_ = false;

    static bool set_error(std::string * out, const std::string & text) {
        if (out) *out = text;
        return false;
    }

    void close_mapped() {
        if (resources_open_) st_close(&file_);
        else file_ = STFile{};
        resources_open_ = false;
    }

    static bool read_u64(yyjson_val * v, uint64_t * out) {
        if (!v || !yyjson_is_uint(v)) return false;
        *out = yyjson_get_uint(v);
        return true;
    }

    bool open_mapped(const char * path, std::string * why) {
        file_ = STFile{};
#ifdef _WIN32
        file_.fh = INVALID_HANDLE_VALUE;
        const std::wstring wpath = hs_widen(path);
        file_.fh = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_.fh == INVALID_HANDLE_VALUE) { *why = "unable to open checkpoint"; return false; }
        resources_open_ = true;
        LARGE_INTEGER li{};
        if (!GetFileSizeEx(file_.fh, &li) || li.QuadPart < 0 ||
            static_cast<unsigned long long>(li.QuadPart) > (std::numeric_limits<size_t>::max)()) {
            *why = "unable to read checkpoint size"; close_mapped(); return false;
        }
        file_.file_size = static_cast<size_t>(li.QuadPart);
        file_.mh = CreateFileMappingW(file_.fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!file_.mh) { *why = "unable to map checkpoint"; close_mapped(); return false; }
        file_.mapping = static_cast<uint8_t *>(MapViewOfFile(file_.mh, FILE_MAP_READ, 0, 0, 0));
        if (!file_.mapping) { *why = "unable to map checkpoint view"; close_mapped(); return false; }
#else
        file_.fd = ::open(path, O_RDONLY);
        if (file_.fd < 0) { *why = "unable to open checkpoint"; return false; }
        resources_open_ = true;
        struct stat sb{};
        if (fstat(file_.fd, &sb) != 0 || sb.st_size < 0) { *why = "unable to read checkpoint size"; close_mapped(); return false; }
        file_.file_size = static_cast<size_t>(sb.st_size);
        file_.mapping = static_cast<uint8_t *>(mmap(nullptr, file_.file_size, PROT_READ, MAP_PRIVATE, file_.fd, 0));
        if (file_.mapping == MAP_FAILED) { file_.mapping = nullptr; *why = "unable to map checkpoint"; close_mapped(); return false; }
#endif
        if (file_.file_size < sizeof(uint64_t)) { *why = "checkpoint is shorter than header length"; return false; }
        uint64_t header_len = 0;
        std::memcpy(&header_len, file_.mapping, sizeof(header_len));
        if (header_len > file_.file_size - sizeof(uint64_t)) { *why = "header length exceeds file"; return false; }
        file_.data_offset = sizeof(uint64_t) + static_cast<size_t>(header_len);
        yyjson_doc * doc = yyjson_read(reinterpret_cast<const char *>(file_.mapping + sizeof(uint64_t)),
                                       static_cast<size_t>(header_len), 0);
        if (!doc) { *why = "truncated or invalid safetensors header JSON"; return false; }
        yyjson_val * root = yyjson_doc_get_root(doc);
        if (!root || !yyjson_is_obj(root)) { yyjson_doc_free(doc); *why = "safetensors header is not an object"; return false; }
        std::unordered_set<std::string> names;
        yyjson_obj_iter it = yyjson_obj_iter_with(root);
        yyjson_val * key = nullptr;
        while ((key = yyjson_obj_iter_next(&it))) {
            const char * name = yyjson_get_str(key);
            yyjson_val * value = yyjson_obj_iter_get_val(key);
            if (!name || !*name || !names.insert(name).second) { yyjson_doc_free(doc); *why = "duplicate or invalid tensor name"; return false; }
            if (std::strcmp(name, "__metadata__") == 0) continue;
            if (!value || !yyjson_is_obj(value)) { yyjson_doc_free(doc); *why = "tensor entry is not an object"; return false; }
            STEntry entry{};
            entry.name = name;
            bool have_dtype = false, have_shape = false, have_offsets = false;
            std::unordered_set<std::string> fields;
            yyjson_obj_iter fi = yyjson_obj_iter_with(value);
            yyjson_val * fk = nullptr;
            while ((fk = yyjson_obj_iter_next(&fi))) {
                const char * field = yyjson_get_str(fk);
                yyjson_val * fv = yyjson_obj_iter_get_val(fk);
                if (!field || !fields.insert(field).second) { yyjson_doc_free(doc); *why = "duplicate tensor field: " + entry.name; return false; }
                if (std::strcmp(field, "dtype") == 0) {
                    if (!fv || !yyjson_is_str(fv)) { yyjson_doc_free(doc); *why = "dtype is not a string: " + entry.name; return false; }
                    entry.dtype = yyjson_get_str(fv); have_dtype = true;
                } else if (std::strcmp(field, "shape") == 0) {
                    if (!fv || !yyjson_is_arr(fv)) { yyjson_doc_free(doc); *why = "shape is not an array: " + entry.name; return false; }
                    yyjson_arr_iter ai = yyjson_arr_iter_with(fv); yyjson_val * av = nullptr;
                    while ((av = yyjson_arr_iter_next(&ai))) {
                        if (entry.n_dims >= 4) { yyjson_doc_free(doc); *why = "rank exceeds four: " + entry.name; return false; }
                        uint64_t dim = 0;
                        if (!read_u64(av, &dim) || dim > static_cast<uint64_t>((std::numeric_limits<int64_t>::max)())) { yyjson_doc_free(doc); *why = "invalid shape dimension: " + entry.name; return false; }
                        entry.shape[entry.n_dims++] = static_cast<int64_t>(dim);
                    }
                    have_shape = true;
                } else if (std::strcmp(field, "data_offsets") == 0) {
                    if (!fv || !yyjson_is_arr(fv)) { yyjson_doc_free(doc); *why = "data_offsets is not an array: " + entry.name; return false; }
                    yyjson_arr_iter ai = yyjson_arr_iter_with(fv); yyjson_val * av = nullptr; uint64_t off[2]{}; int n = 0;
                    while ((av = yyjson_arr_iter_next(&ai))) { if (n >= 2 || !read_u64(av, &off[n])) { yyjson_doc_free(doc); *why = "invalid data_offsets: " + entry.name; return false; } ++n; }
                    if (n != 2 || off[1] < off[0] || off[1] > static_cast<uint64_t>((std::numeric_limits<size_t>::max)())) { yyjson_doc_free(doc); *why = "invalid data_offsets: " + entry.name; return false; }
                    entry.data_start = static_cast<size_t>(off[0]); entry.data_end = static_cast<size_t>(off[1]); have_offsets = true;
                } else { const std::string bad_field = field ? field : ""; yyjson_doc_free(doc); *why = "unknown tensor field: " + bad_field; return false; }
            }
            if (!have_dtype || !have_shape || !have_offsets || entry.n_dims == 0) { yyjson_doc_free(doc); *why = "incomplete tensor entry: " + entry.name; return false; }
            file_.entries.push_back(std::move(entry));
        }
        yyjson_doc_free(doc);
        return true;
    }

    static bool mul_size(size_t a, size_t b, size_t * out) {
        if (a && b > (std::numeric_limits<size_t>::max)() / a) return false;
        *out = a * b;
        return true;
    }

    static size_t dtype_size(const std::string & dtype) {
        if (dtype == "I8" || dtype == "U8" || dtype == "BOOL") return 1;
        if (dtype == "F16" || dtype == "BF16" || dtype == "I16") return 2;
        if (dtype == "F32" || dtype == "I32") return 4;
        if (dtype == "F64" || dtype == "I64") return 8;
        return 0;
    }

    static bool tensor_bytes(const STEntry & e, size_t * out) {
        const size_t elem = dtype_size(e.dtype);
        if (!elem || e.n_dims < 0 || e.n_dims > 4) return false;
        size_t count = 1;
        for (int i = 0; i < e.n_dims; ++i) {
            if (e.shape[i] < 0 || !mul_size(count, static_cast<size_t>(e.shape[i]), &count)) return false;
        }
        return mul_size(count, elem, out);
    }

    const STEntry * find(const std::string & name) const {
        return st_find(file_, name.c_str());
    }

    bool require_exact(const std::string & name, const char * dtype,
                       int64_t d0, int64_t d1, const STEntry ** out, std::string * why) const {
        const STEntry * e = find(name);
        if (!e) { *why = "missing tensor: " + name; return false; }
        if (e->dtype != dtype || e->n_dims != (d1 < 0 ? 1 : 2) || e->shape[0] != d0 ||
            (d1 >= 0 && e->shape[1] != d1)) {
            *why = "tensor shape/dtype mismatch: " + name;
            return false;
        }
        *out = e;
        return true;
    }

    bool validate_marker(const STEntry & marker, std::string * why) const {
        const void * raw = st_data(file_, marker);
        yyjson_doc * doc = yyjson_read(static_cast<const char *>(raw),
                                       marker.data_end - marker.data_start, 0);
        if (!doc) { *why = "invalid ConvRot marker JSON: " + marker.name; return false; }
        yyjson_val * root = yyjson_doc_get_root(doc);
        yyjson_val * format = root ? yyjson_obj_get(root, "format") : nullptr;
        yyjson_val * convrot = root ? yyjson_obj_get(root, "convrot") : nullptr;
        yyjson_val * group = root ? yyjson_obj_get(root, "convrot_groupsize") : nullptr;
        const bool ok = root && yyjson_is_obj(root) && format && yyjson_is_str(format) &&
                        std::strcmp(yyjson_get_str(format), "int8_tensorwise") == 0 &&
                        convrot && yyjson_is_bool(convrot) && yyjson_get_bool(convrot) &&
                        group && yyjson_is_int(group) && yyjson_get_int(group) == 256;
        yyjson_doc_free(doc);
        if (!ok) { *why = "unsupported ConvRot marker: " + marker.name; return false; }
        return true;
    }

    bool validate_triple(const std::string & base, int64_t rows, int64_t cols, std::string * why) {
        const STEntry * marker = find(base + ".comfy_quant");
        const STEntry * weight = nullptr;
        const STEntry * scale = nullptr;
        if (!marker || marker->dtype != "U8" || marker->n_dims != 1 || marker->shape[0] <= 0) {
            *why = "missing or malformed ConvRot marker: " + base;
            return false;
        }
        if (!require_exact(base + ".weight", "I8", rows, cols, &weight, why) ||
            !require_exact(base + ".weight_scale", "F32", rows, 1, &scale, why) ||
            !validate_marker(*marker, why)) return false;
        const size_t weight_bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        const size_t scale_bytes = static_cast<size_t>(rows) * sizeof(float);
        Yue2AitkCheckpointRecord rec;
        rec.name = base;
        rec.rows = static_cast<uint32_t>(rows);
        rec.cols = static_cast<uint32_t>(cols);
        rec.weight_i8 = static_cast<const int8_t *>(st_data(file_, *weight));
        const uintptr_t scale_address = reinterpret_cast<uintptr_t>(st_data(file_, *scale));
        if (scale_address % alignof(float) != 0) { *why = "unaligned F32 scale payload: " + base; return false; }
        rec.row_scales_f32 = static_cast<const float *>(st_data(file_, *scale));
        rec.marker_json = static_cast<const uint8_t *>(st_data(file_, *marker));
        rec.weight_bytes = weight_bytes;
        rec.scale_bytes = scale_bytes;
        rec.marker_bytes = marker->data_end - marker->data_start;
        records_.push_back(rec);
        return true;
    }

    bool validate_file(std::string * why) {
        if (file_.data_offset > file_.file_size) { *why = "header exceeds file"; return false; }
        const size_t payload = file_.file_size - file_.data_offset;
        std::vector<std::pair<size_t, size_t>> ranges;
        ranges.reserve(file_.entries.size());
        for (const STEntry & e : file_.entries) {
            if (e.data_end < e.data_start || e.data_end - e.data_start > payload || e.data_end > payload) {
                *why = "tensor offset outside payload: " + e.name; return false;
            }
            size_t expected = 0;
            if (!tensor_bytes(e, &expected) || expected != e.data_end - e.data_start) {
                *why = "tensor byte size mismatch: " + e.name; return false;
            }
            ranges.emplace_back(e.data_start, e.data_end);
        }
        std::sort(ranges.begin(), ranges.end());
        for (size_t i = 1; i < ranges.size(); ++i) {
            if (ranges[i].first < ranges[i - 1].second) {
                *why = "overlapping tensor offsets";
                return false;
            }
        }

        const char * sites[] = { "self_attn.qkv_proj", "self_attn.o_proj",
                                 "mlp.gate_up_proj", "mlp.down_proj" };
        const int64_t site_rows[] = { 4096, 2048, 12288, 2048 };
        const int64_t site_cols[] = { 2048, 2048, 2048, 6144 };
        const char * experts[] = { "text_encoders.model.layers", "model.diffusion_model.model.layers" };
        for (const char * expert : experts) {
            for (int layer = 0; layer < 28; ++layer) {
                for (int site = 0; site < 4; ++site) {
                    std::string base = std::string(expert) + "." + std::to_string(layer) + "." + sites[site];
                    if (!validate_triple(base, site_rows[site], site_cols[site], why)) return false;
                }
            }
        }
        const char * extra_names[] = {
            "text_encoders.model.embed_tokens", "text_encoders.model.lm_head",
            "model.diffusion_model.llm2vae", "model.diffusion_model.time_embedder.mlp.0",
            "model.diffusion_model.time_embedder.mlp.2" };
        const int64_t extra_rows[] = { 184704, 184704, 64, 2048, 2048 };
        const int64_t extra_cols[] = { 2048, 2048, 2048, 256, 2048 };
        for (size_t i = 0; i < sizeof(extra_names) / sizeof(extra_names[0]); ++i) {
            if (!validate_triple(extra_names[i], extra_rows[i], extra_cols[i], why)) return false;
        }
        size_t marker_count = 0;
        for (const STEntry & e : file_.entries) if (e.name.size() >= 12 && e.name.compare(e.name.size() - 12, 12, ".comfy_quant") == 0) marker_count++;
        if (marker_count != 229 || records_.size() != 229) {
            *why = "ConvRot inventory count mismatch";
            return false;
        }
        return true;
    }
};
