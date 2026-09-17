#pragma once

// Bounded YuE2 optimizer resume container. This is deliberately separate from
// the legacy safetensors/checkpoint readers: it carries optimizer moments,
// parameters, and opaque runner metadata in one versioned binary record.
#include "../hot-step-fsutf8.h"
#include "yue2-aitk-optimizer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace yue2_aitk {

struct ResumeRecord {
    HostStateSnapshot state;
    std::string runner_metadata;
};

namespace resume_detail {
constexpr uint32_t kVersion = 1;
constexpr size_t kMaxFileBytes = size_t(512) * 1024 * 1024;
constexpr size_t kMaxMetadataBytes = size_t(16) * 1024 * 1024;
constexpr size_t kMaxParameters = 65536;
constexpr size_t kMaxNameBytes = 4096;
constexpr std::array<uint8_t, 8> kMagic{{'Y','2','R','S','U','M','E','1'}};

inline bool add_size(size_t a, size_t b, size_t * out) { if (b > (std::numeric_limits<size_t>::max)() - a) return false; *out = a + b; return true; }
inline bool mul_size(size_t a, size_t b, size_t * out) { if (a && b > (std::numeric_limits<size_t>::max)() / a) return false; *out = a * b; return true; }
inline uint64_t hash_bytes(const uint8_t * data, size_t size, uint64_t hash = 1469598103934665603ull) { for (size_t i = 0; i < size; ++i) { hash ^= data[i]; hash *= 1099511628211ull; } return hash; }
inline void put_u32(std::vector<uint8_t> & out, uint32_t value) { for (int i = 0; i < 4; ++i) out.push_back(uint8_t(value >> (8 * i))); }
inline void put_u64(std::vector<uint8_t> & out, uint64_t value) { for (int i = 0; i < 8; ++i) out.push_back(uint8_t(value >> (8 * i))); }
inline bool get_u32(const std::vector<uint8_t> & in, size_t * pos, uint32_t * value) { if (*pos > in.size() || in.size() - *pos < 4) return false; *value = uint32_t(in[*pos]) | uint32_t(in[*pos + 1]) << 8 | uint32_t(in[*pos + 2]) << 16 | uint32_t(in[*pos + 3]) << 24; *pos += 4; return true; }
inline bool get_u64(const std::vector<uint8_t> & in, size_t * pos, uint64_t * value) { if (*pos > in.size() || in.size() - *pos < 8) return false; *value = 0; for (int i = 0; i < 8; ++i) *value |= uint64_t(in[*pos + i]) << (8 * i); *pos += 8; return true; }
inline void put_f32(std::vector<uint8_t> & out, float value) { uint32_t bits = 0; std::memcpy(&bits, &value, sizeof(bits)); put_u32(out, bits); }
inline bool get_f32(const std::vector<uint8_t> & in, size_t * pos, float * value) { uint32_t bits = 0; if (!get_u32(in, pos, &bits)) return false; std::memcpy(value, &bits, sizeof(bits)); return true; }
inline bool append_bytes(std::vector<uint8_t> & out, const void * data, size_t bytes) { const size_t old = out.size(); if (!add_size(old, bytes, &bytes)) return false; out.resize(bytes); if (bytes) std::memcpy(out.data() + old, data, bytes - old); return true; }
inline bool append_string(std::vector<uint8_t> & out, const std::string & value, size_t limit) { if (value.size() > limit || value.size() > (std::numeric_limits<uint32_t>::max)()) return false; put_u32(out, uint32_t(value.size())); return append_bytes(out, value.data(), value.size()); }
inline bool read_bytes(const std::vector<uint8_t> & in, size_t * pos, const uint8_t ** data, size_t bytes) { if (*pos > in.size() || bytes > in.size() - *pos) return false; *data = in.data() + *pos; *pos += bytes; return true; }
inline bool read_string(const std::vector<uint8_t> & in, size_t * pos, std::string * value, size_t limit) { uint32_t size = 0; const uint8_t * data = nullptr; if (!get_u32(in, pos, &size) || size > limit || !read_bytes(in, pos, &data, size)) return false; value->assign(reinterpret_cast<const char *>(data), size); return true; }
inline std::string temporary_path(const char * path) { static std::atomic<uint64_t> serial{0};
#ifdef _WIN32
    const uint64_t pid = static_cast<uint64_t>(GetCurrentProcessId());
#else
    const uint64_t pid = static_cast<uint64_t>(::getpid());
#endif
    return std::string(path) + ".__yue2_resume_writing__" + std::to_string(pid) + "_" + std::to_string(serial.fetch_add(1, std::memory_order_relaxed)); }
inline bool publish_no_replace(const std::string & temp, const char * path) {
#ifdef _WIN32
    return MoveFileExW(hs_widen(temp).c_str(), hs_widen(path).c_str(), MOVEFILE_WRITE_THROUGH) != 0;
#else
    return ::link(temp.c_str(), path) == 0 && ::unlink(temp.c_str()) == 0;
#endif
}
inline bool read_file(const char * path, std::vector<uint8_t> * out) { FILE * file = hs_fopen(path, "rb"); if (!file) return false; if (std::fseek(file, 0, SEEK_END) != 0) { fclose(file); return false; } const long end = std::ftell(file); if (end < 0 || size_t(end) > kMaxFileBytes) { fclose(file); return false; } if (std::fseek(file, 0, SEEK_SET) != 0) { fclose(file); return false; } out->resize(size_t(end)); const bool ok = out->empty() || std::fread(out->data(), 1, out->size(), file) == out->size(); fclose(file); return ok; }
}

inline bool yue2_aitk_write_resume(const char * path, const HostStateSnapshot & state, const std::string & runner_metadata) {
    using namespace resume_detail;
    if (!path || !*path || state.step < 0 || runner_metadata.size() > kMaxMetadataBytes || state.names.size() == 0 || state.names.size() > kMaxParameters || state.names.size() != state.elements.size() || state.names.size() != state.parameters.size() || state.names.size() != state.state1_fp32.size() || state.names.size() != state.state2_fp32.size() || state.names.size() != state.state1_u8.size() || state.names.size() != state.state2_u8.size() || state.names.size() != state.absmax1.size() || state.names.size() != state.absmax2.size()) return false;
    size_t planned = 8;
    for (size_t i = 0; i < state.names.size(); ++i) {
        const size_t n = state.elements[i]; const bool quantized = !state.state1_u8[i].empty() || !state.state2_u8[i].empty();
        size_t slot = 0, bytes = 0;
        if (n == 0 || !add_size(4 + state.names[i].size() + 8 + 1, 0, &slot) || !mul_size(n, quantized ? 6 : 12, &bytes) || !add_size(slot, bytes, &slot) || (quantized && (!mul_size((n + 255) / 256, 8, &bytes) || !add_size(slot, bytes, &slot))) || !add_size(planned, slot, &planned) || planned > kMaxFileBytes) return false;
    }
    std::vector<uint8_t> payload; payload.reserve(planned);
    put_u32(payload, uint32_t(state.step)); put_u32(payload, uint32_t(state.names.size()));
    std::unordered_set<std::string> names;
    for (size_t i = 0; i < state.names.size(); ++i) {
        const auto & name = state.names[i]; const size_t n = state.elements[i];
        if (name.empty() || name.size() > kMaxNameBytes || !names.insert(name).second || n == 0 || n > (std::numeric_limits<uint32_t>::max)() || state.parameters[i].size() != n) return false;
        for (float value : state.parameters[i]) if (!std::isfinite(value)) return false;
        const bool quantized = !state.state1_u8[i].empty() || !state.state2_u8[i].empty();
        const size_t blocks = (n + 255) / 256;
        if (quantized ? (n < 4096 || state.state1_u8[i].size() != n || state.state2_u8[i].size() != n || state.absmax1[i].size() != blocks || state.absmax2[i].size() != blocks || !state.state1_fp32[i].empty() || !state.state2_fp32[i].empty()) : (n >= 4096 || state.state1_fp32[i].size() != n || state.state2_fp32[i].size() != n || !state.state1_u8[i].empty() || !state.state2_u8[i].empty() || !state.absmax1[i].empty() || !state.absmax2[i].empty())) return false;
        if (!append_string(payload, name, kMaxNameBytes)) return false; put_u64(payload, uint64_t(n)); payload.push_back(quantized ? 1 : 0);
        for (float value : state.parameters[i]) put_f32(payload, value);
        if (quantized) { for (float value : state.absmax1[i]) if (!std::isfinite(value) || value < 0) return false; for (float value : state.absmax2[i]) if (!std::isfinite(value) || value < 0) return false; payload.insert(payload.end(), state.state1_u8[i].begin(), state.state1_u8[i].end()); payload.insert(payload.end(), state.state2_u8[i].begin(), state.state2_u8[i].end()); for (float value : state.absmax1[i]) put_f32(payload, value); for (float value : state.absmax2[i]) put_f32(payload, value); }
        else { for (float value : state.state1_fp32[i]) if (!std::isfinite(value)) return false; for (float value : state.state2_fp32[i]) if (!std::isfinite(value) || value < 0) return false; for (float value : state.state1_fp32[i]) put_f32(payload, value); for (float value : state.state2_fp32[i]) put_f32(payload, value); }
        if (payload.size() > kMaxFileBytes) return false;
    }
    std::vector<uint8_t> metadata(runner_metadata.begin(), runner_metadata.end());
    std::vector<uint8_t> file; file.insert(file.end(), kMagic.begin(), kMagic.end()); put_u32(file, kVersion); put_u64(file, uint64_t(payload.size())); put_u64(file, uint64_t(metadata.size())); put_u64(file, hash_bytes(payload.data(), payload.size(), hash_bytes(metadata.data(), metadata.size()))); put_u64(file, uint64_t(metadata.size())); file.insert(file.end(), metadata.begin(), metadata.end()); file.insert(file.end(), payload.begin(), payload.end());
    if (file.size() > kMaxFileBytes) return false;
    const std::string temp = temporary_path(path); FILE * output = hs_fopen(temp.c_str(), "wbx"); if (!output) return false; const bool written = std::fwrite(file.data(), 1, file.size(), output) == file.size() && std::fflush(output) == 0; const bool closed = std::fclose(output) == 0; if (!written || !closed || !publish_no_replace(temp, path)) { hs_remove(temp); return false; } return true;
}

inline bool yue2_aitk_read_resume(const char * path, ResumeRecord * out) {
    using namespace resume_detail;
    if (!out) return false; std::vector<uint8_t> file; if (!read_file(path, &file) || file.size() < 8 + 4 + 8 + 8 + 8 + 8 || !std::equal(kMagic.begin(), kMagic.end(), file.begin())) return false;
    size_t pos = 8; uint32_t version = 0; uint64_t payload_size = 0, metadata_size = 0, checksum = 0, metadata_prefix = 0; if (!get_u32(file, &pos, &version) || version != kVersion || !get_u64(file, &pos, &payload_size) || !get_u64(file, &pos, &metadata_size) || !get_u64(file, &pos, &checksum) || !get_u64(file, &pos, &metadata_prefix) || metadata_prefix != metadata_size || metadata_size > kMaxMetadataBytes || payload_size > kMaxFileBytes) return false;
    if (metadata_size > file.size() - pos) return false; const size_t payload_offset = pos + size_t(metadata_size); if (payload_size > file.size() - payload_offset || size_t(payload_size) != file.size() - payload_offset) return false; const uint64_t actual = hash_bytes(file.data() + payload_offset, size_t(payload_size), hash_bytes(file.data() + pos, size_t(metadata_size))); if (actual != checksum) return false;
    ResumeRecord parsed; parsed.runner_metadata.assign(reinterpret_cast<const char *>(file.data() + pos), size_t(metadata_size)); pos = payload_offset; const size_t end = file.size(); uint32_t step = 0, count = 0; if (!get_u32(file, &pos, &step) || step > uint32_t(INT_MAX) || !get_u32(file, &pos, &count) || !count || count > kMaxParameters) return false; HostStateSnapshot state; state.step = int(step); state.names.reserve(count); state.elements.reserve(count); state.parameters.resize(count); state.state1_fp32.resize(count); state.state2_fp32.resize(count); state.state1_u8.resize(count); state.state2_u8.resize(count); state.absmax1.resize(count); state.absmax2.resize(count); std::unordered_set<std::string> names;
    for (uint32_t i = 0; i < count; ++i) { std::string name; uint64_t n64 = 0; const uint8_t * raw = nullptr; uint8_t type = 0; if (!read_string(file, &pos, &name, kMaxNameBytes) || name.empty() || !names.insert(name).second || !get_u64(file, &pos, &n64) || !n64 || n64 > uint64_t((std::numeric_limits<uint32_t>::max)()) || !read_bytes(file, &pos, &raw, 1)) return false; type = *raw; if (type > 1) return false; const size_t n = size_t(n64); const size_t blocks = (n + 255) / 256; size_t required = 0, temp = 0; if (type == 0) { if (n >= 4096 || !mul_size(n, 8, &required)) return false; } else { if (n < 4096 || !mul_size(n, 2, &required) || !mul_size(blocks, 8, &temp) || !add_size(required, temp, &required)) return false; } if (!mul_size(n, 4, &temp) || !add_size(required, temp, &required) || required > end - pos) return false; state.names.push_back(name); state.elements.push_back(n); state.parameters[i].resize(n); for (float & value : state.parameters[i]) if (!get_f32(file, &pos, &value) || !std::isfinite(value)) return false; if (type == 0) { state.state1_fp32[i].resize(n); state.state2_fp32[i].resize(n); for (float & value : state.state1_fp32[i]) if (!get_f32(file, &pos, &value) || !std::isfinite(value)) return false; for (float & value : state.state2_fp32[i]) if (!get_f32(file, &pos, &value) || !std::isfinite(value) || value < 0) return false; } else { state.state1_u8[i].resize(n); state.state2_u8[i].resize(n); const uint8_t * q1 = nullptr; const uint8_t * q2 = nullptr; if (!read_bytes(file, &pos, &q1, n) || !read_bytes(file, &pos, &q2, n)) return false; std::memcpy(state.state1_u8[i].data(), q1, n); std::memcpy(state.state2_u8[i].data(), q2, n); state.absmax1[i].resize(blocks); state.absmax2[i].resize(blocks); for (float & value : state.absmax1[i]) if (!get_f32(file, &pos, &value) || !std::isfinite(value) || value < 0) return false; for (float & value : state.absmax2[i]) if (!get_f32(file, &pos, &value) || !std::isfinite(value) || value < 0) return false; } }
    if (pos != end) return false; parsed.state = std::move(state); *out = std::move(parsed); return true;
}

} // namespace yue2_aitk

