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
    int version = 1;  // 1 = AdamW8bit two-state format, 2 = LmOptim up-to-four-state format
};

namespace resume_detail {
constexpr uint32_t kVersion = 2;
constexpr size_t kMaxFileBytes = size_t(2) * 1024 * 1024 * 1024;
constexpr size_t kMaxMetadataBytes = size_t(16) * 1024 * 1024;
constexpr size_t kMaxParameters = 65536;
constexpr size_t kMaxNameBytes = 4096;
constexpr size_t kQuantizeBlock = 256;
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

// Version-2 state storage: symmetric 8-bit quantization per 256-element block
// with a per-block absmax, decoding to F32 on restore. This is plain storage
// compression for moments — unlike the v1 file it is NOT the AITK AdamW8bit
// qmap representation, and it only ever round-trips through this header.
inline bool quantize_f32(const std::vector<float> & in, std::vector<uint8_t> * q, std::vector<float> * absmax) {
    const size_t n = in.size();
    const size_t blocks = (n + kQuantizeBlock - 1) / kQuantizeBlock;
    q->assign(n, 0);
    absmax->assign(blocks, 0.0f);
    for (size_t b = 0; b < blocks; ++b) {
        const size_t lo = b * kQuantizeBlock;
        const size_t hi = std::min(n, lo + kQuantizeBlock);
        float am = 0.0f;
        for (size_t i = lo; i < hi; ++i) {
            if (!std::isfinite(in[i])) return false;
            const float a = in[i] < 0.0f ? -in[i] : in[i];
            if (a > am) am = a;
        }
        absmax->at(b) = am;
        if (am > 0.0f) {
            for (size_t i = lo; i < hi; ++i) {
                int v = int(std::lround((double) in[i] / (double) am * 127.0));
                if (v < -128) v = -128;
                if (v > 127) v = 127;
                q->at(i) = uint8_t(v + 128);
            }
        }
    }
    return true;
}

inline bool dequantize_f32(const std::vector<uint8_t> & q, const std::vector<float> & absmax, std::vector<float> * out) {
    const size_t n = q.size();
    const size_t blocks = (n + kQuantizeBlock - 1) / kQuantizeBlock;
    if (absmax.size() != blocks) return false;
    out->resize(n);
    for (size_t b = 0; b < blocks; ++b) {
        if (!std::isfinite(absmax[b]) || absmax[b] < 0.0f) return false;
        const double am = absmax[b];
        const size_t lo = b * kQuantizeBlock;
        const size_t hi = std::min(n, lo + kQuantizeBlock);
        for (size_t i = lo; i < hi; ++i) (*out)[i] = float((double)((int) q[i] - 128) / 127.0 * am);
    }
    return true;
}

// One version-2 state slot: absent, full F32 (n < 4096) or quantized (n >= 4096).
struct V2Slot { const std::vector<float> * fp32; const std::vector<uint8_t> * u8; const std::vector<float> * absmax; };
inline bool v2_slot_valid(const V2Slot & s, size_t n) {
    const bool has_fp32 = s.fp32 && !s.fp32->empty();
    const bool has_u8 = s.u8 && !s.u8->empty();
    if (has_fp32 && has_u8) return false;
    if (!has_fp32 && !has_u8) return true;  // absent slot
    if (has_fp32) {
        if (n >= 4096 || s.fp32->size() != n) return false;
        for (float value : *s.fp32) if (!std::isfinite(value)) return false;
        return true;
    }
    if (n < 4096 || s.u8->size() != n) return false;
    if (s.absmax->size() != (n + kQuantizeBlock - 1) / kQuantizeBlock) return false;
    for (float value : *s.absmax) if (!std::isfinite(value) || value < 0) return false;
    return true;
}

// Size of one v2 slot's file payload (length prefix included).
inline bool v2_slot_bytes(const V2Slot & s, size_t n, size_t * out) {
    size_t bytes = 4;
    if (!s.fp32->empty()) { if (!mul_size(n, 4, out)) return false; return add_size(bytes, *out, out); }
    if (!s.u8->empty()) { size_t partial = 0; if (!mul_size(n, 1, &partial)) return false; size_t am = 0; if (!mul_size((n + kQuantizeBlock - 1) / kQuantizeBlock, 4, &am)) return false; if (!add_size(partial, am, out)) return false; return add_size(bytes, *out, out); }
    *out = bytes;
    return true;
}
}

inline bool yue2_aitk_write_resume(const char * path, const HostStateSnapshot & state, const std::string & runner_metadata) {
    using namespace resume_detail;
    if (!path || !*path || state.step < 0 || runner_metadata.size() > kMaxMetadataBytes || state.names.size() == 0 || state.names.size() > kMaxParameters || state.names.size() != state.elements.size() || state.names.size() != state.parameters.size() || state.names.size() != state.state1_fp32.size() || state.names.size() != state.state2_fp32.size() || state.names.size() != state.state1_u8.size() || state.names.size() != state.state2_u8.size() || state.names.size() != state.absmax1.size() || state.names.size() != state.absmax2.size() || state.names.size() != state.state3_fp32.size() || state.names.size() != state.state4_fp32.size() || state.names.size() != state.state3_u8.size() || state.names.size() != state.state4_u8.size() || state.names.size() != state.absmax3.size() || state.names.size() != state.absmax4.size()) return false;
    for (size_t i = 0; i < state.names.size(); ++i) { if (!state.state3_fp32[i].empty() || !state.state4_fp32[i].empty() || !state.state3_u8[i].empty() || !state.state4_u8[i].empty() || !state.absmax3[i].empty() || !state.absmax4[i].empty()) return false; }
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

// Version-2 writer (LmOptim state). Parameters are always F32; each of the
// four state slots is absent, full F32 (n < 4096) or 8-bit quantized above.
inline bool yue2_aitk_write_resume_v2(const char * path, const HostStateSnapshot & state, const std::string & runner_metadata) {
    using namespace resume_detail;
    if (!path || !*path || state.step < 0 || runner_metadata.size() > kMaxMetadataBytes || state.names.size() == 0 || state.names.size() > kMaxParameters || state.names.size() != state.elements.size() || state.names.size() != state.parameters.size() || state.names.size() != state.state1_fp32.size() || state.names.size() != state.state2_fp32.size() || state.names.size() != state.state3_fp32.size() || state.names.size() != state.state4_fp32.size() || state.names.size() != state.state1_u8.size() || state.names.size() != state.state2_u8.size() || state.names.size() != state.state3_u8.size() || state.names.size() != state.state4_u8.size() || state.names.size() != state.absmax1.size() || state.names.size() != state.absmax2.size() || state.names.size() != state.absmax3.size() || state.names.size() != state.absmax4.size()) return false;
    std::vector<std::array<V2Slot, 4>> slotset;
    slotset.resize(state.names.size());
    std::unordered_set<std::string> names;
    size_t planned = 8;
    for (size_t i = 0; i < state.names.size(); ++i) {
        const size_t n = state.elements[i];
        if (state.names[i].empty() || state.names[i].size() > kMaxNameBytes || !names.insert(state.names[i]).second || n == 0 || n > (std::numeric_limits<uint32_t>::max)() || state.parameters[i].size() != n) return false;
        for (float value : state.parameters[i]) if (!std::isfinite(value)) return false;
        const V2Slot one{ &state.state1_fp32[i], state.state1_u8[i].empty() ? nullptr : &state.state1_u8[i], state.absmax1[i].empty() ? nullptr : &state.absmax1[i] };
        const V2Slot two{ &state.state2_fp32[i], state.state2_u8[i].empty() ? nullptr : &state.state2_u8[i], state.absmax2[i].empty() ? nullptr : &state.absmax2[i] };
        const V2Slot three{ &state.state3_fp32[i], state.state3_u8[i].empty() ? nullptr : &state.state3_u8[i], state.absmax3[i].empty() ? nullptr : &state.absmax3[i] };
        const V2Slot four{ &state.state4_fp32[i], state.state4_u8[i].empty() ? nullptr : &state.state4_u8[i], state.absmax4[i].empty() ? nullptr : &state.absmax4[i] };
        slotset[i] = { one, two, three, four };
        for (size_t k = 0; k < 4; ++k) if (!v2_slot_valid(slotset[i][k], n)) return false;
        size_t header = 0, params = 0, body = 0;
        if (!add_size(4, state.names[i].size(), &header) || !add_size(header, 8, &header) || !mul_size(n, 4, &params) || !add_size(header, params, &body)) return false;
        for (size_t k = 0; k < 4; ++k) { size_t part = 0; if (!v2_slot_bytes(slotset[i][k], n, &part) || !add_size(body, part, &body)) return false; }
        if (!add_size(planned, body, &planned) || planned > kMaxFileBytes) return false;
    }
    std::vector<uint8_t> payload; payload.reserve(planned);
    put_u32(payload, uint32_t(state.step)); put_u32(payload, uint32_t(state.names.size()));
    for (size_t i = 0; i < state.names.size(); ++i) {
        const size_t n = state.elements[i];
        if (!append_string(payload, state.names[i], kMaxNameBytes)) return false;
        put_u64(payload, uint64_t(n));
        for (float value : state.parameters[i]) put_f32(payload, value);
        for (size_t k = 0; k < 4; ++k) {
            const V2Slot & s = slotset[i][k];
            if (s.fp32 && !s.fp32->empty()) {
                put_u32(payload, uint32_t(n));
                for (float value : *s.fp32) put_f32(payload, value);
            } else if (s.u8 && !s.u8->empty()) {
                put_u32(payload, uint32_t(n));
                payload.insert(payload.end(), s.u8->begin(), s.u8->end());
                for (float value : *s.absmax) put_f32(payload, value);
            } else {
                put_u32(payload, 0);
            }
        }
        if (payload.size() > kMaxFileBytes) return false;
    }
    std::vector<uint8_t> metadata(runner_metadata.begin(), runner_metadata.end());
    std::vector<uint8_t> file; file.insert(file.end(), kMagic.begin(), kMagic.end()); put_u32(file, kVersion); put_u64(file, uint64_t(payload.size())); put_u64(file, uint64_t(metadata.size())); put_u64(file, hash_bytes(payload.data(), payload.size(), hash_bytes(metadata.data(), metadata.size()))); put_u64(file, uint64_t(metadata.size())); file.insert(file.end(), metadata.begin(), metadata.end()); file.insert(file.end(), payload.begin(), payload.end());
    if (file.size() > kMaxFileBytes) return false;
    const std::string temp = temporary_path(path); FILE * output = hs_fopen(temp.c_str(), "wbx"); if (!output) return false; const bool written = std::fwrite(file.data(), 1, file.size(), output) == file.size() && std::fflush(output) == 0; const bool closed = std::fclose(output) == 0; if (!written || !closed || !publish_no_replace(temp, path)) { hs_remove(temp); return false; } return true;
}

// Version-1 payload: step, count, then per tensor name, elements, one shared
// type byte, F32 parameters, and two state blocks (FP32 under 4096 elements,
// AITK 8-bit qmap quantization above).
inline bool read_resume_v1(const std::vector<uint8_t> & file, size_t pos, size_t end, HostStateSnapshot * state) {
    using namespace resume_detail;
    uint32_t step = 0, count = 0; if (!get_u32(file, &pos, &step) || step > uint32_t(INT_MAX) || !get_u32(file, &pos, &count) || !count || count > kMaxParameters) return false; state->step = int(step); state->names.reserve(count); state->elements.reserve(count); state->parameters.resize(count); state->state1_fp32.resize(count); state->state2_fp32.resize(count); state->state1_u8.resize(count); state->state2_u8.resize(count); state->absmax1.resize(count); state->absmax2.resize(count); state->state3_fp32.resize(count); state->state4_fp32.resize(count); state->state3_u8.resize(count); state->state4_u8.resize(count); state->absmax3.resize(count); state->absmax4.resize(count); std::unordered_set<std::string> names;
    for (uint32_t i = 0; i < count; ++i) { std::string name; uint64_t n64 = 0; const uint8_t * raw = nullptr; uint8_t type = 0; if (!read_string(file, &pos, &name, kMaxNameBytes) || name.empty() || !names.insert(name).second || !get_u64(file, &pos, &n64) || !n64 || n64 > uint64_t((std::numeric_limits<uint32_t>::max)()) || !read_bytes(file, &pos, &raw, 1)) return false; type = *raw; if (type > 1) return false; const size_t n = size_t(n64); const size_t blocks = (n + 255) / 256; size_t required = 0, temp = 0; if (type == 0) { if (n >= 4096 || !mul_size(n, 8, &required)) return false; } else { if (n < 4096 || !mul_size(n, 2, &required) || !mul_size(blocks, 8, &temp) || !add_size(required, temp, &required)) return false; } if (!mul_size(n, 4, &temp) || !add_size(required, temp, &required) || required > end - pos) return false; state->names.push_back(name); state->elements.push_back(n); state->parameters[i].resize(n); for (float & value : state->parameters[i]) if (!get_f32(file, &pos, &value) || !std::isfinite(value)) return false; if (type == 0) { state->state1_fp32[i].resize(n); state->state2_fp32[i].resize(n); for (float & value : state->state1_fp32[i]) if (!get_f32(file, &pos, &value) || !std::isfinite(value)) return false; for (float & value : state->state2_fp32[i]) if (!get_f32(file, &pos, &value) || !std::isfinite(value) || value < 0) return false; } else { state->state1_u8[i].resize(n); state->state2_u8[i].resize(n); const uint8_t * q1 = nullptr; const uint8_t * q2 = nullptr; if (!read_bytes(file, &pos, &q1, n) || !read_bytes(file, &pos, &q2, n)) return false; std::memcpy(state->state1_u8[i].data(), q1, n); std::memcpy(state->state2_u8[i].data(), q2, n); state->absmax1[i].resize(blocks); state->absmax2[i].resize(blocks); for (float & value : state->absmax1[i]) if (!get_f32(file, &pos, &value) || !std::isfinite(value) || value < 0) return false; for (float & value : state->absmax2[i]) if (!get_f32(file, &pos, &value) || !std::isfinite(value) || value < 0) return false; } }
    if (pos != end) return false; return true;
}

// Version-2 payload: step, count, then per tensor name, elements, F32
// parameters and four independent state slots, each absent or full.
inline bool read_resume_v2(const std::vector<uint8_t> & file, size_t pos, size_t end, HostStateSnapshot * state) {
    using namespace resume_detail;
    uint32_t step = 0, count = 0; if (!get_u32(file, &pos, &step) || step > uint32_t(INT_MAX) || !get_u32(file, &pos, &count) || !count || count > kMaxParameters) return false; state->step = int(step); state->names.reserve(count); state->elements.reserve(count); state->parameters.resize(count); state->state1_fp32.resize(count); state->state2_fp32.resize(count); state->state3_fp32.resize(count); state->state4_fp32.resize(count); state->state1_u8.resize(count); state->state2_u8.resize(count); state->state3_u8.resize(count); state->state4_u8.resize(count); state->absmax1.resize(count); state->absmax2.resize(count); state->absmax3.resize(count); state->absmax4.resize(count); std::unordered_set<std::string> names;
    for (uint32_t i = 0; i < count; ++i) {
        std::string name; uint64_t n64 = 0; if (!read_string(file, &pos, &name, kMaxNameBytes) || name.empty() || !names.insert(name).second || !get_u64(file, &pos, &n64) || !n64 || n64 > uint64_t((std::numeric_limits<uint32_t>::max)())) return false;
        const size_t n = size_t(n64);
        if (n * 4 > end - pos) return false;
        state->names.push_back(name); state->elements.push_back(n); state->parameters[i].resize(n);
        for (float & value : state->parameters[i]) if (!get_f32(file, &pos, &value) || !std::isfinite(value)) return false;
        const size_t blocks = (n + kQuantizeBlock - 1) / kQuantizeBlock;
        std::vector<float> * fp32s[4] = { &state->state1_fp32[i], &state->state2_fp32[i], &state->state3_fp32[i], &state->state4_fp32[i] };
        std::vector<uint8_t> * u8s[4] = { &state->state1_u8[i], &state->state2_u8[i], &state->state3_u8[i], &state->state4_u8[i] };
        std::vector<float> * absmaxs[4] = { &state->absmax1[i], &state->absmax2[i], &state->absmax3[i], &state->absmax4[i] };
        for (int k = 0; k < 4; ++k) {
            uint32_t len = 0; if (!get_u32(file, &pos, &len)) return false;
            if (len == 0) continue;
            if (size_t(len) != n) return false;
            if (n < 4096) {
                if (n * 4 > end - pos) return false;
                fp32s[k]->resize(n);
                for (float & value : *fp32s[k]) if (!get_f32(file, &pos, &value) || !std::isfinite(value)) return false;
            } else {
                if (n > end - pos || n + blocks * 4 > end - pos) return false;
                u8s[k]->resize(n);
                const uint8_t * q = nullptr;
                if (!read_bytes(file, &pos, &q, n)) return false;
                std::memcpy(u8s[k]->data(), q, n);
                absmaxs[k]->resize(blocks);
                for (float & value : *absmaxs[k]) if (!get_f32(file, &pos, &value) || !std::isfinite(value) || value < 0) return false;
            }
        }
    }
    if (pos != end) return false; return true;
}

inline bool yue2_aitk_read_resume(const char * path, ResumeRecord * out) {
    using namespace resume_detail;
    if (!out) return false; std::vector<uint8_t> file; if (!read_file(path, &file) || file.size() < 8 + 4 + 8 + 8 + 8 + 8 || !std::equal(kMagic.begin(), kMagic.end(), file.begin())) return false;
    size_t pos = 8; uint32_t version = 0; uint64_t payload_size = 0, metadata_size = 0, checksum = 0, metadata_prefix = 0; if (!get_u32(file, &pos, &version) || (version != 1 && version != 2) || !get_u64(file, &pos, &payload_size) || !get_u64(file, &pos, &metadata_size) || !get_u64(file, &pos, &checksum) || !get_u64(file, &pos, &metadata_prefix) || metadata_prefix != metadata_size || metadata_size > kMaxMetadataBytes || payload_size > kMaxFileBytes) return false;
    if (metadata_size > file.size() - pos) return false; const size_t payload_offset = pos + size_t(metadata_size); if (payload_size > file.size() - payload_offset || size_t(payload_size) != file.size() - payload_offset) return false; const uint64_t actual = hash_bytes(file.data() + payload_offset, size_t(payload_size), hash_bytes(file.data() + pos, size_t(metadata_size))); if (actual != checksum) return false;
    ResumeRecord record; record.runner_metadata.assign(reinterpret_cast<const char *>(file.data() + pos), size_t(metadata_size)); record.version = int(version);
    if (version == 1 && !read_resume_v1(file, payload_offset, file.size(), &record.state)) return false;
    if (version == 2 && !read_resume_v2(file, payload_offset, file.size(), &record.state)) return false;
    *out = std::move(record); return true;
}

} // namespace yue2_aitk

