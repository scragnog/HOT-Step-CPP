#pragma once
// The held-out gate's best planner, kept beside a checkpoint whose live
// planner has moved past it, so a run resumed there can still write it back
// at the freeze (GH #346). A parameter-only resume record: per-slot names, no
// optimizer state; its step is the best reading's. A copy is written once and
// never changed, so later checkpoints of the same best step hard-link it:
// every folder keeps its own copy, one best step's data is stored once, and
// deleting any folder leaves the others whole.

#include "yue2-aitk-train-state.h"
#include "yue2-aitk-resume.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <utility>

namespace yue2_aitk_planner_best {

constexpr const char * kFile = "planner-best.resume";

inline bool write(const std::filesystem::path & path, const Yue2AitkTrainState & state, const Yue2AitkTrainState::PlannerCopy & copy, int best_step) {
    const auto named = state.named_tensors();
    const size_t n = copy.size();
    yue2_aitk::HostStateSnapshot snap; snap.step = best_step;
    snap.names.resize(n); snap.elements.resize(n); snap.parameters.resize(n);
    snap.state1_fp32.resize(n); snap.state2_fp32.resize(n); snap.state3_fp32.resize(n); snap.state4_fp32.resize(n);
    snap.state1_u8.resize(n); snap.state2_u8.resize(n); snap.state3_u8.resize(n); snap.state4_u8.resize(n);
    snap.absmax1.resize(n); snap.absmax2.resize(n); snap.absmax3.resize(n); snap.absmax4.resize(n);
    for (size_t k = 0; k < n; ++k) { snap.names[k] = named[copy[k].first].name; snap.elements[k] = copy[k].second.size(); snap.parameters[k] = copy[k].second; }
    return yue2_aitk::yue2_aitk_write_resume_v2(path.u8string().c_str(), snap, "{}");
}

// A hard link to `same_step` (an earlier copy of this best step; empty when
// there is none), or a fresh write when linking fails (another volume, a
// file system without links, the earlier copy gone).
inline bool publish(const std::filesystem::path & path, const std::filesystem::path & same_step, const Yue2AitkTrainState & state, const Yue2AitkTrainState::PlannerCopy & copy, int best_step) {
    std::error_code ec;
    if (!same_step.empty()) { std::filesystem::create_hard_link(same_step, path, ec); if (!ec) return true; }
    return write(path, state, copy, best_step);
}

// False when the file is missing, from another step, or does not match this
// train state's planner slots.
inline bool read(const std::filesystem::path & path, const Yue2AitkTrainState & state, int best_step, Yue2AitkTrainState::PlannerCopy * out) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) return false;
    yue2_aitk::ResumeRecord record;
    if (!yue2_aitk::yue2_aitk_read_resume(path.u8string().c_str(), &record) || record.version != 3 || record.state.step != best_step) return false;
    const auto named = state.named_tensors();
    Yue2AitkTrainState::PlannerCopy copy = state.copy_planner();
    if (record.state.names.size() != copy.size()) return false;
    for (size_t k = 0; k < copy.size(); ++k) {
        if (record.state.names[k] != named[copy[k].first].name || record.state.parameters[k].size() != copy[k].second.size()) return false;
        copy[k].second = std::move(record.state.parameters[k]);
    }
    *out = std::move(copy);
    return true;
}

// A safetensors file's tensor bytes (all after its header), empty when it
// cannot be read. Little-endian hosts only, as everywhere else here.
inline std::string tensor_bytes(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary); uint64_t n = 0;
    if (!in.read(reinterpret_cast<char *>(&n), sizeof n) || !in.seekg(std::streamoff(sizeof n + n))) return {};
    return std::string(std::istreambuf_iterator<char>(in), {});
}

// A frozen planner's AR file, written again at every later checkpoint (GH
// #346): `fresh` becomes a hard link to `same` (the previous frozen
// checkpoint's AR) when their tensors are byte-identical, so the run stores
// it once and every folder still holds a complete pair; deleting a folder
// removes only its own link. Only the header differs: the linked file keeps
// the earlier checkpoint's `steps`. False (and `fresh` as written) when the
// tensors differ or linking fails.
inline bool link_same_tensors(const std::filesystem::path & same, const std::filesystem::path & fresh) {
    if (same.empty()) return false;
    const std::string held = tensor_bytes(same);
    if (held.empty() || held != tensor_bytes(fresh)) return false;
    auto link = fresh; link += ".link";
    std::error_code ec;
    std::filesystem::create_hard_link(same, link, ec);
    if (!ec) std::filesystem::rename(link, fresh, ec);
    if (ec) { std::error_code rm; std::filesystem::remove(link, rm); return false; }
    return true;
}

} // namespace yue2_aitk_planner_best
