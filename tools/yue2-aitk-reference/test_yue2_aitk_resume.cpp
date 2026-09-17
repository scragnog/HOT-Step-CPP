#include "../../engine/src/train/yue2-aitk-resume.h"
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2 || !argv[1] || !*argv[1]) { std::fprintf(stderr, "usage: resume_fixture <new-output-dir>\n"); return 1; }
    std::error_code ec;
    if (!std::filesystem::create_directory(argv[1],ec)) { std::fprintf(stderr,"output directory already exists or cannot be created\n"); return 1; }
    yue2_aitk::HostStateSnapshot source;
    source.step = 7; source.names = {"small", "large"}; source.elements = {4095, 4097};
    source.parameters.resize(2); source.state1_fp32.resize(2); source.state2_fp32.resize(2);
    source.state1_u8.resize(2); source.state2_u8.resize(2); source.absmax1.resize(2); source.absmax2.resize(2);
    source.parameters[0].assign(4095, .25f); source.parameters[1].assign(4097, -.5f);
    source.state1_fp32[0].assign(4095, .01f); source.state2_fp32[0].assign(4095, .0025f);
    source.state1_u8[1].assign(4097, 3); source.state2_u8[1].assign(4097, 4);
    source.absmax1[1].assign(17, .75f); source.absmax2[1].assign(17, .125f);
    const std::string path = std::string(argv[1]) + "/resume-roundtrip.bin";
    const std::string metadata = R"({"rng":"seed-19","sampler":"native","provenance":"fixture"})";
    if (!yue2_aitk::yue2_aitk_write_resume(path.c_str(), source, metadata)) { std::fprintf(stderr, "unable to create new output: %s\n", path.c_str()); return 2; }
    yue2_aitk::ResumeRecord loaded;
    if (!yue2_aitk::yue2_aitk_read_resume(path.c_str(), &loaded)) return 3;
    const auto & read_state = loaded.state;
    if (loaded.runner_metadata != metadata || read_state.step != source.step || read_state.names != source.names || read_state.elements != source.elements || read_state.parameters != source.parameters || read_state.state1_fp32 != source.state1_fp32 || read_state.state2_fp32 != source.state2_fp32 || read_state.state1_u8 != source.state1_u8 || read_state.state2_u8 != source.state2_u8 || read_state.absmax1 != source.absmax1 || read_state.absmax2 != source.absmax2) return 4;
    if (yue2_aitk::yue2_aitk_write_resume(path.c_str(), source, metadata)) return 5;
    int serial = 0;
    auto corrupt = [&](auto mutate) {
        const std::string bad_path = path + ".corrupt-" + std::to_string(++serial);
        if (!yue2_aitk::yue2_aitk_write_resume(bad_path.c_str(), source, metadata)) return false;
        std::ifstream in(bad_path, std::ios::binary); std::vector<unsigned char> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()); in.close(); mutate(raw);
        std::ofstream out(bad_path, std::ios::binary | std::ios::trunc); out.write(reinterpret_cast<const char *>(raw.data()), std::streamsize(raw.size())); out.close(); if(!out){std::fputs("cannot write corrupt fixture\n",stderr);return false;}
        yue2_aitk::ResumeRecord unchanged; unchanged.runner_metadata = "sentinel";
        const bool accepted=yue2_aitk::yue2_aitk_read_resume(bad_path.c_str(), &unchanged);
        std::printf("corrupt case %d: accepted=%d metadata=%s\n",serial,int(accepted),unchanged.runner_metadata.c_str());
        return !accepted && unchanged.runner_metadata == "sentinel";
    };
    if (!corrupt([](std::vector<unsigned char> & raw) { raw[28] ^= 1; })) return 6;
    if (!corrupt([](std::vector<unsigned char> & raw) { raw.resize(raw.size() - 1); })) return 7;
    if (!corrupt([](std::vector<unsigned char> & raw) { for (int i = 0; i < 8; ++i) raw[20 + i] = 0xff; })) return 8;
    std::printf("YuE2 resume corruption and roundtrip fixture passed; artifacts: %s\n", path.c_str()); return 0;
}
