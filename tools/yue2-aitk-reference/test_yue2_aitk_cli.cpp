#include "../../engine/src/train/yue2-aitk-runtime.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {
struct Args {
    std::vector<std::string> text;
    std::vector<char *> argv;
    explicit Args(std::initializer_list<const char *> values) {
        for (const char * value : values) text.emplace_back(value);
        for (std::string & value : text) argv.push_back(value.data());
    }
};

bool expect_error(std::initializer_list<const char *> values, const char * label) {
    Args args(values); yue2_aitk_runtime::Config before;
    before.checkpoint = "sentinel"; before.dataset = "sentinel"; before.output = "sentinel";
    before.steps = 77; before.save_every = 11; before.seed = 9; before.cuda_index = 4;
    yue2_aitk_runtime::Config after = before; std::string error;
    const auto result = yue2_aitk_runtime::parse(static_cast<int>(args.argv.size()), args.argv.data(), &after, &error);
    if (result != yue2_aitk_runtime::ParseResult::error || after.checkpoint != before.checkpoint ||
        after.dataset != before.dataset || after.output != before.output || after.steps != before.steps ||
        after.save_every != before.save_every || after.seed != before.seed || after.cuda_index != before.cuda_index) {
        std::fprintf(stderr, "CLI contract failure: %s (result=%d error=%s)\n", label, static_cast<int>(result), error.c_str());
        return false;
    }
    return true;
}
}

int main() {
    int failures = 0;
    {
        Args args({"yue2-joint-train", "--checkpoint", "cp", "--dataset", "ds", "--output", "out",
                   "--steps", "3", "--save-every", "2", "--seed", "18446744073709551615", "--device", "CUDA:7"});
        yue2_aitk_runtime::Config config; std::string error;
        if (yue2_aitk_runtime::parse(static_cast<int>(args.argv.size()), args.argv.data(), &config, &error) != yue2_aitk_runtime::ParseResult::ok ||
            config.cuda_index != 7 || config.steps != 3 || config.save_every != 2 || config.seed != UINT64_MAX) ++failures;
    }
    Args help({"yue2-joint-train", "--help"}); yue2_aitk_runtime::Config untouched; untouched.output = "keep"; std::string help_error;
    if (yue2_aitk_runtime::parse(static_cast<int>(help.argv.size()), help.argv.data(), &untouched, &help_error) != yue2_aitk_runtime::ParseResult::help || untouched.output != "keep") ++failures;
    failures += !expect_error({"yue2-joint-train", "--checkpoint", "cp", "--dataset", "ds", "--output", "out", "--save-every", "1", "--device", "CUDA0"}, "missing steps");
    failures += !expect_error({"yue2-joint-train", "--checkpoint", "cp", "--dataset", "ds", "--output", "out", "--steps", "1", "--save-every", "1", "--device", "CUDA0", "--steps", "2"}, "duplicate steps");
    failures += !expect_error({"yue2-joint-train", "--checkpoint", "cp", "--dataset", "ds", "--output", "out", "--steps", "1", "--save-every", "1", "--seed", "18446744073709551616", "--device", "CUDA0"}, "seed overflow");
    failures += !expect_error({"yue2-joint-train", "--checkpoint", "cp", "--dataset", "ds", "--output", "out", "--steps", "1", "--save-every", "1", "--device", "GPU0"}, "bad device");
    std::puts(failures ? "YuE2 CLI parser contract FAILED" : "YuE2 CLI parser contract passed");
    return failures ? 1 : 0;
}
