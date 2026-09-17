#include "../../engine/src/train/yue2-aitk-dataset.h"
#include "../../engine/src/train/yue2-aitk-sampler.h"
#include <cstdio>
#include <cstdlib>

// Exercise a prepared dataset through the actual loader and training sampler.
// Pass the manifest and expected number of tracks without usable lead sheets.
int main(int argc, char ** argv) {
    if (argc != 3) return 1;
    yue2_aitk::Dataset dataset;
    std::string error;
    if (!yue2_aitk::read_dataset(argv[1], &dataset, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str()); return 2;
    }
    yue2_aitk::Yue2NativeSampler sampler(20260917);
    const auto schedule = sampler.sigmoid_schedule(32);
    size_t missing = 0;
    for (const auto & item : dataset.items) {
        const bool has_sheet = !item.prompt.abc_ids.empty();
        if (!has_sheet) ++missing;
        // Disable stochastic dropout: an absent sheet must still force CoT off.
        const auto sample = sampler.sample(item.song, item.prompt, 1500, schedule, 0.0);
        if (sample.abc_retained != has_sheet) return 3;
        const auto & expected = has_sheet ? item.prompt.retained_prefix_ids : item.prompt.dropped_prefix_ids;
        if (sample.batch.ar.prefix_ids != expected) return 4;
        if (sample.batch.nar.latents.empty() || sample.batch.ar.target_ids.empty()) return 5;
    }
    if (missing != static_cast<size_t>(std::atoi(argv[2]))) return 6;
    std::printf("PASS: %zu tracks sampled; %zu without sheets use CoT off, %zu retain sheets\n",
                dataset.items.size(), missing, dataset.items.size() - missing);
    return 0;
}
