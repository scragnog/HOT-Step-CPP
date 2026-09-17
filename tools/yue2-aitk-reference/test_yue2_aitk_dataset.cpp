#include "../../engine/src/train/yue2-aitk-dataset.h"

#include <iostream>
#include <string>

// Source-only contract harness.  The guarded root runner supplies a fresh
// manifest directory when this is exercised; this file never removes inputs.
int main(int argc, char ** argv) {
    if (argc != 4) { std::cerr << "usage: test_yue2_aitk_dataset accept|reject EXPECTED_COUNT MANIFEST\n"; return 2; }
    const std::string mode = argv[1];
    const size_t expected = std::stoull(argv[2]);
    yue2_aitk::Dataset dataset;
    dataset.recipe_version = "sentinel";
    dataset.items.push_back({"sentinel", {}, {}});
    const auto before = dataset.items.size();
    std::string error;
    if (!yue2_aitk::read_dataset(argv[3], &dataset, &error)) {
        if (mode != "reject") { std::cerr << "unexpected rejection: " << error << "\n"; return 1; }
        if (dataset.items.size() != before || dataset.recipe_version != "sentinel") {
            std::cerr << "failure mutated output: " << error << "\n";
            return 1;
        }
        std::cout << "rejected: " << error << "\n";
        return 0;
    }
    if (mode != "accept" || dataset.items.size() != expected) {
        std::cerr << "unexpected acceptance/count: " << dataset.items.size() << "\n";
        return 1;
    }
    if (dataset.items.empty()) { std::cerr << "accepted empty dataset\n"; return 1; }
    for (const auto & item : dataset.items) {
        if (item.song.semantic_tokens.empty() || item.song.latents.size() !=
            item.song.semantic_tokens.size() * yue2_aitk::kLatentChannels ||
            item.prompt.retained_prefix_ids.empty() || item.prompt.dropped_prefix_ids.empty()) {
            std::cerr << "accepted malformed item " << item.id << "\n";
            return 1;
        }
    }
    std::cout << "accepted " << dataset.items.size() << " immutable dataset items\n";
    return 0;
}
