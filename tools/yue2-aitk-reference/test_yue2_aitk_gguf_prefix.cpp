// Compare schema-1 prefixes against the YuE2 BPE loaded from the production
// LM GGUF.  This is intentionally a CPU-only tokenizer test: it does not load
// model inference and never changes the input dataset or caption directory.

#include "../../engine/src/train/yue2-aitk-dataset.h"
#include "../../engine/src/train/yue2-aitk-import.h"
#include "../../engine/src/yue2/yue2-tokenizer.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static bool read_text(const fs::path & path, std::string * out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out->assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return f.good() || f.eof();
}

static bool same_ids(const std::vector<int32_t> & actual, const std::vector<int> & expected,
                     const std::string & label, const std::string & id) {
    if (actual.size() == expected.size()) {
        bool equal = true;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (actual[i] != expected[i]) { equal = false; break; }
        }
        if (equal) return true;
    }
    size_t first = 0;
    while (first < actual.size() && first < expected.size() && actual[first] == expected[first]) ++first;
    std::cerr << id << ' ' << label << " mismatch: actual=" << actual.size()
              << " expected=" << expected.size() << " first=" << first;
    if (first < actual.size()) std::cerr << " actual_id=" << actual[first];
    if (first < expected.size()) std::cerr << " expected_id=" << expected[first];
    std::cerr << '\n';
    return false;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::cerr << "usage: test_yue2_aitk_gguf_prefix DATASET_JSON CAPTION_DIR LM_GGUF\n";
        return 2;
    }
    const fs::path dataset_path = fs::u8path(argv[1]);
    const fs::path caption_dir = fs::u8path(argv[2]);
    const fs::path lm_path = fs::u8path(argv[3]);

    yue2_aitk::Dataset dataset;
    std::string error;
    if (!yue2_aitk::read_dataset(dataset_path.u8string(), &dataset, &error)) {
        std::cerr << "dataset rejected: " << error << '\n';
        return 1;
    }
    // The pinned imported Toolkit audit contains 15 rows.  Keep this check
    // explicit so a partial fixture cannot accidentally report parity.
    if (dataset.items.size() != 15) {
        std::cerr << "expected 15 imported rows, got " << dataset.items.size() << '\n';
        return 1;
    }

    BPETokenizer tokenizer{};
    if (!yue2_tokenizer_load_from_gguf(&tokenizer, lm_path.u8string())) {
        std::cerr << "failed to load BPE vocabulary/merges from LM GGUF: " << lm_path.u8string() << '\n';
        return 1;
    }

    size_t checked = 0;
    for (const auto & item : dataset.items) {
        // Toolkit captions are named after the audio filename's stem.
        const fs::path caption = caption_dir / (fs::path(item.id).stem().string() + ".txt");
        std::string source;
        if (!read_text(caption, &source)) {
            std::cerr << "missing Toolkit caption for " << item.id << ": " << caption.u8string() << '\n';
            return 1;
        }
        std::string style, lyrics;
        if (!yue2_aitk_import::detail::parse_caption(source, &style, &lyrics, &error)) {
            std::cerr << "caption parse failed for " << item.id << ": " << error << '\n';
            return 1;
        }

        const std::vector<int> full = yue2_token_prefixes(&tokenizer, style, lyrics, YUE2_COT_FULL, nullptr);
        std::vector<int> off = yue2_token_prefixes(&tokenizer, style, lyrics, YUE2_COT_OFF, nullptr);
        if (off.size() < 2 || off[off.size()-2] != YUE2_ABC_END || off.back() != YUE2_MUSIC_START) return 1;
        off.resize(off.size()-2); // Training stores prefixes through ABC_START.
        if (!same_ids(item.prompt.retained_prefix_ids, full, "prefix_full_ids", item.id) ||
            !same_ids(item.prompt.dropped_prefix_ids, off, "prefix_off_ids", item.id)) return 1;
        ++checked;
    }

    std::cout << "exact GGUF BPE prefix parity: " << checked << "/" << dataset.items.size()
              << " rows (full and off)\n";
    return 0;
}
