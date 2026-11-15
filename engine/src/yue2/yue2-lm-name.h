#pragma once
// yue2/yue2-lm-name.h — what a YuE2 LM file name says about the model.
// Shared by the renderer (yue2-adapter.h, yue2-model.h) and the joint trainer
// (train/yue2-aitk-runtime.cpp), so both read a fine-tune the same way.

#include <string>

// A variant is an LM file's type minus its quant: yue2-lm-q8_0.gguf is the base
// model (""), yue2-lm-rs-4.0x0.7-Q6_K.gguf is "rs-4.0x0.7". Nothing inside the
// converted files tells variants apart (the RS fine-tune's norms are
// bit-identical to the base's; it changed exactly the 392 block projections).
// ponytail: identity by filename; add a yue2.variant GGUF key at conversion if a
// variant ever ships under a name this rule misreads.
static inline std::string yue2_lm_variant(std::string type) {
    if (type.size() > 5 && type.compare(type.size() - 5, 5, "-imat") == 0) {
        type.resize(type.size() - 5);
    }
    const size_t dash = type.rfind('-');
    return dash == std::string::npos ? "" : type.substr(0, dash);
}

// "rs-4.0x0.7-q8_0" for .../yue2-lm-rs-4.0x0.7-q8_0.gguf, "" for anything else.
static inline std::string yue2_lm_type_of(const std::string & path) {
    const size_t      slash = path.find_last_of("/\\");
    const std::string n     = slash == std::string::npos ? path : path.substr(slash + 1);
    if (n.size() > 13 && n.compare(0, 8, "yue2-lm-") == 0 && n.compare(n.size() - 5, 5, ".gguf") == 0) {
        return n.substr(8, n.size() - 13);
    }
    return "";
}
