#pragma once
// alignment-config.h: cross-attention layer/head configuration for LRC alignment
//
// Specifies which DiT cross-attention heads produce useful lyric-to-audio
// alignment. The default config targets the 2B model (24 layers, 16 heads).
// XL models (32 layers, 32 heads) must provide their config via GGUF metadata.

#include <cstdio>
#include <string>
#include <vector>

struct AlignLayerHeads {
    int              layer;
    std::vector<int> heads;
};

struct AlignmentConfig {
    std::vector<AlignLayerHeads> targets;
    int  max_layer;    // max(layer indices) — early exit boundary
    int  total_heads;  // sum of all head counts
    bool valid;        // false if config is unavailable/unsupported
};

// Default config for the 2B DiT model (24 layers, 16 heads per layer).
// Python reference: {2: [6], 3: [10, 11], 4: [3], 5: [8, 9], 6: [8]}
static AlignmentConfig alignment_config_default_2b() {
    AlignmentConfig cfg;
    cfg.targets = {
        {2, {6}},
        {3, {10, 11}},
        {4, {3}},
        {5, {8, 9}},
        {6, {8}},
    };
    cfg.max_layer   = 6;
    cfg.total_heads = 7;
    cfg.valid       = true;
    return cfg;
}

// Default config for the XL (4B) DiT model (32 layers, 32 heads per layer).
// From the XL model's config.json lyric_alignment_layers_config.
// The alignment-relevant heads are completely different from the 2B model.
static AlignmentConfig alignment_config_default_xl() {
    AlignmentConfig cfg;
    cfg.targets = {
        {3, {18, 27}},
        {4, {22}},
        {5, {5, 6, 7}},
        {6, {2, 12, 13}},
        {7, {20, 21}},
    };
    cfg.max_layer   = 7;
    cfg.total_heads = 11;
    cfg.valid       = true;
    return cfg;
}

// Parse alignment config from a JSON string (from GGUF metadata).
// Format: {"2": [6], "3": [10, 11], "4": [3], ...}
// Returns invalid config on parse failure.
static AlignmentConfig alignment_config_parse_json(const std::string & json) {
    AlignmentConfig cfg;
    cfg.max_layer   = -1;
    cfg.total_heads = 0;
    cfg.valid       = false;

    if (json.empty()) {
        return cfg;
    }

    // Minimal JSON object parser: { "key": [int, ...], ... }
    // Good enough for the simple nested structure, no external deps.
    size_t pos = json.find('{');
    if (pos == std::string::npos) {
        return cfg;
    }
    pos++;

    while (pos < json.size()) {
        // skip whitespace/commas
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == ',' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t')) {
            pos++;
        }
        if (pos >= json.size() || json[pos] == '}') {
            break;
        }

        // parse key (quoted string or bare number)
        if (json[pos] != '"') {
            break;  // unexpected
        }
        pos++;
        size_t key_end = json.find('"', pos);
        if (key_end == std::string::npos) {
            break;
        }
        int layer = std::atoi(json.substr(pos, key_end - pos).c_str());
        pos       = key_end + 1;

        // skip to ':'
        while (pos < json.size() && json[pos] != ':') {
            pos++;
        }
        if (pos >= json.size()) {
            break;
        }
        pos++;

        // skip to '['
        while (pos < json.size() && json[pos] != '[') {
            pos++;
        }
        if (pos >= json.size()) {
            break;
        }
        pos++;

        // parse head indices
        AlignLayerHeads lh;
        lh.layer = layer;
        while (pos < json.size() && json[pos] != ']') {
            while (pos < json.size() && (json[pos] == ' ' || json[pos] == ',')) {
                pos++;
            }
            if (pos >= json.size() || json[pos] == ']') {
                break;
            }
            int  head = 0;
            bool neg  = false;
            if (json[pos] == '-') {
                neg = true;
                pos++;
            }
            while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
                head = head * 10 + (json[pos] - '0');
                pos++;
            }
            if (neg) {
                head = -head;
            }
            lh.heads.push_back(head);
        }
        if (pos < json.size()) {
            pos++;  // skip ']'
        }

        if (!lh.heads.empty()) {
            cfg.targets.push_back(lh);
            cfg.total_heads += (int) lh.heads.size();
            if (layer > cfg.max_layer) {
                cfg.max_layer = layer;
            }
        }
    }

    cfg.valid = !cfg.targets.empty();
    if (cfg.valid) {
        fprintf(stderr, "[AlignConfig] Parsed: %d layer groups, %d total heads, max_layer=%d\n",
                (int) cfg.targets.size(), cfg.total_heads, cfg.max_layer);
    }
    return cfg;
}

// Resolve alignment config: use GGUF config if available, else fall back
// to model-specific defaults based on architecture.
static AlignmentConfig alignment_config_resolve(const std::string & gguf_json, int n_layers, int n_heads = 0) {
    // Try GGUF config first
    if (!gguf_json.empty()) {
        AlignmentConfig cfg = alignment_config_parse_json(gguf_json);
        if (cfg.valid) {
            return cfg;
        }
        fprintf(stderr, "[AlignConfig] WARNING: failed to parse GGUF alignment config, trying default\n");
    }

    // Detect XL model: 32 layers + 32 heads (vs 2B: 24 layers, 16 heads).
    // XL models have completely different alignment-relevant cross-attention heads.
    // Using the 2B config on XL produces flat/uninformative attention scores.
    if (n_layers >= 32 && n_heads >= 28) {
        fprintf(stderr, "[AlignConfig] Using default XL config (%d layers, %d heads)\n", n_layers, n_heads);
        return alignment_config_default_xl();
    }

    // Fall back to 2B default for models with enough layers/heads.
    const int required_layers = 7;   // need layers 0..6
    const int required_heads  = 12;  // max head index is 11

    if (n_layers >= required_layers && (n_heads == 0 || n_heads >= required_heads)) {
        fprintf(stderr, "[AlignConfig] Using default 2B config (%d layers, %d heads)\n", n_layers, n_heads);
        return alignment_config_default_2b();
    }

    // Unknown model — can't guess
    fprintf(stderr, "[AlignConfig] WARNING: no alignment config for %d-layer / %d-head model\n", n_layers, n_heads);
    AlignmentConfig invalid;
    invalid.max_layer   = -1;
    invalid.total_heads = 0;
    invalid.valid       = false;
    return invalid;
}
