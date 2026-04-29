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
// to 2B default (only safe for 24-layer models).
static AlignmentConfig alignment_config_resolve(const std::string & gguf_json, int n_layers) {
    // Try GGUF config first
    if (!gguf_json.empty()) {
        AlignmentConfig cfg = alignment_config_parse_json(gguf_json);
        if (cfg.valid) {
            return cfg;
        }
        fprintf(stderr, "[AlignConfig] WARNING: failed to parse GGUF alignment config, trying default\n");
    }

    // Fall back to 2B default only for 24-layer models
    if (n_layers == 24) {
        fprintf(stderr, "[AlignConfig] Using default 2B config (24 layers)\n");
        return alignment_config_default_2b();
    }

    // Unknown model — can't guess
    fprintf(stderr, "[AlignConfig] WARNING: no alignment config for %d-layer model\n", n_layers);
    AlignmentConfig invalid;
    invalid.max_layer   = -1;
    invalid.total_heads = 0;
    invalid.valid       = false;
    return invalid;
}
