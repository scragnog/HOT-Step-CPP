#pragma once
// lrc-alignment.h: LRC lyric timestamp alignment (pure C++, no GGML)
//
// Port of Python MusicStampsAligner + _dtw.py:
//   - Dynamic Time Warping (DTW)
//   - Bidirectional consensus denoising
//   - Median filter
//   - Token→sentence grouping
//   - LRC formatting
//
// All operations run on CPU after attention scores are read back from GPU.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

// ====================== DTW ======================

// Dynamic Time Warping on a cost matrix [N, M].
// Returns aligned (text_indices, time_indices) arrays.
// Port of _dtw.py::dtw_cpu + _backtrace.
static void dtw_cpu(const double * cost_matrix, int N, int M,
                    std::vector<int> & text_indices,
                    std::vector<int> & time_indices) {
    // Allocate cost and trace matrices (1-indexed, so +1)
    std::vector<float> cost((N + 1) * (M + 1), std::numeric_limits<float>::infinity());
    std::vector<int>   trace((N + 1) * (M + 1), -1);

    auto idx = [&](int i, int j) { return i * (M + 1) + j; };

    cost[idx(0, 0)] = 0.0f;

    for (int j = 1; j <= M; j++) {
        for (int i = 1; i <= N; i++) {
            float c0 = cost[idx(i - 1, j - 1)];
            float c1 = cost[idx(i - 1, j)];
            float c2 = cost[idx(i, j - 1)];

            float c;
            int   t;
            if (c0 < c1 && c0 < c2) {
                c = c0; t = 0;
            } else if (c1 < c0 && c1 < c2) {
                c = c1; t = 1;
            } else {
                c = c2; t = 2;
            }

            cost[idx(i, j)]  = (float) cost_matrix[(i - 1) * M + (j - 1)] + c;
            trace[idx(i, j)] = t;
        }
    }

    // Backtrace
    // Set boundary conditions
    for (int j = 0; j <= M; j++) {
        trace[idx(0, j)] = 2;
    }
    for (int i = 0; i <= N; i++) {
        trace[idx(i, 0)] = 1;
    }

    int max_path = N + M;
    std::vector<int> path_text(max_path);
    std::vector<int> path_time(max_path);

    int i = N, j = M;
    int path_idx = max_path - 1;

    while (i > 0 || j > 0) {
        path_text[path_idx] = i - 1;
        path_time[path_idx] = j - 1;
        path_idx--;

        int t = trace[idx(i, j)];
        if (t == 0) {
            i--; j--;
        } else if (t == 1) {
            i--;
        } else if (t == 2) {
            j--;
        } else {
            break;
        }
    }

    int start = path_idx + 1;
    int len   = max_path - start;
    text_indices.assign(path_text.begin() + start, path_text.begin() + start + len);
    time_indices.assign(path_time.begin() + start, path_time.begin() + start + len);
}

// ====================== SOFTMAX ======================

// Row-wise softmax: out[i][j] = exp(x[i][j]) / sum_j(exp(x[i][j]))
static void softmax_rows(const float * x, float * out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        const float * row = x + i * cols;
        float *       dst = out + i * cols;

        float mx = row[0];
        for (int j = 1; j < cols; j++) {
            if (row[j] > mx) mx = row[j];
        }

        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            dst[j] = expf(row[j] - mx);
            sum += dst[j];
        }
        float inv = 1.0f / (sum + 1e-9f);
        for (int j = 0; j < cols; j++) {
            dst[j] *= inv;
        }
    }
}

// Column-wise softmax: out[i][j] = exp(x[i][j]) / sum_i(exp(x[i][j]))
static void softmax_cols(const float * x, float * out, int rows, int cols) {
    for (int j = 0; j < cols; j++) {
        float mx = x[0 * cols + j];
        for (int i = 1; i < rows; i++) {
            if (x[i * cols + j] > mx) mx = x[i * cols + j];
        }

        float sum = 0.0f;
        for (int i = 0; i < rows; i++) {
            float v = expf(x[i * cols + j] - mx);
            out[i * cols + j] = v;
            sum += v;
        }
        float inv = 1.0f / (sum + 1e-9f);
        for (int i = 0; i < rows; i++) {
            out[i * cols + j] *= inv;
        }
    }
}

// ====================== MEDIAN ======================

// Median of a small array (selection, not full sort for small n)
static float median_small(std::vector<float> & buf) {
    int n = (int) buf.size();
    if (n == 0) return 0.0f;
    std::nth_element(buf.begin(), buf.begin() + n / 2, buf.end());
    return buf[n / 2];
}

// Row-wise median: compute median of each row
static void row_medians(const float * x, float * out, int rows, int cols) {
    std::vector<float> buf(cols);
    for (int i = 0; i < rows; i++) {
        buf.assign(x + i * cols, x + (i + 1) * cols);
        out[i] = median_small(buf);
    }
}

// Column-wise median: compute median of each column
static void col_medians(const float * x, float * out, int rows, int cols) {
    std::vector<float> buf(rows);
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            buf[i] = x[i * cols + j];
        }
        out[j] = median_small(buf);
    }
}

// 1D median filter on a row with reflect padding
static void median_filter_row(const float * in, float * out, int n, int width) {
    int pad = width / 2;
    if (n <= pad || width <= 1) {
        memcpy(out, in, n * sizeof(float));
        return;
    }

    std::vector<float> padded(n + 2 * pad);
    // reflect padding
    for (int i = 0; i < pad; i++) {
        padded[pad - 1 - i]     = in[i + 1 < n ? i + 1 : 0];
        padded[n + pad + i]     = in[n - 2 - i >= 0 ? n - 2 - i : n - 1];
    }
    memcpy(padded.data() + pad, in, n * sizeof(float));

    std::vector<float> window(width);
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < width; k++) {
            window[k] = padded[i + k];
        }
        out[i] = median_small(window);
    }
}

// 2D median filter on last dimension (column-wise per row)
static void median_filter_2d(float * data, int rows, int cols, int width) {
    if (width <= 1) return;
    std::vector<float> row_buf(cols);
    for (int i = 0; i < rows; i++) {
        median_filter_row(data + i * cols, row_buf.data(), cols, width);
        memcpy(data + i * cols, row_buf.data(), cols * sizeof(float));
    }
}

// 3D median filter: [heads, tokens, frames] — filter along frames dimension
static void median_filter_3d(float * data, int heads, int tokens, int frames, int width) {
    if (width <= 1) return;
    for (int h = 0; h < heads; h++) {
        median_filter_2d(data + h * tokens * frames, tokens, frames, width);
    }
}

// ====================== BIDIRECTIONAL CONSENSUS ======================

// Apply bidirectional consensus denoising.
// weights_stack: [n_heads, n_tokens, n_frames]
// Returns: calc_matrix [n_tokens, n_frames] (averaged across heads)
// Port of MusicStampsAligner._apply_bidirectional_consensus
static void bidirectional_consensus(const float * weights_stack,
                                    int n_heads, int n_tokens, int n_frames,
                                    float violence_level, int medfilt_width,
                                    float * calc_matrix) {
    int HxTxF = n_heads * n_tokens * n_frames;
    int TxF   = n_tokens * n_frames;

    // Work buffer: [heads, tokens, frames]
    std::vector<float> processed(HxTxF);

    // For each head: row_softmax * col_softmax
    std::vector<float> row_prob(TxF);
    std::vector<float> col_prob(TxF);

    for (int h = 0; h < n_heads; h++) {
        const float * src = weights_stack + h * TxF;
        float *       dst = processed.data() + h * TxF;

        // A. Bidirectional consensus
        softmax_rows(src, row_prob.data(), n_tokens, n_frames);  // Token → Frame
        softmax_cols(src, col_prob.data(), n_tokens, n_frames);  // Frame → Token

        for (int i = 0; i < TxF; i++) {
            dst[i] = row_prob[i] * col_prob[i];
        }

        // 1. Row suppression: subtract violence * row_median, ReLU
        std::vector<float> r_med(n_tokens);
        row_medians(dst, r_med.data(), n_tokens, n_frames);
        for (int t = 0; t < n_tokens; t++) {
            float threshold = violence_level * r_med[t];
            for (int f = 0; f < n_frames; f++) {
                float v = dst[t * n_frames + f] - threshold;
                dst[t * n_frames + f] = v > 0.0f ? v : 0.0f;
            }
        }

        // 2. Column suppression: subtract violence * col_median, ReLU
        std::vector<float> c_med(n_frames);
        col_medians(dst, c_med.data(), n_tokens, n_frames);
        for (int t = 0; t < n_tokens; t++) {
            for (int f = 0; f < n_frames; f++) {
                float v = dst[t * n_frames + f] - violence_level * c_med[f];
                dst[t * n_frames + f] = v > 0.0f ? v : 0.0f;
            }
        }

        // C. Power sharpening (^2)
        for (int i = 0; i < TxF; i++) {
            dst[i] = dst[i] * dst[i];
        }
    }

    // D. Z-Score normalization (across all heads)
    {
        double sum = 0.0, sum_sq = 0.0;
        for (int i = 0; i < HxTxF; i++) {
            sum    += processed[i];
            sum_sq += (double) processed[i] * processed[i];
        }
        float mean = (float) (sum / HxTxF);
        float var  = (float) (sum_sq / HxTxF - (double) mean * mean);
        float std  = sqrtf(var > 0.0f ? var : 0.0f);
        float inv  = 1.0f / (std + 1e-9f);
        for (int i = 0; i < HxTxF; i++) {
            processed[i] = (processed[i] - mean) * inv;
        }
    }

    // E. Median filtering along frames dimension
    median_filter_3d(processed.data(), n_heads, n_tokens, n_frames, medfilt_width);

    // Average across heads → calc_matrix [tokens, frames]
    memset(calc_matrix, 0, TxF * sizeof(float));
    float inv_heads = 1.0f / (float) n_heads;
    for (int h = 0; h < n_heads; h++) {
        const float * src = processed.data() + h * TxF;
        for (int i = 0; i < TxF; i++) {
            calc_matrix[i] += src[i] * inv_heads;
        }
    }
}

// ====================== TOKEN TIMESTAMPS ======================

struct TokenTimestamp {
    int         token_id;
    std::string text;
    float       start;
    float       end;
};

// Generate per-token timestamps using DTW.
// calc_matrix: [n_tokens, n_frames]
// Returns a TokenTimestamp for each token.
static std::vector<TokenTimestamp> token_timestamps(
    const float *                      calc_matrix,
    int                                n_tokens,
    int                                n_frames,
    const std::vector<int> &           lyric_token_ids,
    const std::vector<std::string> &   token_texts,
    float                              total_duration) {

    // Negate calc_matrix for DTW (DTW minimizes cost)
    std::vector<double> neg_matrix(n_tokens * n_frames);
    for (int i = 0; i < n_tokens * n_frames; i++) {
        neg_matrix[i] = -(double) calc_matrix[i];
    }

    std::vector<int> text_idx, time_idx;
    dtw_cpu(neg_matrix.data(), n_tokens, n_frames, text_idx, time_idx);

    float seconds_per_frame = total_duration / (float) n_frames;

    std::vector<TokenTimestamp> results(n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        results[i].token_id = lyric_token_ids[i];
        results[i].text     = (i < (int) token_texts.size()) ? token_texts[i] : "";
        results[i].start    = 0.0f;
        results[i].end      = 0.0f;
    }

    // For each token, find the time indices that map to it
    for (int i = 0; i < n_tokens; i++) {
        float first_time = -1.0f;
        float last_time  = -1.0f;
        for (int p = 0; p < (int) text_idx.size(); p++) {
            if (text_idx[p] == i) {
                float t = (float) time_idx[p] * seconds_per_frame;
                if (first_time < 0.0f) first_time = t;
                last_time = t;
            }
        }
        if (first_time >= 0.0f) {
            results[i].start = first_time;
            results[i].end   = last_time;
        } else if (i > 0) {
            results[i].start = results[i - 1].end;
            results[i].end   = results[i - 1].end;
        }
        if (results[i].end < results[i].start) {
            results[i].end = results[i].start;
        }
    }

    return results;
}

// ====================== SENTENCE GROUPING ======================

struct SentenceTimestamp {
    std::string text;
    float       start;
    float       end;
};

// Group token timestamps into sentences (split on \n boundaries).
static std::vector<SentenceTimestamp> sentence_timestamps(
    const std::vector<TokenTimestamp> & token_stamps) {

    std::vector<SentenceTimestamp> results;
    std::string current_text;
    float       sentence_start = -1.0f;
    float       sentence_end   = 0.0f;

    for (const auto & tok : token_stamps) {
        if (sentence_start < 0.0f) {
            sentence_start = tok.start;
        }
        sentence_end = tok.end;
        current_text += tok.text;

        if (tok.text.find('\n') != std::string::npos) {
            // Trim whitespace
            std::string trimmed = current_text;
            size_t s = trimmed.find_first_not_of(" \t\n\r");
            size_t e = trimmed.find_last_not_of(" \t\n\r");
            if (s != std::string::npos && e != std::string::npos) {
                trimmed = trimmed.substr(s, e - s + 1);
            } else {
                trimmed.clear();
            }

            if (!trimmed.empty()) {
                results.push_back({trimmed, sentence_start, sentence_end});
            }
            current_text.clear();
            sentence_start = -1.0f;
        }
    }

    // Handle last sentence
    if (!current_text.empty()) {
        std::string trimmed = current_text;
        size_t s = trimmed.find_first_not_of(" \t\n\r");
        size_t e = trimmed.find_last_not_of(" \t\n\r");
        if (s != std::string::npos && e != std::string::npos) {
            trimmed = trimmed.substr(s, e - s + 1);
        } else {
            trimmed.clear();
        }
        if (!trimmed.empty()) {
            results.push_back({trimmed, sentence_start, sentence_end});
        }
    }

    return results;
}

// ====================== LRC FORMATTING ======================

// Format timestamp as [mm:ss.xx]
static std::string format_lrc_time(float seconds) {
    if (seconds < 0.0f) seconds = 0.0f;
    int   mins = (int) (seconds / 60.0f);
    float secs = seconds - (float) mins * 60.0f;
    char  buf[32];
    snprintf(buf, sizeof(buf), "[%02d:%05.2f]", mins, secs);
    return buf;
}

// ====================== MAIN ENTRY POINT ======================

struct LrcResult {
    std::string lrc_text;
    bool        success;
    std::string error;
};

// Full LRC alignment pipeline:
//   attention_scores [n_heads, n_tokens, n_frames] →
//   bidirectional consensus → DTW → sentence grouping → LRC text
//
// lyric_token_ids: pure lyric tokens (header stripped, endoftext stripped)
// token_texts: decoded text per token (incremental decode)
// total_duration: audio duration in seconds
static LrcResult lrc_align(
    const float *                      attention_scores,
    int                                n_heads,
    int                                n_tokens,
    int                                n_frames,
    const std::vector<int> &           lyric_token_ids,
    const std::vector<std::string> &   token_texts,
    float                              total_duration,
    float                              violence_level = 2.0f,
    int                                medfilt_width  = 1) {

    LrcResult result;
    result.success = false;

    if (n_heads <= 0 || n_tokens <= 0 || n_frames <= 0) {
        result.error = "Invalid dimensions";
        return result;
    }
    if ((int) lyric_token_ids.size() != n_tokens) {
        result.error = "Token count mismatch";
        return result;
    }

    fprintf(stderr, "[LRC-Align] n_heads=%d, n_tokens=%d, n_frames=%d, duration=%.1fs\n",
            n_heads, n_tokens, n_frames, total_duration);

    // Step 1: Bidirectional consensus denoising
    std::vector<float> calc_matrix(n_tokens * n_frames);
    bidirectional_consensus(attention_scores, n_heads, n_tokens, n_frames,
                            violence_level, medfilt_width, calc_matrix.data());

    // Step 2: Token timestamps via DTW
    auto tok_stamps = token_timestamps(calc_matrix.data(), n_tokens, n_frames,
                                        lyric_token_ids, token_texts, total_duration);

    // Step 3: Group into sentences
    auto sent_stamps = sentence_timestamps(tok_stamps);

    if (sent_stamps.empty()) {
        result.error   = "No sentences found";
        result.lrc_text = "";
        result.success  = true;  // not an error, just no content
        return result;
    }

    // Step 4: Format as LRC
    std::string lrc;
    for (const auto & sent : sent_stamps) {
        lrc += format_lrc_time(sent.start);
        lrc += sent.text;
        lrc += "\n";
    }

    result.lrc_text = lrc;
    result.success  = true;
    fprintf(stderr, "[LRC-Align] Generated %d lines\n", (int) sent_stamps.size());
    return result;
}
