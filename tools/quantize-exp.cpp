// quantize-exp.cpp : Experimental GGUF requantizer for ACE-Step
// Extended from upstream quantize.cpp with IQ, NVFP4, MXFP4, and ternary types.
// Generates a synthetic weight-magnitude iMatrix inline for types that need it.
//
// Usage: quantize-exp <input.gguf> <output.gguf> <type>
// Types: Q2_K Q3_K_S Q3_K_M Q3_K_L Q4_K_S Q4_K_M Q5_K_S Q5_K_M Q6_K Q8_0
//        IQ4_NL IQ4_XS IQ3_S IQ3_XXS IQ2_S IQ2_XS IQ2_XXS IQ1_M IQ1_S
//        NVFP4 MXFP4 TQ1_0 TQ2_0 Q1_0

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#ifdef _WIN32
#    include <windows.h>
#    define strcasecmp _stricmp
#else
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

#include "ggml.h"
#include "gguf.h"
#include "version.h"

// Quant variant: base type + optional bump rules for important tensors
struct QuantVariant {
    const char *   name;
    enum ggml_type base;
    enum ggml_type bump;   // type for "important" tensors (or COUNT = no bump)
    enum ggml_type embed;  // type for embed_tokens (or COUNT = same as base)
    // bump_mode: 0=none, 1=first N layers, 2=first+last+every 3rd, 3=all important
    int            bump_mode;
    int            bump_n;  // for mode 1: number of layers to bump
    bool           needs_imatrix; // whether to generate synthetic iMatrix
};

static const QuantVariant VARIANTS[] = {
    // === Original K-quants ===
    // name       base            bump            embed           mode  n  imat
    { "Q2_K",   GGML_TYPE_Q2_K, GGML_TYPE_Q4_K,  GGML_TYPE_Q6_K, 1, 4, false },
    { "Q3_K_S", GGML_TYPE_Q3_K, GGML_TYPE_COUNT, GGML_TYPE_Q6_K, 0, 0, false },
    { "Q3_K_M", GGML_TYPE_Q3_K, GGML_TYPE_Q5_K,  GGML_TYPE_Q6_K, 2, 0, false },
    { "Q3_K_L", GGML_TYPE_Q3_K, GGML_TYPE_Q5_K,  GGML_TYPE_Q6_K, 3, 0, false },
    { "Q4_K_S", GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,  GGML_TYPE_Q6_K, 1, 4, false },
    { "Q4_K_M", GGML_TYPE_Q4_K, GGML_TYPE_Q6_K,  GGML_TYPE_Q6_K, 2, 0, false },
    { "Q5_K_S", GGML_TYPE_Q5_K, GGML_TYPE_COUNT, GGML_TYPE_Q6_K, 0, 0, false },
    { "Q5_K_M", GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,  GGML_TYPE_Q6_K, 2, 0, false },
    { "Q6_K",   GGML_TYPE_Q6_K, GGML_TYPE_COUNT, GGML_TYPE_Q6_K, 0, 0, false },
    { "Q8_0",   GGML_TYPE_Q8_0, GGML_TYPE_COUNT, GGML_TYPE_Q8_0, 0, 0, false },

    // === IQ types (importance-quantized) ===
    // IQ4: safe without iMatrix, but we provide one anyway for quality
    { "IQ4_NL",  GGML_TYPE_IQ4_NL,  GGML_TYPE_COUNT, GGML_TYPE_Q6_K, 0, 0, true },
    { "IQ4_XS",  GGML_TYPE_IQ4_XS,  GGML_TYPE_COUNT, GGML_TYPE_Q6_K, 0, 0, true },

    // IQ3: works without iMatrix but quality improves with it
    { "IQ3_S",   GGML_TYPE_IQ3_S,   GGML_TYPE_COUNT, GGML_TYPE_Q6_K, 0, 0, true },
    { "IQ3_XXS", GGML_TYPE_IQ3_XXS, GGML_TYPE_COUNT, GGML_TYPE_Q6_K, 0, 0, true },

    // IQ2: REQUIRES iMatrix (will ASSERT without it)
    { "IQ2_S",   GGML_TYPE_IQ2_S,   GGML_TYPE_COUNT, GGML_TYPE_Q6_K, 0, 0, true },
    { "IQ2_XS",  GGML_TYPE_IQ2_XS,  GGML_TYPE_COUNT, GGML_TYPE_Q6_K, 0, 0, true },
    { "IQ2_XXS", GGML_TYPE_IQ2_XXS, GGML_TYPE_COUNT, GGML_TYPE_Q6_K, 0, 0, true },

    // IQ1: extreme compression, REQUIRES iMatrix
    { "IQ1_M",   GGML_TYPE_IQ1_M,   GGML_TYPE_COUNT, GGML_TYPE_Q8_0, 0, 0, true },
    { "IQ1_S",   GGML_TYPE_IQ1_S,   GGML_TYPE_COUNT, GGML_TYPE_Q8_0, 0, 0, true },

    // === Floating point 4-bit ===
    { "NVFP4",  GGML_TYPE_NVFP4,  GGML_TYPE_COUNT, GGML_TYPE_Q6_K, 0, 0, false },
    { "MXFP4",  GGML_TYPE_MXFP4,  GGML_TYPE_COUNT, GGML_TYPE_Q6_K, 0, 0, false },

    // === Ternary and 1-bit ===
    { "TQ2_0",  GGML_TYPE_TQ2_0,  GGML_TYPE_COUNT, GGML_TYPE_Q6_K, 0, 0, false },
    { "TQ1_0",  GGML_TYPE_TQ1_0,  GGML_TYPE_COUNT, GGML_TYPE_Q6_K, 0, 0, false },
    { "Q1_0",   GGML_TYPE_Q1_0,   GGML_TYPE_COUNT, GGML_TYPE_Q8_0, 0, 0, false },
};

static const QuantVariant * find_variant(const char * s) {
    for (const auto & v : VARIANTS) {
        if (strcasecmp(s, v.name) == 0) {
            return &v;
        }
    }
    return nullptr;
}

// Extract layer index from HF tensor name: model.layers.N.xxx -> N, else -1
static int extract_layer(const char * name) {
    const char * p = strstr(name, "layers.");
    if (!p) {
        return -1;
    }
    return atoi(p + 7);
}

// Important tensors for S/M: v_proj + down_proj
static bool is_important_sm(const char * name) {
    return (strstr(name, "v_proj.weight") != nullptr) || (strstr(name, "down_proj.weight") != nullptr);
}

// Important tensors for L: v_proj + down_proj + o_proj
static bool is_important_l(const char * name) {
    return is_important_sm(name) || (strstr(name, "o_proj.weight") != nullptr);
}

static bool is_embed(const char * name) {
    return strstr(name, "embed_tokens.weight") != nullptr;
}

// Should this tensor be quantized at all?
static bool should_quantize(const char * name, int n_dims, const char * arch) {
    if (strstr(arch, "vae")) {
        return false;
    }
    if (n_dims < 2) {
        return false;
    }
    if (strstr(arch, "text-enc") && strstr(name, "embed_tokens")) {
        return false;
    }
    if (strstr(name, "silence_latent")) {
        return false;
    }
    if (strstr(name, "scale_shift_table")) {
        return false;
    }
    if (strstr(name, "null_condition_emb")) {
        return false;
    }
    return true;
}

// Decide target type for a single tensor given the variant + layer info
static enum ggml_type pick_type(const char *         name,
                                int                  n_dims,
                                const char *         arch,
                                const QuantVariant & v,
                                int                  n_layers) {
    if (!should_quantize(name, n_dims, arch)) {
        return GGML_TYPE_COUNT;
    }

    // embed_tokens in LM: use embed type
    if (is_embed(name) && !strstr(arch, "text-enc")) {
        return (v.embed != GGML_TYPE_COUNT) ? v.embed : v.base;
    }

    // Important tensor bump logic
    bool important = (v.bump_mode == 3) ? is_important_l(name) : is_important_sm(name);

    if (important && v.bump != GGML_TYPE_COUNT) {
        int  layer  = extract_layer(name);
        bool bumped = false;
        switch (v.bump_mode) {
            case 1:  // first N layers only
                bumped = (layer >= 0 && layer < v.bump_n);
                break;
            case 2:
                {  // M variant: first few + last few + every 3rd
                    int ql = n_layers;
                    bumped = (layer >= 0) && (layer < ql / 9 || layer >= ql - ql / 7 || layer % 3 == 0);
                    break;
                }
            case 3:  // L variant: all important tensors (v+down+o_proj)
                bumped = true;
                break;
        }
        if (bumped) {
            return v.bump;
        }
    }

    return v.base;
}

// Promote 1D tensors (norms/biases) to F32 for precision
static bool should_promote_f32(int n_dims) {
    return n_dims < 2;
}

// Convert source data to F32
static bool to_f32(const void * src, float * dst, int64_t n, enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_BF16:
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) src, dst, n);
            return true;
        case GGML_TYPE_F16:
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, dst, n);
            return true;
        case GGML_TYPE_F32:
            memcpy(dst, src, (size_t) n * sizeof(float));
            return true;
        default:
            return false;
    }
}

// Compute synthetic weight-magnitude iMatrix for a tensor.
// For each column j: imat[j] = sum_over_rows(w[row][j]^2)
// This gives a rough importance signal based on weight magnitudes.
static void compute_weight_imatrix(const float * f32, int64_t nrows, int64_t n_per_row, std::vector<float> & imat) {
    imat.resize((size_t) n_per_row);
    std::fill(imat.begin(), imat.end(), 0.0f);

    for (int64_t row = 0; row < nrows; row++) {
        const float * rp = f32 + row * n_per_row;
        for (int64_t j = 0; j < n_per_row; j++) {
            imat[(size_t) j] += rp[j] * rp[j];
        }
    }

    // Normalize: divide by nrows so values are in a reasonable range
    // Add small epsilon to avoid zero importance (which can cause division by zero)
    const float inv_nrows = 1.0f / (float) nrows;
    for (int64_t j = 0; j < n_per_row; j++) {
        imat[(size_t) j] = imat[(size_t) j] * inv_nrows + 1e-7f;
    }
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "acestep.cpp %s (experimental quantizer)\n\n", ACE_VERSION);
        fprintf(stderr, "Usage: %s <input.gguf> <output.gguf> <type>\n", argv[0]);
        fprintf(stderr, "Types:");
        for (const auto & v : VARIANTS) {
            fprintf(stderr, " %s", v.name);
        }
        fprintf(stderr, "\n\nIQ types use a synthetic weight-magnitude iMatrix.\n");
        fprintf(stderr, "NVFP4/MXFP4 use native GGML floating-point 4-bit.\n");
        return 1;
    }

    const char *         inp_path = argv[1];
    const char *         out_path = argv[2];
    const QuantVariant * variant  = find_variant(argv[3]);

    if (!variant) {
        fprintf(stderr, "[Quantize] Unknown type: %s\n", argv[3]);
        return 1;
    }

    fprintf(stderr, "[Quantize-Exp] %s -> %s (%s)\n", inp_path, out_path, variant->name);
    if (variant->needs_imatrix) {
        fprintf(stderr, "[Quantize-Exp] Using synthetic weight-magnitude iMatrix for IQ quantization\n");
    }

    // Initialize IQ lookup tables if needed
    ggml_quantize_init(variant->base);

    // Mmap input file
#ifdef _WIN32
    HANDLE fh = CreateFileA(inp_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (fh == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[Quantize-Exp] Failed to open %s\n", inp_path);
        return 1;
    }
    HANDLE mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mh) {
        fprintf(stderr, "[Quantize-Exp] CreateFileMapping failed %s\n", inp_path);
        CloseHandle(fh);
        return 1;
    }
    void * mapping = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!mapping) {
        fprintf(stderr, "[Quantize-Exp] MapViewOfFile failed %s\n", inp_path);
        CloseHandle(mh);
        CloseHandle(fh);
        return 1;
    }
#else
    int fd = open(inp_path, O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    struct stat st;
    fstat(fd, &st);
    size_t file_size = (size_t) st.st_size;
    void * mapping   = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }
#endif

    // Parse input GGUF
    struct gguf_init_params params = { /*no_alloc=*/true, /*ctx=*/nullptr };
    struct ggml_context *   meta   = nullptr;
    params.ctx                     = &meta;

    struct gguf_context * inp = gguf_init_from_file(inp_path, params);
    if (!inp) {
        fprintf(stderr, "[Quantize-Exp] Failed to read %s\n", inp_path);
#ifdef _WIN32
        UnmapViewOfFile(mapping);
        CloseHandle(mh);
        CloseHandle(fh);
#else
        munmap(mapping, file_size);
        close(fd);
#endif
        return 1;
    }

    const size_t data_off  = gguf_get_data_offset(inp);
    const int    n_tensors = (int) gguf_get_n_tensors(inp);

    // Read architecture
    char arch[64] = "unknown";
    {
        int64_t idx = gguf_find_key(inp, "general.architecture");
        if (idx >= 0) {
            const char * s = gguf_get_val_str(inp, (int) idx);
            snprintf(arch, sizeof(arch), "%s", s);
        }
    }

    // Read block count for bump policy
    int n_layers = 0;
    {
        char key[128];
        snprintf(key, sizeof(key), "%s.block_count", arch);
        int64_t idx = gguf_find_key(inp, key);
        if (idx >= 0) {
            n_layers = (int) gguf_get_val_u32(inp, (int) idx);
        }
    }

    fprintf(stderr, "[Quantize-Exp] Arch=%s Layers=%d Tensors=%d\n", arch, n_layers, n_tensors);

    // Create output GGUF: copy KV metadata
    struct gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, inp);
    gguf_set_val_u32(out, "general.quantization_version", 2);
    gguf_set_val_str(out, "general.file_type", variant->name);

    // Plan: for each tensor, decide target type
    struct TensorPlan {
        enum ggml_type target;
        bool           quantize;
        bool           promote;
    };

    std::vector<TensorPlan> plans((size_t) n_tensors);

    for (int i = 0; i < n_tensors; i++) {
        const char *         name   = gguf_get_tensor_name(inp, i);
        struct ggml_tensor * t      = ggml_get_tensor(meta, name);
        const int            n_dims = ggml_n_dims(t);

        gguf_add_tensor(out, t);
        plans[(size_t) i] = { GGML_TYPE_COUNT, false, false };

        enum ggml_type target = pick_type(name, n_dims, arch, *variant, n_layers);

        // Promote 1D norms/biases BF16/F16 -> F32
        if (target == GGML_TYPE_COUNT && should_promote_f32(n_dims) &&
            (t->type == GGML_TYPE_BF16 || t->type == GGML_TYPE_F16)) {
            gguf_set_tensor_type(out, name, GGML_TYPE_F32);
            plans[(size_t) i] = { GGML_TYPE_F32, false, true };
            continue;
        }

        if (target == GGML_TYPE_COUNT) {
            continue;
        }

        bool can_convert = (t->type == GGML_TYPE_BF16 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_F32);
        bool aligned     = (t->ne[0] % ggml_blck_size(target) == 0);

        if (can_convert && aligned) {
            gguf_set_tensor_type(out, name, target);
            plans[(size_t) i] = { target, true, false };
        } else if (can_convert && !aligned) {
            fprintf(stderr, "[Quantize-Exp] WARNING: %s ne[0]=%lld not aligned to block_size=%lld for %s, keeping original\n",
                    name, (long long)t->ne[0], (long long)ggml_blck_size(target), variant->name);
        }
    }

    // Write metadata only (header + tensor info, no data)
    bool ok = gguf_write_to_file(out, out_path, true);
    if (!ok) {
        fprintf(stderr, "[Quantize-Exp] Failed to write metadata %s\n", out_path);
        return 1;
    }

    // Stream tensor data one at a time (low memory)
    FILE * fout = fopen(out_path, "ab");
    if (!fout) {
        fprintf(stderr, "[Quantize-Exp] Failed to open %s for append\n", out_path);
        return 1;
    }

    const size_t alignment   = gguf_get_alignment(out);
    int          n_quantized = 0, n_promoted = 0, n_skipped = 0;
    int64_t      bytes_in = 0, bytes_out = 0;
    size_t       data_pos = 0;

    // Reusable iMatrix buffer
    std::vector<float> imat;

    for (int i = 0; i < n_tensors; i++) {
        const char *         name     = gguf_get_tensor_name(inp, i);
        struct ggml_tensor * t        = ggml_get_tensor(meta, name);
        const int64_t        nel      = ggml_nelements(t);
        const size_t         src_size = ggml_nbytes(t);
        const size_t         t_off    = gguf_get_tensor_offset(inp, i);
        const void *         src      = (const uint8_t *) mapping + data_off + t_off;

        bytes_in += (int64_t) src_size;

        // Pad to alignment boundary
        size_t pad = (alignment - (data_pos % alignment)) % alignment;
        if (pad > 0) {
            uint8_t zeros[64] = {};
            fwrite(zeros, 1, pad, fout);
            data_pos += pad;
        }

        const TensorPlan & plan = plans[(size_t) i];

        if (plan.promote) {
            // BF16/F16 -> F32
            std::vector<float> f32((size_t) nel);
            to_f32(src, f32.data(), nel, t->type);
            size_t out_size = (size_t) nel * sizeof(float);
            fwrite(f32.data(), 1, out_size, fout);
            data_pos += out_size;
            bytes_out += (int64_t) out_size;
            n_promoted++;
        } else if (plan.quantize) {
            // Quantize: src -> f32 -> target
            std::vector<float> f32((size_t) nel);
            to_f32(src, f32.data(), nel, t->type);

            const int64_t n_per_row = t->ne[0];
            const int64_t nrows     = nel / n_per_row;
            const size_t  qsize     = ggml_row_size(plan.target, n_per_row) * (size_t) nrows;

            // Compute iMatrix if this variant uses one
            const float * imat_ptr = nullptr;
            if (variant->needs_imatrix) {
                compute_weight_imatrix(f32.data(), nrows, n_per_row, imat);
                imat_ptr = imat.data();
            }

            std::vector<uint8_t> qbuf(qsize);
            ggml_quantize_chunk(plan.target, f32.data(), qbuf.data(), 0, nrows, n_per_row, imat_ptr);

            fwrite(qbuf.data(), 1, qsize, fout);
            data_pos += qsize;
            bytes_out += (int64_t) qsize;
            n_quantized++;

            // Progress: print every 50th tensor
            if (n_quantized % 50 == 0 || i == n_tensors - 1) {
                fprintf(stderr, "[Quantize-Exp] Progress: %d/%d tensors quantized\r", n_quantized, n_tensors);
            }
        } else {
            // Keep as-is
            fwrite(src, 1, src_size, fout);
            data_pos += src_size;
            bytes_out += (int64_t) src_size;
            n_skipped++;
        }
    }

    fclose(fout);

    // Cleanup IQ lookup tables
    ggml_quantize_free();

    fprintf(stderr, "\n[Quantize-Exp] Quantized %d/%d tensors, promoted %d to F32, kept %d as-is\n",
            n_quantized, n_tensors, n_promoted, n_skipped);
    fprintf(stderr, "[Quantize-Exp] %.1f GB -> %.1f GB (%.1fx compression)\n",
            (double) bytes_in / 1e9, (double) bytes_out / 1e9,
            bytes_out > 0 ? (double) bytes_in / (double) bytes_out : 0.0);
    fprintf(stderr, "[Quantize-Exp] Wrote %s\n", out_path);

    gguf_free(out);
    gguf_free(inp);
    ggml_free(meta);
#ifdef _WIN32
    UnmapViewOfFile(mapping);
    CloseHandle(mh);
    CloseHandle(fh);
#else
    munmap(mapping, file_size);
    close(fd);
#endif

    return 0;
}
