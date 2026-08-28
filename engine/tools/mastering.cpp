// mastering.cpp — Reference-based audio mastering CLI tool
//
// Usage: mastering --target input.wav --reference ref.wav --output mastered.wav
//                  [--no-limiter] [--ceiling <dBFS>] [--pcm24]
//
// Implements the matchering algorithm: spectral + RMS matching against a reference track.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "mastering.h"

// ─── Minimal WAV Reader/Writer ──────────────────────────────────────

#pragma pack(push, 1)
struct WavHeader {
    char     riff[4];        // "RIFF"
    uint32_t file_size;      // file size - 8
    char     wave[4];        // "WAVE"
};
struct WavChunkHdr {
    char     id[4];
    uint32_t size;
};
struct WavFmt {
    uint16_t format;         // 1=PCM, 3=IEEE float
    uint16_t channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};
#pragma pack(pop)

struct WavData {
    std::vector<float> L, R;
    int sample_rate;
    int channels;
};

static bool wav_read(const char * path, WavData & out) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[WAV] Cannot open: %s\n", path);
        return false;
    }

    WavHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1
        || memcmp(hdr.riff, "RIFF", 4) != 0
        || memcmp(hdr.wave, "WAVE", 4) != 0) {
        fprintf(stderr, "[WAV] Invalid WAV header: %s\n", path);
        fclose(f);
        return false;
    }

    WavFmt fmt = {};
    bool found_fmt = false, found_data = false;
    int data_size = 0;

    while (!feof(f)) {
        WavChunkHdr chunk;
        if (fread(&chunk, sizeof(chunk), 1, f) != 1) break;

        if (memcmp(chunk.id, "fmt ", 4) == 0) {
            int to_read = std::min((uint32_t) sizeof(fmt), chunk.size);
            if (fread(&fmt, to_read, 1, f) != 1) break;
            // Skip extra fmt bytes
            if (chunk.size > (uint32_t) to_read) {
                fseek(f, chunk.size - to_read, SEEK_CUR);
            }
            found_fmt = true;
        } else if (memcmp(chunk.id, "data", 4) == 0) {
            data_size = chunk.size;
            found_data = true;
            break;  // data follows
        } else {
            // Skip unknown chunk
            fseek(f, chunk.size, SEEK_CUR);
        }
    }

    if (!found_fmt || !found_data) {
        fprintf(stderr, "[WAV] Missing fmt or data chunk: %s\n", path);
        fclose(f);
        return false;
    }

    // WAVE_FORMAT_EXTENSIBLE (0xFFFE): the real format code is in the sub-format
    // GUID at the end of the extended fmt chunk. The first 2 bytes of the GUID
    // encode the actual format (1=PCM, 3=float). Common for 24/32-bit and
    // multi-channel audio from DAWs.
    if (fmt.format == 0xFFFE) {
        // The extended fmt chunk has: cbSize(2) + validBitsPerSample(2) +
        // channelMask(4) + subFormat GUID(16). We already read the base WavFmt
        // (16 bytes), and we skipped any extra bytes. We need to re-read the
        // extension. Seek back to re-read the extension portion.
        // Actually, we skipped extra fmt bytes — we need to read them before skip.
        // Let's fix: the fmt chunk was already fully consumed (read + skip).
        // We need to handle this in the fmt reading section above. For now,
        // the simplest fix: re-open and re-parse just the fmt extension.
        fprintf(stderr, "[WAV] WAVE_FORMAT_EXTENSIBLE detected, attempting re-parse: %s\n", path);
        fclose(f);

        f = fopen(path, "rb");
        if (!f) return false;
        fseek(f, sizeof(WavHeader), SEEK_SET);

        // Scan for fmt chunk again
        while (!feof(f)) {
            WavChunkHdr chunk2;
            if (fread(&chunk2, sizeof(chunk2), 1, f) != 1) break;
            if (memcmp(chunk2.id, "fmt ", 4) == 0) {
                // Read base fmt (16 bytes)
                WavFmt fmt2 = {};
                int base = std::min((uint32_t)sizeof(fmt2), chunk2.size);
                if (fread(&fmt2, base, 1, f) != 1) break;

                // Read extension: cbSize(2), validBits(2), channelMask(4), subFormat(16)
                if (chunk2.size >= 40) {  // 16 base + 2 cbSize + 2 validBits + 4 mask + 16 GUID
                    uint16_t cb_size = 0;
                    uint16_t valid_bits = 0;
                    uint32_t channel_mask = 0;
                    uint16_t sub_format = 0;

                    fread(&cb_size, 2, 1, f);
                    fread(&valid_bits, 2, 1, f);
                    fread(&channel_mask, 4, 1, f);
                    fread(&sub_format, 2, 1, f);  // first 2 bytes of GUID = real format

                    fmt.format = sub_format;
                    if (valid_bits > 0) {
                        fmt.bits_per_sample = valid_bits;
                    }
                    fprintf(stderr, "[WAV] EXTENSIBLE sub-format: %d (%s), valid bits: %d\n",
                            sub_format, sub_format == 1 ? "PCM" : sub_format == 3 ? "float" : "unknown",
                            valid_bits > 0 ? valid_bits : fmt.bits_per_sample);
                }
                break;
            } else {
                fseek(f, chunk2.size, SEEK_CUR);
            }
        }
        fclose(f);

        // Re-open and seek to data chunk
        f = fopen(path, "rb");
        if (!f) return false;
        fseek(f, sizeof(WavHeader), SEEK_SET);
        found_data = false;
        while (!feof(f)) {
            WavChunkHdr chunk2;
            if (fread(&chunk2, sizeof(chunk2), 1, f) != 1) break;
            if (memcmp(chunk2.id, "data", 4) == 0) {
                data_size = chunk2.size;
                found_data = true;
                break;
            } else {
                fseek(f, chunk2.size, SEEK_CUR);
            }
        }
        if (!found_data) {
            fprintf(stderr, "[WAV] Cannot find data chunk on re-parse: %s\n", path);
            fclose(f);
            return false;
        }
    }

    if (fmt.format != 1 && fmt.format != 3) {
        fprintf(stderr, "[WAV] Unsupported format %d (need PCM=1 or float=3): %s\n",
                fmt.format, path);
        fclose(f);
        return false;
    }

    if (fmt.channels < 1 || fmt.channels > 2) {
        fprintf(stderr, "[WAV] Unsupported channel count %d: %s\n", fmt.channels, path);
        fclose(f);
        return false;
    }

    int bytes_per_sample = fmt.bits_per_sample / 8;
    int n_samples = data_size / (bytes_per_sample * fmt.channels);

    out.sample_rate = fmt.sample_rate;
    out.channels    = fmt.channels;
    out.L.resize(n_samples);
    out.R.resize(n_samples);

    // Read raw data
    std::vector<uint8_t> raw(data_size);
    if (fread(raw.data(), 1, data_size, f) != (size_t) data_size) {
        fprintf(stderr, "[WAV] Truncated data: %s\n", path);
        fclose(f);
        return false;
    }
    fclose(f);

    // Convert to float
    for (int i = 0; i < n_samples; i++) {
        for (int ch = 0; ch < fmt.channels; ch++) {
            int offset = (i * fmt.channels + ch) * bytes_per_sample;
            float val = 0.0f;

            if (fmt.format == 3) {
                // IEEE float
                if (bytes_per_sample == 4) {
                    memcpy(&val, raw.data() + offset, 4);
                } else if (bytes_per_sample == 8) {
                    double dval;
                    memcpy(&dval, raw.data() + offset, 8);
                    val = (float) dval;
                }
            } else {
                // PCM integer
                if (bytes_per_sample == 2) {
                    int16_t ival;
                    memcpy(&ival, raw.data() + offset, 2);
                    val = ival / 32768.0f;
                } else if (bytes_per_sample == 3) {
                    int32_t ival = 0;
                    memcpy(&ival, raw.data() + offset, 3);
                    if (ival & 0x800000) ival |= 0xFF000000;  // sign extend
                    val = ival / 8388608.0f;
                } else if (bytes_per_sample == 4) {
                    int32_t ival;
                    memcpy(&ival, raw.data() + offset, 4);
                    val = (float) ((double) ival / 2147483648.0);
                }
            }

            if (ch == 0) out.L[i] = val;
            else         out.R[i] = val;
        }
    }

    // Mono → stereo
    if (fmt.channels == 1) {
        out.R = out.L;
        out.channels = 2;
    }

    fprintf(stderr, "[WAV] Read %s: %d samples, %d ch, %d Hz, %d-bit %s\n",
            path, n_samples, fmt.channels, fmt.sample_rate,
            fmt.bits_per_sample, fmt.format == 3 ? "float" : "PCM");
    return true;
}

static bool wav_write(const char * path, const float * L, const float * R, int n,
                       int sample_rate, int bits = 16) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[WAV] Cannot create: %s\n", path);
        return false;
    }

    int channels = 2;
    int bytes_per_sample = bits / 8;
    int data_size = n * channels * bytes_per_sample;

    WavHeader hdr;
    memcpy(hdr.riff, "RIFF", 4);
    hdr.file_size = 36 + data_size;
    memcpy(hdr.wave, "WAVE", 4);

    WavChunkHdr fmt_chunk;
    memcpy(fmt_chunk.id, "fmt ", 4);
    fmt_chunk.size = 16;

    WavFmt fmt;
    fmt.format         = (bits == 32) ? 3 : 1;  // float or PCM
    fmt.channels       = channels;
    fmt.sample_rate    = sample_rate;
    fmt.bits_per_sample = bits;
    fmt.block_align    = channels * bytes_per_sample;
    fmt.byte_rate      = sample_rate * fmt.block_align;

    WavChunkHdr data_chunk;
    memcpy(data_chunk.id, "data", 4);
    data_chunk.size = data_size;

    fwrite(&hdr, sizeof(hdr), 1, f);
    fwrite(&fmt_chunk, sizeof(fmt_chunk), 1, f);
    fwrite(&fmt, sizeof(fmt), 1, f);
    fwrite(&data_chunk, sizeof(data_chunk), 1, f);

    // Write interleaved samples
    for (int i = 0; i < n; i++) {
        float l = std::clamp(L[i], -1.0f, 1.0f);
        float r = std::clamp(R[i], -1.0f, 1.0f);

        if (bits == 16) {
            int16_t sl = (int16_t) (l * 32767.0f);
            int16_t sr = (int16_t) (r * 32767.0f);
            fwrite(&sl, 2, 1, f);
            fwrite(&sr, 2, 1, f);
        } else if (bits == 24) {
            int32_t sl = (int32_t) (l * 8388607.0f);
            int32_t sr_v = (int32_t) (r * 8388607.0f);
            fwrite(&sl, 3, 1, f);
            fwrite(&sr_v, 3, 1, f);
        } else if (bits == 32) {
            fwrite(&l, 4, 1, f);
            fwrite(&r, 4, 1, f);
        }
    }

    fclose(f);
    fprintf(stderr, "[WAV] Wrote %s: %d samples, %d Hz, %d-bit\n",
            path, n, sample_rate, bits);
    return true;
}

// ─── CLI ────────────────────────────────────────────────────────────

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s --target input.wav --reference ref.wav --output mastered.wav\n"
        "           [--no-limiter] [--ceiling <dBFS>] [--pcm24] [--pcm32f]\n"
        "\n"
        "Reference-based audio mastering using the matchering algorithm.\n"
        "Matches the RMS level, frequency spectrum, and dynamic range\n"
        "of the target track to the reference track.\n"
        "\n"
        "Options:\n"
        "  --target    PATH   Input audio file to master (WAV)\n"
        "  --reference PATH   Reference track to match against (WAV)\n"
        "  --output    PATH   Output mastered file (WAV)\n"
        "  --no-limiter       Skip peak limiting. wav_write still clamps, so\n"
        "                     any overs become hard clipping.\n"
        "  --ceiling   DBFS   Peak ceiling for the limiter (default: -0.3)\n"
        "  --pcm24            Write 24-bit PCM output (default: 16-bit)\n"
        "  --pcm32f           Write 32-bit float output\n",
        prog);
}

// ─── Look-ahead peak limiter ────────────────────────────────────────
//
// The usage banner has always advertised --no-limiter, but the flag was never
// parsed and there was no limiter to disable: the only thing standing between
// a hot master and the file was the std::clamp in wav_write, i.e. hard
// clipping. Reference-based mastering matches a commercial reference's
// loudness, so overshoot is the normal case, not the exception.
//
// Same construction as audioLevel.ts in the Node chain, deliberately: a
// sliding minimum of the required gain over the look-ahead window, smoothed
// with separate attack and release, then clamped per sample against the
// requirement so the ceiling is a guarantee rather than an aspiration. The
// envelope already accounts for the next lookahead_ms of signal, so the gain
// is down before the peak arrives and the output needs no delay.
//
// Gain is derived from the louder channel at each sample, so stereo image is
// preserved. See docs/plans/2026-08-28-post-processing-gain-staging.md.
static void limit_peaks(std::vector<float> & L, std::vector<float> & R,
                        int sample_rate, float ceiling_db,
                        float lookahead_ms = 5.0f, float release_ms = 50.0f) {
    const int n = (int) L.size();
    if (n <= 0) return;

    const float ceiling = powf(10.0f, ceiling_db / 20.0f);

    std::vector<float> demand((size_t) n, 1.0f);
    float peak_before = 0.0f;
    for (int i = 0; i < n; i++) {
        const float a = std::max(fabsf(L[i]), fabsf(R[i]));
        if (a > peak_before) peak_before = a;
        demand[(size_t) i] = a > ceiling ? ceiling / a : 1.0f;
    }

    if (peak_before <= ceiling) {
        fprintf(stderr, "[Mastering] Limiter: peak %.2f dBFS is under the %.1f dBFS ceiling, not engaged\n",
                20.0f * log10f(std::max(peak_before, 1e-10f)), ceiling_db);
        return;
    }

    // Sliding minimum of demand over [i, i + look], via a monotonic deque.
    const int look = std::max(1, (int) (sample_rate * lookahead_ms / 1000.0f));
    std::vector<float> env((size_t) n, 1.0f);
    std::vector<int>   dq((size_t) n, 0);
    int head = 0, tail = 0;
    for (int i = n - 1; i >= 0; i--) {
        while (tail > head && demand[(size_t) dq[tail - 1]] >= demand[(size_t) i]) tail--;
        dq[tail++] = i;
        while (dq[head] > i + look) head++;
        env[(size_t) i] = demand[(size_t) dq[head]];
    }

    const float attack_coeff  = expf(-1.0f / std::max(1.0f, look / 3.0f));
    const float release_coeff = expf(-1.0f / std::max(1.0f, sample_rate * release_ms / 1000.0f));

    float g = 1.0f, min_gain = 1.0f;
    for (int i = 0; i < n; i++) {
        const float target = env[(size_t) i];
        const float coeff  = target < g ? attack_coeff : release_coeff;
        g = target + (g - target) * coeff;
        if (g > demand[(size_t) i]) g = demand[(size_t) i];   // ceiling guarantee
        if (g < min_gain) min_gain = g;
        L[i] *= g;
        R[i] *= g;
    }

    fprintf(stderr, "[Mastering] Limiter: peak %.2f dBFS -> %.1f dBFS ceiling, %.2f dB max reduction\n",
            20.0f * log10f(std::max(peak_before, 1e-10f)), ceiling_db,
            -20.0f * log10f(std::max(min_gain, 1e-10f)));
}

int main(int argc, char ** argv) {
    const char * target_path = nullptr;
    const char * ref_path    = nullptr;
    const char * output_path = nullptr;
    int          output_bits = 16;
    bool         use_limiter = true;
    float        ceiling_db  = -0.3f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--target") == 0 && i + 1 < argc) {
            target_path = argv[++i];
        } else if (strcmp(argv[i], "--reference") == 0 && i + 1 < argc) {
            ref_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--no-limiter") == 0) {
            use_limiter = false;
        } else if (strcmp(argv[i], "--ceiling") == 0 && i + 1 < argc) {
            ceiling_db = (float) atof(argv[++i]);
        } else if (strcmp(argv[i], "--pcm24") == 0) {
            output_bits = 24;
        } else if (strcmp(argv[i], "--pcm32f") == 0) {
            output_bits = 32;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!target_path || !ref_path || !output_path) {
        fprintf(stderr, "Error: --target, --reference, and --output are required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    // Read input files
    WavData target, reference;
    if (!wav_read(target_path, target)) return 1;
    if (!wav_read(ref_path, reference)) return 1;

    // Run mastering
    auto result = mastering_process(
        target.L.data(), target.R.data(), (int) target.L.size(), target.sample_rate,
        reference.L.data(), reference.R.data(), (int) reference.L.size(), reference.sample_rate
    );

    if (!result.success) {
        fprintf(stderr, "[Mastering] FAILED: %s\n", result.error ? result.error : "unknown");
        return 1;
    }

    // Limit before writing. Without this the only thing between a hot master
    // and the file is the clamp in wav_write, which is hard clipping.
    if (use_limiter) {
        limit_peaks(result.L, result.R, target.sample_rate, ceiling_db);
    } else {
        fprintf(stderr, "[Mastering] Limiter disabled (--no-limiter); wav_write will clamp any overs\n");
    }

    // Write output
    if (!wav_write(output_path, result.L.data(), result.R.data(),
                    (int) result.L.size(), target.sample_rate, output_bits)) {
        return 1;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    fprintf(stderr, "[Mastering] Total time: %.2f seconds\n", elapsed);

    return 0;
}
