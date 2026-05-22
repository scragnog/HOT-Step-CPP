#pragma once
// silence-latent.h: read silence_latent.pt from safetensors model directories
//
// PyTorch .pt files are ZIP archives with a data entry containing raw f32.
// The silence_latent tensor is [64, 15000] f32 in PyTorch layout.
// We transpose to [15000, 64] for ggml (64 contiguous per frame).
//
// Reference: convert.py read_silence_latent()

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Minimal ZIP reader for PyTorch .pt files.
// We only need to find and read the "*/data/0" entry.
//
// PyTorch >= 2.x writes ZIP files with data descriptors (flag bit 3 set),
// which means compressed_size/uncompressed_size in local file headers are 0.
// We use the central directory (at end of ZIP) which always has correct sizes.
//
// Central directory entry (46+ bytes):
//   [PK\x01\x02][2B ver_made][2B ver_needed][2B flags][2B method]
//   [4B datetime][4B crc32][4B comp_size][4B uncomp_size]
//   [2B name_len][2B extra_len][2B comment_len]
//   [2B disk_start][2B int_attr][4B ext_attr][4B local_header_offset]
//   [name][extra][comment]
//
// End of central directory (22 bytes):
//   [PK\x05\x06][2B disk][2B disk_cd][2B n_entries_disk][2B n_entries_total]
//   [4B cd_size][4B cd_offset][2B comment_len]

static bool sl_read_silence_latent(const char * pt_path,
                                   std::vector<float> & out,
                                   int expected_dim0 = 64,
                                   int expected_dim1 = 15000) {
    FILE * f = fopen(pt_path, "rb");
    if (!f) {
        fprintf(stderr, "[SilenceLatent] Cannot open %s\n", pt_path);
        return false;
    }

    // Get file size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    // Read entire file (silence_latent.pt is ~3.7 MB)
    std::vector<uint8_t> buf(file_size);
    if ((long) fread(buf.data(), 1, file_size, f) != file_size) {
        fclose(f);
        fprintf(stderr, "[SilenceLatent] Read error %s\n", pt_path);
        return false;
    }
    fclose(f);

    // Find End of Central Directory record (scan backwards from EOF)
    // EOCD signature: PK\x05\x06
    size_t eocd_pos = 0;
    bool found_eocd = false;
    for (size_t i = (size_t) file_size - 22; i > 0 && i < (size_t) file_size; i--) {
        if (buf[i] == 0x50 && buf[i + 1] == 0x4B &&
            buf[i + 2] == 0x05 && buf[i + 3] == 0x06) {
            eocd_pos = i;
            found_eocd = true;
            break;
        }
    }
    if (!found_eocd) {
        fprintf(stderr, "[SilenceLatent] No EOCD in %s\n", pt_path);
        return false;
    }

    uint32_t cd_size   = *(uint32_t *) &buf[eocd_pos + 12];
    uint32_t cd_offset = *(uint32_t *) &buf[eocd_pos + 16];

    if (cd_offset + cd_size > (size_t) file_size) {
        fprintf(stderr, "[SilenceLatent] Corrupt central directory in %s\n", pt_path);
        return false;
    }

    // Scan central directory entries for "*/data/0"
    size_t pos = cd_offset;
    size_t cd_end = cd_offset + cd_size;

    while (pos + 46 <= cd_end) {
        // Check central directory signature: PK\x01\x02
        if (buf[pos] != 0x50 || buf[pos + 1] != 0x4B ||
            buf[pos + 2] != 0x01 || buf[pos + 3] != 0x02) {
            break;
        }

        uint16_t method          = *(uint16_t *) &buf[pos + 10];
        uint32_t uncomp_size     = *(uint32_t *) &buf[pos + 24];
        uint16_t name_len        = *(uint16_t *) &buf[pos + 28];
        uint16_t extra_len       = *(uint16_t *) &buf[pos + 30];
        uint16_t comment_len     = *(uint16_t *) &buf[pos + 32];
        uint32_t local_offset    = *(uint32_t *) &buf[pos + 42];

        size_t name_start = pos + 46;
        if (name_start + name_len > cd_end) break;

        std::string entry_name((char *) &buf[name_start], name_len);
        bool is_data0 = (entry_name.size() >= 7 &&
                         entry_name.compare(entry_name.size() - 7, 7, "/data/0") == 0);

        if (is_data0 && method == 0) {
            // Found the data entry — compute actual data position from the local header
            // Local header: 30 bytes + name_len + extra_len (may differ from central)
            if (local_offset + 30 > (size_t) file_size) break;
            uint16_t local_name_len  = *(uint16_t *) &buf[local_offset + 26];
            uint16_t local_extra_len = *(uint16_t *) &buf[local_offset + 28];
            size_t data_start = local_offset + 30 + local_name_len + local_extra_len;

            size_t nbytes = uncomp_size;
            size_t expected_bytes = (size_t) expected_dim0 * expected_dim1 * sizeof(float);

            if (nbytes != expected_bytes) {
                fprintf(stderr, "[SilenceLatent] WARNING: expected %zu bytes, got %u in %s\n",
                        expected_bytes, uncomp_size, pt_path);
                if (nbytes < expected_bytes) {
                    return false;
                }
            }
            if (data_start + nbytes > (size_t) file_size) {
                fprintf(stderr, "[SilenceLatent] Truncated data in %s\n", pt_path);
                return false;
            }

            // Source: [64, 15000] f32 (PyTorch row-major: 15000 contiguous per row)
            // Target: [15000, 64] f32 (ggml: 64 contiguous per frame)
            const float * src = (const float *) &buf[data_start];
            out.resize(expected_dim0 * expected_dim1);
            for (int i = 0; i < expected_dim0; i++) {
                for (int j = 0; j < expected_dim1; j++) {
                    out[j * expected_dim0 + i] = src[i * expected_dim1 + j];
                }
            }

            fprintf(stderr, "[SilenceLatent] Loaded [%d, %d] f32 from %s (%.1f MB)\n",
                    expected_dim1, expected_dim0, pt_path,
                    (float) out.size() * sizeof(float) / (1024 * 1024));
            return true;
        }

        // Advance to next central directory entry
        pos = name_start + name_len + extra_len + comment_len;
    }

    fprintf(stderr, "[SilenceLatent] No data/0 entry found in %s\n", pt_path);
    return false;
}
