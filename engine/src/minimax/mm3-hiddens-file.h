#pragma once
// minimax/mm3-hiddens-file.h — the MM3 AR cache, on disk.
//
// HOT-Step file (does not exist upstream).
//
// ── What this is ────────────────────────────────────────────────────────────
//
// mm3-ar-cache.h holds ONE stage-1 plan in host RAM and loses it on restart.
// This serialises that slot to a `.mm3hiddens` file and reads it back, so a
// pinned plan survives a cold start, can be kept beside a song, and can be
// handed to someone else.
//
// The payload is the [frames, num_codebooks, embedding_length] f32 hidden
// block plus the stage-1 byproducts the cache holds alongside it (the AR
// codes, the LRC, the EOS flag), so a file replay is indistinguishable from an
// in-memory hit downstream — plank capture and x-lrc-text keep working.
//
// ── Why it carries the cache key ────────────────────────────────────────────
//
// The in-memory slot is safe because mm3_ar_cache_key() decides what may hit
// it. A file has no such protection unless it carries the key with it: a blob
// made under a different LM quant, or with a different adapter merged, is
// numerically meaningless against the current model but has exactly the same
// SHAPE, so a dimension check would wave it straight through and the render
// would be silently wrong.
//
// So the file stores two keys and they are checked differently:
//
//   • MODEL key (mm3_ar_model_key) — the LM and depth files, their byte counts
//     and quant, and the LM adapter's identity and dials. These decide whether
//     the block means anything at all. A mismatch is a REFUSAL: the run plans
//     fresh rather than rendering from a block the current model did not make.
//
//   • FULL key (mm3_ar_cache_key) — the above plus prompt, max_frames, seed and
//     the LRC flag. These decide what the plan IS, not whether it is valid. A
//     mismatch is a WARNING and the render proceeds: replaying a saved bed
//     against an edited caption is a legitimate thing to do, it just means the
//     caption box no longer describes what you are about to hear. (The caption
//     only ever reaches the flow DiT through these hiddens, so the edit has no
//     other effect.) `max_frames` is in the same boat — the file's frame count
//     is where the render length comes from, so the duration slider loses.
//
// Both keys are compared verbatim against ones built by the running binary, so
// bumping the `v=` field in either invalidates every file made before it.
//
// ── Size ────────────────────────────────────────────────────────────────────
//
// 128 KB per frame at f32, ~3 MB per second of audio, ~600 MB for a 200 s song.
// That is 4096x what the equivalent plank costs (a plank stores one i32 per
// codebook per frame; this stores 4096 f32). Fine for a handful of pinned beds,
// hopeless as a library of past renders — which is why the plank is still the
// right artefact for archival, and this one is for pinning a plan you are about
// to iterate flow settings against.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// 64-bit file offsets: `long` is 32 bits on MSVC, and the duration ceiling
// already allows a 900 MB plan — close enough to 2 GB not to rely on it.
#ifdef _WIN32
#    define MM3_FTELL  _ftelli64
#    define MM3_FSEEK  _fseeki64
#else
#    define MM3_FTELL  ftello
#    define MM3_FSEEK  fseeko
#endif

// "MM3HIDN" + format revision. Bump the trailing digit only for a layout change
// the reader below cannot parse; a semantic change belongs in the key instead.
static const char MM3_HIDDENS_MAGIC[8] = { 'M', 'M', '3', 'H', 'I', 'D', 'N', '1' };

/** Everything a `.mm3hiddens` file holds. Mirrors MM3ArCache minus `hits`.
 *
 *  `hiddens` is filled by the READER only. The writer takes the block by
 *  pointer instead, so saving never has to duplicate it — at ~600 MB for a
 *  200 s song, a copy just to hand it to a function is not a rounding error. */
struct MM3HiddensFile {
    int64_t              frames        = 0;
    int64_t              num_codebooks = 0;
    int64_t              embedding_len = 0;
    std::string          model_key;
    std::string          full_key;
    std::vector<float>   hiddens;
    std::vector<int32_t> semantic_all;
    std::vector<int32_t> acoustic_all;
    std::string          lrc;
    bool                 eos_hit = false;
};

// ── little scalar/blob helpers ──────────────────────────────────────────────
//
// Written host-endian on purpose: these files are a local cache, not a wire
// format, and every platform we build for is little-endian. The magic would not
// catch a byte-swapped file, but the shape check right after it would.

static bool mm3_hf_wr(FILE * fp, const void * p, size_t n) {
    return n == 0 || fwrite(p, 1, n, fp) == n;
}
static bool mm3_hf_rd(FILE * fp, void * p, size_t n) {
    return n == 0 || fread(p, 1, n, fp) == n;
}
static bool mm3_hf_wr_i32(FILE * fp, int32_t v) { return mm3_hf_wr(fp, &v, sizeof(v)); }
static bool mm3_hf_wr_i64(FILE * fp, int64_t v) { return mm3_hf_wr(fp, &v, sizeof(v)); }

static bool mm3_hf_wr_str(FILE * fp, const std::string & s) {
    return mm3_hf_wr_i32(fp, (int32_t) s.size()) && mm3_hf_wr(fp, s.data(), s.size());
}
static bool mm3_hf_wr_codes(FILE * fp, const std::vector<int32_t> & v) {
    return mm3_hf_wr_i32(fp, (int32_t) v.size()) && mm3_hf_wr(fp, v.data(), v.size() * sizeof(int32_t));
}

// A length field is the one thing a truncated or foreign file can turn into a
// gigantic allocation, so every one of them is bounded before it is used.
static bool mm3_hf_rd_len(FILE * fp, int64_t max, int64_t * out) {
    int32_t n = 0;
    if (!mm3_hf_rd(fp, &n, sizeof(n)) || n < 0 || (int64_t) n > max) {
        return false;
    }
    *out = n;
    return true;
}
static bool mm3_hf_rd_str(FILE * fp, int64_t max, std::string * out) {
    int64_t n = 0;
    if (!mm3_hf_rd_len(fp, max, &n)) {
        return false;
    }
    out->resize((size_t) n);
    return n == 0 || mm3_hf_rd(fp, &(*out)[0], (size_t) n);
}
static bool mm3_hf_rd_codes(FILE * fp, int64_t max, std::vector<int32_t> * out) {
    int64_t n = 0;
    if (!mm3_hf_rd_len(fp, max, &n)) {
        return false;
    }
    out->resize((size_t) n);
    return n == 0 || mm3_hf_rd(fp, out->data(), (size_t) n * sizeof(int32_t));
}

/** Write one slot to `path`. `hiddens` points at f.frames * num_codebooks *
 *  embedding_len floats — passed separately so the caller can write straight
 *  out of the live cache slot without copying it.
 *
 *  Writes to `path.tmp` and renames, so an interrupted save leaves the previous
 *  file intact rather than a plausible-looking truncated one. `err` gets a
 *  human-readable reason on failure; the caller logs it and carries on, because
 *  a failed save must never fail the render that produced the audio. */
static bool mm3_hiddens_write(const std::string & path, const MM3HiddensFile & f, const float * hiddens,
                              std::string * err) {
    const int64_t want = f.frames * f.num_codebooks * f.embedding_len;
    if (f.frames <= 0 || want <= 0 || hiddens == nullptr) {
        if (err) { *err = "refusing to write a hidden block of the wrong size"; }
        return false;
    }
    const std::string tmp = path + ".tmp";
    FILE *            fp  = fopen(tmp.c_str(), "wb");
    if (!fp) {
        if (err) { *err = "cannot open " + tmp + " for writing"; }
        return false;
    }
    bool ok = mm3_hf_wr(fp, MM3_HIDDENS_MAGIC, sizeof(MM3_HIDDENS_MAGIC))
           && mm3_hf_wr_i64(fp, f.frames)
           && mm3_hf_wr_i64(fp, f.num_codebooks)
           && mm3_hf_wr_i64(fp, f.embedding_len)
           && mm3_hf_wr_i32(fp, f.eos_hit ? 1 : 0)
           && mm3_hf_wr_str(fp, f.model_key)
           && mm3_hf_wr_str(fp, f.full_key)
           && mm3_hf_wr_codes(fp, f.semantic_all)
           && mm3_hf_wr_codes(fp, f.acoustic_all)
           && mm3_hf_wr_str(fp, f.lrc)
           && mm3_hf_wr(fp, hiddens, (size_t) want * sizeof(float));
    // fflush before fclose so a full disk surfaces here rather than silently.
    ok = (fflush(fp) == 0) && ok;
    fclose(fp);
    if (!ok) {
        std::remove(tmp.c_str());
        if (err) { *err = "write failed (disk full?)"; }
        return false;
    }
    // rename() will not replace an existing file on Windows, so clear the way.
    std::remove(path.c_str());
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        if (err) { *err = "cannot rename " + tmp + " into place"; }
        return false;
    }
    return true;
}

/** Read `path` into `out`. Structural failures only — the KEY checks are the
 *  caller's, because refusing on one and warning on the other is a policy
 *  decision that belongs where the logging is. */
static bool mm3_hiddens_read(const std::string & path, MM3HiddensFile * out, std::string * err) {
    FILE * fp = fopen(path.c_str(), "rb");
    if (!fp) {
        if (err) { *err = "cannot open " + path; }
        return false;
    }
    auto bail = [&](const char * why) {
        fclose(fp);
        out->hiddens.clear();
        if (err) { *err = why; }
        return false;
    };
    char magic[8] = { 0 };
    if (!mm3_hf_rd(fp, magic, sizeof(magic)) || memcmp(magic, MM3_HIDDENS_MAGIC, sizeof(magic)) != 0) {
        return bail("not a .mm3hiddens file (bad magic), or written by an incompatible build");
    }
    if (!mm3_hf_rd(fp, &out->frames, sizeof(int64_t))
        || !mm3_hf_rd(fp, &out->num_codebooks, sizeof(int64_t))
        || !mm3_hf_rd(fp, &out->embedding_len, sizeof(int64_t))) {
        return bail("truncated header");
    }
    // 2^20 frames is ~12 hours of audio and ~137 GB of hiddens; anything past
    // that is a corrupt length field, not a long song.
    if (out->frames <= 0 || out->frames > (1 << 20)
        || out->num_codebooks <= 0 || out->num_codebooks > 64
        || out->embedding_len <= 0 || out->embedding_len > (1 << 16)) {
        return bail("implausible shape in header");
    }
    int32_t eos = 0;
    if (!mm3_hf_rd(fp, &eos, sizeof(eos))) {
        return bail("truncated header");
    }
    out->eos_hit = eos != 0;
    // + slack for the un-emitted iteration 0 and any future per-frame extra.
    const int64_t code_max = out->frames * out->num_codebooks + 1024;
    if (!mm3_hf_rd_str(fp, 1 << 20, &out->model_key)
        || !mm3_hf_rd_str(fp, 1 << 20, &out->full_key)
        || !mm3_hf_rd_codes(fp, code_max, &out->semantic_all)
        || !mm3_hf_rd_codes(fp, code_max, &out->acoustic_all)
        || !mm3_hf_rd_str(fp, 1 << 22, &out->lrc)) {
        return bail("truncated or corrupt metadata block");
    }
    // Size the allocation against the FILE, not against the header. The bounds
    // above still admit shapes worth terabytes, and `resize` on one of those
    // throws bad_alloc out of a read that is supposed to return false — so the
    // remaining byte count is what decides, and it has to match exactly.
    const int64_t want = out->frames * out->num_codebooks * out->embedding_len;
    const int64_t pos = (int64_t) MM3_FTELL(fp);
    if (pos < 0 || MM3_FSEEK(fp, 0, SEEK_END) != 0) {
        return bail("cannot measure the file");
    }
    const int64_t end = (int64_t) MM3_FTELL(fp);
    if (end < 0 || MM3_FSEEK(fp, pos, SEEK_SET) != 0) {
        return bail("cannot measure the file");
    }
    if (end - pos != want * (int64_t) sizeof(float)) {
        return bail("the hidden block is not the size the header claims");
    }
    out->hiddens.resize((size_t) want);
    if (!mm3_hf_rd(fp, out->hiddens.data(), (size_t) want * sizeof(float))) {
        return bail("truncated hidden block");
    }
    fclose(fp);
    return true;
}
