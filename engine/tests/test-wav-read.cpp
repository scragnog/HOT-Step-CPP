// read_wav_buf (wav.h): integer PCM 16/24/32 and float32, classic and
// WAVE_FORMAT_EXTENSIBLE, mono and stereo. GH #359 was 32-bit integer PCM.
// No GPU or models. Exit 0 = every case passed.
#include "../src/wav.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int fails = 0;

static void put16(std::vector<uint8_t> & b, uint32_t v) {
    b.push_back((uint8_t) v);
    b.push_back((uint8_t) (v >> 8));
}

static void put32(std::vector<uint8_t> & b, uint32_t v) {
    put16(b, v);
    put16(b, v >> 16);
}

// tag 1 = PCM, 3 = float; extensible wraps it in 0xFFFE with that subformat.
static std::vector<uint8_t> make_wav(uint16_t tag, bool ext, int ch, int bits, const std::vector<double> & s) {
    std::vector<uint8_t> fmt, dat, w;
    const int            bps = bits / 8;
    put16(fmt, ext ? 0xfffe : tag);
    put16(fmt, (uint32_t) ch);
    put32(fmt, 44100);
    put32(fmt, (uint32_t) (44100 * ch * bps));
    put16(fmt, (uint32_t) (ch * bps));
    put16(fmt, (uint32_t) bits);
    if (ext) {
        put16(fmt, 22);                 // cbSize
        put16(fmt, (uint32_t) bits);    // valid bits
        put32(fmt, ch == 1 ? 4 : 3);    // channel mask
        put16(fmt, tag);                // subformat GUID, first two bytes
        static const uint8_t guid_tail[14] = { 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80,
                                               0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71 };
        fmt.insert(fmt.end(), guid_tail, guid_tail + 14);
    }
    for (double v : s) {
        if (tag == 3) {
            float f = (float) v;
            uint32_t u;
            memcpy(&u, &f, 4);
            put32(dat, u);
        } else {
            const double  full = std::ldexp(1.0, bits - 1);
            const int64_t q    = std::llround(std::fmin(v * full, full - 1));
            for (int i = 0; i < bps; i++) {
                dat.push_back((uint8_t) ((uint64_t) q >> (8 * i)));
            }
        }
    }
    w.insert(w.end(), { 'R', 'I', 'F', 'F' });
    put32(w, (uint32_t) (4 + 8 + fmt.size() + 8 + dat.size()));
    w.insert(w.end(), { 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ' });
    put32(w, (uint32_t) fmt.size());
    w.insert(w.end(), fmt.begin(), fmt.end());
    w.insert(w.end(), { 'd', 'a', 't', 'a' });
    put32(w, (uint32_t) dat.size());
    w.insert(w.end(), dat.begin(), dat.end());
    return w;
}

static void check(const char * name, uint16_t tag, bool ext, int ch, int bits) {
    // Per channel: silence, half scale, negative full scale, a small value.
    const std::vector<double> L = { 0.0, 0.5, -1.0, 0.123 };
    const std::vector<double> R = { 0.25, -0.5, 0.75, -0.001 };
    std::vector<double>       s;
    for (size_t t = 0; t < L.size(); t++) {
        s.push_back(L[t]);
        if (ch == 2) {
            s.push_back(R[t]);
        }
    }
    const std::vector<uint8_t> buf = make_wav(tag, ext, ch, bits, s);
    int                        T = 0, sr = 0;
    float *                    a   = read_wav_buf(buf.data(), buf.size(), &T, &sr);
    const double               tol = std::fmax(1e-7, tag == 3 ? 0.0 : 1.5 / std::ldexp(1.0, bits - 1));  // float32 keeps ~24 bits
    bool                       ok  = a && T == (int) L.size() && sr == 44100;
    for (int t = 0; ok && t < T; t++) {
        const double r = ch == 2 ? R[t] : L[t];
        ok = std::fabs(a[t * 2] - L[t]) <= tol && std::fabs(a[t * 2 + 1] - r) <= tol;
    }
    printf("%s: %s\n", ok ? "PASS" : "FAIL", name);
    fails += ok ? 0 : 1;
    free(a);
}

int main() {
    for (int ch = 1; ch <= 2; ch++) {
        const char * c = ch == 1 ? "mono" : "stereo";
        char         n[64];
        for (int bits : { 16, 24, 32 }) {
            snprintf(n, sizeof n, "PCM%d %s", bits, c);
            check(n, 1, false, ch, bits);
            snprintf(n, sizeof n, "PCM%d EXTENSIBLE %s", bits, c);
            check(n, 1, true, ch, bits);
        }
        snprintf(n, sizeof n, "float32 %s", c);
        check(n, 3, false, ch, 32);
        snprintf(n, sizeof n, "float32 EXTENSIBLE %s", c);
        check(n, 3, true, ch, 32);
    }
    // Unsupported formats must still be refused: 8-bit PCM and 64-bit float.
    for (auto f : { std::pair<uint16_t, int>{ 1, 8 }, std::pair<uint16_t, int>{ 3, 64 } }) {
        std::vector<uint8_t> buf = make_wav(1, false, 1, f.second, { 0.0, 0.0 });
        buf[20]                  = (uint8_t) f.first;
        int  T = 0, sr = 0;
        bool ok = read_wav_buf(buf.data(), buf.size(), &T, &sr) == NULL;
        printf("%s: refuses format=%u bits=%d\n", ok ? "PASS" : "FAIL", f.first, f.second);
        fails += ok ? 0 : 1;
    }
    printf("%s\n", fails ? "FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
