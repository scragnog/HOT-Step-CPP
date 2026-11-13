#pragma once
// yourmt3/inference.h: YourMT3+ end to end, as MuLaCover's MelodyTranscriber
// runs it: mono audio at any rate -> 16 kHz -> 32,767-sample segments in
// batches of 8 -> spectrogram, pre-encoder, encoder, projection, greedy
// 13-channel decode -> token streams -> notes in seconds (events.h), with the
// program-100 vocal and the drums selected from them. Stage times are kept.

#include "yourmt3/decoder.h"
#include "yourmt3/events.h"

namespace yourmt3 {

struct Transcript {
    int                           n_seg = 0, steps = 0, errors = 0;
    std::vector<std::vector<int>> tokens;  // [segment * 13 + channel], each its batch's length
    std::vector<Note>             notes, vocal, drums;
    bool                          finite = true;
    double ms_resample = 0, ms_spec = 0, ms_pre = 0, ms_enc = 0, ms_dec = 0, ms_events = 0;
};

static void transcribe(Encoder & e, Decoder & d, const std::vector<float> & mono, int sr, Transcript & t) {
    double             t0  = now_ms();
    std::vector<float> wav = pad_segments(resample(mono, sr, SR), &t.n_seg);
    double             t1  = now_ms();
    std::vector<float> spec = spectrogram(wav.data(), t.n_seg, e.w.window.data());
    t.ms_resample = t1 - t0;
    t.ms_spec     = now_ms() - t1;
    t.tokens.clear();
    for (int b0 = 0; b0 < t.n_seg; b0 += BATCH) {
        const int          B   = std::min(BATCH, t.n_seg - b0);
        double             a   = now_ms();
        std::vector<float> pre = pre_encode(e, &spec[(size_t) b0 * NFRAME * NBIN], B);
        double             m   = now_ms();
        std::vector<float> enc = encode(e, pre.data(), B);
        double             z   = now_ms();
        decoder_prepare(e, d, enc.data(), B, false);
        DecodeOut o;
        decoder_greedy(e, d, o);
        t.ms_pre += m - a;
        t.ms_enc += z - m;
        t.ms_dec += now_ms() - z;
        t.steps += o.steps;
        t.finite &= o.finite;
        for (auto & s : o.tok) t.tokens.push_back(std::move(s));
    }
    double t2 = now_ms();
    t.errors  = tokens_to_notes(t.tokens, t.n_seg, N_CH, SEG, SR, t.notes);
    t.vocal   = select_vocal(t.notes);
    t.drums   = select_drums(t.notes);
    t.ms_events = now_ms() - t2;
}

}  // namespace yourmt3
