#pragma once
// preprocess-run.h: model context (VAE-Enc + Qwen3 text encoder + BPE +
// condition encoder + silence_latent), prompt construction and the per-song
// preprocessing pipeline for `ace-train preprocess`.
//
// docs/plans/2026-07-27-preprocess-implementation.md §3.4

#include "audio-io.h"
#include "bpe.h"
#include "cond-enc.h"
#include "gguf-weights.h"
#include "qwen3-enc.h"
#include "silence-latent.h"
#include "task-types.h"
#include "vae-enc.h"
#include "version.h"

#include "train/lm-common.h"     // lm_normalize_language — parity with inference
#include "train/preprocess-io.h"
#include "train/st-write.h"

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

struct PreprocessOpts {
    std::string dit_path, vae_path, text_enc_path, out_dir, ffmpeg;
    // 600 (2026-08-29) — 240 truncated a third of the corpus and taught the
    // LM a false mid-phrase ending on every capped track. Matches the
    // engine's 10-minute generation ceiling.
    int         max_duration   = 600;
    bool        normalize_peak = true;
    float       target_db      = -1.0f;
    STWDType    dtype          = STW_F32;
    bool        sidestep_compat = false;  // --compat sidestep
    bool        timbre_zeros    = false;  // --timbre zeros
    int         max_caption_tokens = 256;
    int         max_lyric_tokens   = 512;
    int         vae_chunk = 384, vae_overlap = 48;   // ~15.4s tiles: caps VAE-encode activations ~5GB (1024 cost ~17GB)
    bool        overwrite = false;

    // recorded verbatim into __metadata__
    std::string model_variant;
};

struct PreprocessCtx {
    VAEEncoder         vae = {};
    Qwen3GGML          te  = {};
    CondGGML           ce  = {};
    BPETokenizer       bpe = {};
    std::vector<float> silence;  // 15000 * 64, [t*64+c]
    int                silence_T = 0;

    bool vae_loaded = false, te_loaded = false, ce_loaded = false;
};

struct SongResult {
    bool               ok = false;
    std::string        error;
    int                t_latent = 0, s_total = 0, s_text = 0, s_lyric = 0, s_total_genre = 0, s_text_genre = 0;
    bool               has_genre = false;
    long long          bytes     = 0;
    std::vector<float> latents;  // [T,64] — kept for channel-stats accumulation
};

// ─── model context ──────────────────────────────────────────────────────────

static bool pp_ends_with_gguf(const std::string & p) {
    return p.size() >= 5 && pm_lower(p.substr(p.size() - 5)) == ".gguf";
}

// Stage sink so the caller can time and report each load stage.
// (Minimal extension of §3.4's signature — defaulted, so the documented
//  3-argument call still compiles.)
typedef void (*PCtxStageFn)(const char * stage, const char * path, long long ms);

// Loads DiT meta (silence_latent), VAE-Enc, text encoder, BPE, cond encoder —
// in that order, all kept resident.
static bool pctx_load(PreprocessCtx & c, const PreprocessOpts & o, std::string & err, PCtxStageFn stage = NULL) {
    int64_t t0 = ggml_time_ms();
#define PP_STAGE(name, path)                              \
    do {                                                  \
        int64_t _t1 = ggml_time_ms();                     \
        if (stage) stage((name), (path), (long long) (_t1 - t0)); \
        t0 = _t1;                                         \
    } while (0)

    // 1. silence_latent — read straight out of the DiT GGUF; NO ModelStore.
    if (pp_ends_with_gguf(o.dit_path)) {
        GGUFModel gf = {};
        if (!gf_load(&gf, o.dit_path.c_str())) {
            err = "cannot open DiT " + o.dit_path;
            return false;
        }
        const void * sl = gf_get_data(gf, "silence_latent");  // [15000,64] f32, already transposed
        if (!sl) {
            gf_close(&gf);
            err = "silence_latent missing from " + o.dit_path;
            return false;
        }
        c.silence.assign((const float *) sl, (const float *) sl + 15000 * 64);
        c.silence_T = 15000;
        gf_close(&gf);
    } else {
        std::string pt = o.dit_path + WS_SEP + "silence_latent.pt";
        if (!sl_read_silence_latent(pt.c_str(), c.silence)) {
            err = "cannot read " + pt;
            return false;
        }
        c.silence_T = (int) (c.silence.size() / 64);
    }
    if (c.silence_T <= 0) {
        err = "silence_latent is empty in " + o.dit_path;
        return false;
    }
    PP_STAGE("dit-meta", o.dit_path.c_str());

    // 2. VAE encoder (vae_enc_load exits internally on a bad path — validated by the caller)
    vae_enc_load(&c.vae, o.vae_path.c_str());
    c.vae_loaded = true;
    PP_STAGE("vae", o.vae_path.c_str());

    // 3. Text encoder + BPE (the Qwen3 GGUF carries the tokenizer KV)
    if (!qwen3_load_text_encoder(&c.te, o.text_enc_path.c_str())) {
        err = "cannot load text encoder " + o.text_enc_path;
        return false;
    }
    c.te_loaded = true;

    // cond_ggml_forward hard-codes a 1024-wide text/lyric input (cond-enc.h:261,
    // :331). A larger Qwen3-Embedding (4B = 2560, 8B = 4096) would silently be
    // reinterpreted 1024 floats at a time — no NaN, every NaN check passes, and
    // the whole cache is numerically plausible garbage. Refuse it up front.
    if (c.te.cfg.hidden_size != 1024) {
        char b[128];
        snprintf(b, sizeof(b), "text encoder hidden_size is %d, but the condition encoder requires 1024",
                 c.te.cfg.hidden_size);
        err = std::string(b) + " — use a Qwen3-Embedding-0.6B model (" + o.text_enc_path + ")";
        return false;
    }

    bool ok = pp_ends_with_gguf(o.text_enc_path)
                  ? load_bpe_from_gguf(&c.bpe, o.text_enc_path.c_str())
                  : load_bpe_from_files(&c.bpe, (o.text_enc_path + WS_SEP + "vocab.json").c_str(),
                                        (o.text_enc_path + WS_SEP + "merges.txt").c_str());
    if (!ok) {
        err = "BPE tokenizer not found in " + o.text_enc_path;
        return false;
    }
    PP_STAGE("text-enc", o.text_enc_path.c_str());

    // 4. Condition encoder — rides on the DiT path, not a separate file
    if (!cond_ggml_load(&c.ce, o.dit_path.c_str())) {
        err = "cannot load condition encoder from " + o.dit_path;
        return false;
    }
    c.ce_loaded = true;
    PP_STAGE("cond-enc", o.dit_path.c_str());

#undef PP_STAGE
    return true;
}

static void pctx_free(PreprocessCtx & c) {
    if (c.ce_loaded) {
        cond_ggml_free(&c.ce);
        c.ce_loaded = false;
    }
    if (c.te_loaded) {
        qwen3_free(&c.te);
        c.te_loaded = false;
    }
    if (c.vae_loaded) {
        vae_enc_free(&c.vae);
        c.vae_loaded = false;
    }
    c.silence.clear();
    c.silence_T = 0;
}

// ─── prompt construction (ports build_simple_prompt + build_prompt_strings) ──

static std::string pp_caption_text(const PSample & s, const PManifest & mf, bool use_genre) {
    std::string text;
    if (s.prompt_override == "genre") {
        text = s.genre;
    } else if (s.prompt_override == "caption") {
        text = s.caption;
    } else if (!use_genre) {
        text = s.caption;
    } else {
        text = s.genre.empty() ? s.caption : s.genre;
    }

    const std::string tag = s.custom_tag.empty() ? mf.custom_tag : s.custom_tag;
    std::string       pos = s.tag_position.empty() ? mf.tag_position : s.tag_position;
    if (pos.empty()) {
        pos = "prepend";
    }
    if (!tag.empty()) {
        if (pos == "prepend") {
            text = text.empty() ? tag : tag + ", " + text;
        } else if (pos == "append") {
            text = text.empty() ? tag : text + ", " + tag;
        } else if (pos == "replace") {
            // The caption IS the trigger (2026-08-12). Every descriptive word
            // from the sidecar is dropped, so the trigger token alone has to
            // carry genre/timbre/feel and the adapter is forced to learn them
            // rather than lean on the prose. The Build panel has always
            // labelled this "Instead of the caption"; until now preprocess
            // silently did nothing (verbatim Side-Step), which trained an
            // adapter whose captions carried no trigger at all.
            //
            // The `# Metas` block (bpm/timesignature/keyscale/duration) and
            // the lyric prompt are built separately in pp_prompt_strings and
            // are NOT affected — only the caption text collapses.
            text = tag;
        }
    }
    return text;
}

static void pp_prompt_strings(const PSample & s, const std::string & caption_text, int duration, bool sidestep,
                              std::string & text_out, std::string & lyric_out) {
    char bpm_b[16] = "N/A";
    if (s.bpm > 0) {
        snprintf(bpm_b, sizeof(bpm_b), "%d", s.bpm);
    }
    const char * ks   = s.keyscale.empty() ? "N/A" : s.keyscale.c_str();
    // Both normalized to what INFERENCE conditions on (2026-08-12 parity):
    // timesig numerator ('4/4' → '4') and ISO language ('english' → 'en').
    // Raw values here baked "timesignature: 4/4" / "Languages\nenglish" into
    // the training tensors while every generation sends '4' / 'en'.
    const std::string ts_n   = pm_timesig_numerator(s.timesignature);
    const std::string lang_n = lm_normalize_language(s.language);
    const char * ts   = ts_n.empty() ? "N/A" : ts_n.c_str();
    const char * lang = lang_n.empty() ? "unknown" : lang_n.c_str();

    char metas[512];
    snprintf(metas, sizeof(metas), "- bpm: %s\n- timesignature: %s\n- keyscale: %s\n- duration: %d seconds\n", bpm_b,
             ts, ks, duration);

    text_out = std::string("# Instruction\n") + DIT_INSTR_TEXT2MUSIC + "\n\n" + "# Caption\n" + caption_text + "\n\n" +
               "# Metas\n" + metas + "<|endoftext|>\n";

    const std::string lyr = s.lyrics.empty() ? "[Instrumental]" : s.lyrics;
    lyric_out = sidestep ? lyr : std::string("# Languages\n") + lang + "\n\n# Lyric\n" + lyr + "<|endoftext|>";
}

// ─── numeric guards ─────────────────────────────────────────────────────────

static bool pp_finite(const float * p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!std::isfinite(p[i])) {
            return false;
        }
    }
    return true;
}

// Truncate an id vector to `cap`, overwriting the last kept id with EOS (P13).
static bool pp_truncate_ids(std::vector<int> & ids, int cap, int eos_id) {
    if (cap <= 0 || (int) ids.size() <= cap) {
        return false;
    }
    ids.resize((size_t) cap);
    ids[(size_t) cap - 1] = eos_id;
    return true;
}

// ─── ffmpeg transcode ───────────────────────────────────────────────────────

// Transcode any container ffmpeg understands into a 48 kHz stereo f32 WAV.
// Decode an audio file to 48 kHz planar stereo, UTF-8 path and all.
//
// NOT audio_read_48k(): that opens the path with the CRT's narrow fopen, which
// on Windows decodes it in the ANSI codepage and cannot see a non-ASCII
// filename (hot-step-fsutf8.h). Load the bytes ourselves through hs_fopen and
// hand the buffer to the buffer-based decoder, which sniffs RIFF-vs-MP3 from
// the content rather than the extension. Keeping the fix here — rather than in
// the upstream audio-io.h — also keeps it out of the way of an upstream sync.
static float * pp_audio_read_48k(const std::string & path, int * T_out) {
    *T_out = 0;
    FILE * f = hs_fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    const long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize <= 0) {
        fclose(f);
        return NULL;
    }
    uint8_t * buf = (uint8_t *) malloc((size_t) fsize);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    const size_t nr = fread(buf, 1, (size_t) fsize, f);
    fclose(f);
    if (nr != (size_t) fsize) {
        free(buf);
        return NULL;
    }
    float * planar = audio_read_48k_buf(buf, (size_t) fsize, T_out);
    free(buf);
    return planar;
}

static bool pp_ffmpeg_transcode(const PreprocessOpts & o, const std::string & src, const std::string & tmp_wav,
                                std::string & err) {
    char cmd[4096];
    if (o.max_duration > 0) {
        snprintf(cmd, sizeof(cmd), "\"%s\" -y -v error -i \"%s\" -ac 2 -ar 48000 -t %d -c:a pcm_f32le -f wav \"%s\"",
                 o.ffmpeg.c_str(), src.c_str(), o.max_duration, tmp_wav.c_str());
    } else {
        snprintf(cmd, sizeof(cmd), "\"%s\" -y -v error -i \"%s\" -ac 2 -ar 48000 -c:a pcm_f32le -f wav \"%s\"",
                 o.ffmpeg.c_str(), src.c_str(), tmp_wav.c_str());
    }
#ifdef _WIN32
    std::string wrapped = "\"" + std::string(cmd) + "\"";  // cmd.exe strips the outer pair
    int         rc      = hs_system(wrapped);              // UTF-16 — src may be non-ASCII
#else
    int rc = hs_system(cmd);
#endif
    if (rc != 0) {
        char b[64];
        snprintf(b, sizeof(b), "%d", rc);
        err = std::string("ffmpeg failed (exit ") + b + ") for " + src;
        return false;
    }
    if (!pm_file_exists(tmp_wav)) {
        err = "ffmpeg produced no output for " + src;
        return false;
    }
    return true;
}

// ─── skip classification (§2.2 `reason` values) ─────────────────────────────

// A song that cannot be attempted at all is a SKIP, not a failure: §2.2 freezes
// `reason ∈ "exists" | "unsupported-audio" | "missing-file"`, and reporting
// these as song_fail also flips the process exit code to 1 for a whole FLAC
// dataset run without --ffmpeg. Returns NULL when the song can be attempted.
static const char * pp_skip_reason(const PreprocessOpts & o, const PSample & s) {
    if (!pm_file_exists(s.audio_path)) {
        return "missing-file";
    }
    const std::string ext = pm_ext_of(s.audio_path);
    if (ext != ".wav" && ext != ".mp3" && o.ffmpeg.empty()) {
        return "unsupported-audio";
    }
    return NULL;
}

// ─── per-song pipeline ──────────────────────────────────────────────────────

static SongResult preprocess_song(PreprocessCtx & c, const PreprocessOpts & o, const PManifest & mf, const PSample & s,
                                  const std::string & out_path, void (*warn)(const char *)) {
    SongResult  r;
    std::string tmp_wav;
    char        wbuf[512];

    auto fail = [&](const std::string & msg) -> SongResult {
        r.ok    = false;
        r.error = msg;
        if (!tmp_wav.empty()) {
            hs_remove(tmp_wav);
        }
        return r;
    };

    // 1. Resolve the decodable audio path.
    std::string src = s.audio_path;
    if (!pm_file_exists(src)) {
        return fail("audio file not found: " + src);
    }
    const std::string ext = pm_ext_of(src);
    if (ext != ".wav" && ext != ".mp3") {
        if (o.ffmpeg.empty()) {
            return fail("unsupported audio format " + (ext.empty() ? std::string("(none)") : ext) +
                        "; ffmpeg not configured");
        }
        pm_mkdir_p(o.out_dir + "/.tmp");
        tmp_wav = o.out_dir + "/.tmp/" + (s.id.empty() ? s.out_stem : s.id) + ".wav";
        std::string err;
        if (!pp_ffmpeg_transcode(o, src, tmp_wav, err)) {
            return fail(err);
        }
        src = tmp_wav;
    }

    // 2. Decode to 48 kHz planar stereo, hard-cut at max_duration.
    //
    //    audio_read_48k returns PLANAR [L: T_orig][R: T_orig], so the right
    //    channel lives at planar[T_orig + t] — the channel stride is T_orig and
    //    NEVER the truncated length. Reducing T_audio and then interleaving with
    //    audio_planar_to_interleaved(planar, T_audio) reads the right channel
    //    out of the LEFT channel's tail (proven bit-exactly: a 60 s file cut to
    //    20 s encoded as mono-ised, phase-scrambled audio). Everything below
    //    therefore addresses channel 1 through T_orig and only bounds the loops
    //    by T_audio. Plan §3.4.3 steps 2/4 carry the original defect.
    int     T_orig = 0;
    float * planar = pp_audio_read_48k(src, &T_orig);
    if (!planar || T_orig <= 0) {
        if (planar) {
            free(planar);
        }
        return fail("audio decode failed: " + s.audio_path);
    }
    int T_audio = T_orig;
    if (o.max_duration > 0) {
        // int*int would overflow for --max-duration above ~44739.
        const long long cap = (long long) o.max_duration * 48000LL;
        if ((long long) T_audio > cap) {
            T_audio = (int) cap;
        }
    }
    float * const chL = planar;
    float * const chR = planar + (size_t) T_orig;

    // 3. Peak-normalize the planar buffer over both channels (kept samples only).
    if (o.normalize_peak) {
        float peak = 0.0f;
        for (int i = 0; i < T_audio; i++) {
            float a = fabsf(chL[i]);
            if (a > peak) {
                peak = a;
            }
            a = fabsf(chR[i]);
            if (a > peak) {
                peak = a;
            }
        }
        if (peak >= 1e-6f) {
            const float scale = powf(10.0f, o.target_db / 20.0f) / peak;
            for (int i = 0; i < T_audio; i++) {
                chL[i] *= scale;
                chR[i] *= scale;
            }
        }
    }

    // 4. Interleave (explicit — the helper assumes stride == length).
    float * inter = (float *) malloc((size_t) T_audio * 2 * sizeof(float));
    if (!inter) {
        free(planar);
        return fail("out of memory interleaving audio");
    }
    for (int t = 0; t < T_audio; t++) {
        inter[(size_t) t * 2 + 0] = chL[t];
        inter[(size_t) t * 2 + 1] = chR[t];
    }
    free(planar);

    // 5. VAE encode (tiled).
    const int max_T = T_audio / 1920 + 64;
    r.latents.assign((size_t) max_T * 64, 0.0f);
    int T = vae_enc_encode_tiled(&c.vae, inter, T_audio, r.latents.data(), max_T, o.vae_chunk, o.vae_overlap);
    free(inter);
    if (T <= 0) {
        return fail("VAE encode failed");
    }
    r.latents.resize((size_t) T * 64);
    if (!pp_finite(r.latents.data(), r.latents.size())) {  // CHECK 1
        return fail("VAE encode produced NaN/Inf");
    }
    r.t_latent = T;

    // 6. Prompts + text/lyric encode.
    const int duration = s.duration > 0 ? s.duration : (int) llround((double) T / 25.0);

    const std::string caption_text = pp_caption_text(s, mf, /*use_genre=*/false);
    std::string       text_str, lyric_str;
    pp_prompt_strings(s, caption_text, duration, o.sidestep_compat, text_str, lyric_str);

    std::vector<int> text_ids  = bpe_encode(&c.bpe, text_str, true);
    std::vector<int> lyric_ids = bpe_encode(&c.bpe, lyric_str, true);
    {
        size_t before = text_ids.size();
        if (pp_truncate_ids(text_ids, o.max_caption_tokens, c.bpe.eos_id) && warn) {
            snprintf(wbuf, sizeof(wbuf), "caption truncated to %d tokens (was %zu)", o.max_caption_tokens, before);
            warn(wbuf);
        }
        before = lyric_ids.size();
        if (pp_truncate_ids(lyric_ids, o.max_lyric_tokens, c.bpe.eos_id) && warn) {
            snprintf(wbuf, sizeof(wbuf), "lyrics truncated to %d tokens (was %zu)", o.max_lyric_tokens, before);
            warn(wbuf);
        }
    }
    const int S_text  = (int) text_ids.size();
    const int S_lyric = (int) lyric_ids.size();
    if (S_text <= 0 || S_lyric <= 0) {
        return fail("tokenizer produced an empty sequence");
    }

    const int          H_text = c.te.cfg.hidden_size;
    std::vector<float> text_hidden((size_t) H_text * S_text);
    std::vector<float> lyric_embed((size_t) H_text * S_lyric);
    qwen3_forward(&c.te, text_ids.data(), S_text, text_hidden.data());
    qwen3_embed_lookup(&c.te, lyric_ids.data(), S_lyric, lyric_embed.data());
    if (!pp_finite(text_hidden.data(), text_hidden.size()) ||       // CHECK 2
        !pp_finite(lyric_embed.data(), lyric_embed.size())) {
        return fail("text encoder produced NaN/Inf");
    }

    // 7. Condition encoder (one timbre frame, always).
    float timbre[64];
    if (o.timbre_zeros) {
        memset(timbre, 0, sizeof(timbre));
    } else {
        memcpy(timbre, c.silence.data(), sizeof(timbre));
    }

    std::vector<float> enc_hidden;
    int                S_total = 0;
    cond_ggml_forward(&c.ce, text_hidden.data(), S_text, lyric_embed.data(), S_lyric, timbre, /*S_ref=*/1, enc_hidden,
                      &S_total);
    if (S_total != S_lyric + 1 + S_text) {
        char b[128];
        snprintf(b, sizeof(b), "cond-enc returned S_total=%d, expected %d (=%d+1+%d)", S_total, S_lyric + 1 + S_text,
                 S_lyric, S_text);
        return fail(b);
    }

    // 8. context_latents [T,128]: silence_latent + an all-ones mask half.
    const int Tc = T < c.silence_T ? T : c.silence_T;
    if (T > c.silence_T && warn) {
        snprintf(wbuf, sizeof(wbuf), "T=%d exceeds silence_latent frames (%d); clamping the context tail", T,
                 c.silence_T);
        warn(wbuf);
    }
    std::vector<float> context((size_t) T * 128);
    for (int t = 0; t < T; t++) {
        const float * srcf = c.silence.data() + (size_t) (t < Tc ? t : Tc - 1) * 64;
        float *       dst  = context.data() + (size_t) t * 128;
        for (int ch = 0; ch < 64; ch++) {
            dst[ch] = srcf[ch];
        }
        for (int ch = 0; ch < 64; ch++) {
            dst[64 + ch] = 1.0f;
        }
    }
    if (!pp_finite(enc_hidden.data(), enc_hidden.size()) ||  // CHECK 3
        !pp_finite(context.data(), context.size())) {
        return fail("condition encoder produced NaN/Inf");
    }

    // 9. Genre variant (shares the lyric + timbre halves).
    std::vector<float> enc_hidden_g;
    std::vector<float> mask_g;
    int                S_total_g = 0, S_text_g = 0;
    std::string        genre_prompt;
    bool               has_genre = !pm_trim(s.genre).empty();
    if (has_genre) {
        const std::string caption_text_g = pp_caption_text(s, mf, /*use_genre=*/true);
        std::string       text_str_g, lyric_dummy;
        pp_prompt_strings(s, caption_text_g, duration, o.sidestep_compat, text_str_g, lyric_dummy);
        genre_prompt = text_str_g;

        std::vector<int> text_ids_g = bpe_encode(&c.bpe, text_str_g, true);
        size_t           before     = text_ids_g.size();
        if (pp_truncate_ids(text_ids_g, o.max_caption_tokens, c.bpe.eos_id) && warn) {
            snprintf(wbuf, sizeof(wbuf), "genre caption truncated to %d tokens (was %zu)", o.max_caption_tokens,
                     before);
            warn(wbuf);
        }
        S_text_g = (int) text_ids_g.size();

        std::vector<float> text_hidden_g((size_t) H_text * S_text_g);
        qwen3_forward(&c.te, text_ids_g.data(), S_text_g, text_hidden_g.data());
        if (!pp_finite(text_hidden_g.data(), text_hidden_g.size())) {
            has_genre = false;
            if (warn) {
                warn("genre text encode produced NaN/Inf; dropping the genre variant");
            }
        } else {
            cond_ggml_forward(&c.ce, text_hidden_g.data(), S_text_g, lyric_embed.data(), S_lyric, timbre, 1,
                              enc_hidden_g, &S_total_g);
            if (S_total_g != S_lyric + 1 + S_text_g || !pp_finite(enc_hidden_g.data(), enc_hidden_g.size())) {
                has_genre = false;
                if (warn) {
                    warn("genre condition encode produced NaN/Inf; dropping the genre variant");
                }
            }
        }
        if (!has_genre) {
            enc_hidden_g.clear();
            S_total_g = 0;
            S_text_g  = 0;
        }
    }

    // 10. Masks.
    std::vector<float> attn_mask((size_t) T, 1.0f);
    std::vector<float> enc_mask((size_t) S_total, 1.0f);
    if (has_genre) {
        mask_g.assign((size_t) S_total_g, 1.0f);
    }

    // 11. __metadata__ (§2.4) + write.
    auto istr = [](long long v) {
        char b[32];
        snprintf(b, sizeof(b), "%lld", v);
        return std::string(b);
    };
    auto dstr = [](double v) {
        char b[48];
        snprintf(b, sizeof(b), "%g", v);
        return std::string(b);
    };

    std::vector<std::pair<std::string, std::string>> md;
    md.push_back({ "format", "hot-step-preprocess-v1" });
    md.push_back({ "producer", std::string("ace-train ") + ACE_VERSION });
    md.push_back({ "created_at", pm_iso8601_utc_now() });
    md.push_back({ "compat", o.sidestep_compat ? "sidestep" : "hotstep" });
    md.push_back({ "dtype", o.dtype == STW_F32 ? "F32" : "BF16" });
    md.push_back({ "timbre", o.timbre_zeros ? "zeros" : "silence" });
    md.push_back({ "model_variant", o.model_variant });
    md.push_back({ "dit_path", o.dit_path });
    md.push_back({ "vae_path", o.vae_path });
    md.push_back({ "text_encoder_path", o.text_enc_path });
    md.push_back({ "sample_id", s.id });
    md.push_back({ "audio_path", s.audio_path });
    md.push_back({ "filename", s.filename });
    md.push_back({ "rel_path", s.rel_path });
    md.push_back({ "src_bytes", istr(s.src_bytes) });
    md.push_back({ "src_mtime_ms", istr(s.src_mtime_ms) });
    md.push_back({ "caption", s.caption });
    md.push_back({ "genre", s.genre });
    md.push_back({ "lyrics", s.lyrics });
    md.push_back({ "caption_prompt", text_str });
    md.push_back({ "lyric_prompt", lyric_str });
    md.push_back({ "genre_prompt", has_genre ? genre_prompt : std::string("") });
    md.push_back({ "duration", istr(duration) });
    md.push_back({ "bpm", istr(s.bpm) });
    md.push_back({ "keyscale", s.keyscale });
    md.push_back({ "timesignature", s.timesignature });
    md.push_back({ "language", s.language });
    md.push_back({ "is_instrumental", s.is_instrumental ? "true" : "false" });
    md.push_back({ "custom_tag", s.custom_tag.empty() ? mf.custom_tag : s.custom_tag });
    md.push_back({ "tag_position", s.tag_position.empty() ? mf.tag_position : s.tag_position });
    md.push_back({ "prompt_override", s.prompt_override });
    md.push_back({ "repeat", istr(s.repeat) });
    md.push_back({ "t_latent", istr(T) });
    md.push_back({ "s_total", istr(S_total) });
    md.push_back({ "s_text", istr(S_text) });
    md.push_back({ "s_lyric", istr(S_lyric) });
    md.push_back({ "s_total_genre", has_genre ? istr(S_total_g) : std::string("") });
    md.push_back({ "s_text_genre", has_genre ? istr(S_text_g) : std::string("") });
    md.push_back({ "max_duration", istr(o.max_duration) });
    md.push_back({ "normalize", o.normalize_peak ? "peak" : "none" });
    md.push_back({ "target_db", dstr(o.target_db) });
    md.push_back({ "max_caption_tokens", istr(o.max_caption_tokens) });
    md.push_back({ "max_lyric_tokens", istr(o.max_lyric_tokens) });
    md.push_back({ "vae_chunk", istr(o.vae_chunk) });
    md.push_back({ "vae_overlap", istr(o.vae_overlap) });
    md.push_back({ "latent_fps", "25.0" });

    std::vector<STWTensor> tensors;
    tensors.push_back({ "target_latents", { T, 64 }, r.latents.data() });
    tensors.push_back({ "attention_mask", { T }, attn_mask.data() });
    tensors.push_back({ "encoder_hidden_states", { S_total, 2048 }, enc_hidden.data() });
    tensors.push_back({ "encoder_attention_mask", { S_total }, enc_mask.data() });
    tensors.push_back({ "context_latents", { T, 128 }, context.data() });
    if (has_genre) {
        tensors.push_back({ "encoder_hidden_states_genre", { S_total_g, 2048 }, enc_hidden_g.data() });
        tensors.push_back({ "encoder_attention_mask_genre", { S_total_g }, mask_g.data() });
    }

    if (!st_write_file(out_path.c_str(), tensors, md, o.dtype)) {
        return fail("failed to write " + out_path);
    }

    // 12. Cleanup + result.
    if (!tmp_wav.empty()) {
        hs_remove(tmp_wav);
    }
    long long bytes = 0;
    pm_stat_file(out_path, &bytes, NULL);

    r.ok            = true;
    r.s_total       = S_total;
    r.s_text        = S_text;
    r.s_lyric       = S_lyric;
    r.s_total_genre = has_genre ? S_total_g : 0;
    r.s_text_genre  = has_genre ? S_text_g : 0;
    r.has_genre     = has_genre;
    r.bytes         = bytes;
    return r;
}
