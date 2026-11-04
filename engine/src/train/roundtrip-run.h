#pragma once
// roundtrip-run.h — `ace-train roundtrip`: hear what adapter training keeps.
//
// Takes one source track, encodes it the way that backend's training prep does
// (same decode, crop, level and encoder), decodes the result straight back to
// audio and writes two WAVs:
//
//   <out>.wav         the reconstruction: everything the adapter can learn from
//   <out>.source.wav  the audio the encoder was fed, at the same rate and level
//
// What each backend stores, and so what is decoded here:
//
//   ace        VAE latents (target_latents)       -> VAE decoder
//   sa3        SAME latents (.sa3c cache)         -> SAME decoder
//   mm3        DAV latents (the flow DiT target)  -> vocoder
//   yue2       VAE latents (the NAR target)       -> VAE decoder
//   heartmula  HeartCodec RVQ codes               -> flow-matching codec decoder
//
// The first four decode deterministically. HeartCodec's decoder starts from
// noise, so its reconstruction changes with --seed. The LM-side token streams
// (ACE 5 Hz codes, MM3 RVQ codes, YuE2 semantic codes) need a generative model
// to turn back into audio and are not covered.
//
// Header-only like every train/*.h. ace-train.cpp includes it after its own
// resolve_model(), which the ace path uses to find the VAE.

#include "train/heartmula-prepare.h"
#include "train/preprocess-run.h"
#include "train/sa3-train-data.h"
#include "train/yue2-preprocess-run.h"

#include <string>
#include <vector>

struct RoundtripArgs {
    std::string backend, audio, models_dir, out, ffmpeg = "ffmpeg";
    std::string vae;           // ace: VAE name or path (required); yue2: standard|legacy
    std::string enc;           // mm3: DAV encoder GGUF (default: newest mm3-enc-* in <models>/mm3)
    int         max_duration = -1;  // -1 = the backend's prep default
    // ace preprocess knobs, defaults as PreprocessOpts
    std::string normalize = "none", dtype = "f32";  // the app's defaults (2026-11-04): no normalization
    float       target_db = -1.0f;
    int         vae_chunk = 384, vae_overlap = 48;
    double      loudness_lufs = 0.0;    // yue2: 0 = off, as the app prepares
    uint64_t    seed          = 42;     // heartmula
};

// ── shared ───────────────────────────────────────────────────────────────────

static bool rt_write_wav(const std::string & path, const float * planar, int64_t T, int sr) {
    const std::string bytes = audio_encode_wav(planar, (int) T, sr, WAV_F32);
    FILE *            f     = hs_fopen(path, "wb");
    const bool        ok    = f && !bytes.empty() && fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
    if (f) fclose(f);
    if (ok) fprintf(stderr, "[roundtrip] wrote %s (%.2f s at %d Hz)\n", path.c_str(), (double) T / sr, sr);
    else fprintf(stderr, "[roundtrip] cannot write %s\n", path.c_str());
    return ok;
}

static std::string rt_source_path(const std::string & out) {
    const std::string ext = pm_ext_of(out);
    return (ext == ".wav" ? out.substr(0, out.size() - 4) : out) + ".source.wav";
}

static bool rt_write_pair(const RoundtripArgs & a, const float * src, int64_t T_src, int sr_src, const float * rec,
                          int64_t T_rec, int sr_rec) {
    return rt_write_wav(rt_source_path(a.out), src, T_src, sr_src) && rt_write_wav(a.out, rec, T_rec, sr_rec);
}

static std::string rt_tmp_dir(const RoundtripArgs & a) {
    std::string d = a.out;
    const size_t s = d.find_last_of("/\\");
    d = (s == std::string::npos ? std::string(".") : d.substr(0, s)) + "/.roundtrip-tmp";
    pm_mkdir_p(d);
    return d;
}

// ffmpeg to a stereo WAV at `sr` (0 = keep the rate), `-t` when max_duration > 0.
static bool rt_ffmpeg(const RoundtripArgs & a, const std::string & src, const std::string & dst, int sr,
                      const char * codec, int max_duration) {
    std::string cmd = "\"" + a.ffmpeg + "\" -y -v error -i \"" + src + "\" -vn";
    if (max_duration > 0) cmd += " -t " + std::to_string(max_duration);
    if (sr > 0) cmd += " -ar " + std::to_string(sr);
    cmd += std::string(" -ac 2 -c:a ") + codec + " -f wav \"" + dst + "\"";
#ifdef _WIN32
    const int rc = hs_system("\"" + cmd + "\"");  // cmd.exe strips the outer pair
#else
    const int rc = hs_system(cmd);
#endif
    if (rc != 0 || !pm_file_exists(dst)) {
        fprintf(stderr, "[roundtrip] ffmpeg failed (exit %d) for %s\n", rc, src.c_str());
        return false;
    }
    return true;
}

static float * rt_read_wav(const std::string & path, int * T, int * sr) {
    std::vector<uint8_t> buf;
    if (!hp_read_file(path, &buf)) return nullptr;
    return audio_io_read_wav_buf(buf.data(), buf.size(), T, sr);
}

// ── ace: preprocess_song's encode (pp_encode_audio), then the VAE decoder ────

static int rt_ace(const RoundtripArgs & a) {
    PreprocessOpts o;
    o.ffmpeg         = a.ffmpeg;
    o.out_dir        = rt_tmp_dir(a);
    o.max_duration   = a.max_duration >= 0 ? a.max_duration : o.max_duration;
    o.normalize_peak = a.normalize == "peak";
    o.target_db      = a.target_db;
    o.vae_chunk      = a.vae_chunk;
    o.vae_overlap    = a.vae_overlap;

    ModelRegistry reg;
    registry_scan(&reg, a.models_dir.c_str());
    std::string vae_name;
    if (!resolve_model(a.vae, reg.vae, true, o.vae_path, vae_name)) {
        fprintf(stderr, "[roundtrip] cannot resolve --vae '%s' in %s (ONNX VAEs cannot encode)\n", a.vae.c_str(),
                a.models_dir.c_str());
        return 2;
    }
    fprintf(stderr, "[roundtrip] ace: VAE %s\n", o.vae_path.c_str());

    VAEEncoder enc = {};
    vae_enc_load(&enc, o.vae_path.c_str());
    std::vector<float> lat, inter;
    std::string        tmp_wav, err;
    const bool         ok = pp_encode_audio(&enc, o, a.audio, "roundtrip", tmp_wav, lat, err, &inter);
    vae_enc_free(&enc);
    if (!tmp_wav.empty()) hs_remove(tmp_wav);
    if (!ok) {
        fprintf(stderr, "[roundtrip] %s\n", err.c_str());
        return 1;
    }
    const int T = (int) (lat.size() / 64);
    // --dtype bf16 caches round every latent to BF16; the trainer reads them back as that.
    if (a.dtype == "bf16") {
        for (float & v : lat) {
            const uint32_t bits = (uint32_t) stw_f32_to_bf16(v) << 16;
            memcpy(&v, &bits, sizeof(v));
        }
    }
    fprintf(stderr, "[roundtrip] ace: %d latent frames x 64 (%s)\n", T, a.dtype.c_str());

    VAEGGML dec = {};
    vae_ggml_load(&dec, o.vae_path.c_str());
    const int          max_T = (T + 64) * 1920;
    std::vector<float> audio((size_t) max_T * 2);
    const int          T_a = vae_ggml_decode_tiled(&dec, lat.data(), T, audio.data(), max_T, 1024, 64);
    vae_ggml_free(&dec);
    if (T_a <= 0) {
        fprintf(stderr, "[roundtrip] VAE decode failed\n");
        return 1;
    }
    const int64_t      Ts = (int64_t) inter.size() / 2;
    std::vector<float> src((size_t) Ts * 2);
    for (int64_t t = 0; t < Ts; t++) {
        src[(size_t) t]      = inter[(size_t) t * 2];
        src[(size_t) (Ts + t)] = inter[(size_t) t * 2 + 1];
    }
    return rt_write_pair(a, src.data(), Ts, 48000, audio.data(), T_a, 48000) ? 0 : 1;
}

// ── sa3: sa3t_preencode's clip prep + sa3t_same_encode, then the SAME decoder ─
//
// SA3 datasets are the SuperSep instrumental stem, converted by sa3Stems.ts
// (`-t 384 -ar 44100 -ac 2 pcm_s16le`). This runs the same conversion on
// whatever it is given, so pass a prepared instrumental.wav to hear exactly
// what training reads; a full mix shows the codec on the vocals as well.

static int rt_sa3(const RoundtripArgs & a) {
    const std::string tmp = rt_tmp_dir(a) + "/sa3.wav";
    if (!rt_ffmpeg(a, a.audio, tmp, SA3T_SR, "pcm_s16le", a.max_duration >= 0 ? a.max_duration : 384)) return 1;
    int     T = 0, sr = 0;
    float * audio = rt_read_wav(tmp, &T, &sr);
    hs_remove(tmp);
    if (!audio || sr != SA3T_SR || T <= 0) {
        free(audio);
        fprintf(stderr, "[roundtrip] sa3: cannot read the 44.1 kHz conversion\n");
        return 1;
    }
    float peak = 0.0f;
    for (int64_t j = 0; j < 2 * (int64_t) T; j++) peak = std::max(peak, std::fabs(audio[j]));
    if (peak < 1e-3f) fprintf(stderr, "[roundtrip] sa3: below -60 dBFS; sa3-train would skip this clip as silent\n");
    const int64_t      n = ((int64_t) T + SA3T_DS - 1) / SA3T_DS * SA3T_DS;
    std::vector<float> planar((size_t) (2 * n), 0.0f);
    for (int ch = 0; ch < 2; ch++)
        for (int64_t j = 0; j < T; j++)
            planar[(size_t) (ch * n + j)] = std::min(1.0f, std::max(-1.0f, audio[ch * (int64_t) T + j]));
    free(audio);

    SA3Same enc{};
    if (!sa3_same_load(&enc, (a.models_dir + "/sa3-same-enc-F16.gguf").c_str(), true)) {
        fprintf(stderr, "[roundtrip] sa3: cannot load %s/sa3-same-enc-F16.gguf\n", a.models_dir.c_str());
        return 1;
    }
    std::vector<float> lat;
    sa3t_same_encode(&enc, planar.data(), n, lat);
    sa3_same_free(&enc);
    const int L = (int) (n / SA3T_DS);
    fprintf(stderr, "[roundtrip] sa3: %d latent frames x %d\n", L, SA3T_LAT);

    SA3Same dec{};
    if (!sa3_same_load(&dec, (a.models_dir + "/sa3-same-dec-F16.gguf").c_str(), false)) {
        fprintf(stderr, "[roundtrip] sa3: cannot load %s/sa3-same-dec-F16.gguf\n", a.models_dir.c_str());
        return 1;
    }
    // sa3_decode_tiled (sa3-refine.h) with the SAME decoder called directly:
    // fixed 128-latent chunks, 32 overlap, short input zero-padded to one chunk.
    const int64_t      CS = (int64_t) SA3T_CHUNK * SA3T_DS, S = (int64_t) L * SA3T_DS;
    std::vector<float> out((size_t) (2 * S), 0.0f), chunk((size_t) SA3T_LAT * SA3T_CHUNK), aud((size_t) (2 * CS));
    std::vector<int>   starts;
    if (L < SA3T_CHUNK) starts.push_back(0);
    else {
        for (int s = 0; s + SA3T_CHUNK <= L; s += SA3T_CHUNK - SA3T_OVERLAP) starts.push_back(s);
        if (starts.back() != L - SA3T_CHUNK) starts.push_back(L - SA3T_CHUNK);
    }
    const int64_t half = (int64_t) (SA3T_OVERLAP / 2) * SA3T_DS;
    for (size_t i = 0; i < starts.size(); i++) {
        std::fill(chunk.begin(), chunk.end(), 0.0f);
        const int w = std::min(SA3T_CHUNK, L);
        for (int c = 0; c < SA3T_LAT; c++)
            std::copy(lat.begin() + (size_t) c * L + starts[i], lat.begin() + (size_t) c * L + starts[i] + w,
                      chunk.begin() + (size_t) c * SA3T_CHUNK);
        sa3_same_forward(&dec, chunk.data(), aud.data(), SA3T_CHUNK);
        const bool    first = i == 0, last = i + 1 == starts.size();
        const int64_t os    = (int64_t) starts[i] * SA3T_DS;
        const int64_t left = first ? 0 : half, right = std::min(last ? CS : CS - half, S - os);
        for (int ch = 0; ch < 2; ch++)
            std::copy(aud.begin() + ch * CS + left, aud.begin() + ch * CS + right, out.begin() + ch * S + os + left);
    }
    sa3_same_free(&dec);
    // Both trimmed to the clip; the pad to a 4096 multiple is silence.
    std::vector<float> rec((size_t) T * 2), src((size_t) T * 2);
    for (int ch = 0; ch < 2; ch++) {
        std::copy(out.begin() + ch * S, out.begin() + ch * S + T, rec.begin() + (size_t) ch * T);
        std::copy(planar.begin() + ch * n, planar.begin() + ch * n + T, src.begin() + (size_t) ch * T);
    }
    return rt_write_pair(a, src.data(), T, SA3T_SR, rec.data(), T, SA3T_SR) ? 0 : 1;
}

// ── heartmula: heartmula-prepare's decode, discovery and encode(), then the codec ─

static int rt_heartmula(const RoundtripArgs & a) {
    HeartmulaPrepareArgs pa;
    pa.ffmpeg = a.ffmpeg;
    const HeartMuLaCatalogue cat = heartmula_discover(a.models_dir);
    const HeartMuLaFile *    rvq_file = nullptr;
    for (const auto & f : cat.files)
        if (f.compatible && f.role == HM_RVQ && !rvq_file) rvq_file = &f;
    HeartMuLaResolved res;
    int               status = 0;
    std::string       err;
    if (!rvq_file || !heartmula_resolve(cat, HeartMuLaPick{}, &res, &status, &err)) {
        fprintf(stderr, "[roundtrip] heartmula: no usable codec under %s%s%s\n", a.models_dir.c_str(),
                err.empty() ? "" : ": ", err.c_str());
        return 1;
    }
    const std::string enc_dir = fs::exists(fs::u8path(a.models_dir) / "heartmula")
                                    ? (fs::u8path(a.models_dir) / "heartmula").u8string()
                                    : a.models_dir;
    heartcodec::EncoderPaths paths = heartcodec::encoder_paths(enc_dir);
    paths.rvq                      = rvq_file->path;

    int     T = 0, sr = 0;
    float * planar = hp_decode(pa, a.audio, rt_tmp_dir(a) + "/heartmula.wav", &T, &sr, &err);
    if (!planar) {
        fprintf(stderr, "[roundtrip] heartmula: %s\n", err.c_str());
        return 1;
    }
    std::vector<float> audio(planar, planar + (size_t) 2 * T);
    free(planar);

    BackendPair bp = backend_init("HeartMuLa roundtrip");
    int         rc = 1;
    try {
        auto                     encoders = std::make_unique<heartcodec::Encoders>();
        heartcodec::EncodeResult r = heartcodec::encode(paths, bp.backend, audio, 2, T, sr, {}, {}, encoders.get());
        encoders.reset();
        fprintf(stderr, "[roundtrip] heartmula: %lld frames x 8 codebooks\n", (long long) r.frames);
        // heartmula-prepare skips a track whose prompt plus frames overruns the context.
        if (r.frames - 1 > heartmula_global_config().n_ctx + 2)
            fprintf(stderr, "[roundtrip] heartmula: %lld frames is past the %lld-position context; "
                            "heartmula-prepare would skip this track\n",
                    (long long) r.frames, (long long) heartmula_global_config().n_ctx);
        heartcodec::Weights rvq, flow, scalar;
        rvq.load(rvq_file->path, k_heartmula_roles[HM_RVQ], bp.backend);
        flow.load(res.file[HM_FLOW].path, k_heartmula_roles[HM_FLOW], bp.backend);
        scalar.load(res.file[HM_SCALAR].path, k_heartmula_roles[HM_SCALAR], bp.backend);
        fprintf(stderr, "[roundtrip] heartmula: decoding with %s, seed %llu\n", res.file[HM_FLOW].filename.c_str(),
                (unsigned long long) a.seed);
        heartmula_pipeline_detail::Gaussian noise(a.seed);  // the generation pipeline's noise
        const std::vector<float> wav = heartcodec::decode(rvq, flow, scalar, bp.backend, r.codes, r.frames,
                                                          std::ref(noise));
        rc = rt_write_pair(a, audio.data(), T, sr, wav.data(), (int64_t) wav.size() / 2, 48000) ? 0 : 1;
    } catch (const std::exception & e) {
        fprintf(stderr, "[roundtrip] heartmula: %s\n", e.what());
    }
    backend_release(bp.backend, bp.cpu_backend);
    return rc;
}

// ── mm3: mm3-preprocess's transcode + DAV encode, then the vocoder ──────────

static int rt_mm3(const RoundtripArgs & a) {
    std::string enc_path = a.enc;
    if (enc_path.empty()) {
        for (const std::string & dir : { a.models_dir + "/mm3", a.models_dir }) {
            if (!pm_path_exists(dir)) continue;
            for (const auto & e : fs::directory_iterator(fs::u8path(dir))) {
                const std::string name = e.path().filename().u8string();
                if (name.rfind("mm3-enc-", 0) == 0 && pm_ext_of(name) == ".gguf" &&
                    (enc_path.empty() || name > pm_basename(enc_path)))
                    enc_path = e.path().u8string();
            }
            if (!enc_path.empty()) break;
        }
    }
    MM3Enc      enc = {};
    std::string err;
    if (enc_path.empty() || !mm3_enc_load(&enc, enc_path.c_str(), &err)) {
        fprintf(stderr, "[roundtrip] mm3: cannot load the DAV encoder %s%s\n", enc_path.c_str(), err.c_str());
        return 1;
    }
    const int         SR  = (int) enc.cfg.sampling_rate;
    const std::string tmp = rt_tmp_dir(a) + "/mm3.wav";
    int               T = 0, sr = 0;
    float *           planar = nullptr;
    if (rt_ffmpeg(a, a.audio, tmp, SR, "pcm_f32le", a.max_duration)) planar = rt_read_wav(tmp, &T, &sr);
    hs_remove(tmp);
    if (!planar || T <= 0 || sr != SR) {
        free(planar);
        mm3_enc_free(&enc);
        fprintf(stderr, "[roundtrip] mm3: cannot decode the %d Hz transcode\n", SR);
        return 1;
    }
    const float *      chans[2] = { planar, planar + T };
    std::vector<float> lat;
    int64_t            L  = 0;
    const bool         ok = mm3_enc_encode(&enc, chans, (int64_t) T, &lat, &L, &err);
    const unsigned     LC = enc.cfg.latent_channels;
    mm3_enc_free(&enc);
    if (!ok) {
        free(planar);
        fprintf(stderr, "[roundtrip] mm3: encode failed: %s\n", err.c_str());
        return 1;
    }
    fprintf(stderr, "[roundtrip] mm3: %lld latent frames x %u\n", (long long) L, LC);

    // The vocoder lives in the "rest" parts; pin the DiT that loads beside it to
    // the smallest quant, as mm3-condition does, since it is never run here.
    static MM3Model model;
    std::string     roles[MM3_R_COUNT];
    roles[MM3_R_DIT] = "Q2_K";
    mm3_discover(&model, a.models_dir.c_str(), "", roles);
    std::vector<float> wav;
    if (!mm3_load_parts(&model, false, false, true, &err) || !mm3_vocoder_decode(model, lat.data(), L, wav, &err)) {
        free(planar);
        fprintf(stderr, "[roundtrip] mm3: vocoder: %s\n", err.c_str());
        return 1;
    }
    const int rc = rt_write_pair(a, planar, T, SR, wav.data(), (int64_t) wav.size() / 2,
                                 (int) model.synth_cfg.voc.sampling_rate)
                       ? 0
                       : 1;
    free(planar);
    return rc;
}

// ── yue2: yue2-preprocess's decode, loudness and VAE encode, then the decoder ─

static int rt_yue2(const RoundtripArgs & a) {
    if (!yue2_vae_enc_tf32_disabled()) {
        fprintf(stderr, "[roundtrip] yue2: NVIDIA_TF32_OVERRIDE is not 0\n");
        return 1;
    }
    Yue2PreprocessArgs pa;
    pa.models_dir = a.models_dir;
    pa.vae_arg    = a.vae.empty() ? "standard" : a.vae;
    Yue2PpVae   vs;
    std::string err;
    if (!yp_parse_vae_arg(pa, &vs, &err)) {
        fprintf(stderr, "[roundtrip] yue2: %s\n", err.c_str());
        return 2;
    }
    static Yue2Model m;
    yue2_discover(&m, vs.models_dir.c_str());
    if (!m.vae_file[vs.variant].found ||
        !yue2_load_parts(&m, /*want_lm=*/false, /*want_vae=*/true, vs.variant, /*want_encoder=*/true, &err)) {
        fprintf(stderr, "[roundtrip] yue2: VAE (%s) load failed: %s\n", YUE2_VAE_VARIANT_NAME[vs.variant], err.c_str());
        return 1;
    }
    const int64_t SR = (int64_t) m.vae_cfg.sample_rate, RATIO = (int64_t) m.vae_cfg.downsampling_ratio;
    const std::string ext = pm_ext_of(a.audio), tmp = rt_tmp_dir(a) + "/yue2.wav";
    int               T = 0;
    float * planar = (ext == ".wav" || ext == ".mp3") ? yp_read_native_48k(a.audio, &T)
                                                      : yp_read_ffmpeg(a.ffmpeg, a.audio, tmp, (int) SR, &T, &err);
    hs_remove(tmp);
    if (!planar || T <= 0) {
        free(planar);
        fprintf(stderr, "[roundtrip] yue2: decode failed %s\n", err.c_str());
        yue2_unload(&m);
        return 1;
    }
    if (a.loudness_lufs != 0.0) {
        double     lufs = 0.0;
        bool       capped = false;
        const double g = yue2_loudness_gain_db(planar, T, a.loudness_lufs, -1.0, &lufs, &capped);
        yue2_apply_gain_db(planar, T, g);
        fprintf(stderr, "[roundtrip] yue2: %.1f LUFS, gain %+.1f dB\n", lufs, g);
    }
    yue2_peak_scale(planar, (size_t) T * 2);

    std::vector<float> lat, out;
    int64_t            got = 0, samples = 0;
    const int64_t      core = std::min<int64_t>((int64_t) m.vae_cfg.decode_core_frames, 512);
    const bool ok = yue2_vae_encode_tiled(m, planar, (int64_t) T, 750 * RATIO, 20 * RATIO, &lat, &got, nullptr, &err) &&
                    yue2_vae_decode_tiled(m, lat.data(), got, core, (int64_t) m.vae_cfg.decode_halo_frames, &out,
                                          &samples, nullptr, &err);
    yue2_unload(&m);
    if (!ok) {
        free(planar);
        fprintf(stderr, "[roundtrip] yue2: %s\n", err.c_str());
        return 1;
    }
    fprintf(stderr, "[roundtrip] yue2: %lld latent frames (%s VAE)\n", (long long) got,
            YUE2_VAE_VARIANT_NAME[vs.variant]);
    const int rc = rt_write_pair(a, planar, T, (int) SR, out.data(), samples, (int) SR) ? 0 : 1;
    free(planar);
    return rc;
}

static int roundtrip_run(const RoundtripArgs & a) {
    // Every prep but ACE's runs with TF32 off; the driver reads it when the
    // first CUDA context is created, so it is set before any model loads.
    if (a.backend != "ace") {
#ifdef _WIN32
        _putenv_s("NVIDIA_TF32_OVERRIDE", "0");
#else
        setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
#endif
    }
    ggml_time_init();
    int rc = 2;
    if (a.backend == "ace") rc = rt_ace(a);
    else if (a.backend == "sa3") rc = rt_sa3(a);
    else if (a.backend == "heartmula") rc = rt_heartmula(a);
    else if (a.backend == "mm3") rc = rt_mm3(a);
    else if (a.backend == "yue2") rc = rt_yue2(a);
    else fprintf(stderr, "[roundtrip] --backend must be ace, sa3, heartmula, mm3 or yue2\n");
    // Empty dirs only: pp_encode_audio's .tmp, then ours.
    std::error_code ec;
    fs::remove(fs::u8path(rt_tmp_dir(a)) / ".tmp", ec);
    fs::remove(fs::u8path(rt_tmp_dir(a)), ec);
    return rc;
}
