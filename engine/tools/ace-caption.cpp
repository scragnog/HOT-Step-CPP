// ace-caption.cpp: MOSS-Music captioning CLI. Audio in, caption out.
//
// Usage:
//   ace-caption --models <dir> --src-audio <file> [--mode prose|mm3|lyrics[,...]]
//               [--prompt "..."] [--prompt-file [<mode>=]<path>]... [-o <out.txt>]
//               [--max-tokens N] [--max-seconds S]
//               [--temperature T] [--rep-penalty R] [--freq-penalty F]
//
// Shaped like ace-understand.cpp (audio -> metadata/lyrics), which it
// complements: ace-understand goes through ACE's own VAE+FSQ+LM, this one runs
// the MOSS-Music tower.
//
// ── Prompt sources, and why --prompt-file exists ─────────────────────────────
//
// The ACE-Step 1.5 caption format is defined by CAPTION_SENTENCE_PLAN in
// server/src/services/training/captionPrompt.ts, which is byte-identical to
// Side-Step's caption_config.py so a caption produced here is indistinguishable
// from one Side-Step produced. Duplicating that wording in C++ would guarantee
// it drifts. So the built-in modes cover prose/mm3/lyrics, and the server passes
// the AS1.5 prompt through --prompt-file, keeping one source of truth.
//
// --prompt-file is repeatable and takes an optional `<mode>=` prefix. A bare path
// overrides every mode; `prose=<path>` overrides just that one, leaving the other
// modes on their built-ins. That is what lets a single run emit the AS1.5 block
// AND an MM3 Structured Caption off one encode — the Training Studio's whole ask.
// A bare Windows path still parses as the blanket form: the split only happens
// when the text before the first '=' is literally prose, mm3 or lyrics.
//
// ── Cost shape ───────────────────────────────────────────────────────────────
//
// The encoder is prompt-independent: mel + 32 Whisper layers depend only on the
// audio. Only the LM sees the prompt. So --mode takes a COMMA-SEPARATED list and
// runs one encode followed by N decodes:
//
//   ace-caption --mode prose,mm3,lyrics -o out.txt
//     -> out.prose.txt, out.mm3.txt, out.lyrics.txt
//
// On a 4-minute track the encode is the expensive half, so asking for all three
// together is far cheaper than three invocations. This is the shape the Training
// Studio pipeline should use: dataset tracks want the AS1.5 caption AND the MM3
// Structured Caption, and there is no reason to pay for the audio twice.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

#include "audio-io.h"
#include "audio-resample.h"
#include "backend.h"
#include "bpe.h"
#include "moss/moss-encoder-graph.h"
#include "moss/moss-lm-graph.h"
#include "moss/moss-mel.h"
#include "moss/moss-model.h"

static const char * PROMPT_PROSE =
    "Analyse this music track and describe it in detail: genre and energy, the drum and "
    "percussion pattern, bass, harmony and lead instrumentation, vocal style and timbre, "
    "production and mix character, and how the arrangement develops from intro to ending. "
    "State the tempo in BPM, the key and scale, and the time signature. Be specific and "
    "concrete about what you actually hear.";

// Plain-text section labels, NOT markdown headings: the 1000 official templates
// in .claude/skills/mm3-captioning/upstream/templates/ use bare labels, and MM3
// was trained on that. Lyrics are forbidden inside an MM3 caption.
static const char * PROMPT_MM3 =
    "Listen to this track and write a Structured Caption in exactly the format below.\n\n"
    "Use these three section labels on their own lines, with no markdown, no '#', and no "
    "extra sections:\n\n"
    "Global Metadata\n"
    "Basic Attributes: bpm is <N>. key is <X>, and scale is <major|minor>. <Genre / Subgenre>.\n"
    "Global Emotional Progression: <how the emotional arc moves across the song>\n"
    "Application Scenarios & Imagery: <where this music would be used; concrete visual imagery>\n"
    "Sonics & Production Profile: <soundstage, density, frequency balance, dynamics, mix character>\n"
    "Vocal Details\n"
    "Vocal Gender & Timbre: <Singer A (Male/Female). timbre, register, texture>\n"
    "Vocal Style: <delivery and how it evolves across sections>\n"
    "Harmony/Backing Vocals: <layering, doubling, gang vocals, call-and-response, or state none>\n"
    "Vocal FX: <reverb, delay, distortion, pitch correction, or state minimal>\n"
    "Arrangement\n"
    "Instrument Lifecycle Description (Primary/Secondary Layering):\n"
    "Primary: <the instruments carrying the track and how they change>\n"
    "Secondary: <supporting instruments and when they enter or drop>\n"
    "Groove & Foundation Progression: <rhythm section, feel changes, section by section>\n"
    "Embellishments, Textures & Spatial FX: <risers, sweeps, reverse reverb, glitches, ambience>\n\n"
    "Rules:\n"
    "- Describe only what you actually hear. Do not invent an exact BPM or key if unsure.\n"
    "- If the track is instrumental, say so under Vocal Details and name the lead instrument.\n"
    "- Do NOT quote or summarise the lyrics anywhere.\n"
    "- Write flowing prose inside each field.";

static const char * PROMPT_LYRICS =
    "Transcribe the complete lyrics of this song.\n\n"
    "Mark every section with a bracketed tag on its own line: [Intro], [Verse], "
    "[Pre-Chorus], [Chorus], [Bridge], [Breakdown], [Instrumental], [Outro]. Put the sung "
    "words underneath each tag, one line per sung line. Use [Instrumental] for passages "
    "with no vocals. Transcribe only what is actually sung; do not invent lines, translate, "
    "or add commentary.\n\n"
    "If the track has no vocals at all, reply with exactly: [Instrumental]";

static bool ends_with_ci(const std::string & s, const char * suf) {
    const size_t n = strlen(suf);
    if (s.size() < n) {
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        if (tolower((unsigned char) s[s.size() - n + i]) != tolower((unsigned char) suf[i])) {
            return false;
        }
    }
    return true;
}

// audio-io.h reads WAV and MP3 only. Everything else goes through ffmpeg, which
// is the same contract ace-train uses ("--ffmpeg <path> required for non-WAV/MP3
// input") -- worth matching rather than vendoring a second decode path, since
// dataset tracks are overwhelmingly FLAC.
static bool transcode_to_wav16k(const std::string & ffmpeg, const std::string & src,
                                std::string & out_path) {
    // L_tmpnam_s is MSVC's Annex K spelling and does not exist on glibc, so the
    // buffer has to be declared inside the branch -- naming it outside broke the
    // Linux build.
#ifdef _WIN32
    char tmpl[L_tmpnam_s];
    if (tmpnam_s(tmpl, sizeof(tmpl)) != 0) {
        return false;
    }
#else
    char tmpl[L_tmpnam];
    if (!tmpnam(tmpl)) {
        return false;
    }
#endif
    out_path = std::string(tmpl) + ".wav";
    const std::string cmd = "\"\"" + ffmpeg + "\" -y -loglevel error -i \"" + src +
                            "\" -ac 1 -ar 16000 -c:a pcm_s16le \"" + out_path + "\"\"";
    const int rc = system(cmd.c_str());
    if (rc != 0) {
        fprintf(stderr, "ace-caption: ffmpeg failed (%d) on %s\n", rc, src.c_str());
        return false;
    }
    return true;
}

static bool read_text_file(const std::string & path, std::string & out) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize((size_t) std::max(0L, n));
    const size_t got = out.empty() ? 0 : fread(&out[0], 1, out.size(), f);
    fclose(f);
    return got == out.size();
}

static void push_text(const BPETokenizer & tok, const std::string & s,
                      std::vector<int32_t> & ids) {
    for (int id : bpe_encode(&tok, s, false)) {
        ids.push_back((int32_t) id);
    }
}

static bool push_special(const BPETokenizer & tok, const char * s,
                         std::vector<int32_t> & ids) {
    auto it = tok.vocab.find(s);
    if (it == tok.vocab.end()) {
        fprintf(stderr, "ace-caption: tokenizer has no '%s'\n", s);
        return false;
    }
    ids.push_back((int32_t) it->second);
    return true;
}

// GPT-2 byte-level decode: map each token's UTF-8 surrogate chars back to bytes.
static std::string detokenize(const BPETokenizer & tok, const std::vector<int32_t> & ids) {
    std::unordered_map<std::string, unsigned char> str2byte;
    for (int b = 0; b < 256; ++b) {
        str2byte[tok.byte2str[b]] = (unsigned char) b;
    }
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || (size_t) id >= tok.id_to_str.size()) {
            continue;
        }
        const std::string & piece = tok.id_to_str[(size_t) id];
        // Walk UTF-8 codepoints; each maps back to exactly one byte.
        size_t i = 0;
        while (i < piece.size()) {
            int adv = 1;
            utf8_codepoint(piece.c_str() + i, &adv);
            const std::string ch = piece.substr(i, (size_t) adv);
            auto it = str2byte.find(ch);
            if (it != str2byte.end()) {
                out.push_back((char) it->second);
            } else {
                out += ch;  // a special token or something unmapped: pass through
            }
            i += (size_t) adv;
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    std::string models, src, mode = "prose", prompt, out_path, ffmpeg, src_list;
    std::vector<std::string> prompt_files;   // repeatable; see the per-mode note below
    int max_tokens = 1024;
    int top_k_debug = 0;
    std::string dump_ids, dump_encoder;   // parity instrumentation, see --dump-* below
    double max_seconds = 420.0;   // hard cap, see below
    int enc_window = 400;         // mel frames per encoder chunk; 0 = whole-track (see below)
    moss::SamplerParams sp;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char * what) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "ace-caption: %s needs a value\n", what);
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--models") { models = next("--models"); }
        else if (a == "--src-audio") { src = next("--src-audio"); }
        else if (a == "--mode") { mode = next("--mode"); }
        else if (a == "--prompt") { prompt = next("--prompt"); }
        else if (a == "--prompt-file") { prompt_files.push_back(next("--prompt-file")); }
        else if (a == "--src-list") { src_list = next("--src-list"); }
        else if (a == "-o") { out_path = next("-o"); }
        else if (a == "--ffmpeg") { ffmpeg = next("--ffmpeg"); }
        else if (a == "--top-k-debug") { top_k_debug = atoi(next("--top-k-debug").c_str()); }
        else if (a == "--dump-ids") { dump_ids = next("--dump-ids"); }
        else if (a == "--dump-encoder") { dump_encoder = next("--dump-encoder"); }
        else if (a == "--max-tokens") { max_tokens = atoi(next("--max-tokens").c_str()); }
        else if (a == "--max-seconds") { max_seconds = atof(next("--max-seconds").c_str()); }
        else if (a == "--encoder-window") { enc_window = atoi(next("--encoder-window").c_str()); }
        else if (a == "--temperature") { sp.temperature = (float) atof(next("--temperature").c_str()); }
        else if (a == "--rep-penalty") { sp.repetition_penalty = (float) atof(next("--rep-penalty").c_str()); }
        else if (a == "--freq-penalty") { sp.frequency_penalty = (float) atof(next("--freq-penalty").c_str()); }
        else {
            fprintf(stderr,
                    "usage: ace-caption --models <dir> --src-audio <file>\n"
                    "       [--mode prose|mm3|lyrics[,...]] [--prompt \"...\"]\n"
                    "       [--prompt-file [prose=|mm3=|lyrics=]<path>]   (repeatable)\n"
                    "       [-o <out.txt>] [--max-tokens N] [--max-seconds S]\n"
                    "       [--temperature T] [--rep-penalty R] [--freq-penalty F]\n"
                    "       [--dump-ids <path>] [--dump-encoder <path>]   (parity debug)\n"
                    "       [--encoder-window FRAMES]   (0 = whole-track legacy encode)\n");
            return 2;
        }
    }
    if (models.empty() || (src.empty() && src_list.empty())) {
        fprintf(stderr, "ace-caption: --models and one of --src-audio / --src-list are required\n");
        return 2;
    }

    // ---- the work list ----------------------------------------------------
    // Each entry is (audio path, output base). --src-list is TAB-separated:
    //   D:\a\01.flac<TAB>D:\out\01
    // The caller supplies the output base explicitly rather than it being derived
    // from the audio stem, because dataset filenames collide across albums and a
    // derived name would silently overwrite one track's caption with another's.
    std::vector<std::pair<std::string, std::string>> jobs;
    if (!src_list.empty()) {
        std::string listing;
        if (!read_text_file(src_list, listing)) {
            fprintf(stderr, "ace-caption: cannot read %s\n", src_list.c_str());
            return 1;
        }
        std::string line;
        std::istringstream ls(listing);
        while (std::getline(ls, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) { line.pop_back(); }
            if (line.empty()) { continue; }
            const size_t tab = line.find('\t');
            if (tab == std::string::npos) {
                const size_t dot = line.find_last_of('.');
                jobs.emplace_back(line, dot == std::string::npos ? line : line.substr(0, dot));
            } else {
                jobs.emplace_back(line.substr(0, tab), line.substr(tab + 1));
            }
        }
        if (jobs.empty()) {
            fprintf(stderr, "ace-caption: --src-list %s has no entries\n", src_list.c_str());
            return 1;
        }
    } else {
        jobs.emplace_back(src, out_path);
    }

    // --prompt / --prompt-file override every mode; otherwise each mode picks its
    // own built-in prompt. --modes takes a comma-separated list so all formats
    // come off ONE encode (see the cost-shape note at the top).
    //
    // --prompt-file also takes a PER-MODE form, `--prompt-file prose=<path>`,
    // which is what makes "one encode, two different caption formats" possible.
    // The Training Studio needs Side-Step's 5-line `caption:/genre:/bpm:/key:/
    // signature:` block AND an MM3 Structured Caption from the same audio; with
    // only the blanket override those are two full encodes, and the encode is
    // the expensive half of the run.
    std::string prompt_override = prompt;
    std::unordered_map<std::string, std::string> mode_prompt;
    for (const std::string & spec : prompt_files) {
        // Split on the FIRST '=' only when the left side names a mode, so a
        // bare Windows path (`D:\x\p.txt`) still parses as the blanket form.
        const size_t eq = spec.find('=');
        const std::string head = eq == std::string::npos ? std::string() : spec.substr(0, eq);
        const bool per_mode = head == "prose" || head == "mm3" || head == "lyrics";
        std::string text;
        if (!read_text_file(per_mode ? spec.substr(eq + 1) : spec, text)) {
            fprintf(stderr, "ace-caption: cannot read %s\n",
                    (per_mode ? spec.substr(eq + 1) : spec).c_str());
            return 1;
        }
        if (per_mode) { mode_prompt[head] = text; } else { prompt_override = text; }
    }
    std::vector<std::string> modes;
    {
        std::string cur;
        for (char c : mode) {
            if (c == ',') {
                if (!cur.empty()) { modes.push_back(cur); }
                cur.clear();
            } else {
                cur.push_back(c);
            }
        }
        if (!cur.empty()) { modes.push_back(cur); }
        if (modes.empty()) { modes.push_back("prose"); }
    }
    // Precedence: per-mode override > blanket override > the mode's built-in.
    auto prompt_for_mode = [&](const std::string & m) -> std::string {
        const auto it = mode_prompt.find(m);
        if (it != mode_prompt.end()) { return it->second; }
        if (!prompt_override.empty()) { return prompt_override; }
        if (m == "mm3") { return PROMPT_MM3; }
        if (m == "lyrics") { return PROMPT_LYRICS; }
        return PROMPT_PROSE;
    };

    // ---- audio: any format -> 16 kHz mono ---------------------------------
    // A lambda, not straight-line code, because the model load below must happen
    // ONCE and every source file then reuses it (see the --src-list note above).
    // Returns false on a bad file so a batch run can skip it and carry on rather
    // than throwing away a loaded 8 GB model over one unreadable track.
    auto load_pcm = [&](const std::string & path, std::vector<float> & pcm) -> bool {
        std::string audio_path = path, tmp_wav;
        if (!ends_with_ci(path, ".wav") && !ends_with_ci(path, ".mp3")) {
            if (ffmpeg.empty()) {
                fprintf(stderr, "ace-caption: %s is not WAV/MP3; pass --ffmpeg <path>\n",
                        path.c_str());
                return false;
            }
            if (!transcode_to_wav16k(ffmpeg, path, tmp_wav)) {
                return false;
            }
            audio_path = tmp_wav;
        }
        int T_in = 0, sr = 0;
        float * planar = audio_read(audio_path.c_str(), &T_in, &sr);  // planar [L:T][R:T]
        if (!tmp_wav.empty()) {
            remove(tmp_wav.c_str());
        }
        if (!planar || T_in <= 0) {
            fprintf(stderr, "ace-caption: cannot read audio %s\n", path.c_str());
            return false;
        }
        int T16 = T_in;
        float * res = planar;
        bool owned = false;
        if (sr != 16000) {
            res = audio_resample(planar, T_in, sr, 16000, 2, &T16);
            owned = true;
            if (!res) {
                fprintf(stderr, "ace-caption: resample %d -> 16000 failed\n", sr);
                free(planar);
                return false;
            }
        }
        // A 9:57 track pushed 78 GB through WDDM shared-memory spill on a 32 GB
        // card and never finished. On Windows that fails as silent slowdown, not
        // an error, so the cap is enforced here rather than trusted to the user.
        const int max_samples = (int) (max_seconds * 16000.0);
        const int T_use = std::min(T16, max_samples);
        if (T_use < T16) {
            fprintf(stderr, "[MOSS] input is %.0f s; using the first %.0f s (--max-seconds)\n",
                    (double) T16 / 16000.0, max_seconds);
        }
        pcm.assign((size_t) T_use, 0.0f);
        for (int t = 0; t < T_use; ++t) {
            pcm[(size_t) t] = 0.5f * (res[t] + res[T16 + t]);   // planar L/R -> mono
        }
        if (owned) {
            free(res);
        }
        free(planar);
        return true;
    };

    // ---- models -----------------------------------------------------------
    const std::string sep =
#ifdef _WIN32
        "\\";
#else
        "/";
#endif
    BackendPair bp = backend_init("MOSS");
    if (!bp.backend) {
        fprintf(stderr, "ace-caption: no backend\n");
        return 1;
    }

    moss::AudioTower tower;
    moss::LmModel lm;
    std::string lm_path = models + sep + "moss-lm-q8_0.gguf";
    {
        FILE * probe = fopen(lm_path.c_str(), "rb");
        if (probe) {
            fclose(probe);
        } else {
            lm_path = models + sep + "moss-lm-f16.gguf";
        }
    }
    if (!moss::moss_load_audio_tower(&tower, (models + sep + "moss-aud-f16.gguf").c_str(),
                                     bp.backend) ||
        !moss::moss_load_lm(&lm, lm_path.c_str(), bp.backend)) {
        moss::moss_free_lm(&lm);
        moss::moss_free_audio_tower(&tower);
        backend_release(bp.backend, bp.cpu_backend);
        return 1;
    }

    BPETokenizer tok;
    if (!load_bpe_from_gguf(&tok, lm_path.c_str())) {
        fprintf(stderr, "ace-caption: tokenizer load failed\n");
        moss::moss_free_lm(&lm);
        moss::moss_free_audio_tower(&tower);
        backend_release(bp.backend, bp.cpu_backend);
        return 1;
    }

    // ---- per-file loop ----------------------------------------------------
    // The model is loaded ABOVE this line and reused for every entry. That is the
    // whole point of --src-list: on a warm cache the load costs ~4.6 s against
    // ~6 s of actual decoding, so re-spawning per track spent roughly 40 % of a
    // bulk run re-reading 8.3 GB it had just discarded.
    //
    // Scope is deliberately ONE DATASET, not the whole bulk run: a bulk job
    // interleaves captioning with LM/DiT training per dataset, and holding ~10 GB
    // of captioner resident while a trainer wants the card is the wrong trade.
    // The process exits between datasets and the memory goes back.
    int batch_rc = 0;
    for (size_t ji = 0; ji < jobs.size(); ++ji) {
    const std::string & src_file = jobs[ji].first;
    const std::string & out_file = jobs[ji].second;
    if (jobs.size() > 1) {
        fprintf(stderr, "[MOSS] (%zu/%zu) %s\n", ji + 1, jobs.size(), src_file.c_str());
    }
    std::vector<float> pcm;
    if (!load_pcm(src_file, pcm)) {
        // Skip, don't abort: one unreadable track must not cost the whole batch
        // the model load it has already paid for.
        batch_rc = 1;
        continue;
    }

    // ---- mel + encode -----------------------------------------------------
    moss::MelParams mp;
    mp.sample_rate = (int) tower.hp.sample_rate;
    mp.n_fft = (int) tower.hp.n_fft;
    mp.hop_length = (int) tower.hp.hop_length;
    mp.n_mels = (int) tower.hp.n_mels;

    int frames = 0;
    std::vector<float> mel = moss::log_mel(pcm.data(), pcm.size(), mp, &frames);

    // CHUNKED ENCODE -- this matches how the model was TRAINED, and it is the
    // single biggest quality lever in this tool. SGLang's serving path (the
    // runtime MOSS's authors recommend) splits the mel into n_window*2 = 400
    // frame (4 s) chunks and runs the encoder on each independently: positions
    // restart per chunk and attention is local to the chunk
    // (sglang/srt/models/moss_audio.py). One whole-track pass -- what vanilla
    // transformers' modeling code does, and what this tool did before -- feeds
    // the encoder positions far past max_source_positions (1500) and global
    // attention spans it never saw in training. Measured on
    // 08-the_clash-lost_in_the_supermarket (230 s): whole-track collapses to
    // the prompt's example genre ("bass house, electro house", q8_0 AND f16),
    // chunked matches SGLang ("indie rock, post-punk revival"). At 30 s clips
    // both paths agree, which is why the original fixtures never caught it.
    // 400 is a multiple of 8, so per-chunk token counts sum to the whole-track
    // count and the time-marker interleave downstream is unchanged.
    // The adapter and deepstack mergers are per-token GatedMLPs, so running
    // them per chunk and concatenating equals running them on the full track.
    const int chunk = (enc_window > 0) ? enc_window : frames;
    moss::EncoderOutput eo;
    bool enc_ok = true;
    for (int off = 0; off < frames && enc_ok; off += chunk) {
        const int len = std::min(chunk, frames - off);
        // mel is row-major [n_mels, frames]; slice columns [off, off+len).
        std::vector<float> piece((size_t) mp.n_mels * len);
        for (int m = 0; m < mp.n_mels; ++m) {
            memcpy(piece.data() + (size_t) m * len,
                   mel.data() + (size_t) m * frames + off,
                   sizeof(float) * (size_t) len);
        }
        moss::EncoderOutput part;
        if (!moss::moss_encode_audio(tower, piece.data(), mp.n_mels, len, bp.backend, &part)) {
            enc_ok = false;
            break;
        }
        eo.n_tokens += part.n_tokens;
        eo.encoder_out.insert(eo.encoder_out.end(), part.encoder_out.begin(), part.encoder_out.end());
        eo.adapter_out.insert(eo.adapter_out.end(), part.adapter_out.begin(), part.adapter_out.end());
        if (eo.deepstack_taps.size() < part.deepstack_taps.size()) eo.deepstack_taps.resize(part.deepstack_taps.size());
        for (size_t k = 0; k < part.deepstack_taps.size(); ++k) {
            eo.deepstack_taps[k].insert(eo.deepstack_taps[k].end(), part.deepstack_taps[k].begin(), part.deepstack_taps[k].end());
        }
        if (eo.merger_out.size() < part.merger_out.size()) eo.merger_out.resize(part.merger_out.size());
        for (size_t k = 0; k < part.merger_out.size(); ++k) {
            eo.merger_out[k].insert(eo.merger_out[k].end(), part.merger_out[k].begin(), part.merger_out[k].end());
        }
    }
    if (!enc_ok) {
        fprintf(stderr, "ace-caption: encode failed for %s\n", src_file.c_str());
        batch_rc = 1;
        continue;
    }
    fprintf(stderr, "[MOSS] %.1f s audio -> %d mel frames -> %d audio tokens (%.2f/s)\n",
            (double) pcm.size() / 16000.0, frames, eo.n_tokens,
            (double) eo.n_tokens / ((double) pcm.size() / 16000.0));

    // ---- parity instrumentation (--dump-encoder) --------------------------
    // Raw little-endian f32 of encoder_out [n_tokens, d_model] in row-major
    // (token-major) order, plus a one-line manifest. Written for the FIRST job
    // only -- this exists to diff a single track against the reference at FULL
    // length, which is the parity bar the 30 s fixtures never tested.
    if (!dump_encoder.empty()) {
        const int D = eo.n_tokens > 0 ? (int) (eo.encoder_out.size() / (size_t) eo.n_tokens) : 0;
        FILE * df = fopen(dump_encoder.c_str(), "wb");
        if (df) {
            fwrite(eo.encoder_out.data(), sizeof(float), eo.encoder_out.size(), df);
            fclose(df);
            double s1 = 0.0, s2 = 0.0;
            for (float v : eo.encoder_out) { s1 += v; s2 += (double) v * v; }
            const double n = (double) eo.encoder_out.size();
            fprintf(stderr, "[MOSS] dumped encoder_out [%d,%d] mean=%.6f rms=%.6f -> %s\n",
                    eo.n_tokens, D, s1 / n, sqrt(s2 / n), dump_encoder.c_str());
        } else {
            fprintf(stderr, "[MOSS] could not open %s for writing\n", dump_encoder.c_str());
        }
        dump_encoder.clear();   // first job only
    }


    // ---- prompt assembly (per mode) ---------------------------------------
    // Everything from here down runs once PER OUTPUT MODE, but `eo` above is
    // computed once and shared: the encoder is prompt-independent, so N formats
    // cost one encode plus N decodes rather than N full passes. On a 4-minute
    // track the encode is the expensive half, so asking for prose+mm3+lyrics
    // together is far cheaper than three invocations.
    auto build_ids = [&](const std::string & user_prompt, std::vector<int32_t> & ids) -> bool {
    // Mirrors processing_moss_music.py::_build_default_prompt.
    bool ok = true;
    ok = ok && push_special(tok, "<|im_start|>", ids);
    push_text(tok, "system\nYou are a helpful assistant.", ids);
    ok = ok && push_special(tok, "<|im_end|>", ids);
    push_text(tok, "\n", ids);
    ok = ok && push_special(tok, "<|im_start|>", ids);
    push_text(tok, "user\n", ids);
    // The audio markers are pushed BY ID, not by looking up "<|audio_bos|>" etc.
    // Those strings are NOT in the vocabulary: processing_moss_music.py monkey-
    // patches convert_tokens_to_ids with an alias map precisely because the
    // tokenizer cannot resolve them. Ids come from the GGUF KVs instead.
    ids.push_back((int32_t) tower.hp.audio_bos_id);
    {
        // TIME MARKERS. Every `every_s` seconds the elapsed second count is
        // written into the audio stream as ordinary digit tokens, splitting the
        // <|AUDIO|> run into segments.
        //
        // THIS IS NOT OPTIONAL. SGLang's processor -- the runtime MOSS's authors
        // recommend -- always emits them. The HF processor has the identical
        // routine but `from_pretrained` pops `enable_time_marker` defaulting to
        // FALSE (its __init__ signature says True), so the Transformers path
        // silently feeds the model an out-of-distribution token stream. That is
        // what makes the HF path confidently wrong rather than noisy: on a 30 s
        // clip of Daft Punk it answers "Arabic pop" where SGLang answers
        // "Eurodance". Omit these and the port reproduces the wrong reference.
        const int every_s = 2;
        const double tps = (double) tower.hp.tokens_per_second;   // 12.5
        const int every_tok = (int) (tps * every_s);              // 25
        const int total_s = (int) ((double) eo.n_tokens / tps);
        int consumed = 0;
        for (int second = every_s; second <= total_s; second += every_s) {
            const int marker_pos = (second / every_s) * every_tok;
            for (int i = consumed; i < marker_pos && i < eo.n_tokens; ++i) {
                ids.push_back((int32_t) tower.hp.audio_token_id);
            }
            consumed = std::min(marker_pos, eo.n_tokens);
            // The second count as decimal digits, each its own ordinary token.
            const std::string digits = std::to_string(second);
            for (char d : digits) {
                push_text(tok, std::string(1, d), ids);
            }
        }
        for (int i = consumed; i < eo.n_tokens; ++i) {
            ids.push_back((int32_t) tower.hp.audio_token_id);
        }
    }
    ids.push_back((int32_t) tower.hp.audio_eos_id);
    push_text(tok, "\n" + user_prompt, ids);
    ok = ok && push_special(tok, "<|im_end|>", ids);
    push_text(tok, "\n", ids);
    ok = ok && push_special(tok, "<|im_start|>", ids);
    push_text(tok, "assistant\n", ids);
    return ok;
    };  // build_ids

    // Runs one output mode against the shared EncoderOutput.
    auto run_mode = [&](const std::string & user_prompt, std::string & out_text) -> bool {
    std::vector<int32_t> ids;
    if (!build_ids(user_prompt, ids)) {
        return false;
    }

    // ---- parity instrumentation (--dump-ids) ------------------------------
    // One decimal token id per line, in prompt order. Diffed against the
    // reference processor's input_ids to prove the time-marker interleave and
    // chat wrapper agree token-for-token at full track length.
    if (!dump_ids.empty()) {
        FILE * df = fopen(dump_ids.c_str(), "w");
        if (df) {
            for (int32_t id : ids) {
                fprintf(df, "%d\n", (int) id);
            }
            fclose(df);
            fprintf(stderr, "[MOSS] dumped %zu input ids -> %s\n", ids.size(), dump_ids.c_str());
        } else {
            fprintf(stderr, "[MOSS] could not open %s for writing\n", dump_ids.c_str());
        }
        dump_ids.clear();   // first mode only
    }

    // ---- join tensors -----------------------------------------------------
    const int64_t T = (int64_t) ids.size();
    const int64_t H = (int64_t) lm.hp.n_embd;
    std::vector<float> text_mask((size_t) T, 1.0f);
    std::vector<float> audio_vals((size_t) (H * T), 0.0f);
    std::vector<std::vector<float>> merge(eo.merger_out.size(),
                                          std::vector<float>((size_t) (H * T), 0.0f));
    int64_t seen = 0;
    for (int64_t i = 0; i < T; ++i) {
        if ((uint32_t) ids[(size_t) i] == tower.hp.audio_token_id) {
            text_mask[(size_t) i] = 0.0f;
            memcpy(&audio_vals[(size_t) (i * H)], &eo.adapter_out[(size_t) (seen * H)],
                   (size_t) H * sizeof(float));
            for (size_t k = 0; k < merge.size(); ++k) {
                memcpy(&merge[k][(size_t) (i * H)], &eo.merger_out[k][(size_t) (seen * H)],
                       (size_t) H * sizeof(float));
            }
            ++seen;
        }
    }

    // ---- generate ---------------------------------------------------------
    moss::LmKv kv;
    if (!moss::moss_lm_kv_init(&kv, lm, T + max_tokens + 8, bp.backend)) {
        fprintf(stderr, "ace-caption: kv init failed (needs %lld positions)\n",
                (long long) (T + max_tokens + 8));
        return false;
    }
    // The cache is per-mode: each mode has its own prompt and therefore its own
    // KV contents. Only `eo` is shared.
    struct KvGuard {
        moss::LmKv * p;
        ~KvGuard() { moss::moss_lm_kv_free(p); }
    } kv_guard{ &kv };

    std::vector<float> logits;
    if (!moss::moss_lm_eval(lm, kv, ids, audio_vals, merge, text_mask, bp.backend, &logits)) {
        return false;
    }

    // --top-k-debug: dump the first generated position's top-N as a stable,
    // comparable distribution. Free-text agreement is the WRONG parity bar for a
    // 400-token greedy generation -- two runs that differ in the last bits will
    // fork at the first genuinely ambiguous token and never reconverge. The
    // distribution at step 0 is the thing that actually says whether the graphs
    // agree, and it is directly comparable with SGLang's return_logprob output.
    if (top_k_debug > 0) {
        std::vector<int> idx((size_t) logits.size());
        for (size_t i = 0; i < idx.size(); ++i) {
            idx[i] = (int) i;
        }
        const int n = std::min<int>(top_k_debug, (int) idx.size());
        std::partial_sort(idx.begin(), idx.begin() + n, idx.end(),
                          [&](int a, int b) { return logits[(size_t) a] > logits[(size_t) b]; });
        printf("TOPK %d\n", n);
        for (int i = 0; i < n; ++i) {
            const std::string piece = detokenize(tok, { (int32_t) idx[(size_t) i] });
            printf("%d\t%.6f\t%s\n", idx[(size_t) i], logits[(size_t) idx[(size_t) i]],
                   piece.c_str());
        }
        out_text.clear();
        return true;   // caller exits; --top-k-debug is a diagnostic, not a caption
    }

    const int32_t eos = (int32_t) 151645;   // <|im_end|>
    std::vector<int32_t> counts((size_t) lm.hp.n_vocab, 0);
    std::vector<int32_t> gen;
    for (int step = 0; step < max_tokens; ++step) {
        const int32_t next = moss::moss_sample(logits, counts, sp);
        if (next == eos) {
            break;
        }
        gen.push_back(next);
        if ((size_t) next < counts.size()) {
            counts[(size_t) next]++;
        }
        if (step + 1 >= max_tokens) {
            break;
        }
        if (!moss::moss_lm_eval(lm, kv, { next }, {}, {}, {}, bp.backend, &logits)) {
            break;
        }
    }

    out_text = detokenize(tok, gen);
    return true;
    };  // run_mode

    // ---- run every requested mode off the one encode ----------------------
    int rc = 0;
    for (size_t mi = 0; mi < modes.size(); ++mi) {
        const std::string & mname = modes[mi];
        const std::string p = prompt_for_mode(mname);   // handles both override forms
        std::string text;
        const double t0 = (double) clock() / CLOCKS_PER_SEC;
        if (!run_mode(p, text)) {
            fprintf(stderr, "ace-caption: mode '%s' failed\n", mname.c_str());
            rc = 1;
            break;
        }
        const double el = (double) clock() / CLOCKS_PER_SEC - t0;
        if (top_k_debug > 0) {
            break;   // diagnostic already printed
        }

        if (out_file.empty()) {
            if (modes.size() > 1) {
                printf("===== %s =====\n", mname.c_str());
            }
            printf("%s\n", text.c_str());
        } else {
            // One mode -> -o is the file. Several -> -o is a prefix, and each
            // mode appends its own suffix, so a caller can ask for all formats
            // in one invocation and get predictable filenames.
            std::string dst = out_file;
            if (modes.size() > 1) {
                const size_t dot = out_file.find_last_of('.');
                const std::string stem = (dot == std::string::npos) ? out_file
                                                                    : out_file.substr(0, dot);
                const std::string ext = (dot == std::string::npos) ? ".txt"
                                                                   : out_file.substr(dot);
                dst = stem + "." + mname + ext;
            }
            FILE * f = fopen(dst.c_str(), "wb");
            if (!f) {
                fprintf(stderr, "ace-caption: cannot write %s\n", dst.c_str());
                rc = 1;
                break;
            }
            fwrite(text.data(), 1, text.size(), f);
            fputc('\n', f);
            fclose(f);
            fprintf(stderr, "[MOSS] %-7s -> %s (%.1fs)\n", mname.c_str(), dst.c_str(), el);
        }
    }
    if (rc != 0) {
        batch_rc = rc;
    }
    if (top_k_debug > 0) {
        break;   // diagnostic mode is single-shot by design
    }
    }   // ---- end per-file loop ----------------------------------------------

    moss::moss_free_lm(&lm);
    moss::moss_free_audio_tower(&tower);
    backend_release(bp.backend, bp.cpu_backend);
    // Non-zero if ANY file failed, but only after every other file has been
    // captioned — a batch reports partial failure, it does not discard good work.
    return batch_rc;
}
