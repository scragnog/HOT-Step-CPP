// YuE2 plan continuation (plan_only + plan_continue + abc). No GPU, no model:
//  - the request parser's accept and reject cases, and the old plan_only/abc
//    exclusion without the flag;
//  - the section stop rule (yue2_plan_section_cut) on token-by-token text;
//  - the tokenizer prefix property: encoding a score cut after a label line
//    gives a token prefix of encoding the whole score, so the prefill sees
//    the tokens the planner would have had. Needs the YuE2 tokenizer dir
//    (vocab.json + merges.txt) in YUE2_TOKENIZER_DIR; skips that part (77)
//    without it.
#include "yue2-request.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

static int g_failed = 0;
// No #c: MSVC mis-stringizes raw string literals in macro arguments.
static void check(bool ok, int line) {
    if (!ok) {
        std::fprintf(stderr, "FAIL line %d\n", line);
        g_failed++;
    }
}
#define CHECK(c) check((c), __LINE__)

static bool parses(const std::string & body, Yue2Request * r = nullptr) {
    Yue2Request tmp;
    std::string err;
    return yue2_parse_request(body, r ? r : &tmp, &err);
}

// A planned score as the engine writes it (shortened), three labels.
static const char * kScore =
    "X:1\nT:\nM:2/4\nL:1/16\nQ:1/4=90\n"
    "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\n"
    "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\nK:D\n"
    "% intro\nV: Vocal\n\"D\"z8|\"D\"z8|\"D\"z8|\"D\"z8|\nV: Ins\nz2A3A3|AB2d2BA2|z2A3A3|AB2d2BA2|\n"
    "% verse\nV: Vocal\n\"D\"d4a4|\"D\"f4e4|\"D\"d4a4|\"D\"f4z4|\nV: Ins\nZ4|\n"
    "V: Vocal\n\"Bm\"z8|\"Bm\"z8|\"Bm\"z8|\"Bm\"z8|\nV: Ins\nz2B3B3|B3d2B3|z2B3B3|B3d2BB2|\n"
    "% chorus\nV: Vocal\n\"G\"d4e4|\"G\"f4a4|\"A\"c4d4|\"A\"e4z4|\nV: Ins\nZ4|\n"
    "V: Vocal\n\"G\"d4e4|\"G\"f4a4|\"A\"c8-|\"A\"c2B2B2d2|\nV: Ins\nZ4|\n";

static void test_parse() {
    const std::string pre = "X:1\\nK:D\\n% chorus\\n";
    const std::string base = R"("style":"pop","cot":"melody","plan_only":true)";
    Yue2Request r;
    // accept: request abc, and every songs entry with its own prefix
    CHECK(parses("{" + base + R"(,"plan_continue":true,"abc":")" + pre + "\"}", &r));
    CHECK(r.plan_continue && r.plan_only && r.abc_provided && r.abc == "X:1\nK:D\n% chorus\n");
    CHECK(parses("{" + base + R"(,"plan_continue":true,"plan_max_tokens":1024,"songs":[{"style":"a","abc":")" + pre +
                 R"(","seed":1},{"style":"b","abc":")" + pre + R"(","seed":2}]})", &r));
    CHECK(r.songs.size() == 2 && r.plan.max_tokens == 1024);
    // plan_continue false is the old request
    CHECK(parses("{" + base + R"(,"plan_continue":false})"));
    // reject: old exclusion without the flag (also with it false)
    CHECK(!parses("{" + base + R"(,"abc":")" + pre + "\"}"));
    CHECK(!parses("{" + base + R"(,"plan_continue":false,"abc":")" + pre + "\"}"));
    // reject: without plan_only, without abc, a songs entry without abc
    CHECK(!parses(R"({"style":"pop","cot":"melody","plan_continue":true,"abc":")" + pre + "\"}"));
    CHECK(!parses("{" + base + R"(,"plan_continue":true})"));
    CHECK(!parses("{" + base + R"(,"plan_continue":true,"abc":")" + pre + R"(","songs":[{"style":"a","abc":")" + pre +
                  R"("},{"style":"b"}]})"));
    // reject: last non-empty line is not a label, no trailing newline, not a bool
    CHECK(!parses("{" + base + R"(,"plan_continue":true,"abc":"X:1\nK:D\nabc|\n"})"));
    CHECK(!parses("{" + base + R"(,"plan_continue":true,"abc":"X:1\nK:D\n% chorus"})"));
    CHECK(!parses("{" + base + R"(,"plan_continue":"yes","abc":")" + pre + "\"}"));
    // a blank line after the label still ends in the label
    CHECK(parses("{" + base + R"(,"plan_continue":true,"abc":"X:1\n% chorus\n\n"})"));
}

// Feed `pieces` as successive decoded tokens after `prefix`; return the cut
// the plan loop would make (npos = no stop).
static size_t feed(const std::string & prefix, std::initializer_list<const char *> pieces) {
    std::string t = prefix;
    for (const char * s : pieces) {
        const size_t from = t.size();
        t += s;
        const size_t c = yue2_plan_section_cut(t, from);
        if (c != std::string::npos) return c;
    }
    return std::string::npos;
}

static void test_stop() {
    const std::string pre = "K:D\n% chorus\n";
    const size_t      n   = pre.size();
    // marker split across two tokens
    CHECK(feed(pre, { "V:", " Vocal", "\n", "a|", "\n", "% verse" }) == n + 12);
    CHECK(feed(pre, { "V: Vocal\na|\n", "%" }) == n + 12);
    // marker inside one token
    CHECK(feed(pre, { "V: Vocal\na", "|\n%" }) == n + 12);
    // '%' mid-line is no stop
    CHECK(feed(pre, { "V: Vocal\n", "a % b|", "\n" }) == std::string::npos);
    // immediate '%': the prefix's final newline counts, the section is empty
    CHECK(feed(pre, { "%" }) == n);
    CHECK(feed(pre, { "% outro" }) == n);
    // a label inside the prefix is never a stop
    CHECK(yue2_plan_section_cut(pre, n) == std::string::npos);
    // the cut keeps the section's last newline
    std::string t = pre + "a|\n% v";
    CHECK(t.substr(0, yue2_plan_section_cut(t, n)) == pre + "a|\n");
}

static int test_tokenizer() {
    const char * dir = std::getenv("YUE2_TOKENIZER_DIR");
    BPETokenizer tok;
    if (!dir || !yue2_tokenizer_load_from_dir(&tok, dir)) {
        std::printf("tokenizer prefix: SKIP (set YUE2_TOKENIZER_DIR)\n");
        return 77;
    }
    const std::string      score = kScore;
    const std::vector<int> whole = yue2_bpe_encode(&tok, score);
    int                    labels = 0;
    for (size_t at = score.find("\n%"); at != std::string::npos; at = score.find("\n%", at + 1)) {
        const size_t           eol = score.find('\n', at + 1);
        const std::string      cut = score.substr(0, eol + 1);  // through the label line and its newline
        const std::vector<int> ids = yue2_bpe_encode(&tok, cut);
        CHECK(ids.size() <= whole.size() && std::equal(ids.begin(), ids.end(), whole.begin()));
        CHECK(yue2_continue_prefix_ok(cut));
        labels++;
    }
    CHECK(labels == 3);
    // decode(encode) is lossless, so the returned prefix is byte-for-byte
    CHECK(yue2_bpe_decode(&tok, std::vector<int32_t>(whole.begin(), whole.end())) == score);
    return 0;
}

int main() {
    test_parse();
    test_stop();
    const int tok = test_tokenizer();
    std::printf("yue2 plan continue: %s\n", g_failed ? "FAIL" : "PASS");
    if (g_failed) return 1;
    return tok;
}
