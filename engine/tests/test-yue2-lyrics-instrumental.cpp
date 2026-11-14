// yue2_lyrics_say_instrumental: non-Latin lyrics are words, not an
// instrumental (GH #355). No models or GPU.
#include "../src/train/yue2-sidecar.h"

int main() {
    struct { const char * lyrics; bool want; } cases[] = {
        { "[Verse 1]\n\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82 \xD0\xBC\xD0\xB8\xD1\x80", false },  // Cyrillic
        { "[Chorus]\n\xE6\x98\xA5\xE7\x9C\xA0\xE4\xB8\x8D\xE8\xA7\x89\xE6\x99\x93", false },              // CJK
        { "[Verse]\nhello world", false },
        { "[Instrumental]", true },
        { "[Intro]\n[Instrumental]\n[Outro]", true },
        { "This song is an instrumental", true },
        { "", true },
        { "[Verse]\n123 ...", true },
    };
    int fails = 0;
    for (auto & c : cases) {
        bool got = yue2_lyrics_say_instrumental(c.lyrics);
        if (got != c.want) {
            printf("FAIL: want %d got %d for \"%s\"\n", c.want, got, c.lyrics);
            fails++;
        }
    }
    printf(fails ? "FAIL: %d case(s)\n" : "PASS: yue2 lyrics instrumental (%d)\n", fails ? fails : (int) (sizeof cases / sizeof cases[0]));
    return fails ? 1 : 0;
}
