#include "../../engine/src/train/yue2-aitk-import.h"

#include <cassert>
#include <string>

int main() {
    using yue2_aitk_import::detail::parse_caption;
    std::string style, lyrics, error;

    assert(parse_caption("  Punk rock  \n[Lyrics]\n[Verse 1]\nline\n", &style, &lyrics, &error));
    assert(style == "Punk rock");
    assert(lyrics == "[Verse 1]\nline");

    assert(parse_caption("Style\n[Duration]\n120\n[Lyrics]\n[BRIDGE]\nline", &style, &lyrics, &error));
    assert(style == "Style");
    assert(lyrics == "[Bridge]\nline");

    assert(parse_caption("style\n[Verse soft]\nline", &style, &lyrics, &error));
    assert(style == "style");
    assert(lyrics == "[Verse soft]\nline");

    assert(parse_caption("Instrumental", &style, &lyrics, &error));
    assert(style == "Instrumental");
    assert(lyrics.empty());
    return 0;
}
