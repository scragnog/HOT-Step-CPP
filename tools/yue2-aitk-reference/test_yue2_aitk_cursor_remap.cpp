// Isolated CPU fixture for Legacy cursor_words coordinate remapping.
#include "../../engine/src/train/yue2-aitk-native-import.h"
#include <iostream>
#include <vector>

using yue2_aitk_native_import::detail::remap_words5;

static bool check(bool value, const char * label) {
    if (!value) std::cerr << "FAIL: " << label << "\n";
    return value;
}
static std::vector<float> word(float c0, float c1) { return {0.0f, 1.0f, 1.0f, c0, c1}; }

int main() {
    bool ok = true; std::string error;
    auto unchanged = word(0, 5);
    ok &= check(remap_words5("hello world", "hello world", &unchanged, &error) &&
                unchanged[3] == 0.0f && unchanged[4] == 5.0f, "unchanged offsets");
    auto tag_case = word(8, 12); error.clear();
    ok &= check(remap_words5("[VERSE] Caf\xC3\xA9", "[Verse] Caf\xC3\xA9", &tag_case, &error) &&
                tag_case[3] == 8.0f && tag_case[4] == 12.0f, "ASCII tag case");
    auto trim_crlf = word(8, 13); error.clear();
    ok &= check(remap_words5(" hello\r\nworld ", "hello\nworld", &trim_crlf, &error) &&
                trim_crlf[3] == 6.0f && trim_crlf[4] == 11.0f, "trim and CRLF removal");
    auto unicode = word(0, 4); error.clear();
    ok &= check(remap_words5("caf\xC3\xA9", "caf\xC3\xA9", &unicode, &error) &&
                unicode[3] == 0.0f && unicode[4] == 4.0f, "UTF-8 codepoint identity");
    auto changed = word(0, 5); error.clear();
    ok &= check(!remap_words5("hello", "hullo", &changed, &error), "changed lyric rejection");
    auto removed_word = word(5, 6); error.clear();
    ok &= check(!remap_words5("hello world", "helloworld", &removed_word, &error), "empty removed word rejection");
    return ok ? 0 : 1;
}

