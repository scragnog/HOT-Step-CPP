#include "../../engine/src/train/yue2-aitk-sha256.h"

#include <array>
#include <cstring>
#include <iostream>
#include <string>

namespace {
using yue2_aitk::sha256::digest;
using yue2_aitk::sha256::hasher;

bool expect(const char *name, const digest &actual, const char *wanted) {
    if (actual.hex() != wanted) {
        std::cerr << name << ": got " << actual.hex() << ", expected " << wanted << '\n';
        return false;
    }
    return true;
}

bool test_vectors() {
    bool ok = true;
    ok &= expect("empty", yue2_aitk::sha256::bytes("", 0),
                 "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    ok &= expect("abc", yue2_aitk::sha256::bytes("abc", 3),
                 "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    hasher million;
    std::array<char, 1000> a{};
    a.fill('a');
    for (int i = 0; i < 1000; ++i) million.update(a.data(), a.size());
    ok &= expect("million-a", million.final(),
                 "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");

    // Exercise a multi-block message with deliberately fragmented updates.
    std::array<std::uint8_t, 4097> pattern{};
    for (std::size_t i = 0; i < pattern.size(); ++i) pattern[i] = static_cast<std::uint8_t>(i * 37u + 11u);
    const digest one_shot = yue2_aitk::sha256::bytes(pattern.data(), pattern.size());
    hasher fragmented;
    std::size_t offset = 0;
    for (const std::size_t part : {1u, 63u, 64u, 127u, 511u, 1024u, 2307u}) {
        if (offset == pattern.size()) break;
        const std::size_t count = (part < pattern.size() - offset) ? part : pattern.size() - offset;
        fragmented.update(pattern.data() + offset, count);
        offset += count;
    }
    if (offset != pattern.size()) fragmented.update(pattern.data() + offset, pattern.size() - offset);
    if (fragmented.final().hex() != one_shot.hex()) {
        std::cerr << "fragmented multi-block digest differs from one-shot\n";
        ok = false;
    }
    return ok;
}
} // namespace

int main(int argc, char **argv) {
    if (!test_vectors()) return 1;
    digest unused;
    std::string error;
    if (yue2_aitk::sha256::file({}, unused, &error) || error != "empty path") {
        std::cerr << "empty-path I/O validation failed\n";
        return 2;
    }
    if (argc > 1) {
        digest actual;
        error.clear();
        if (!yue2_aitk::sha256::file(argv[1], actual, &error)) {
            std::cerr << error << '\n';
            return 2;
        }
        std::cout << actual.hex() << '\n';
    }
    return 0;
}
