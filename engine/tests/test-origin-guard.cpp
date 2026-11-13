// test-origin-guard.cpp - unit test for engine/tools/hot-step-origin-guard.h.
// Pure functions, no server, no models.
// Build (MSVC): cl /nologo /EHsc /std:c++17 /I engine/tools engine/tests/test-origin-guard.cpp
// Build (gcc/clang): c++ -std=c++17 -I engine/tools engine/tests/test-origin-guard.cpp

#include "hot-step-origin-guard.h"

#include <cstdio>

using namespace origin_guard;

static int g_pass = 0, g_fail = 0;

static void check(bool got, bool want, const char * what) {
    if (got == want) {
        g_pass++;
    } else {
        g_fail++;
        printf("FAIL: %s (got %d, want %d)\n", what, (int) got, (int) want);
    }
}

int main() {
    const std::string cfg = "127.0.0.1";
    // Host header
    check(host_allowed("evil.com", cfg), false, "host evil.com");
    check(host_allowed("evil.com.", cfg), false, "host evil.com.");
    check(host_allowed("evil.com:8095", cfg), false, "host evil.com:8095");
    check(host_allowed("EVIL.COM", cfg), false, "host EVIL.COM");
    check(host_allowed("localhost.evil.com", cfg), false, "host localhost.evil.com");
    check(host_allowed("127.0.0.1.evil.com", cfg), false, "host 127.0.0.1.evil.com");
    check(host_allowed("localhost:8095", cfg), true, "host localhost:8095");
    check(host_allowed("LocalHost:8095", cfg), true, "host LocalHost:8095");
    check(host_allowed("localhost.", cfg), true, "host localhost.");
    check(host_allowed("127.0.0.1", cfg), true, "host 127.0.0.1");
    check(host_allowed("127.0.0.1:8085", cfg), true, "host 127.0.0.1:8085");
    check(host_allowed("[::1]:8095", cfg), true, "host [::1]:8095");
    check(host_allowed("[::1]", cfg), true, "host [::1]");
    check(host_allowed("192.168.1.5", cfg), true, "host 192.168.1.5");
    check(host_allowed("192.168.1.5:8085", cfg), true, "host 192.168.1.5:8085");
    check(host_allowed("999.1.1.1", cfg), false, "host 999.1.1.1");
    check(host_allowed("", cfg), true, "host empty");
    check(host_allowed("myhost.test:8085", "myhost.test"), true, "host configured myhost.test");
    check(host_allowed("MYHOST.TEST", "myhost.test"), true, "host configured case");
    check(host_allowed("other.test", "myhost.test"), false, "host other.test vs configured");

    // Origin header
    const std::vector<int>         ports = { 8085, 3001, 3000 };
    const std::vector<std::string> none;
    check(origin_allowed("http://evil.com", ports, none), false, "origin evil.com");
    check(origin_allowed("null", ports, none), false, "origin null");
    check(origin_allowed("NULL", ports, none), false, "origin NULL");
    check(origin_allowed("http://127.0.0.1:3001", ports, none), true, "origin 127.0.0.1:3001");
    check(origin_allowed("http://localhost:3000", ports, none), true, "origin localhost:3000");
    check(origin_allowed("HTTP://LOCALHOST:3000", ports, none), true, "origin upper localhost:3000");
    check(origin_allowed("http://localhost:3000/", ports, none), true, "origin trailing slash");
    check(origin_allowed("http://[::1]:8085", ports, none), true, "origin [::1]:8085");
    check(origin_allowed("https://127.0.0.1:3001", ports, none), true, "origin https 3001");
    check(origin_allowed("http://localhost:8080", ports, none), false, "origin localhost:8080");
    check(origin_allowed("http://localhost", ports, none), false, "origin localhost no port");
    check(origin_allowed("http://127.0.0.1:30011", ports, none), false, "origin port prefix 30011");
    check(origin_allowed("http://localhost:3000.evil.com", ports, none), false, "origin suffix trick");
    check(origin_allowed("", ports, none), true, "origin empty");

    // Opt-in CSV
    std::vector<std::string> allowed = parse_csv(" http://MyHost.test:3001/ , ,https://a.example ");
    check(allowed.size() == 2, true, "parse_csv count 2");
    check(!allowed.empty() && allowed[0] == "http://myhost.test:3001", true, "parse_csv lower+trim+slash");
    check(origin_allowed("http://myhost.test:3001", ports, allowed), true, "origin opt-in csv");
    check(origin_allowed("HTTP://MYHOST.TEST:3001", ports, allowed), true, "origin opt-in csv case");
    check(origin_allowed("http://myhost.test:3002", ports, allowed), false, "origin opt-in wrong port");
    check(origin_allowed("http://evil.com", ports, allowed), false, "origin evil with csv");

    // host_name
    check(host_name("[::1]:8085") == "::1", true, "host_name [::1]:8085");
    check(host_name("Example.com:80") == "example.com", true, "host_name Example.com:80");

    printf("origin-guard: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
