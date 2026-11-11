// Parser regression: empty and absent style are valid, wrong types are not.
#include "yue2-request.h"
#include <cstdio>

int main() {
    int failed = 0;
    for (const char * body : { "{}", "{\"style\":\"\",\"lyrics\":\"\"}",
                               "{\"songs\":[{\"style\":\"\"},{\"lyrics\":\"\"}]}" }) {
        Yue2Request req;
        std::string err;
        if (!yue2_parse_request(body, &req, &err)) {
            std::fprintf(stderr, "FAIL %s: %s\n", body, err.c_str());
            failed++;
        }
    }
    Yue2Request req;
    std::string err;
    if (yue2_parse_request("{\"style\":5}", &req, &err)) failed++;
    std::printf("yue2 empty style: %s\n", failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
