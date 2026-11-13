#pragma once
// Host and Origin guard for the engine HTTP server.
//
// The engine listens on loopback by default, but a web page open in the
// user's browser can still reach it: DNS rebinding (an attacker hostname
// that resolves to 127.0.0.1) or a cross-site request from any origin.
// These pure functions decide whether a request should be refused, so the
// logic can be unit tested without starting a server.
//
// Rules:
//   Host header: missing is allowed. Otherwise the hostname (port stripped,
//   [v6] brackets handled) must be localhost, a 127.x address, ::1, the
//   configured --host value, or any IP literal. A hostname that is not an
//   IP literal is what DNS rebinding needs, so it is refused.
//   Origin header: missing is allowed. "null" is refused. Otherwise it must
//   be http(s)://127.0.0.1|localhost|[::1] on one of the trusted ports
//   (engine port, Node server port, Vite dev port), or appear exactly in
//   the opt-in list (--allow-origin / ACESTEPCPP_ALLOWED_ORIGINS).

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace origin_guard {

inline std::string to_lower(std::string s) {
    for (char & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

inline std::string trim(const std::string & s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char) s[b])) {
        b++;
    }
    while (e > b && std::isspace((unsigned char) s[e - 1])) {
        e--;
    }
    return s.substr(b, e - b);
}

// Split a comma separated list, trim entries, lowercase, drop empties and
// any trailing slash.
inline std::vector<std::string> parse_csv(const std::string & csv) {
    std::vector<std::string> out;
    size_t                   start = 0;
    while (start <= csv.size()) {
        size_t      comma = csv.find(',', start);
        std::string item  = trim(csv.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        while (!item.empty() && item.back() == '/') {
            item.pop_back();
        }
        if (!item.empty()) {
            out.push_back(to_lower(item));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return out;
}

// Returns the hostname part of a Host header value: "[::1]:8085" -> "::1",
// "Example.com:80" -> "example.com". Lowercased.
inline std::string host_name(const std::string & host_header) {
    std::string h = to_lower(trim(host_header));
    if (!h.empty() && h[0] == '[') {
        size_t close = h.find(']');
        return close == std::string::npos ? h.substr(1) : h.substr(1, close - 1);
    }
    size_t colon = h.find(':');
    // More than one colon without brackets: treat as a bare IPv6 literal.
    if (colon != std::string::npos && h.find(':', colon + 1) == std::string::npos) {
        h = h.substr(0, colon);
    }
    while (!h.empty() && h.back() == '.') {
        h.pop_back();
    }
    return h;
}

inline bool is_ipv4_literal(const std::string & s) {
    int    parts = 0;
    size_t i     = 0;
    while (i <= s.size()) {
        size_t j = i;
        while (j < s.size() && std::isdigit((unsigned char) s[j])) {
            j++;
        }
        size_t len = j - i;
        if (len == 0 || len > 3 || std::stoi(s.substr(i, len)) > 255) {
            return false;
        }
        parts++;
        if (j == s.size()) {
            break;
        }
        if (s[j] != '.') {
            return false;
        }
        i = j + 1;
    }
    return parts == 4;
}

inline bool is_ipv6_literal(const std::string & s) {
    if (s.find(':') == std::string::npos) {
        return false;
    }
    for (char c : s) {
        if (!(std::isxdigit((unsigned char) c) || c == ':' || c == '.' || c == '%')) {
            return false;
        }
    }
    return true;
}

// True if a request with this Host header may proceed.
inline bool host_allowed(const std::string & host_header, const std::string & configured_host) {
    if (trim(host_header).empty()) {
        return true;
    }
    std::string h = host_name(host_header);
    if (h == "localhost" || h == "::1") {
        return true;
    }
    if (h.rfind("127.", 0) == 0 && is_ipv4_literal(h)) {
        return true;
    }
    if (!configured_host.empty() && h == host_name(configured_host)) {
        return true;
    }
    return is_ipv4_literal(h) || is_ipv6_literal(h);
}

// True if a request with this Origin header may proceed.
// trusted_ports: ports on loopback whose pages may call the engine.
// allowed: extra origins from the opt-in list (already lowercased).
inline bool origin_allowed(const std::string &              origin_header,
                           const std::vector<int> &         trusted_ports,
                           const std::vector<std::string> & allowed) {
    std::string o = to_lower(trim(origin_header));
    if (o.empty()) {
        return true;
    }
    if (o == "null") {
        return false;
    }
    while (!o.empty() && o.back() == '/') {
        o.pop_back();
    }
    if (std::find(allowed.begin(), allowed.end(), o) != allowed.end()) {
        return true;
    }
    static const char * schemes[] = { "http://", "https://" };
    static const char * hosts[]   = { "127.0.0.1", "localhost", "[::1]" };
    for (const char * sc : schemes) {
        for (const char * ho : hosts) {
            for (int p : trusted_ports) {
                if (p <= 0) {
                    continue;
                }
                if (o == std::string(sc) + ho + ":" + std::to_string(p)) {
                    return true;
                }
            }
        }
    }
    return false;
}

}  // namespace origin_guard
