#pragma once

// Small, dependency-free SHA-256 used for YuE2 checkpoint and dataset
// provenance.  The implementation is intentionally streaming: callers do not
// need to materialize a multi-gigabyte model or cache file.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

namespace yue2_aitk::sha256 {

struct digest {
    std::array<std::uint8_t, 32> bytes{};

    std::string hex() const {
        static constexpr char digits[] = "0123456789abcdef";
        std::string result;
        result.resize(bytes.size() * 2);
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            result[2 * i] = digits[bytes[i] >> 4];
            result[2 * i + 1] = digits[bytes[i] & 15];
        }
        return result;
    }
};

class hasher {
public:
    hasher() = default;

    void update(const void *data, std::size_t size) noexcept {
        if (size == 0) {
            return;
        }
        if (data == nullptr) {
            invalid_ = true;
            return;
        }

        if (size > (std::numeric_limits<std::uint64_t>::max() - bit_count_) / 8u) {
            invalid_ = true;
            return;
        }

        const auto *input = static_cast<const std::uint8_t *>(data);
        bit_count_ += static_cast<std::uint64_t>(size) * 8u;
        while (size != 0) {
            const std::size_t room = block_.size() - block_used_;
            const std::size_t take = size < room ? size : room;
            for (std::size_t i = 0; i < take; ++i) {
                block_[block_used_ + i] = input[i];
            }
            block_used_ += take;
            input += take;
            size -= take;
            if (block_used_ == block_.size()) {
                transform(state_, block_.data());
                block_used_ = 0;
            }
        }
    }

    digest final() const noexcept {
        hasher copy = *this;
        digest result;
        if (copy.invalid_) {
            return result;
        }

        copy.block_[copy.block_used_++] = 0x80;
        if (copy.block_used_ > 56) {
            while (copy.block_used_ < copy.block_.size()) {
                copy.block_[copy.block_used_++] = 0;
            }
            transform(copy.state_, copy.block_.data());
            copy.block_used_ = 0;
        }
        while (copy.block_used_ < 56) {
            copy.block_[copy.block_used_++] = 0;
        }
        for (int i = 0; i < 8; ++i) {
            copy.block_[56 + i] = static_cast<std::uint8_t>(copy.bit_count_ >> (56 - 8 * i));
        }
        transform(copy.state_, copy.block_.data());

        for (std::size_t i = 0; i < copy.state_.size(); ++i) {
            result.bytes[4 * i] = static_cast<std::uint8_t>(copy.state_[i] >> 24);
            result.bytes[4 * i + 1] = static_cast<std::uint8_t>(copy.state_[i] >> 16);
            result.bytes[4 * i + 2] = static_cast<std::uint8_t>(copy.state_[i] >> 8);
            result.bytes[4 * i + 3] = static_cast<std::uint8_t>(copy.state_[i]);
        }
        return result;
    }

private:
    static constexpr std::array<std::uint32_t, 8> initial_state_ = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };

    static constexpr std::array<std::uint32_t, 64> round_constants_ = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };

    static constexpr std::uint32_t rotate_right(std::uint32_t x, unsigned n) noexcept {
        return (x >> n) | (x << (32 - n));
    }

    static void transform(std::array<std::uint32_t, 8> &state,
                          const std::uint8_t *block) noexcept {
        std::uint32_t schedule[64]{};
        for (int i = 0; i < 16; ++i) {
            schedule[i] = (static_cast<std::uint32_t>(block[4 * i]) << 24) |
                          (static_cast<std::uint32_t>(block[4 * i + 1]) << 16) |
                          (static_cast<std::uint32_t>(block[4 * i + 2]) << 8) |
                          static_cast<std::uint32_t>(block[4 * i + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotate_right(schedule[i - 15], 7) ^
                rotate_right(schedule[i - 15], 18) ^ (schedule[i - 15] >> 3);
            const std::uint32_t s1 = rotate_right(schedule[i - 2], 17) ^
                rotate_right(schedule[i - 2], 19) ^ (schedule[i - 2] >> 10);
            schedule[i] = schedule[i - 16] + s0 + schedule[i - 7] + s1;
        }

        std::uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
        std::uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const std::uint32_t choose = (e & f) ^ (~e & g);
            const std::uint32_t temp1 = h + s1 + choose + round_constants_[i] + schedule[i];
            const std::uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = s0 + majority;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }

    std::array<std::uint32_t, 8> state_ = initial_state_;
    std::array<std::uint8_t, 64> block_{};
    std::size_t block_used_ = 0;
    std::uint64_t bit_count_ = 0;
    bool invalid_ = false;
};

inline digest bytes(const void *data, std::size_t size) noexcept {
    hasher state;
    state.update(data, size);
    return state.final();
}

inline bool file(const std::filesystem::path &path, digest &result,
                 std::string *error = nullptr) {
    if (path.empty()) {
        if (error) *error = "empty path";
        return false;
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        if (error) *error = "cannot open file: " + path.string();
        return false;
    }
    hasher state;
    std::array<std::uint8_t, 64 * 1024> buffer{};
    while (input) {
        input.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) state.update(buffer.data(), static_cast<std::size_t>(count));
    }
    if (!input.eof()) {
        if (error) *error = "read failed: " + path.string();
        return false;
    }
    result = state.final();
    return true;
}

} // namespace yue2_aitk::sha256
