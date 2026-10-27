// Complete (R+1)-round SIMON Markov table from a complete dense R-round table.
// Same transition as tools/simon_queried_row_8r.cpp, evaluated for every output
// row instead of queried rows only, and written left-major as doubles.
//
// Input:  SMDT0002 (u64 mass / 2^60, R = 7) or SMDT0003 (double, any R).
// Output: SMDT0003, header bits=16, rounds=R+1, initial=0x40, exponent=0,
//         count=2^32, then 2^32 little-endian doubles indexed (left<<16)|right.
//
// For input row a (left difference a), the averaged row over a's image
// subspace gives out[c][a] = avg_a[c ^ f(a)] for every c, so one pass over the
// input fills column a of the output. Columns are buffered in blocks and
// written with pwrite.
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
constexpr uint32_t kWordCount = 1 << 16;
constexpr double kInvDenominator = 1.0 / static_cast<double>(uint64_t{1} << 60);

uint32_t decode_u32(const unsigned char* p) {
    uint32_t v = 0;
    for (unsigned i = 0; i < 4; ++i) v |= uint32_t{p[i]} << (8 * i);
    return v;
}
uint64_t decode_u64(const unsigned char* p) {
    uint64_t v = 0;
    for (unsigned i = 0; i < 8; ++i) v |= uint64_t{p[i]} << (8 * i);
    return v;
}
void encode_u32(unsigned char* p, uint32_t v) {
    for (unsigned i = 0; i < 4; ++i) p[i] = static_cast<unsigned char>(v >> (8 * i));
}
void encode_u64(unsigned char* p, uint64_t v) {
    for (unsigned i = 0; i < 8; ++i) p[i] = static_cast<unsigned char>(v >> (8 * i));
}
uint32_t rol(uint32_t x, unsigned s) { return ((x << s) | (x >> (16 - s))) & 0xFFFF; }
uint32_t simon_f(uint32_t x) { return (rol(x, 1) & rol(x, 8)) ^ rol(x, 2); }

// Identical to image_basis in tools/simon_queried_row_8r.cpp.
std::vector<uint32_t> image_basis(uint32_t a) {
    std::array<uint32_t, 16> pivots{};
    const uint32_t offset = simon_f(a);
    for (unsigned bit = 0; bit < 16; ++bit) {
        const uint32_t e = uint32_t{1} << bit;
        uint32_t column = simon_f(a ^ e) ^ simon_f(e) ^ offset;
        for (int pivot = 15; pivot >= 0; --pivot) {
            if (!(column & (uint32_t{1} << pivot))) continue;
            if (pivots[pivot]) column ^= pivots[pivot];
            else {
                pivots[pivot] = column;
                break;
            }
        }
    }
    std::vector<uint32_t> basis;
    for (unsigned bit = 0; bit < 16; ++bit)
        if (pivots[bit]) basis.push_back(pivots[bit]);
    return basis;
}

void read_exact(int fd, void* buf, size_t n, off_t off) {
    auto* p = static_cast<char*>(buf);
    while (n) {
        const ssize_t got = pread(fd, p, n, off);
        if (got <= 0) throw std::runtime_error("short read");
        p += got; n -= static_cast<size_t>(got); off += got;
    }
}
void write_exact(int fd, const void* buf, size_t n, off_t off) {
    auto* p = static_cast<const char*>(buf);
    while (n) {
        const ssize_t put = pwrite(fd, p, n, off);
        if (put <= 0) throw std::runtime_error("short write");
        p += put; n -= static_cast<size_t>(put); off += put;
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: simon_full_table_next INPUT_DENSE OUTPUT_TABLE [BLOCK_ROWS=1024]\n";
        return 2;
    }
    try {
        const std::string in_path = argv[1], out_path = argv[2];
        const uint32_t block = argc == 4 ? static_cast<uint32_t>(std::stoul(argv[3])) : 1024;
        if (block == 0 || kWordCount % block) throw std::invalid_argument("block must divide 65536");
        const auto t0 = std::chrono::steady_clock::now();

        const int in_fd = open(in_path.c_str(), O_RDONLY);
        if (in_fd < 0) throw std::runtime_error("cannot open input");
        unsigned char header[32];
        read_exact(in_fd, header, 32, 0);
        const bool fixed = std::memcmp(header, "SMDT0002", 8) == 0;
        const bool dbl = std::memcmp(header, "SMDT0003", 8) == 0;
        if (!fixed && !dbl) throw std::runtime_error("input must be SMDT0002 or SMDT0003");
        const uint32_t bits = decode_u32(header + 8), rounds = decode_u32(header + 12);
        const uint32_t initial = decode_u32(header + 16), exponent = decode_u32(header + 20);
        const uint64_t count = decode_u64(header + 24);
        if (bits != 16 || initial != 0x40 || count != (uint64_t{1} << 32) ||
            (fixed && (rounds != 7 || exponent != 60)) || (dbl && exponent != 0))
            throw std::runtime_error("unexpected input metadata");

        const std::string partial = out_path + ".partial." + std::to_string(getpid());
        const int out_fd = open(partial.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
        if (out_fd < 0) throw std::runtime_error("cannot create output (exists?)");
        unsigned char out_header[32] = {};
        std::memcpy(out_header, "SMDT0003", 8);
        encode_u32(out_header + 8, 16);
        encode_u32(out_header + 12, rounds + 1);
        encode_u32(out_header + 16, 0x40);
        encode_u32(out_header + 20, 0);
        encode_u64(out_header + 24, uint64_t{1} << 32);
        write_exact(out_fd, out_header, 32, 0);

        std::vector<uint64_t> raw(kWordCount);
        std::vector<double> row(kWordCount);
        // cols[c * block + (a - a0)] = out[c][a]
        std::vector<double> cols(static_cast<size_t>(block) * kWordCount);
        long double in_mass = 0.0L, out_mass = 0.0L;
        for (uint32_t a0 = 0; a0 < kWordCount; a0 += block) {
            for (uint32_t da = 0; da < block; ++da) {
                const uint32_t a = a0 + da;
                const off_t off = 32 + static_cast<off_t>(a) * kWordCount * 8;
                if (fixed) {
                    read_exact(in_fd, raw.data(), kWordCount * 8, off);
                    for (uint32_t b = 0; b < kWordCount; ++b)
                        row[b] = static_cast<double>(raw[b]) * kInvDenominator;
                } else {
                    read_exact(in_fd, row.data(), kWordCount * 8, off);
                }
                for (uint32_t b = 0; b < kWordCount; ++b) in_mass += row[b];
                for (uint32_t vec : image_basis(a)) {
                    for (uint32_t i = 0; i < kWordCount; ++i) {
                        const uint32_t j = i ^ vec;
                        if (i >= j) continue;
                        const double avg = 0.5 * (row[i] + row[j]);
                        row[i] = row[j] = avg;
                    }
                }
                const uint32_t offset = simon_f(a);
                for (uint32_t c = 0; c < kWordCount; ++c) {
                    const double v = row[c ^ offset];
                    cols[static_cast<size_t>(c) * block + da] = v;
                    out_mass += v;
                }
            }
            for (uint32_t c = 0; c < kWordCount; ++c) {
                const off_t off = 32 + (static_cast<off_t>(c) * kWordCount + a0) * 8;
                write_exact(out_fd, &cols[static_cast<size_t>(c) * block], block * 8, off);
            }
            std::cerr << "rows " << a0 + block << "/65536\n";
        }
        close(in_fd);
        if (fsync(out_fd) != 0 || close(out_fd) != 0) throw std::runtime_error("flush failed");
        if (std::rename(partial.c_str(), out_path.c_str()) != 0) throw std::runtime_error("rename failed");
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::cout << "{\"schema\": \"simon_full_table_next.v1\", \"input_rounds\": " << rounds
                  << ", \"output_rounds\": " << rounds + 1 << ", \"input_mass\": "
                  << static_cast<double>(in_mass) << ", \"output_mass\": " << static_cast<double>(out_mass)
                  << ", \"wall_seconds\": " << secs << "}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
