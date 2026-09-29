// Source-aligned double-precision queried rows for the SIMON 8-round Markov prediction.
// Evaluates the Bao transition from the complete 7R table for attack-dependent rows.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <unistd.h>

namespace {
constexpr uint64_t kDenominator = uint64_t{1} << 60;
constexpr double kInvDenominator = 1.0 / static_cast<double>(kDenominator);
constexpr uint32_t kWordCount = 1 << 16;
namespace fs = std::filesystem;

uint32_t decode_u32(const unsigned char* data) {
    uint32_t val = 0;
    for (unsigned i = 0; i < 4; ++i) val |= uint32_t{data[i]} << (8 * i);
    return val;
}

uint64_t decode_u64(const unsigned char* data) {
    uint64_t val = 0;
    for (unsigned i = 0; i < 8; ++i) val |= uint64_t{data[i]} << (8 * i);
    return val;
}

uint32_t rol(uint32_t x, unsigned shift, unsigned bits) {
    const uint32_t mask = (uint32_t{1} << bits) - 1;
    shift %= bits;
    return shift ? ((x << shift) | (x >> (bits - shift))) & mask : x;
}

uint32_t simon_f(uint32_t x, unsigned bits) {
    return (rol(x, 1, bits) & rol(x, 8, bits)) ^ rol(x, 2, bits);
}

std::vector<uint32_t> image_basis(uint32_t a, unsigned bits) {
    std::array<uint32_t, 16> pivots{};
    const uint32_t offset = simon_f(a, bits);
    for (unsigned bit = 0; bit < bits; ++bit) {
        const uint32_t e = uint32_t{1} << bit;
        uint32_t column = simon_f(a ^ e, bits) ^ simon_f(e, bits) ^ offset;
        for (int pivot = static_cast<int>(bits) - 1; pivot >= 0; --pivot) {
            if (!(column & (uint32_t{1} << pivot))) continue;
            if (pivots[pivot]) column ^= pivots[pivot];
            else {
                pivots[pivot] = column;
                break;
            }
        }
    }
    std::vector<uint32_t> basis;
    for (unsigned bit = 0; bit < bits; ++bit) {
        if (pivots[bit]) basis.push_back(pivots[bit]);
    }
    return basis;
}


} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: simon_queried_row_8r INPUT_7R_DENSE OUTPUT_ROWS_BIN ROW_C0 [ROW_C1 ...]\n";
        return 2;
    }
    try {
        const std::string input_7r_path = argv[1];
        const std::string output_bin_path = argv[2];

        std::vector<uint32_t> query_rows;
        for (int i = 3; i < argc; ++i) {
            query_rows.push_back(std::stoul(argv[i]));
        }
        if (query_rows.empty()) {
            throw std::invalid_argument("no query rows provided");
        }
        for (uint32_t r : query_rows) {
            if (r >= kWordCount) throw std::invalid_argument("query row out of 16-bit range");
        }

        const size_t num_queries = query_rows.size();

        const auto t_start = std::chrono::steady_clock::now();

        FILE* in_file = std::fopen(input_7r_path.c_str(), "rb");
        if (!in_file) throw std::runtime_error("cannot open 7R dense input");

        std::vector<char> in_buf(4 * 1024 * 1024);
        std::setvbuf(in_file, in_buf.data(), _IOFBF, in_buf.size());

        unsigned char header[32];
        if (std::fread(header, 1, 32, in_file) != 32) throw std::runtime_error("truncated header");
        if (std::memcmp(header, "SMDT0002", 8) != 0) {
            if (std::memcmp(header, "SMDT0001", 8) == 0) {
                throw std::runtime_error("invalid table magic: received 6R sparse table SMDT0001, expected 7R dense SMDT0002");
            }
            throw std::runtime_error("invalid table magic: expected SMDT0002");
        }
        const uint32_t bits = decode_u32(header + 8);
        const uint32_t rounds = decode_u32(header + 12);
        const uint32_t initial = decode_u32(header + 16);
        const uint32_t exponent = decode_u32(header + 20);
        const uint64_t count = decode_u64(header + 24);

        if (bits != 16 || rounds != 7 || initial != 0x40 || exponent != 60 || count != (uint64_t{1} << 32)) {
            throw std::runtime_error(
                "unexpected 7R metadata: expected bits=16, rounds=7, initial=0x40, exponent=60, count=2^32");
        }

        // Output matrix in double: shape (num_queries, 65536)
        std::vector<double> queried_matrix(num_queries * kWordCount, 0.0);

        std::vector<uint64_t> raw_row(kWordCount);
        std::vector<double> double_row(kWordCount);

        double read_seconds = 0.0;
        double arithmetic_seconds = 0.0;

        for (uint32_t a = 0; a < kWordCount; ++a) {
            const auto t_r0 = std::chrono::steady_clock::now();
            if (std::fread(raw_row.data(), sizeof(uint64_t), kWordCount, in_file) != kWordCount) {
                throw std::runtime_error("truncated 7R read at a = " + std::to_string(a));
            }
            const auto t_r1 = std::chrono::steady_clock::now();
            read_seconds += std::chrono::duration<double>(t_r1 - t_r0).count();

            const auto t_a0 = std::chrono::steady_clock::now();
            // Convert to double (normalized exact probability: mass / 2^60)
            for (uint32_t b = 0; b < kWordCount; ++b) {
                double_row[b] = static_cast<double>(raw_row[b]) * kInvDenominator;
            }

            const std::vector<uint32_t> basis = image_basis(a, 16);
            for (uint32_t vec : basis) {
                for (uint32_t i = 0; i < kWordCount; ++i) {
                    const uint32_t j = i ^ vec;
                    if (i >= j) continue;
                    const double avg = 0.5 * (double_row[i] + double_row[j]);
                    double_row[i] = double_row[j] = avg;
                }
            }

            const uint32_t offset = simon_f(a, 16);
            for (size_t q = 0; q < num_queries; ++q) {
                const uint32_t c = query_rows[q];
                queried_matrix[q * kWordCount + a] = double_row[c ^ offset];
            }
            const auto t_a1 = std::chrono::steady_clock::now();
            arithmetic_seconds += std::chrono::duration<double>(t_a1 - t_a0).count();
        }
        std::fclose(in_file);

        // Write output binary file atomically if requested
        if (!output_bin_path.empty() && output_bin_path != "/dev/null") {
            const std::string partial_path = output_bin_path + ".partial." + std::to_string(getpid());
            if (fs::exists(partial_path)) {
                fs::remove(partial_path);
            }
            FILE* out_file = std::fopen(partial_path.c_str(), "wb");
            if (!out_file) throw std::runtime_error("cannot open temporary output file: " + partial_path);

            std::vector<char> out_buf(4 * 1024 * 1024);
            std::setvbuf(out_file, out_buf.data(), _IOFBF, out_buf.size());

            const size_t expected_elements = queried_matrix.size();
            const size_t written = std::fwrite(queried_matrix.data(), sizeof(double), expected_elements, out_file);
            if (written != expected_elements) {
                std::fclose(out_file);
                fs::remove(partial_path);
                throw std::runtime_error("incomplete write to output file: wrote " + std::to_string(written) +
                                         " of " + std::to_string(expected_elements) + " doubles");
            }
            if (std::fclose(out_file) != 0) {
                fs::remove(partial_path);
                throw std::runtime_error("failed to flush/close output file");
            }
            fs::rename(partial_path, output_bin_path);
        }

        const auto t_end = std::chrono::steady_clock::now();
        const double wall_seconds = std::chrono::duration<double>(t_end - t_start).count();

        struct rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
        const int64_t peak_rss = usage.ru_maxrss;
#else
        const int64_t peak_rss = usage.ru_maxrss * 1024;
#endif

        std::cout << "{\n"
                  << "  \"schema\": \"simon_queried_row_8r.v1\",\n"
                  << "  \"num_query_rows\": " << num_queries << ",\n"
                  << "  \"input_7r_bytes_read\": 34359738400,\n"
                  << "  \"wall_seconds\": " << wall_seconds << ",\n"
                  << "  \"read_seconds\": " << read_seconds << ",\n"
                  << "  \"arithmetic_seconds\": " << arithmetic_seconds << ",\n"
                  << "  \"peak_rss_bytes\": " << peak_rss << ",\n"
                  << "  \"peak_rss_gib\": " << (double)peak_rss / (1024.0 * 1024.0 * 1024.0) << "\n"
                  << "}\n";
        return 0;
    } catch (const std::exception& err) {
        std::cerr << "error: " << err.what() << '\n';
        return 1;
    }
}
