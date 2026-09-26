// Exact, sparse evaluation of Bao et al.'s SIMON Markov DDT through six rounds.
// Source semantics: simonNDvsDD/simon_ddt.cpp at 09cc81320be9ba65eb4b5c932ac4a0b85d5e4bc8.
// This is an offline distribution generator; it never reads attack instances.
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint64_t kDenominator = uint64_t{1} << 60;
constexpr uint64_t kDefaultRecordCap = 80000000;

struct State {
    uint32_t index;
    uint64_t mass;
};

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
    for (unsigned bit = 0; bit < bits; ++bit) {
        const uint32_t e = uint32_t{1} << bit;
        uint32_t column = simon_f(a ^ e, bits) ^ simon_f(e, bits) ^ simon_f(a, bits);
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

void append_u32(std::ostream& out, uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) out.put(static_cast<char>(value >> (i * 8)));
}
void append_u64(std::ostream& out, uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) out.put(static_cast<char>(value >> (i * 8)));
}

void write_distribution(const std::string& path, unsigned bits, unsigned rounds,
                        uint32_t initial, const std::vector<State>& states) {
    if (std::filesystem::exists(path) || std::filesystem::exists(path + ".partial"))
        throw std::runtime_error("output or partial file already exists: " + path);
    std::ofstream out(path + ".partial", std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open partial output: " + path);
    out.write("SMDT0001", 8);
    append_u32(out, bits);
    append_u32(out, rounds);
    append_u32(out, initial);
    append_u32(out, 60);
    append_u64(out, states.size());
    std::array<char, 12 * 8192> buffer{};
    size_t used = 0;
    for (const State& state : states) {
        for (unsigned i = 0; i < 4; ++i) buffer[used++] = static_cast<char>(state.index >> (i * 8));
        for (unsigned i = 0; i < 8; ++i) buffer[used++] = static_cast<char>(state.mass >> (i * 8));
        if (used == buffer.size()) {
            out.write(buffer.data(), static_cast<std::streamsize>(used));
            used = 0;
        }
    }
    if (used) out.write(buffer.data(), static_cast<std::streamsize>(used));
    out.close();
    if (!out) throw std::runtime_error("incomplete output: " + path);
    std::filesystem::rename(path + ".partial", path);
}

void report(unsigned round, const std::vector<State>& states, unsigned max_rank) {
    uint64_t total = 0;
    uint64_t min_mass = std::numeric_limits<uint64_t>::max();
    uint64_t max_mass = 0;
    uint32_t previous = 0;
    bool first = true;
    for (const State& state : states) {
        if (!state.index || !state.mass || (!first && state.index <= previous))
            throw std::runtime_error("zero state/mass or duplicate/unsorted output index");
        if (state.mass > kDenominator - total)
            throw std::runtime_error("total mass exceeds denominator");
        total += state.mass;
        min_mass = std::min(min_mass, state.mass);
        max_mass = std::max(max_mass, state.mass);
        previous = state.index;
        first = false;
    }
    if (total != kDenominator) throw std::runtime_error("distribution lost mass");
    std::cout << "{\"round\":" << round << ",\"support\":" << states.size()
              << ",\"mass_numerator\":" << total << ",\"denominator_exponent\":60"
              << ",\"min_numerator\":" << min_mass << ",\"max_numerator\":" << max_mass
              << ",\"max_image_rank\":" << max_rank << "}" << std::endl;
}

void run(unsigned bits, unsigned rounds, uint32_t left, uint32_t right,
         uint64_t record_cap, const std::string& output) {
    if (bits < 3 || bits > 16 || rounds > (bits == 16 ? 6 : 7) || record_cap == 0)
        throw std::invalid_argument("bits must be 3..16; rounds <=6 at full width or <=7 at toy width; cap positive");
    const uint32_t word_count = uint32_t{1} << bits;
    if (left >= word_count || right >= word_count)
        throw std::invalid_argument("initial difference words must fit word width");
    if (record_cap > std::numeric_limits<size_t>::max() / sizeof(State))
        throw std::invalid_argument("record cap overflows address space");
    const uint32_t initial = (left << bits) | right;
    if (!initial) throw std::invalid_argument("Bao source excludes zero input difference");
    std::vector<State> states{{initial, kDenominator}};
    report(0, states, 0);
    std::vector<uint64_t> row(word_count);
    for (unsigned round = 1; round <= rounds; ++round) {
        std::vector<State> next;
        if (bits == 16 && round == 6) next.reserve(std::min<uint64_t>(record_cap, kDefaultRecordCap));
        unsigned max_rank = 0;
        for (size_t start = 0; start < states.size();) {
            const uint32_t a = states[start].index >> bits;
            std::fill(row.begin(), row.end(), 0);
            size_t end = start;
            while (end < states.size() && (states[end].index >> bits) == a) {
                row[states[end].index & (word_count - 1)] = states[end].mass;
                ++end;
            }
            const std::vector<uint32_t> basis = image_basis(a, bits);
            max_rank = std::max(max_rank, static_cast<unsigned>(basis.size()));
            for (const uint32_t vector : basis) {
                for (uint32_t i = 0; i < word_count; ++i) {
                    const uint32_t j = i ^ vector;
                    if (i >= j) continue;
                    const uint64_t sum = row[i] + row[j];
                    if (sum & 1) throw std::runtime_error("2^60 denominator insufficient");
                    row[i] = row[j] = sum >> 1;
                }
            }
            const uint32_t offset = simon_f(a, bits);
            for (uint32_t y = 0; y < word_count; ++y) {
                if (!row[y]) continue;
                if (next.size() >= record_cap) throw std::runtime_error("record cap exceeded");
                next.push_back({((y ^ offset) << bits) | a, row[y]});
            }
            start = end;
        }
        std::sort(next.begin(), next.end(), [](const State& x, const State& y) {
            return x.index < y.index;
        });
        report(round, next, max_rank);
        states.swap(next);
    }
    if (!output.empty()) write_distribution(output, bits, rounds, initial, states);
}

uint64_t parse_unsigned(const char* input) {
    const std::string text(input);
    if (text.empty() || text[0] == '-') throw std::invalid_argument("expected unsigned integer");
    size_t used = 0;
    const uint64_t value = std::stoull(text, &used, 0);
    if (used != text.size()) throw std::invalid_argument("trailing characters in integer argument");
    return value;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "usage: simon_predicted_ddt BITS ROUNDS INPUT_LEFT INPUT_RIGHT MAX_RECORDS OUTPUT_OR_DASH\n";
        return 2;
    }
    try {
        const uint64_t bits_arg = parse_unsigned(argv[1]);
        const uint64_t rounds_arg = parse_unsigned(argv[2]);
        const uint64_t left_arg = parse_unsigned(argv[3]);
        const uint64_t right_arg = parse_unsigned(argv[4]);
        if (bits_arg > std::numeric_limits<unsigned>::max() ||
            rounds_arg > std::numeric_limits<unsigned>::max() ||
            left_arg > std::numeric_limits<uint32_t>::max() ||
            right_arg > std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument("argument overflows integer width");
        const unsigned bits = static_cast<unsigned>(bits_arg);
        const unsigned rounds = static_cast<unsigned>(rounds_arg);
        const uint32_t left = static_cast<uint32_t>(left_arg);
        const uint32_t right = static_cast<uint32_t>(right_arg);
        const uint64_t record_cap = parse_unsigned(argv[5]);
        run(bits, rounds, left, right, record_cap, argv[6] == std::string("-") ? "" : argv[6]);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
