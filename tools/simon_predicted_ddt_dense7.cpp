// Seven-round dense continuation of the exact sparse SIMON Markov DDT.
// The output is canonical Bao index order (left << word_bits) | right.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint64_t kDenominator = uint64_t{1} << 60;
constexpr uint32_t kTileWidth = 1024;
namespace fs = std::filesystem;

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
    for (unsigned bit = 0; bit < bits; ++bit) if (pivots[bit]) basis.push_back(pivots[bit]);
    return basis;
}

uint32_t decode_u32(const unsigned char* data) {
    uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i) value |= uint32_t{data[i]} << (8 * i);
    return value;
}
uint64_t decode_u64(const unsigned char* data) {
    uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) value |= uint64_t{data[i]} << (8 * i);
    return value;
}
void encode_u32(unsigned char* data, uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) data[i] = static_cast<unsigned char>(value >> (8 * i));
}
void encode_u64(unsigned char* data, uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) data[i] = static_cast<unsigned char>(value >> (8 * i));
}

class SparseReader {
public:
    explicit SparseReader(const std::string& path) : path_(path), file_(std::fopen(path.c_str(), "rb")) {
        if (!file_) throw std::runtime_error("cannot open sparse input");
        buffer_.resize(1 << 20);
        if (std::setvbuf(file_, buffer_.data(), _IOFBF, buffer_.size()))
            throw std::runtime_error("cannot buffer sparse input");
        unsigned char header[32];
        read_exact(header, sizeof(header));
        if (std::memcmp(header, "SMDT0001", 8)) throw std::runtime_error("bad sparse magic");
        bits = decode_u32(header + 8);
        rounds = decode_u32(header + 12);
        initial = decode_u32(header + 16);
        exponent = decode_u32(header + 20);
        count = decode_u64(header + 24);
        if (bits < 3 || bits > 16 || rounds != 6 || exponent != 60 || !initial)
            throw std::runtime_error("incompatible sparse metadata");
        const uint64_t expected_size = 32 + 12 * count;
        if (fs::file_size(path) != expected_size) throw std::runtime_error("sparse file length mismatch");
        advance();
    }
    ~SparseReader() { if (file_) std::fclose(file_); }
    SparseReader(const SparseReader&) = delete;
    SparseReader& operator=(const SparseReader&) = delete;

    void advance() {
        if (read_count_ == count) { has_record = false; return; }
        unsigned char record[12];
        read_exact(record, sizeof(record));
        index = decode_u32(record);
        mass = decode_u64(record + 4);
        if (!mass || (read_count_ && index <= previous_index_))
            throw std::runtime_error("zero mass or unordered/duplicate sparse record");
        if (index >= (uint64_t{1} << (2 * bits)))
            throw std::runtime_error("sparse index outside domain");
        previous_index_ = index;
        ++read_count_;
        has_record = true;
    }
    void finish() const {
        if (has_record || read_count_ != count || input_mass_ != kDenominator)
            throw std::runtime_error("sparse input mass/count mismatch");
    }
    void add_input_mass(uint64_t value) {
        if (value > kDenominator - input_mass_) throw std::runtime_error("sparse input mass exceeds one");
        input_mass_ += value;
    }

    unsigned bits = 0, rounds = 0;
    uint32_t initial = 0, exponent = 0, index = 0;
    uint64_t count = 0, mass = 0;
    bool has_record = false;
private:
    void read_exact(void* target, size_t bytes) {
        if (std::fread(target, 1, bytes, file_) != bytes) throw std::runtime_error("truncated sparse input");
    }
    std::string path_;
    FILE* file_;
    std::vector<char> buffer_;
    uint64_t read_count_ = 0, input_mass_ = 0;
    uint32_t previous_index_ = 0;
};

class TileWriter {
public:
    TileWriter(const std::string& directory, uint32_t word_count)
        : directory_(directory), tile_width_(std::min(kTileWidth, word_count)),
          tile_count_(word_count / tile_width_) {
        if (fs::exists(directory_)) throw std::runtime_error("tile directory already exists");
        fs::create_directory(directory_);
        files_.resize(tile_count_);
        buffers_.resize(tile_count_, std::vector<char>(1 << 20));
        for (uint32_t tile = 0; tile < tile_count_; ++tile) {
            const std::string path = tile_path(tile);
            files_[tile] = std::fopen(path.c_str(), "wb");
            if (!files_[tile] || std::setvbuf(files_[tile], buffers_[tile].data(), _IOFBF, buffers_[tile].size()))
                throw std::runtime_error("cannot open/buffer tile");
        }
    }
    ~TileWriter() { close_noexcept(); }
    TileWriter(const TileWriter&) = delete;
    TileWriter& operator=(const TileWriter&) = delete;

    void append_row(const std::vector<uint64_t>& row, uint32_t offset) {
        std::array<unsigned char, kTileWidth * 8> encoded{};
        for (uint32_t tile = 0; tile < tile_count_; ++tile) {
            for (uint32_t local = 0; local < tile_width_; ++local) {
                const uint32_t c = tile * tile_width_ + local;
                encode_u64(encoded.data() + 8 * local, row[c ^ offset]);
            }
            if (std::fwrite(encoded.data(), 8, tile_width_, files_[tile]) != tile_width_)
                throw std::runtime_error("tile write failed");
        }
    }
    void close_checked(uint32_t word_count) {
        for (uint32_t tile = 0; tile < tile_count_; ++tile) {
            FILE* file = files_[tile];
            files_[tile] = nullptr;
            if (std::fclose(file)) throw std::runtime_error("tile close failed");
            if (fs::file_size(tile_path(tile)) != uint64_t{word_count} * tile_width_ * 8)
                throw std::runtime_error("tile length mismatch");
        }
    }
    std::string tile_path(uint32_t tile) const {
        return directory_ + "/tile_" + std::to_string(tile) + ".bin";
    }
    uint32_t tile_width() const { return tile_width_; }
    uint32_t tile_count() const { return tile_count_; }
private:
    void close_noexcept() {
        for (FILE*& file : files_) if (file) { std::fclose(file); file = nullptr; }
    }
    std::string directory_;
    uint32_t tile_width_, tile_count_;
    std::vector<FILE*> files_;
    std::vector<std::vector<char>> buffers_;
};

void assemble(const TileWriter& tiles, const std::string& output,
              unsigned bits, uint32_t initial) {
    if (fs::exists(output) || fs::exists(output + ".partial"))
        throw std::runtime_error("dense output or partial file already exists");
    FILE* out = std::fopen((output + ".partial").c_str(), "wb");
    if (!out) throw std::runtime_error("cannot create dense partial output");
    std::vector<char> out_buffer(1 << 20);
    if (std::setvbuf(out, out_buffer.data(), _IOFBF, out_buffer.size()))
        throw std::runtime_error("cannot buffer dense output");
    const uint32_t word_count = uint32_t{1} << bits;
    unsigned char header[32]{};
    std::memcpy(header, "SMDT0002", 8);
    encode_u32(header + 8, bits);
    encode_u32(header + 12, 7);
    encode_u32(header + 16, initial);
    encode_u32(header + 20, 60);
    encode_u64(header + 24, uint64_t{word_count} * word_count);
    if (std::fwrite(header, 1, sizeof(header), out) != sizeof(header))
        throw std::runtime_error("dense header write failed");
    const uint32_t tile_width = tiles.tile_width();
    std::vector<unsigned char> tile_data(uint64_t{word_count} * tile_width * 8);
    std::vector<unsigned char> row_data(uint64_t{word_count} * 8);
    for (uint32_t tile = 0; tile < tiles.tile_count(); ++tile) {
        FILE* in = std::fopen(tiles.tile_path(tile).c_str(), "rb");
        if (!in) throw std::runtime_error("cannot read completed tile");
        if (std::fread(tile_data.data(), 1, tile_data.size(), in) != tile_data.size())
            throw std::runtime_error("incomplete tile read");
        if (std::fclose(in)) throw std::runtime_error("tile read close failed");
        for (uint32_t c_local = 0; c_local < tile_width; ++c_local) {
            for (uint32_t a = 0; a < word_count; ++a) {
                std::memcpy(row_data.data() + uint64_t{a} * 8,
                            tile_data.data() + (uint64_t{a} * tile_width + c_local) * 8, 8);
            }
            if (std::fwrite(row_data.data(), 8, word_count, out) != word_count)
                throw std::runtime_error("dense row write failed");
        }
        std::cout << "{\"assembled_tile\":" << tile + 1 << ",\"tile_count\":"
                  << tiles.tile_count() << "}" << std::endl;
    }
    if (std::fclose(out)) throw std::runtime_error("dense output close failed");
    if (fs::file_size(output + ".partial") != 32 + uint64_t{word_count} * word_count * 8)
        throw std::runtime_error("dense output length mismatch");
    fs::rename(output + ".partial", output);
}

void run(const std::string& input, const std::string& mode,
         const std::string& tile_directory, const std::string& output) {
    if (mode != "probe" && mode != "generate") throw std::invalid_argument("mode must be probe or generate");
    SparseReader source(input);
    const uint32_t word_count = uint32_t{1} << source.bits;
    if (mode == "generate") {
        if (fs::exists(output) || fs::exists(output + ".partial") || fs::exists(tile_directory))
            throw std::runtime_error("output, partial output, or tile directory already exists");
        const uint64_t dense_bytes = uint64_t{word_count} * word_count * 8;
        const uint64_t safety_bytes = source.bits == 16 ? uint64_t{20} << 30 : 0;
        if (fs::space(fs::path(output).parent_path()).available < 2 * dense_bytes + safety_bytes)
            throw std::runtime_error("insufficient free disk for tiles, final table, and 20 GiB reserve");
        if (!fs::equivalent(fs::path(output).parent_path(), fs::path(tile_directory).parent_path()))
            throw std::runtime_error("tiles and final output must share a filesystem");
    }
    std::vector<uint64_t> row(word_count);
    std::unique_ptr<TileWriter> tiles;
    if (mode == "generate") tiles = std::make_unique<TileWriter>(tile_directory, word_count);
    uint64_t output_mass = 0, support = 0, min_mass = std::numeric_limits<uint64_t>::max(), max_mass = 0;
    unsigned max_rank = 0;
    for (uint32_t a = 0; a < word_count; ++a) {
        std::fill(row.begin(), row.end(), 0);
        while (source.has_record && (source.index >> source.bits) == a) {
            row[source.index & (word_count - 1)] = source.mass;
            source.add_input_mass(source.mass);
            source.advance();
        }
        const std::vector<uint32_t> basis = image_basis(a, source.bits);
        max_rank = std::max(max_rank, static_cast<unsigned>(basis.size()));
        for (uint32_t vector : basis) {
            for (uint32_t i = 0; i < word_count; ++i) {
                const uint32_t j = i ^ vector;
                if (i >= j) continue;
                const uint64_t sum = row[i] + row[j];
                if (sum & 1) throw std::runtime_error("2^60 denominator insufficient for seven rounds");
                row[i] = row[j] = sum >> 1;
            }
        }
        for (uint64_t mass : row) {
            if (!mass) continue;
            if (mass > kDenominator - output_mass) throw std::runtime_error("seven-round mass exceeds one");
            output_mass += mass;
            ++support;
            min_mass = std::min(min_mass, mass);
            max_mass = std::max(max_mass, mass);
        }
        if (tiles) tiles->append_row(row, simon_f(a, source.bits));
    }
    source.finish();
    if (output_mass != kDenominator) throw std::runtime_error("seven-round mass lost");
    std::cout << "{\"round\":7,\"support\":" << support
              << ",\"mass_numerator\":" << output_mass << ",\"denominator_exponent\":60"
              << ",\"min_numerator\":" << min_mass << ",\"max_numerator\":" << max_mass
              << ",\"max_image_rank\":" << max_rank << "}" << std::endl;
    if (tiles) {
        tiles->close_checked(word_count);
        assemble(*tiles, output, source.bits, source.initial);
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: simon_predicted_ddt_dense7 SPARSE6 probe|generate TILE_DIRECTORY OUTPUT\n";
        return 2;
    }
    try {
        run(argv[1], argv[2], argv[3], argv[4]);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
