#pragma once

// configure retrieval-test to use xxhash instead of murmur
#define USE_XXHASH 1

// prevent clang-format from reordering...
#include "bloom/bloom.h"
#include "bloom/counting_bloom.h"
#include "bloom/simd-block-fixed-fpp.h"
#include "bumpribbon/DySECT/utils/hash/xx_hash.hpp" // for quotient filter
#include "bumpribbon/ribbon.hpp"
#include "cuckoo/cuckoofilter.h"
#include "cuckoo/cuckoofilter_stable.h"
#include "gcs/gcs.h"
#include "lpqfilter/templated_qfilter_seq.hpp"
#include "morton/compressed_cuckoo_filter.h"
#include "morton/morton_sample_configs.h"
#include "random.h"
#include "retrieval-test/BPZStrategy.h"
#include "retrieval-test/ChunkInfo.h"
#include "retrieval-test/CoupledStrategy.h"
#include "retrieval-test/GOVStrategy.h"
#include "retrieval-test/LMSSStrategy.h"
#include "retrieval-test/Retriever.h"
#include "retrieval-test/RetrieverChunked.h"
#include "retrieval-test/TwoBlockStrategy.h"
#undef DEBUG // the joys of including multiple projects...

#include "ribbon/bloom_impl.h"
#include "ribbon/ribbon_impl.h"
#include "timing.h"
#include "xorfilter/xor_fuse_filter.h"
#include "xorfilter/xorfilter.h"
#include "xorfilter/xorfilter_10_666bit.h"
#include "xorfilter/xorfilter_10bit.h"
#include "xorfilter/xorfilter_13bit.h"
#include "xorfilter/xorfilter_2.h"
#include "xorfilter/xorfilter_2n.h"
#include "xorfilter/xorfilter_plus.h"
#include "xorfilter/xorfilter_plus2.h"
#include "xorfilter/xorfilter_singleheader.h"

#ifdef __AVX2__
#include "bloom/simd-block.h"
#include "gqf/gqf_cpp.h"
#endif

#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <vector>

// The number of items sampled when determining the lookup performance
const size_t MAX_SAMPLE_SIZE = 100 * 1000 * 1000;
const size_t MIN_SAMPLE_SIZE = 10 * 1000 * 1000;

// The statistics gathered for each table type:
struct Statistics {
    size_t add_count;
    double nanos_per_add;
    double nanos_per_remove;
    // key: percent of queries that were expected to be positive
    map<int, double> nanos_per_finds;
    double false_positive_probabilty;
    double bits_per_item;

    void printKV(std::ostream &os) const {
        // we get some nonsensical result for very small fpps but ah well
        const double minbits = log2(1 / false_positive_probabilty);
        os << " n=" << add_count << " fpp=" << false_positive_probabilty
           << " bits=" << bits_per_item << " minbits=" << minbits
           << " wasted=" << 100 * (bits_per_item / minbits - 1)
           << " tadd=" << nanos_per_add;

        double add_and_find = nanos_per_add;
        for (const auto &fps : nanos_per_finds) {
            os << " tfind" << static_cast<int>(fps.first) << "=" << fps.second;
            add_and_find += fps.second;
        }
        os << " tcombined=" << add_and_find;
    }
};

// Inlining the "contains" which are executed within a tight loop can be both
// very detrimental or very beneficial, and which ways it goes depends on the
// compiler. It is unclear whether we want to benchmark the inlining of
// Contains, as it depends very much on how "contains" is used. So it is maybe
// reasonable to benchmark it without inlining.
//
#define CONTAIN_ATTRIBUTES __attribute__((noinline))

// Output for the first row of the table of results. type_width is the maximum
// number of characters of the description of any table type, and
// find_percent_count is the number of different lookup statistics gathered for
// each table. This function assumes the lookup expected positive probabiilties
// are evenly distributed, with the first being 0% and the last 100%.
string StatisticsTableHeader(int type_width,
                             const std::vector<double> &found_probabilities) {
    ostringstream os;

    os << string(type_width, ' ');
    os << setw(10) << right << "";
    for (size_t i = 0; i < found_probabilities.size(); ++i) {
        os << setw(8) << "find";
    }
    os << setw(10) << "1Xadd+";
    os << setw(8) << "" << setw(3) << "" << setw(8) << "" << setw(12)
       << "optimal" << setw(9) << "wasted" << setw(8) << "million" << endl;

    os << string(type_width, ' ');
    os << setw(10) << right << "add";
    // os << setw(8) << right << "remove";
    for (double prob : found_probabilities) {
        os << setw(8 - 1) << static_cast<int>(prob * 100.0) << '%';
    }
    os << setw(10 - 5) << found_probabilities.size() << "Xfind";
    os << setw(10) << "ε%" << setw(11) << "bits/item" << setw(11) << "bits/item"
       << setw(9) << "space%" << setw(8) << "keys";
    return os.str();
}

// Overloading the usual operator<< as used in "std::cout << foo", but for
// Statistics
template <class CharT, class Traits>
basic_ostream<CharT, Traits> &operator<<(basic_ostream<CharT, Traits> &os,
                                         const Statistics &stats) {
    os << fixed << setprecision(2) << setw(10) << right << stats.nanos_per_add;
    double add_and_find = stats.nanos_per_add;
    // os << fixed << setprecision(2) << setw(8) << right << stats.nanos_per_remove;
    for (const auto &fps : stats.nanos_per_finds) {
        os << setw(8) << fps.second;
        add_and_find += fps.second;
    }
    os << setw(10) << add_and_find << setw(9) << setprecision(4)
       << stats.false_positive_probabilty * 100 << setw(11) << setprecision(3)
       << stats.bits_per_item << setw(11);

    // we get some nonsensical result for very small fpps
    if (stats.false_positive_probabilty > 0.0000001) {
        const auto minbits = log2(1 / stats.false_positive_probabilty);
        os << minbits << setw(9) << 100 * (stats.bits_per_item / minbits - 1);
    } else {
        os << 64 << setw(9) << 0;
    }
    os << " " << setw(7) << (stats.add_count / 1000000.);
    return os;
}

template <typename Table>
struct FilterAPI {};

template <typename ItemType, size_t bits_per_item,
          template <size_t> class TableType, typename HashFamily>
struct FilterAPI<cuckoofilter::CuckooFilter<ItemType, bits_per_item, TableType, HashFamily>> {
    using Table =
        cuckoofilter::CuckooFilter<ItemType, bits_per_item, TableType, HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        if (0 != table->Add(key)) {
            throw logic_error(
                "The filter is too small to hold all of the elements");
        }
    }
    static void AddAll(const vector<ItemType> &keys, const size_t start,
                       const size_t end, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void Remove(uint64_t key, Table *table) {
        table->Delete(key);
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return (0 == table->Contain(key));
    }
};

template <typename ItemType, size_t bits_per_item,
          template <size_t> class TableType, typename HashFamily>
struct FilterAPI<cuckoofilter::CuckooFilterStable<ItemType, bits_per_item, TableType, HashFamily>> {
    using Table =
        cuckoofilter::CuckooFilterStable<ItemType, bits_per_item, TableType, HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        if (0 != table->Add(key)) {
            throw logic_error(
                "The filter is too small to hold all of the elements");
        }
    }
    static void AddAll(const vector<ItemType> &keys, const size_t start,
                       const size_t end, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void Remove(uint64_t key, Table *table) {
        table->Delete(key);
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return (0 == table->Contain(key));
    }
};

template <typename ItemType, size_t bits_per_item, size_t percent_extra_pad,
          template <size_t> class TableType = cuckoofilter::SingleTable,
          typename HashFamily = hashing::TwoIndependentMultiplyShift>
class CuckooFilterStablePad
    : public cuckoofilter::CuckooFilterStable<ItemType, bits_per_item, TableType, HashFamily> {
public:
    explicit CuckooFilterStablePad(const size_t max_num_keys)
        : cuckoofilter::CuckooFilterStable<ItemType, bits_per_item, TableType, HashFamily>(
              max_num_keys + (percent_extra_pad * max_num_keys / 100)) {}
};

template <typename ItemType, size_t bits_per_item, size_t percent_extra_pad,
          template <size_t> class TableType, typename HashFamily>
struct FilterAPI<CuckooFilterStablePad<ItemType, bits_per_item, percent_extra_pad,
                                       TableType, HashFamily>> {
    using Table = CuckooFilterStablePad<ItemType, bits_per_item,
                                        percent_extra_pad, TableType, HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        if (0 != table->Add(key)) {
            throw logic_error(
                "The filter is too small to hold all of the elements");
        }
    }
    static void AddAll(const vector<ItemType> &keys, const size_t start,
                       const size_t end, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void Remove(uint64_t key, Table *table) {
        table->Delete(key);
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return (0 == table->Contain(key));
    }
};

#ifdef __aarch64__
template <typename HashFamily>
struct FilterAPI<SimdBlockFilterFixed<HashFamily>> {
    using Table = SimdBlockFilterFixed<HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        Table ans(ceil(add_count * 8.0 / CHAR_BIT));
        return ans;
    }
    static void Add(uint64_t key, Table *table) {
        table->Add(key);
    }
    static void AddAll(const vector<uint64_t> &keys, const size_t start,
                       const size_t end, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return table->Find(key);
    }
};

#endif

#ifdef __AVX2__
template <typename HashFamily>
struct FilterAPI<SimdBlockFilter<HashFamily>> {
    using Table = SimdBlockFilter<HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        Table ans(ceil(log2(add_count * 8.0 / CHAR_BIT)));
        return ans;
    }
    static void Add(uint64_t key, Table *table) {
        table->Add(key);
    }
    static void AddAll(const vector<uint64_t> &keys, const size_t start,
                       const size_t end, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return table->Find(key);
    }
};

template <typename HashFamily>
struct FilterAPI<SimdBlockFilterFixed64<HashFamily>> {
    using Table = SimdBlockFilterFixed64<HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        Table ans(ceil(add_count * 8.0 / CHAR_BIT));
        return ans;
    }
    static void Add(uint64_t key, Table *table) {
        table->Add(key);
    }
    static void AddAll(const vector<uint64_t> &keys, const size_t start,
                       const size_t end, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return table->Find(key);
    }
};

template <typename HashFamily>
struct FilterAPI<SimdBlockFilterFixed16<HashFamily>> {
    using Table = SimdBlockFilterFixed16<HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        Table ans(ceil(add_count * 8.0 / CHAR_BIT));
        return ans;
    }
    static void Add(uint64_t key, Table *table) {
        table->Add(key);
    }
    static void AddAll(const vector<uint64_t> &keys, const size_t start,
                       const size_t end, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return table->Find(key);
    }
};

template <typename HashFamily>
struct FilterAPI<SimdBlockFilterFixed<HashFamily>> {
    using Table = SimdBlockFilterFixed<HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        Table ans(ceil(add_count * 8.0 / CHAR_BIT));
        return ans;
    }
    static void Add(uint64_t key, Table *table) {
        table->Add(key);
    }
    static void AddAll(const vector<uint64_t> &keys, const size_t start,
                       const size_t end, Table *table) {
        table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return table->Find(key);
    }
};
#endif

template <typename ItemType, typename FingerprintType>
struct FilterAPI<xorfilter::XorFilter<ItemType, FingerprintType>> {
    using Table = xorfilter::XorFilter<ItemType, FingerprintType>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void AddAll(const vector<ItemType> keys, const size_t start,
                       const size_t end, Table *table) {
        table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return (0 == table->Contain(key));
    }
};

template <typename ItemType, typename FingerprintType>
struct FilterAPI<xorfusefilter::XorFuseFilter<ItemType, FingerprintType>> {
    using Table = xorfusefilter::XorFuseFilter<ItemType, FingerprintType>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void AddAll(const vector<ItemType> keys, const size_t start,
                       const size_t end, Table *table) {
        table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return (0 == table->Contain(key));
    }
};

template <typename CoeffType, uint32_t kNumColumns>
struct RibbonTsHomog {
    static constexpr bool kIsFilter = true;
    static constexpr bool kHomogeneous = true;
    static constexpr bool kFirstCoeffAlwaysOne = true;
    static constexpr bool kUseSmash = false;
    using CoeffRow = CoeffType;
    using Hash = uint64_t;
    using Key = uint64_t;
    using Seed = uint32_t;
    using Index = size_t;
    using ResultRow = uint32_t;
    static constexpr bool kAllowZeroStarts = false;
    static constexpr uint32_t kFixedNumColumns = kNumColumns;

    static Hash HashFn(const Hash &input, Seed raw_seed) {
        // No re-seeding for Homogeneous, because it can be skipped in practice
        return input;
    }
};

template <typename CoeffType, uint32_t kNumColumns, bool kSmash = false>
struct RibbonTsSeeded
    : public r2::StandardRehasherAdapter<RibbonTsHomog<CoeffType, kNumColumns>> {
    static constexpr bool kHomogeneous = false;
    static constexpr bool kUseSmash = kSmash;
};

template <typename CoeffType, uint32_t kNumColumns, uint32_t kMilliBitsPerKey = 7700>
class HomogRibbonFilter {
    using TS = RibbonTsHomog<CoeffType, kNumColumns>;
    IMPORT_RIBBON_IMPL_TYPES(TS);

    size_t num_slots;
    size_t bytes;
    unique_ptr<char[]> ptr;
    InterleavedSoln soln;
    Hasher hasher;

public:
    static constexpr double kFractionalCols =
        kNumColumns == 0 ? kMilliBitsPerKey / 1000.0 : kNumColumns;

    static double GetBestOverheadFactor() {
        double overhead =
            (4.0 + kFractionalCols * 0.25) / (8.0 * sizeof(CoeffType));
        return 1.0 + overhead;
    }

    HomogRibbonFilter(size_t add_count)
        : num_slots(InterleavedSoln::RoundUpNumSlots(
              (size_t)(GetBestOverheadFactor() * add_count))),
          bytes(static_cast<size_t>((num_slots * kFractionalCols + 7) / 8)),
          ptr(new char[bytes]), soln(ptr.get(), bytes) {}

    void AddAll(const vector<uint64_t> &keys, const size_t start, const size_t end) {
        Banding b(num_slots);
        (void)b.AddRange(keys.begin() + start, keys.begin() + end);
        soln.BackSubstFrom(b);
    }
    bool Contain(uint64_t key) const {
        return soln.FilterQuery(key, hasher);
    }
    size_t SizeInBytes() const {
        return bytes;
    }
};

template <typename CoeffType, uint32_t kNumColumns, uint32_t kMilliBitsPerKey>
struct FilterAPI<HomogRibbonFilter<CoeffType, kNumColumns, kMilliBitsPerKey>> {
    using Table = HomogRibbonFilter<CoeffType, kNumColumns, kMilliBitsPerKey>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void AddAll(const vector<uint64_t> &keys, const size_t start,
                       const size_t end, Table *table) {
        table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return table->Contain(key);
    }
};

template <typename CoeffType, uint32_t kNumColumns, uint32_t kMinPctOverhead,
          uint32_t kMilliBitsPerKey = 7700>
class BalancedRibbonFilter {
    using TS = RibbonTsSeeded<CoeffType, kNumColumns>;
    IMPORT_RIBBON_IMPL_TYPES(TS);
    static constexpr uint32_t kBitsPerVshard = 8;
    using BalancedBanding = r2::BalancedBanding<TS, kBitsPerVshard>;
    using BalancedHasher = r2::BalancedHasher<TS, kBitsPerVshard>;

    uint32_t log2_vshards;
    size_t num_slots;

    size_t bytes;
    unique_ptr<char[]> ptr;
    InterleavedSoln soln;

    size_t meta_bytes;
    unique_ptr<char[]> meta_ptr;
    BalancedHasher hasher;

public:
    static constexpr double kFractionalCols =
        kNumColumns == 0 ? kMilliBitsPerKey / 1000.0 : kNumColumns;

    static double GetNumSlots(size_t add_count, uint32_t log2_vshards) {
        size_t add_per_vshard = add_count >> log2_vshards;

        double overhead;
        if (sizeof(CoeffType) == 8) {
            overhead = 0.0000055 * add_per_vshard; // FIXME?
        } else if (sizeof(CoeffType) == 4) {
            overhead = 0.00005 * add_per_vshard;
        } else if (sizeof(CoeffType) == 2) {
            overhead = 0.00010 * add_per_vshard; // FIXME?
        } else {
            assert(sizeof(CoeffType) == 16);
            overhead = 0.0000013 * add_per_vshard;
        }
        overhead = std::max(overhead, 0.01 * kMinPctOverhead);
        return InterleavedSoln::RoundUpNumSlots(
            (size_t)(add_count + overhead * add_count + add_per_vshard / 5));
    }

    BalancedRibbonFilter(size_t add_count)
        : log2_vshards(
              (uint32_t)r2::FloorLog2((add_count + add_count / 3 + add_count / 5) /
                                      (128 * sizeof(CoeffType)))),
          num_slots(GetNumSlots(add_count, log2_vshards)),
          bytes(static_cast<size_t>((num_slots * kFractionalCols + 7) / 8)),
          ptr(new char[bytes]), soln(ptr.get(), bytes),
          meta_bytes(BalancedHasher(log2_vshards, nullptr).GetMetadataLength()),
          meta_ptr(new char[meta_bytes]), hasher(log2_vshards, meta_ptr.get()) {}

    void AddAll(const vector<uint64_t> &keys, const size_t start, const size_t end) {
        for (uint32_t seed = 0;; ++seed) {
            BalancedBanding b(log2_vshards);
            b.SetOrdinalSeed(seed);
            b.BalancerAddRange(keys.begin() + start, keys.begin() + end);
            if (b.Balance(num_slots)) {
                if (seed > 0) {
                    fprintf(stderr, "Success after %d tries\n", (int)seed + 1);
                }
                hasher.SetOrdinalSeed(seed);
                soln.BackSubstFrom(b);
                memcpy(meta_ptr.get(), b.GetMetadata(), b.GetMetadataLength());
                return;
            }
        }
    }
    bool Contain(uint64_t key) const {
        return soln.FilterQuery(key, hasher);
    }
    size_t SizeInBytes() const {
        return bytes + meta_bytes;
    }
};

template <typename CoeffType, uint32_t kNumColumns, uint32_t kMinPctOverhead, uint32_t kMilliBitsPerKey>
struct FilterAPI<BalancedRibbonFilter<CoeffType, kNumColumns, kMinPctOverhead, kMilliBitsPerKey>> {
    using Table =
        BalancedRibbonFilter<CoeffType, kNumColumns, kMinPctOverhead, kMilliBitsPerKey>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void AddAll(const vector<uint64_t> &keys, const size_t start,
                       const size_t end, Table *table) {
        table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return table->Contain(key);
    }
};

template <typename CoeffType, uint32_t kNumColumns, uint32_t kMinPctOverhead,
          bool kUseSmash = false>
class StandardRibbonFilter {
    using TS = RibbonTsSeeded<CoeffType, kNumColumns, kUseSmash>;
    IMPORT_RIBBON_IMPL_TYPES(TS);

    size_t num_slots;

    size_t bytes;
    unique_ptr<char[]> ptr;
    InterleavedSoln soln;
    Hasher hasher;

public:
    static constexpr double kFractionalCols = kNumColumns == 0 ? 7.7 : kNumColumns;

    static double GetNumSlots(size_t add_count) {
        double overhead;
        if (sizeof(CoeffType) == 8) {
            overhead = -0.0251 + std::log(1.0 * add_count) * 1.4427 * 0.0083;
        } else {
            assert(sizeof(CoeffType) == 16);
            overhead = -0.0176 + std::log(1.0 * add_count) * 1.4427 * 0.0038;
        }
        overhead = std::max(overhead, 0.01 * kMinPctOverhead);
        return InterleavedSoln::RoundUpNumSlots(
            (size_t)(add_count + overhead * add_count));
    }

    StandardRibbonFilter(size_t add_count)
        : num_slots(GetNumSlots(add_count)),
          bytes(static_cast<size_t>((num_slots * kFractionalCols + 7) / 8)),
          ptr(new char[bytes]), soln(ptr.get(), bytes) {}

    void AddAll(const vector<uint64_t> &keys, const size_t start, const size_t end) {
        Banding b;
        if (b.ResetAndFindSeedToSolve(num_slots, keys.begin() + start,
                                      keys.begin() + end)) {
            uint32_t seed = b.GetOrdinalSeed();
            if (seed > 0) {
                fprintf(stderr, "Success after %d tries\n", (int)seed + 1);
            }
            hasher.SetOrdinalSeed(seed);
            soln.BackSubstFrom(b);
        } else {
            fprintf(stderr, "Failed!\n");
        }
    }
    bool Contain(uint64_t key) const {
        return soln.FilterQuery(key, hasher);
    }
    size_t SizeInBytes() const {
        return bytes;
    }
};

template <typename CoeffType, uint32_t kNumColumns, uint32_t kMinPctOverhead, bool kUseSmash>
struct FilterAPI<StandardRibbonFilter<CoeffType, kNumColumns, kMinPctOverhead, kUseSmash>> {
    using Table =
        StandardRibbonFilter<CoeffType, kNumColumns, kMinPctOverhead, kUseSmash>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void AddAll(const vector<uint64_t> &keys, const size_t start,
                       const size_t end, Table *table) {
        table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return table->Contain(key);
    }
};

template <int kProbes, int kBlocks, int kMilliBitsPerKey = 0>
class RocksBloomFilter {
    size_t bytes;
    unique_ptr<char[]> ptr;

public:
    static double GetBitsPerKey() {
        double bpk = 0;
        if (kMilliBitsPerKey > 0) {
            return kMilliBitsPerKey / 1000.0;
        }
        // Else, best bpk for probes
        for (int i = 0; i < kBlocks; ++i) {
            int probes = (kProbes + i) / kBlocks;
            switch (probes) {
                case 1: bpk += 1.44; break;
                // Based roughly on ChooseNumProbes
                case 2: bpk += 2.83; break;
                case 3: bpk += 4.34; break;
                case 4: bpk += 5.87; break;
                case 5: bpk += 7.47; break;
                case 6: bpk += 9.19; break;
                case 7: bpk += 10.90; break;
                case 8: bpk += 12.76; break;
                case 9: bpk += 14.93; break;
                case 10: bpk += 17.18; break;
                case 11: bpk += 20.15; break;
                case 12: bpk += 23.75; break;
                default: bpk += 27.50 + 3.75 * (probes - 13); break;
            }
        }
        return bpk;
    }

    RocksBloomFilter(size_t add_count)
        : bytes(static_cast<size_t>(GetBitsPerKey() * add_count / 8.0)),
          ptr(new char[bytes]()) {}

    static constexpr uint32_t kMixFactor = 0x12345673U;
    inline void Add(uint64_t key) {
        uint32_t a = static_cast<uint32_t>(key);
        uint32_t b = static_cast<uint32_t>(key >> 32);
        for (int i = 0; i < kBlocks; ++i) {
            int probes = (kProbes + i) / kBlocks;
            r2::FastLocalBloomImpl::AddHash(a, b, bytes, probes, ptr.get());
            a *= kMixFactor;
            b *= kMixFactor;
        }
    }
    void AddAll(const vector<uint64_t> &keys, const size_t start, const size_t end) {
        for (size_t i = start; i < end; ++i) {
            Add(keys[i]);
        }
    }
    bool Contain(uint64_t key) const {
        uint32_t a = static_cast<uint32_t>(key);
        uint32_t b = static_cast<uint32_t>(key >> 32);
        bool rv = true;
        for (int i = 0; i < kBlocks; ++i) {
            int probes = (kProbes + i) / kBlocks;
            rv &= r2::FastLocalBloomImpl::HashMayMatch(a, b, bytes, probes,
                                                       ptr.get());
            a *= kMixFactor;
            b *= kMixFactor;
        }
        return rv;
    }
    size_t SizeInBytes() const {
        return bytes;
    }
};

template <int kProbes, int kBlocks, int kMilliBitsPerKey>
struct FilterAPI<RocksBloomFilter<kProbes, kBlocks, kMilliBitsPerKey>> {
    using Table = RocksBloomFilter<kProbes, kBlocks, kMilliBitsPerKey>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        table->Add(key);
    }
    static void AddAll(const vector<uint64_t> &keys, const size_t start,
                       const size_t end, Table *table) {
        table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return table->Contain(key);
    }
};

template <typename morton_t = CompressedCuckoo::Morton3_8>
class MortonFilter {
    morton_t *filter;
    size_t size;

public:
    MortonFilter(const size_t size) {
        filter = new morton_t((size_t)(size / 0.95) + 64);
        this->size = size;
    }
    ~MortonFilter() {
        delete filter;
    }
    void Add(uint64_t key) {
        filter->insert(key);
    }
    void AddAll(const vector<uint64_t> &keys, const size_t start, const size_t end) {
        size_t size = end - start;
        // Morton filter batch interface requires the vectors' size to be a
        // multiple of 128 or it will crash - work around that.
        size_t alloc_size = ((size + 127) / 128) * 128;
        std::vector<uint64_t> k(alloc_size);
        std::vector<bool> status(alloc_size);
        for (size_t i = start; i < end; i++) {
            k[i - start] = keys[i];
        }
        // TODO return value and status is ignored currently
        filter->insert_many(k, status, size);
    }
    inline bool Contain(uint64_t item) const {
        return filter->likely_contains(item);
    };
    size_t SizeInBytes() const {
        // according to morton_sample_configs.h:
        // Morton3_8 - 3-slot buckets with 8-bit fingerprints: 11.7 bits/item
        // (load factor = 0.95)
        // so in theory we could just hardcode the size here,
        // and don't measure it
        // return (size_t)((size * 11.7) / 8);

        return filter->SizeInBytes();
    }
};

template <typename morton_t>
struct FilterAPI<MortonFilter<morton_t>> {
    using Table = MortonFilter<morton_t>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        table->Add(key);
    }
    static void AddAll(const vector<uint64_t> &keys, const size_t start,
                       const size_t end, Table *table) {
        table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, Table *table) {
        return table->Contain(key);
    }
};

class XorSingle {
public:
    xor8_s filter; // let us expose the struct. to avoid indirection
    explicit XorSingle(const size_t size) {
        if (!xor8_allocate(size, &filter)) {
            throw std::runtime_error("Allocation failed");
        }
    }
    ~XorSingle() {
        xor8_free(&filter);
    }
    bool AddAll(const uint64_t *data, const size_t start, const size_t end) {
        return xor8_buffered_populate(data + start, end - start, &filter);
    }
    inline bool Contain(uint64_t item) const {
        return xor8_contain(item, &filter);
    }
    inline size_t SizeInBytes() const {
        return xor8_size_in_bytes(&filter);
    }
    XorSingle(XorSingle &&o) : filter(o.filter) {
        o.filter.fingerprints = nullptr; // we take ownership for the data
    }

private:
    XorSingle(const XorSingle &o) = delete;
};

template <>
struct FilterAPI<XorSingle> {
    using Table = XorSingle;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void AddAll(const vector<uint64_t> &keys, const size_t start,
                       const size_t end, Table *table) {
        table->AddAll(keys.data(), start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        // some compilers are not smart enough to do the inlining properly
        return xor8_contain(key, &table->filter);
    }
};

template <size_t blocksize, int k, typename HashFamily>
struct FilterAPI<bloomfilter::SimpleBlockFilter<blocksize, k, HashFamily>> {
    using Table = bloomfilter::SimpleBlockFilter<blocksize, k, HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        Table ans(ceil(add_count * 8.0 / CHAR_BIT));
        return ans;
    }
    static void Add(uint64_t key, Table *table) {
        table->Add(key);
    }
    static void AddAll(const vector<uint64_t> &keys, const size_t start,
                       const size_t end, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return table->Find(key);
    }
};

template <typename ItemType, typename FingerprintType, typename HashFamily>
struct FilterAPI<xorfilter::XorFilter<ItemType, FingerprintType, HashFamily>> {
    using Table = xorfilter::XorFilter<ItemType, FingerprintType, HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void AddAll(const vector<ItemType> keys, const size_t start,
                       const size_t end, Table *table) {
        table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return (0 == table->Contain(key));
    }
};

template <typename ItemType, typename FingerprintType,
          typename FingerprintStorageType, typename HashFamily>
struct FilterAPI<xorfilter2::XorFilter2<ItemType, FingerprintType,
                                        FingerprintStorageType, HashFamily>> {
    using Table =
        xorfilter2::XorFilter2<ItemType, FingerprintType, FingerprintStorageType, HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void AddAll(const vector<ItemType> keys, const size_t start,
                       const size_t end, Table *table) {
        table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return (0 == table->Contain(key));
    }
};

template <typename ItemType, typename HashFamily>
struct FilterAPI<xorfilter::XorFilter10<ItemType, HashFamily>> {
    using Table = xorfilter::XorFilter10<ItemType, HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void AddAll(const vector<ItemType> keys, const size_t start,
                       const size_t end, Table *table) {
        table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return (0 == table->Contain(key));
    }
};

template <typename ItemType, typename HashFamily>
struct FilterAPI<xorfilter::XorFilter13<ItemType, HashFamily>> {
    using Table = xorfilter::XorFilter13<ItemType, HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void AddAll(const vector<ItemType> keys, const size_t start,
                       const size_t end, Table *table) {
        table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return (0 == table->Contain(key));
    }
};

template <typename ItemType, typename HashFamily>
struct FilterAPI<xorfilter::XorFilter10_666<ItemType, HashFamily>> {
    using Table = xorfilter::XorFilter10_666<ItemType, HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void AddAll(const vector<ItemType> keys, const size_t start,
                       const size_t end, Table *table) {
        table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return (0 == table->Contain(key));
    }
};

template <typename ItemType, typename FingerprintType,
          typename FingerprintStorageType, typename HashFamily>
struct FilterAPI<xorfilter2n::XorFilter2n<ItemType, FingerprintType,
                                          FingerprintStorageType, HashFamily>> {
    using Table = xorfilter2n::XorFilter2n<ItemType, FingerprintType,
                                           FingerprintStorageType, HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void AddAll(const vector<ItemType> keys, const size_t start,
                       const size_t end, Table *table) {
        table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return (0 == table->Contain(key));
    }
};

template <typename ItemType, typename FingerprintType, typename HashFamily>
struct FilterAPI<xorfilter_plus::XorFilterPlus<ItemType, FingerprintType, HashFamily>> {
    using Table =
        xorfilter_plus::XorFilterPlus<ItemType, FingerprintType, HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void AddAll(const vector<ItemType> keys, const size_t start,
                       const size_t end, Table *table) {
        table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return (0 == table->Contain(key));
    }
};

template <typename ItemType, typename FingerprintType,
          typename FingerprintStorageType, typename HashFamily>
struct FilterAPI<xorfilter_plus2::XorFilterPlus2<ItemType, FingerprintType,
                                                 FingerprintStorageType, HashFamily>> {
    using Table =
        xorfilter_plus2::XorFilterPlus2<ItemType, FingerprintType,
                                        FingerprintStorageType, HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void AddAll(const vector<ItemType> keys, const size_t start,
                       const size_t end, Table *table) {
        table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return (0 == table->Contain(key));
    }
};

template <typename ItemType, size_t bits_per_item, typename HashFamily>
struct FilterAPI<gcsfilter::GcsFilter<ItemType, bits_per_item, HashFamily>> {
    using Table = gcsfilter::GcsFilter<ItemType, bits_per_item, HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void AddAll(const vector<ItemType> keys, const size_t start,
                       const size_t end, Table *table) {
        table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return (0 == table->Contain(key));
    }
};

#ifdef __AVX2__
template <typename ItemType, size_t bits_per_item, typename HashFamily>
struct FilterAPI<gqfilter::GQFilter<ItemType, bits_per_item, HashFamily>> {
    using Table = gqfilter::GQFilter<ItemType, bits_per_item, HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        table->Add(key);
    }
    static void AddAll(const vector<ItemType> keys, const size_t start,
                       const size_t end, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void Remove(uint64_t key, Table *table) {
        table->Remove(key);
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return (0 == table->Contain(key));
    }
};
#endif

template <typename ItemType, size_t bits_per_item, bool branchless, typename HashFamily>
struct FilterAPI<bloomfilter::BloomFilter<ItemType, bits_per_item, branchless, HashFamily>> {
    using Table =
        bloomfilter::BloomFilter<ItemType, bits_per_item, branchless, HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        table->Add(key);
    }
    static void AddAll(const vector<ItemType> keys, const size_t start,
                       const size_t end, Table *table) {
        table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return (0 == table->Contain(key));
    }
};

template <typename ItemType, size_t bits_per_item, bool branchless, typename HashFamily>
struct FilterAPI<counting_bloomfilter::CountingBloomFilter<ItemType, bits_per_item,
                                                           branchless, HashFamily>> {
    using Table =
        counting_bloomfilter::CountingBloomFilter<ItemType, bits_per_item,
                                                  branchless, HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        table->Add(key);
    }
    static void AddAll(const vector<ItemType> keys, const size_t start,
                       const size_t end, Table *table) {
        table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        table->Remove(key);
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return (0 == table->Contain(key));
    }
};

template <typename ItemType, size_t bits_per_item, bool branchless, typename HashFamily>
struct FilterAPI<counting_bloomfilter::SuccinctCountingBloomFilter<
    ItemType, bits_per_item, branchless, HashFamily>> {
    using Table =
        counting_bloomfilter::SuccinctCountingBloomFilter<ItemType, bits_per_item,
                                                          branchless, HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        table->Add(key);
    }
    static void AddAll(const vector<ItemType> keys, const size_t start,
                       const size_t end, Table *table) {
        table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        table->Remove(key);
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return (0 == table->Contain(key));
    }
};

template <typename ItemType, size_t bits_per_item, typename HashFamily>
struct FilterAPI<counting_bloomfilter::SuccinctCountingBlockedBloomFilter<
    ItemType, bits_per_item, HashFamily>> {
    using Table = counting_bloomfilter::SuccinctCountingBlockedBloomFilter<
        ItemType, bits_per_item, HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        table->Add(key);
    }
    static void AddAll(const vector<ItemType> keys, const size_t start,
                       const size_t end, Table *table) {
        throw std::runtime_error("Unsupported");
        // table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        table->Remove(key);
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return table->Contain(key);
    }
};

template <typename ItemType, size_t bits_per_item, typename HashFamily>
struct FilterAPI<counting_bloomfilter::SuccinctCountingBlockedBloomRankFilter<
    ItemType, bits_per_item, HashFamily>> {
    using Table = counting_bloomfilter::SuccinctCountingBlockedBloomRankFilter<
        ItemType, bits_per_item, HashFamily>;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        table->Add(key);
    }
    static void AddAll(const vector<ItemType> keys, const size_t start,
                       const size_t end, Table *table) {
        throw std::runtime_error("Unsupported");
        // table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        table->Remove(key);
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return table->Contain(key);
    }
};

/******************************************************************************/

template <ribbon::ThreshMode mode, bool sparse_coeffs, bool interleaved,
          bool cls, size_t coeff_bits, size_t result_bits>
struct BumpRibbonConfig
    : public ribbon::RConfig<coeff_bits, result_bits, mode, sparse_coeffs,
                             interleaved, cls, 0, uint64_t> {
    static constexpr bool log = false; // quiet

    // disable MSH to ensure that no-op hashing is used (see below)
    static constexpr bool kUseMultiplyShiftHash = false;
    static uint64_t HashFn(const uint64_t &input, uint64_t raw_seed) {
        // Does not use re-seeding here, to be comparable to
        // other implementations here
        return input;
    }
};

template <uint8_t depth, ribbon::ThreshMode mode, bool sparse_coeffs,
          bool interleaved, bool cls, size_t coeff_bits, size_t result_bits>
struct BumpRibbonFilter {
    using Config =
        BumpRibbonConfig<mode, sparse_coeffs, interleaved, cls, coeff_bits, result_bits>;
    using Table = ribbon::ribbon_filter<depth, Config>;

    // eps = -0.666 L / (4B + L)
    const double slots_per_item =
        mode == ribbon::ThreshMode::onebit
            ? 1.0 - 0.666 * coeff_bits / (4.0 * Config::kBucketSize + coeff_bits)
            : 1.0 - (coeff_bits <= 32 ? 3.0 : 4.0) / coeff_bits;

    Table filter;

    BumpRibbonFilter(size_t add_count, size_t seed = 42)
        : filter(add_count * slots_per_item, slots_per_item, seed) {}
    void AddAll(const vector<uint64_t> &keys, const size_t start, const size_t end) {
        // if the input is too small for the filter, abort before backsubstitution
        if (!filter.AddRange(keys.begin() + start, keys.begin() + end))
            throw std::runtime_error("Input size too small for this filter");
        filter.BackSubst();
    }
    inline bool Contain(uint64_t key) const {
        return filter.QueryFilter(key);
    }
    size_t SizeInBytes() const {
        return filter.Size();
    }
};

template <uint8_t depth, ribbon::ThreshMode mode, bool sparse_coeffs,
          bool interleaved, bool cls, size_t coeff_bits, size_t result_bits>
struct FilterAPI<BumpRibbonFilter<depth, mode, sparse_coeffs, interleaved, cls,
                                  coeff_bits, result_bits>> {
    using Table = BumpRibbonFilter<depth, mode, sparse_coeffs, interleaved, cls,
                                   coeff_bits, result_bits>;

    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void AddAll(const vector<uint64_t> &keys, const size_t start,
                       const size_t end, Table *table) {
        table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return table->Contain(key);
    }
};

/******************************************************************************/
/*** Stefan Walzer's retrieval tests ******************************************/
/******************************************************************************/

// specialise this and provide a constructor that provides this class's
// constructor with an appropriate config
template <typename Table, typename data_t = bool>
struct RetrievalTestAdapter {
    static constexpr unsigned bits =
        std::is_same_v<data_t, bool> ? 1 : 8u * sizeof(data_t);
    static constexpr uint64_t mask = (1ul << bits) - 1;
    template <typename Config>
    RetrievalTestAdapter(Config config, size_t num_items)
        : R(config), values(num_items) {}

    void AddAll(const vector<uint64_t> &keys) {
        for (size_t i = 0; i < keys.size(); i++) {
            values[i] = ((keys[i] * 0xc367844a6e52731dU) >> 32) & mask;
        }
        R.Construct(keys, values);
        if (!R.hasSucceeded()) {
            throw std::runtime_error("Failed to construct");
        }
    }

    inline bool Contain(uint64_t key) const {
        const data_t expected = ((key * 0xc367844a6e52731dU) >> 32) & mask;
        bool result = expected == R.retrieve(key);
        return result;
    }
    size_t SizeInBytes() const {
        // bits to bytes
        return (R.memoryUsage() + 7) / 8;
    }

protected:
    Table R;
    std::vector<data_t> values;
};

// Inherit from this to add more filters
template <typename Table>
struct RetrievalTestFilterAPI {
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    static void AddAll(const vector<uint64_t> &keys, const size_t /* start */,
                       const size_t /* end */, Table *table) {
        table->AddAll(keys);
    }
    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return table->Contain(key);
    }
};

template <uint32_t C>
struct TwoBlockRetrieval
    : RetrievalTestAdapter<walzer::RetrieverChunked<walzer::TwoBlockStrategy<16, uint64_t>>> {
    // static constexpr int l = 16; ← hardcoded in template above
    static constexpr double c = 1 / 1.0005;
    using Strat = walzer::TwoBlockStrategy<16, uint64_t>;
    using Ret = walzer::RetrieverChunked<Strat>;

    TwoBlockRetrieval(size_t num_items)
        : RetrievalTestAdapter<Ret>(
              Ret::Configuration{Strat::Configuration{c}, C}, num_items) {}
};

template <uint32_t C>
struct FilterAPI<TwoBlockRetrieval<C>>
    : public RetrievalTestFilterAPI<TwoBlockRetrieval<C>> {
    using Table = TwoBlockRetrieval<C>;
};

// e.g. 800, 997
template <int D, int milliC, typename data_t = bool>
struct LMSSRetrieval
    : RetrievalTestAdapter<walzer::Retriever<walzer::RetrieverLMSS<uint64_t, data_t>>, data_t> {
    using Strat = walzer::RetrieverLMSS<uint64_t, data_t>;
    using Ret = walzer::Retriever<Strat>;
    static constexpr unsigned bits =
        std::is_same_v<data_t, bool> ? 1 : 8 * sizeof(data_t);

    LMSSRetrieval(size_t num_items)
        : RetrievalTestAdapter<Ret, data_t>(
              typename Strat::Configuration{milliC / 1000.0, D}, num_items) {}
};

template <int D, int milliC, typename data_t>
struct FilterAPI<LMSSRetrieval<D, milliC, data_t>>
    : public RetrievalTestFilterAPI<LMSSRetrieval<D, milliC, data_t>> {
    using Table = LMSSRetrieval<D, milliC, data_t>;
};


// e.g. {3, 910, 10000, false} or {3, 900, 1000, true}
// or {4, 970, 10000, false} or {4, 960, 1000, true}
template <int k, int millic, uint32_t C, bool compress>
struct GovRetrieval
    : RetrievalTestAdapter<walzer::RetrieverChunked<
          walzer::GOVStrategy<k, uint64_t>,
          std::conditional_t<compress, walzer::ChunkInfoCompressed, walzer::ChunkInfoPacked>>> {
    using Strat = walzer::GOVStrategy<k, uint64_t>;
    using ChunkInfoManager =
        std::conditional_t<compress, walzer::ChunkInfoCompressed, walzer::ChunkInfoPacked>;
    using Ret = walzer::RetrieverChunked<Strat, ChunkInfoManager>;

    GovRetrieval(size_t num_items)
        : RetrievalTestAdapter<Ret>(
              typename Ret::Configuration{
                  typename Strat::Configuration{millic / 1000.0}, C},
              num_items) {}
};

template <int k, int millic, uint32_t C, bool compress>
struct FilterAPI<GovRetrieval<k, millic, C, compress>>
    : public RetrievalTestFilterAPI<GovRetrieval<k, millic, C, compress>> {
    using Table = GovRetrieval<k, millic, C, compress>;
};


// e.g. {3, 120, 905}, {4, 120, 960}, or {7, 120, 979}
template <int k, int z, int millic, typename data_t = bool>
struct CoupledRetrieval
    : RetrievalTestAdapter<
          walzer::Retriever<walzer::CoupledStrategy<k, uint64_t, data_t>>, data_t> {
    using Strat = walzer::CoupledStrategy<k, uint64_t, data_t>;
    using Ret = walzer::Retriever<Strat>;

    CoupledRetrieval(size_t num_items)
        : RetrievalTestAdapter<Ret, data_t>(
              typename Strat::Configuration{millic / 1000.0, 1.0 / z}, num_items) {
    }
};

template <int k, int z, int millic, typename data_t>
struct FilterAPI<CoupledRetrieval<k, z, millic, data_t>>
    : public RetrievalTestFilterAPI<CoupledRetrieval<k, z, millic, data_t>> {
    using Table = CoupledRetrieval<k, z, millic, data_t>;
};

// e.g. {3, 120, 905}, {4, 120, 960}, or {7, 120, 979}
template <int k, typename data_t = bool>
struct SimpleCoupledRetrieval
    : RetrievalTestAdapter<
          walzer::Retriever<walzer::CoupledStrategy<k, uint64_t, data_t>>, data_t> {
    using Strat = walzer::CoupledStrategy<k, uint64_t, data_t>;
    using Ret = walzer::Retriever<Strat>;

     static constexpr typename Strat::Configuration c3_s = {0.885, 1.0 / 60};
     static constexpr typename Strat::Configuration c3_m = {0.905, 1.0 / 120};
     static constexpr typename Strat::Configuration c3_l = {0.912, 1.0 / 240};

     static constexpr typename Strat::Configuration c4_s = {0.940, 1.0 / 60};
     static constexpr typename Strat::Configuration c4_m = {0.960, 1.0 / 120};
     static constexpr typename Strat::Configuration c4_l = {0.970, 1.0 / 240};

     static constexpr typename Strat::Configuration c7_s = {0.959, 1.0 / 60};
     static constexpr typename Strat::Configuration c7_m = {0.979, 1.0 / 120};
     static constexpr typename Strat::Configuration c7_l = {0.989, 1.0 / 240};

    SimpleCoupledRetrieval(size_t n)
        : RetrievalTestAdapter<Ret, data_t>(
              k == 3   ? (n < 8'000'000 ? c3_s : (n < 80'000'000 ? c3_m : c3_l))
              : k == 4 ? (n < 8'000'000 ? c4_s : (n < 80'000'000 ? c4_m : c4_l))
                       : (n < 8'000'000 ? c7_s : (n < 80'000'000 ? c7_m : c7_l)),
              n) {}
};

template <int k, typename data_t>
struct FilterAPI<SimpleCoupledRetrieval<k, data_t>>
    : public RetrievalTestFilterAPI<SimpleCoupledRetrieval<k, data_t>> {
    using Table = SimpleCoupledRetrieval<k, data_t>;
};

template <int millic = 810, typename data_t = bool>
struct BPZRetrieval
    : RetrievalTestAdapter<walzer::Retriever<walzer::BPZStrategy<uint64_t, data_t>>, data_t> {
    using Strat = walzer::BPZStrategy<uint64_t, data_t>;
    using Ret = walzer::Retriever<Strat>;
    static constexpr unsigned bits =
        std::is_same_v<data_t, bool> ? 1 : 8 * sizeof(data_t);

    BPZRetrieval(size_t num_items)
        : RetrievalTestAdapter<Ret, data_t>(
              typename Strat::Configuration{millic / 1000.0}, num_items) {}
};

template <int millic, typename data_t>
struct FilterAPI<BPZRetrieval<millic, data_t>>
    : public RetrievalTestFilterAPI<BPZRetrieval<millic, data_t>> {
    using Table = BPZRetrieval<millic, data_t>;
};

/******************************************************************************/
// Quotient Filter
template <size_t remainder_bits = qf::DEFAULT_REMAINDER_BITS>
struct QuotientFilter {
    /*
    struct NoOpHash {
        static constexpr std::string_view name = "nohash";
        static constexpr size_t significant_digits = 64;
        NoOpHash() {}
        inline uint64_t operator()(const uint64_t k) const {
            return k;
        }
    };
    */

    using QF_type =
        qf::templated_qfilter_seq<uint64_t, remainder_bits, hashing::SimpleXorMul>;
    QF_type filter;

    QuotientFilter(size_t add_count) : filter(add_count) {}

    void Add(uint64_t key) {
        filter.insert(key);
    }

    inline bool Contain(uint64_t key) const {
        return filter.contains(key);
    }

    size_t SizeInBytes() const {
        return filter.memory_usage_bytes();
    }
};


template <size_t remainder_bits>
struct FilterAPI<QuotientFilter<remainder_bits>> {
    using Table = QuotientFilter<remainder_bits>;

    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table *table) {
        table->Add(key);
    }
    static void AddAll(const vector<uint64_t> &, const size_t, const size_t,
                       Table *) {
        throw std::runtime_error("Unsupported");
        // table->AddAll(keys, start, end);
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return table->Contain(key);
    }
};


/******************************************************************************/

// assuming that first1,last1 and first2, last2 are sorted,
// this tries to find out how many of first1,last1 can be
// found in first2, last2, this includes duplicates
template <class InputIt1, class InputIt2>
size_t match_size_iter(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                       InputIt2 last2) {
    size_t answer = 0;
    while (first1 != last1 && first2 != last2) {
        if (*first1 < *first2) {
            ++first1;
        } else if (*first2 < *first1) {
            ++first2;
        } else {
            answer++;
            ++first1;
        }
    }
    return answer;
}

template <class InputIt>
size_t count_distinct(InputIt first, InputIt last) {
    if (last == first)
        return 0;
    size_t answer = 1;
    auto val = *first;
    first++;

    while (first != last) {
        if (val != *first)
            ++answer;
        first++;
    }
    return answer;
}

// Intentionally copies vectors
size_t match_size(vector<uint64_t> a, vector<uint64_t> b, size_t *distincta,
                  size_t *distinctb) {
    // could obviously be accelerated with a Bloom filter
    // But this is surprisingly fast!
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    if (distincta != NULL)
        *distincta = count_distinct(a.begin(), a.end());
    if (distinctb != NULL)
        *distinctb = count_distinct(b.begin(), b.end());
    return match_size_iter(a.begin(), a.end(), b.begin(), b.end());
}

// b must be sorted lexicographically. Intentionally copies a but not b
template <typename T1, typename T2>
size_t match_size_pairs(std::vector<std::pair<T1, T2>> a,
                        const std::vector<std::pair<T1, T2>> &b) {
    // could obviously be accelerated with a Bloom filter
    // But this is surprisingly fast!
    std::sort(a.begin(), a.end(), [](const auto &x, const auto &y) {
        return (x.first != y.first) ? x.first < y.first : x.second < y.second;
    });
    return match_size_iter(a.begin(), a.end(), b.begin(), b.end());
}


// intentionally copies vector
bool has_duplicates(vector<uint64_t> a) {
    std::sort(a.begin(), a.end());
    return count_distinct(a.begin(), a.end()) < a.size();
}
struct samples {
    double found_probability;
    std::vector<uint64_t> to_lookup_mixed;
    size_t true_match;
    size_t actual_sample_size;
};

typedef struct samples samples_t;

struct psamples {
    double found_probability;
    std::vector<std::pair<uint64_t, uint64_t>> to_lookup_mixed;
    size_t true_match;
    size_t actual_sample_size;
};

typedef struct psamples psamples_t;


uint64_t reverseBitsSlow(uint64_t v) {
    // r will be reversed bits of v; first get LSB of v
    uint64_t r = v & 1;
    int s = sizeof(v) * CHAR_BIT - 1; // extra shift needed at end
    for (v >>= 1; v; v >>= 1) {
        r <<= 1;
        r |= v & 1;
        s--;
    }
    r <<= s; // shift when v's highest bits are zero
    return r;
}

void parse_comma_separated(char *c, std::set<int> &answer) {
    std::stringstream ss(c);
    int i;
    while (ss >> i) {
        answer.insert(i);
        if (ss.peek() == ',')
            ss.ignore();
    }
}


template <template <typename> typename FilterBenchmark, typename GenIn, typename Sorter>
int do_main(GenIn &&GenerateInputs, Sorter &&sorter, int argc, char *argv[],
            std::string outfn_base = "results_") {
    std::map<int, std::string> names = //
        {// Xor
         {0, "Xor8"},
         {1, "Xor12"},
         {2, "Xor16"},
         {3, "XorPlus8"},
         {4, "XorPlus16"},
         {5, "Xor10"},
         {6, "Xor10.666"},
         {7, "Xor10(NBitArray)"},
         {8, "Xor14(NBitArray)"},
         {9, "XorPowTwo8"},
         // Cuckooo
         {10, "Cuckoo8"},
         {11, "Cuckoo10"},
         {12, "Cuckoo12"},
         {13, "Cuckoo14"},
         {14, "Cuckoo16"},
         {15, "CuckooSemiSort13"},
         {16, "CuckooPowTwo8"},
         {17, "CuckooPowTwo12"},
         {18, "CuckooPowTwo16"},
         {19, "CuckooSemiSortPowTwo13"},
         // GCS
         {20, "GCS"},
#ifdef __AVX2__
         // CQF
         {30, "CQF8"},
         {31, "CQF7"},
         {32, "CQF10"},
         {33, "CQF13"},
#endif
         // Bloom
         {39, "Bloom10"},
         {40, "Bloom8"},
         {41, "Bloom12"},
         {42, "Bloom16"},
         {43, "Bloom8(addall)"},
         {44, "Bloom12(addall)"},
         {45, "Bloom16(addall)"},
         {46, "BranchlessBloom8(addall)"},
         {47, "BranchlessBloom12(addall)"},
         {48, "BranchlessBloom16(addall)"},
         // Blocked Bloom
         {50, "BlockedBloom(simple)"},
#ifdef __aarch64__
         {51, "BlockedBloom"},
         {52, "BlockedBloom(addall)"},
#elif defined(__AVX2__)
         {51, "BlockedBloom"},
         {52, "BlockedBloom(addall)"},
         {53, "BlockedBloom64"},
#endif
#ifdef __SSE41__
         {54, "BlockedBloom16"},
#endif

         // Counting Bloom
         {60, "CountingBloom10(addall)"},
         {61, "SuccCountingBloom10(addall)"},
         {62, "SuccCountBlockBloom10"},
         {63, "SuccCountBlockBloomRank10"},

         {70, "Xor8-singleheader"},

         {90, "XorFuse8"},
         {91, "XorFuse16"},

         {101, "Xor1(NBitArray)"},
         {103, "Xor3(NBitArray)"},
         {105, "Xor5(NBitArray)"},
         {107, "Xor7(NBitArray)"},
         {109, "Xor9(NBitArray)"},
         {111, "Xor11(NBitArray)"},
         {113, "Xor13(NBitArray)"},
         {115, "Xor15(NBitArray)"},

         {205, "XorPlus5(NBitArray)"},
         {207, "XorPlus7(NBitArray)"},
         {209, "XorPlus9(NBitArray)"},
         {211, "XorPlus11(NBitArray)"},
         {213, "XorPlus13(NBitArray)"},
         {215, "XorPlus15(NBitArray)"},

         {308, "Cuckoo8(Extra5Pct)"},
         {310, "Cuckoo10(Extra5Pct)"},
         {312, "Cuckoo12(Extra5Pct)"},
         {314, "Cuckoo14(Extra5Pct)"},
         {316, "Cuckoo16(Extra5Pct)"},

         {802, "TwoBlockBloom2K(Rocks)"},
         {803, "TwoBlockBloom3K(Rocks)"},
         {804, "TwoBlockBloom4K(Rocks)"},
         {805, "TwoBlockBloom5K(Rocks)"},
         {806, "TwoBlockBloom6K(Rocks)"},
         {807, "TwoBlockBloom7K(Rocks)"},
         {808, "TwoBlockBloom8K(Rocks)"},
         {809, "TwoBlockBloom9K(Rocks)"},
         {810, "TwoBlockBloom10K(Rocks)"},
         {811, "TwoBlockBloom11K(Rocks)"},
         {812, "TwoBlockBloom12K(Rocks)"},
         {813, "TwoBlockBloom13K(Rocks)"},
         {814, "TwoBlockBloom14K(Rocks)"},
         {815, "TwoBlockBloom15K(Rocks)"},
         {816, "TwoBlockBloom16K(Rocks)"},

         {901, "BlockedBloom1K(Rocks)"},
         {902, "BlockedBloom2K(Rocks)"},
         {903, "BlockedBloom3K(Rocks)"},
         {904, "BlockedBloom4K(Rocks)"},
         {905, "BlockedBloom5K(Rocks)"},
         {906, "BlockedBloom6K(Rocks)"},
         {907, "BlockedBloom7K(Rocks)"},
         {908, "BlockedBloom8K(Rocks)"},
         {909, "BlockedBloom9K(Rocks)"},
         {910, "BlockedBloom10K(Rocks)"},
         {911, "BlockedBloom11K(Rocks)"},
         {912, "BlockedBloom12K(Rocks)"},
         {913, "BlockedBloom13K(Rocks)"},
         {914, "BlockedBloom14K(Rocks)"},
         {915, "BlockedBloom15K(Rocks)"},
         {916, "BlockedBloom16K(Rocks)"},
         {917, "BlockedBloom17K(Rocks)"},
         {999, "BlockedBloom6KCompare(Rocks)"},

         {1014, "HomogRibbon16_1"},
         {1015, "HomogRibbon32_1"},
         {1016, "HomogRibbon64_1"},
         {1017, "HomogRibbon128_1"},
         {1034, "HomogRibbon16_3"},
         {1035, "HomogRibbon32_3"},
         {1036, "HomogRibbon64_3"},
         {1037, "HomogRibbon128_3"},
         {1054, "HomogRibbon16_5"},
         {1055, "HomogRibbon32_5"},
         {1056, "HomogRibbon64_5"},
         {1057, "HomogRibbon128_5"},
         {1074, "HomogRibbon16_7"},
         {1075, "HomogRibbon32_7"},
         {1076, "HomogRibbon64_7"},
         {1077, "HomogRibbon128_7"},
         {1084, "HomogRibbon16_8"},
         {1085, "HomogRibbon32_8"},
         {1086, "HomogRibbon64_8"},
         {1087, "HomogRibbon128_8"},
         {1094, "HomogRibbon16_9"},
         {1095, "HomogRibbon32_9"},
         {1096, "HomogRibbon64_9"},
         {1097, "HomogRibbon128_9"},
         {1114, "HomogRibbon16_11"},
         {1115, "HomogRibbon32_11"},
         {1116, "HomogRibbon64_11"},
         {1117, "HomogRibbon128_11"},
         {1135, "HomogRibbon32_13"},
         {1136, "HomogRibbon64_13"},
         {1137, "HomogRibbon128_13"},
         {1155, "HomogRibbon32_15"},
         {1156, "HomogRibbon64_15"},
         {1157, "HomogRibbon128_15"},
         {1165, "HomogRibbon32_16"},
         {1166, "HomogRibbon64_16"},
         {1167, "HomogRibbon128_16"},
         {1275, "HomogRibbon32_2.7"},
         {1276, "HomogRibbon64_2.7"},
         {1335, "HomogRibbon32_3.3"},
         {1336, "HomogRibbon64_3.3"},
         {1774, "HomogRibbon16_7.7"},
         {1775, "HomogRibbon32_7.7"},
         {1776, "HomogRibbon64_7.7"},
         {1777, "HomogRibbon128_7.7"},

         {2015, "BalancedRibbon32Pack_1"},
         {2016, "BalancedRibbon64Pack_1"},
         {2035, "BalancedRibbon32Pack_3"},
         {2036, "BalancedRibbon64Pack_3"},
         {2055, "BalancedRibbon32Pack_5"},
         {2056, "BalancedRibbon64Pack_5"},
         {2071, "BalancedRibbon32_25PctPad_7"},
         {2072, "BalancedRibbon32_20PctPad_7"},
         {2073, "BalancedRibbon32_15PctPad_7"},
         {2074, "BalancedRibbon32_10PctPad_7"},
         {2075, "BalancedRibbon32Pack_7"},
         {2076, "BalancedRibbon64Pack_7"},
         {2077, "BalancedRibbon128Pack_7"},
         {2085, "BalancedRibbon32Pack_8"},
         {2086, "BalancedRibbon64Pack_8"},
         {2095, "BalancedRibbon32Pack_9"},
         {2096, "BalancedRibbon64Pack_9"},
         {2115, "BalancedRibbon32Pack_11"},
         {2116, "BalancedRibbon64Pack_11"},
         {2117, "BalancedRibbon128Pack_11"},
         {2135, "BalancedRibbon32Pack_13"},
         {2136, "BalancedRibbon64Pack_13"},
         {2155, "BalancedRibbon32Pack_15"},
         {2156, "BalancedRibbon64Pack_15"},
         {2775, "BalancedRibbon32Pack_7.7"},
         {2776, "BalancedRibbon64Pack_7.7"},

         {3016, "StandardRibbon64_1"},
         {3017, "StandardRibbon128_1"},
         {3036, "StandardRibbon64_3"},
         {3037, "StandardRibbon128_3"},
         {3056, "StandardRibbon64_5"},
         {3057, "StandardRibbon128_5"},
         {3072, "StandardRibbon64_25PctPad_7"},
         {3073, "StandardRibbon64_20PctPad_7"},
         {3074, "StandardRibbon64_15PctPad_7"},
         {3075, "StandardRibbon64_10PctPad_7"},
         {3076, "StandardRibbon64_7"},
         {3077, "StandardRibbon128_7"},
         {3086, "StandardRibbon64_8"},
         {3087, "StandardRibbon128_8"},
         {3088, "StandardRibbon64_8_Smash"},
         {3089, "StandardRibbon128_8_Smash"},
         {3096, "StandardRibbon64_9"},
         {3097, "StandardRibbon128_9"},
         {3116, "StandardRibbon64_11"},
         {3117, "StandardRibbon128_11"},
         {3136, "StandardRibbon64_13"},
         {3137, "StandardRibbon128_13"},
         {3156, "StandardRibbon64_15"},
         {3157, "StandardRibbon128_15"},
         {3776, "StandardRibbon64_7.7"},
         {3777, "StandardRibbon128_7.7"},

         // Stefan Walzer's 1-bit retrieval tests
         {8000, "TwoBlock10k_1"},
         {8001, "TwoBlock20k_1"},
         {8010, "LMSS_12_0.91_1"},
         {8011, "LMSS_150_0.99_1"},
         {8012, "LMSS_800_0.997_1"},
         {8013, "LMSS_12_0.91_8"},
         {8014, "LMSS_150_0.99_8"},
         {8015, "LMSS_800_0.997_8"},
         {8016, "LMSS_12_0.91_16"},
         {8017, "LMSS_150_0.99_16"},
         {8018, "LMSS_800_0.997_16"},
         {8020, "GOV3_0.91_10k_pack_1"},
         {8021, "GOV3_0.9_1k_compress_1"},
         {8022, "GOV4_0.97_10k_pack_1"},
         {8023, "GOV4_0.96_1k_compress_1"},
         {8030, "Coupled_3_1"},
         {8031, "Coupled_4_1"},
         {8032, "Coupled_7_1"},
         {8033, "Coupled_3_8"},
         {8034, "Coupled_4_8"},
         {8035, "Coupled_7_8"},
         {8036, "Coupled_3_16"},
         {8037, "Coupled_4_16"},
         {8038, "Coupled_7_16"},
         {8060, "BPZ_0.81_1"},
         {8061, "BPZ_0.81_8"},
         {8062, "BPZ_0.81_16"},

         {8500, "QuotientFilter7"},
         {8501, "QuotientFilter10"},
         {8502, "QuotientFilter13"},

         // Sort
         {9000, "Sort"},

         // At the end because it tends to crash
         {9800, "Morton1_8"},
         {9801, "Morton3_8"},
         {9802, "Morton7_8"},
         // {9803, "Morton15_8"},

         {9811, "Morton3_6"},
         {9812, "Morton7_6"},
         {9813, "Morton15_6"},

         {9821, "Morton3_12"},
         {9822, "Morton7_12"},
         {9823, "Morton15_12"}};

    auto addl = [&names](ribbon::ThreshMode mode, bool sparse, unsigned coeffbits,
                         unsigned resbits, bool interleaved, bool cls) {
        int id = 4000 + 1000 * interleaved + 2000 * cls +
                 200 * static_cast<int>(mode) + 100 * sparse + coeffbits + resbits;
        // hack to prevent conflicts
        if (coeffbits == 128)
            id -= 32;
        std::stringstream name;
        name << "BumpRibbon"
             << (mode == ribbon::ThreshMode::onebit
                     ? "1B"
                     : (mode == ribbon::ThreshMode::twobit ? "2B" : ""))
             << (sparse ? "_SC" : "") << (interleaved ? "_int" : "")
             << (cls ? "_cls" : "") << "_" << coeffbits << "_" << resbits;
        if (names.find(id) != names.end()) {
            cerr << "dupe for id " << id << " already have " << names[id]
                 << " want to add " << name.str() << std::endl;
        } else {
            names[id] = name.str();
        }
    };

    auto addseq = [&addl](ribbon::ThreshMode mode, bool sparse,
                          bool interleaved, bool cls) {
        for (size_t logcb = 4; logcb <= 7; ++logcb) {
            unsigned coeffbits = (1u << logcb);
            auto resbits =
                interleaved
                    ? std::vector<unsigned>{1, 3, 5, 7, 8, 9, 11, 13, 15, 16}
                    : std::vector<unsigned>{1, 2, 4, 8, 16};
            for (unsigned rb : resbits) {
                addl(mode, sparse, coeffbits, rb, interleaved, cls);
            }
        }
    };

    // sparse + interleaved makes no sense -> interleaved only non-sparse
    addseq(ribbon::ThreshMode::onebit, false, true, false);
    addseq(ribbon::ThreshMode::twobit, false, true, false);
    addseq(ribbon::ThreshMode::normal, false, true, false);

    for (bool sparse : {false, true}) {
        for (bool cls : {false, true}) {
            addseq(ribbon::ThreshMode::onebit, sparse, false, cls);
            addseq(ribbon::ThreshMode::twobit, sparse, false, cls);
            addseq(ribbon::ThreshMode::normal, sparse, false, cls);
        }
    }

    // Parameter Parsing
    // ----------------------------------------------------------

    if (argc < 2) {
        cout << "Usage: " << argv[0]
             << " <numberOfEntries> [<algorithmId> [<seed>]]" << endl;
        cout << " numberOfEntries: number of keys, we recommend at least "
                "100000000"
             << endl;
        cout << " algorithmId: -1 for all default algos, or 0..n to only run "
                "this "
                "algorithm"
             << endl;
        cout << " algorithmId: can also be a comma-separated list of "
                "non-negative "
                "integers"
             << endl;
        for (auto i : names) {
            cout << "           " << i.first << " : " << i.second << endl;
        }
        cout << " algorithmId: can also be set to the string 'all' if you want "
                "to "
                "run them all, including some that are excluded by default"
             << endl;
        cout << " seed: seed for the PRNG; -1 for random seed (default)" << endl;
        return 1;
    }
    stringstream input_string(argv[1]);
    size_t add_count;
    input_string >> add_count;
    if (input_string.fail()) {
        cerr << "Invalid number: " << argv[1];
        return 2;
    }
    int algorithmId = -1; // -1 is just the default
    std::set<int> algos;
    if (argc > 2) {
        if (strcmp(argv[2], "all") == 0) {
            for (auto i : names) { // we add all the named algos.
                algos.insert(i.first);
            }
        } else if (strstr(argv[2], ",") != NULL) {
            // we have a list of algos
            algorithmId = 9999999; // disabling
            parse_comma_separated(argv[2], algos);
            if (algos.size() == 0) {
                cerr << " no algo selected " << endl;
                return -3;
            }
        } else {
            // we select just one
            stringstream input_string_2(argv[2]);
            input_string_2 >> algorithmId;
            if (input_string_2.fail()) {
                cerr << "Invalid number: " << argv[2];
                return 2;
            }
        }
    }
    int int_seed = -1;
    if (argc > 3) {
        stringstream input_string_3(argv[3]);
        input_string_3 >> int_seed;
        if (input_string_3.fail()) {
            cerr << "Invalid number: " << argv[3];
            return 2;
        }
    }
    if (int_seed == -1) {
        int_seed = std::random_device{}();
        cout << "Using seed " << int_seed << endl;
    }
    // Turn the seed into a positive number
    uint64_t seed = static_cast<int64_t>(int_seed) -
                    static_cast<int64_t>(std::numeric_limits<int>::min());
    size_t actual_sample_size = std::min(add_count, MAX_SAMPLE_SIZE);

    // Generating Samples
    // ----------------------------------------------------------
    auto [to_add, mixed_sets, found_probabilities, intersectionsize] =
        GenerateInputs(add_count, seed, actual_sample_size);

    // make sure the seed for queries is different than for input generation
    seed ^= (seed * 1337) + 13579;

    constexpr int NAME_WIDTH = 32;
    cout << StatisticsTableHeader(NAME_WIDTH, found_probabilities) << endl;

    std::fstream outfile;
    outfile.open(outfn_base + std::to_string(add_count) + "_" +
                     std::to_string(seed) + ".txt",
                 ios::out);
    const auto print = [&](const std::string &name, const Statistics &stats) {
        cout << setw(NAME_WIDTH) << name << stats << endl;
        outfile << "RESULT name=" << name;
        stats.printKV(outfile);
        outfile << endl;
    };

    // Algorithms ----------------------------------------------------------
    int a;

    // Xor ----------------------------------------------------------
    a = 0;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<
            xorfilter::XorFilter<uint64_t, uint8_t, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf =
            FilterBenchmark<xorfilter2::XorFilter2<uint64_t, uint32_t, UInt12Array,
                                                   hashing::SimpleXorMul>>()(
                add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<
            xorfilter::XorFilter<uint64_t, uint16_t, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<
            xorfilter_plus::XorFilterPlus<uint64_t, uint8_t, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 4;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<
            xorfilter_plus::XorFilterPlus<uint64_t, uint16_t, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 5;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf =
            FilterBenchmark<xorfilter::XorFilter10<uint64_t, hashing::SimpleXorMul>>()(
                add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 6;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf =
            FilterBenchmark<xorfilter::XorFilter10_666<uint64_t, hashing::SimpleXorMul>>()(
                add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 7;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<xorfilter2::XorFilter2<
            uint64_t, uint16_t, NBitArray<uint16_t, 10>, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<xorfilter2::XorFilter2<
            uint64_t, uint16_t, NBitArray<uint16_t, 14>, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 9;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<xorfilter2n::XorFilter2n<
            uint64_t, uint8_t, UIntArray<uint8_t>, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }

    // Cuckoo ----------------------------------------------------------
    a = 10;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<cuckoofilter::CuckooFilterStable<
            uint64_t, 8, cuckoofilter::SingleTable, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 11;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<cuckoofilter::CuckooFilterStable<
            uint64_t, 10, cuckoofilter::SingleTable, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 12;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<cuckoofilter::CuckooFilterStable<
            uint64_t, 12, cuckoofilter::SingleTable, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 13;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<cuckoofilter::CuckooFilterStable<
            uint64_t, 14, cuckoofilter::SingleTable, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 14;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<cuckoofilter::CuckooFilterStable<
            uint64_t, 16, cuckoofilter::SingleTable, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 15;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<cuckoofilter::CuckooFilterStable<
            uint64_t, 13, cuckoofilter::PackedTable, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 16;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<cuckoofilter::CuckooFilter<
            uint64_t, 8, cuckoofilter::SingleTable, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 17;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<cuckoofilter::CuckooFilter<
            uint64_t, 12, cuckoofilter::SingleTable, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 18;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<cuckoofilter::CuckooFilter<
            uint64_t, 16, cuckoofilter::SingleTable, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 19;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<cuckoofilter::CuckooFilter<
            uint64_t, 13, cuckoofilter::PackedTable, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }

    // GCS ----------------------------------------------------------
    a = 20;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf =
            FilterBenchmark<gcsfilter::GcsFilter<uint64_t, 8, hashing::SimpleXorMul>>()(
                add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }

    // CQF ----------------------------------------------------------
#ifdef __AVX2__
    a = 30;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf =
            FilterBenchmark<gqfilter::GQFilter<uint64_t, 8, hashing::SimpleXorMul>>()(
                add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 31;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf =
            FilterBenchmark<gqfilter::GQFilter<uint64_t, 7, hashing::SimpleXorMul>>()(
                add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 32;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf =
            FilterBenchmark<gqfilter::GQFilter<uint64_t, 10, hashing::SimpleXorMul>>()(
                add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 33;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf =
            FilterBenchmark<gqfilter::GQFilter<uint64_t, 13, hashing::SimpleXorMul>>()(
                add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
#endif

    // Bloom ----------------------------------------------------------
    a = 39;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<
            bloomfilter::BloomFilter<uint64_t, 10, false, hashing::NoopHash>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 40;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<
            bloomfilter::BloomFilter<uint64_t, 8, false, hashing::NoopHash>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 41;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<
            bloomfilter::BloomFilter<uint64_t, 12, false, hashing::NoopHash>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 42;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<
            bloomfilter::BloomFilter<uint64_t, 16, false, hashing::NoopHash>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 43;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<
            bloomfilter::BloomFilter<uint64_t, 8, false, hashing::NoopHash>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 44;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<
            bloomfilter::BloomFilter<uint64_t, 12, false, hashing::NoopHash>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 45;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<
            bloomfilter::BloomFilter<uint64_t, 16, false, hashing::NoopHash>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 46;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<
            bloomfilter::BloomFilter<uint64_t, 8, true, hashing::NoopHash>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 47;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<
            bloomfilter::BloomFilter<uint64_t, 12, true, hashing::NoopHash>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 48;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<
            bloomfilter::BloomFilter<uint64_t, 16, true, hashing::NoopHash>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }

    a = 48;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<
            bloomfilter::BloomFilter<uint64_t, 16, true, hashing::NoopHash>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }

    // Blocked Bloom ----------------------------------------------------------
    a = 50;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf =
            FilterBenchmark<bloomfilter::SimpleBlockFilter<8, 8, hashing::NoopHash>>()(
                add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
#ifdef __aarch64__
    a = 51;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<SimdBlockFilterFixed<NoopHash>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 52;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<SimdBlockFilterFixed<NoopHash>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
#endif
#ifdef __AVX2__
    a = 51;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<SimdBlockFilterFixed<hashing::NoopHash>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 52;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<SimdBlockFilterFixed<hashing::NoopHash>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 53;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<SimdBlockFilterFixed64<hashing::NoopHash>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
#endif
#ifdef __SSE41__
    a = 54;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<SimdBlockFilterFixed16<NoopHash>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
#endif

    // Counting Bloom ----------------------------------------------------------
    a = 60;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<counting_bloomfilter::CountingBloomFilter<
            uint64_t, 10, true, hashing::NoopHash>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 61;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<counting_bloomfilter::SuccinctCountingBloomFilter<
            uint64_t, 10, true, hashing::NoopHash>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 62;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf =
            FilterBenchmark<counting_bloomfilter::SuccinctCountingBlockedBloomFilter<
                uint64_t, 10, hashing::NoopHash>>()(
                add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 63;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf =
            FilterBenchmark<counting_bloomfilter::SuccinctCountingBlockedBloomRankFilter<
                uint64_t, 10, hashing::NoopHash>>()(
                add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }

    a = 70;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<XorSingle>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }

    // Xor Fuse Filter ----------------------------------------------------------
    a = 90;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf =
            FilterBenchmark<xorfusefilter::XorFuseFilter<uint64_t, uint8_t>>()(
                add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 91;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf =
            FilterBenchmark<xorfusefilter::XorFuseFilter<uint64_t, uint16_t>>()(
                add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }

    // Specific Xor/XorPlus bit widths
    a = 101;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<xorfilter2::XorFilter2<
            uint64_t, uint8_t, NBitArray<uint8_t, 1>, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 103;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<xorfilter2::XorFilter2<
            uint64_t, uint8_t, NBitArray<uint8_t, 3>, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 105;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<xorfilter2::XorFilter2<
            uint64_t, uint8_t, NBitArray<uint8_t, 5>, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 107;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<xorfilter2::XorFilter2<
            uint64_t, uint8_t, NBitArray<uint8_t, 7>, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 109;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<xorfilter2::XorFilter2<
            uint64_t, uint16_t, NBitArray<uint16_t, 9>, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 111;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<xorfilter2::XorFilter2<
            uint64_t, uint16_t, NBitArray<uint16_t, 11>, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 113;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<xorfilter2::XorFilter2<
            uint64_t, uint16_t, NBitArray<uint16_t, 13>, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 115;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<xorfilter2::XorFilter2<
            uint64_t, uint16_t, NBitArray<uint16_t, 15>, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 205;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<xorfilter_plus2::XorFilterPlus2<
            uint64_t, uint8_t, NBitArray<uint8_t, 5>, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 207;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<xorfilter_plus2::XorFilterPlus2<
            uint64_t, uint8_t, NBitArray<uint8_t, 7>, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 209;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<xorfilter_plus2::XorFilterPlus2<
            uint64_t, uint16_t, NBitArray<uint16_t, 9>, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 211;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<xorfilter_plus2::XorFilterPlus2<
            uint64_t, uint16_t, NBitArray<uint16_t, 11>, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 213;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<xorfilter_plus2::XorFilterPlus2<
            uint64_t, uint16_t, NBitArray<uint16_t, 13>, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 215;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<xorfilter_plus2::XorFilterPlus2<
            uint64_t, uint16_t, NBitArray<uint16_t, 15>, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    // Cuckoo (Extra5Pct) --------------------------------------------------
    a = 308;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<CuckooFilterStablePad<
            uint64_t, 8, 5, cuckoofilter::SingleTable, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 310;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<CuckooFilterStablePad<
            uint64_t, 10, 5, cuckoofilter::SingleTable, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 312;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<CuckooFilterStablePad<
            uint64_t, 12, 5, cuckoofilter::SingleTable, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 314;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<CuckooFilterStablePad<
            uint64_t, 14, 5, cuckoofilter::SingleTable, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 316;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<CuckooFilterStablePad<
            uint64_t, 16, 5, cuckoofilter::SingleTable, hashing::SimpleXorMul>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }

    // TwoBlockBloom(Rocks)
#define ADD(k)                                                                 \
    a = 800 + k;                                                               \
    if (algorithmId == a || (algos.find(a) != algos.end())) {                  \
        auto cf = FilterBenchmark<RocksBloomFilter<k, 2>>()(                   \
            add_count, to_add, intersectionsize, mixed_sets, false, seed);     \
        print(names[a], cf);                                                   \
    }
    ADD(2);
    ADD(3);
    ADD(4);
    ADD(5);
    ADD(6);
    ADD(7);
    ADD(8);
    ADD(9);
    ADD(10);
    ADD(11);
    ADD(12);
    ADD(13);
    ADD(14);
    ADD(15);
    ADD(16);

    // BlockedBloom(Rocks)
#undef ADD
#define ADD(k)                                                                 \
    a = 900 + k;                                                               \
    if (algorithmId == a || (algos.find(a) != algos.end())) {                  \
        auto cf = FilterBenchmark<RocksBloomFilter<k, 1>>()(                   \
            add_count, to_add, intersectionsize, mixed_sets, false, seed);     \
        print(names[a], cf);                                                   \
    }
    ADD(1);
    ADD(2);
    ADD(3);
    ADD(4);
    ADD(5);
    ADD(6);
    ADD(7);
    ADD(8);
    ADD(9);
    ADD(10);
    ADD(11);
    ADD(12);
    ADD(13);
    ADD(14);
    ADD(15);
    ADD(16);

    // For direct comparison with BlockedBloom64
    a = 999;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<RocksBloomFilter<6, 1, 10240>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }

    // Homogeneous Ribbon
    a = 1014;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint16_t, 1>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1015;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint32_t, 1>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1016;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint64_t, 1>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1017;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<r2::Unsigned128, 1>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1034;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint16_t, 3>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1035;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint32_t, 3>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1036;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint64_t, 3>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1037;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<r2::Unsigned128, 3>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1054;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint16_t, 5>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1055;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint32_t, 5>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1056;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint64_t, 5>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1057;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<r2::Unsigned128, 5>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1074;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint16_t, 7>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1075;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint32_t, 7>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1076;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint64_t, 7>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1077;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<r2::Unsigned128, 7>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1084;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint16_t, 8>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1085;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint32_t, 8>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1086;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint64_t, 8>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1087;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<r2::Unsigned128, 8>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1094;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint16_t, 9>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1095;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint32_t, 9>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1096;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint64_t, 9>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1097;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<r2::Unsigned128, 9>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1114;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint16_t, 11>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1115;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint32_t, 11>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1116;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint64_t, 11>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1117;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<r2::Unsigned128, 11>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1135;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint32_t, 13>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1136;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint64_t, 13>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1137;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<r2::Unsigned128, 13>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1155;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint32_t, 15>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1156;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint64_t, 15>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1157;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<r2::Unsigned128, 15>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1165;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint32_t, 16>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1166;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint64_t, 16>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1167;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<r2::Unsigned128, 16>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1275;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint32_t, 0, 2700>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1276;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint64_t, 0, 2700>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1335;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint32_t, 0, 3300>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1336;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint64_t, 0, 3300>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1774;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint16_t, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1775;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint32_t, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1776;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<uint64_t, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 1777;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<HomogRibbonFilter<r2::Unsigned128, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }

    // BalancedRibbon
    a = 2015;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint32_t, 1, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2016;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint64_t, 1, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2035;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint32_t, 3, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2036;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint64_t, 3, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2055;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint32_t, 5, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2056;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint64_t, 5, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2071;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint32_t, 7, 25>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2072;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint32_t, 7, 20>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2073;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint32_t, 7, 15>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2074;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint32_t, 7, 10>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2075;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint32_t, 7, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2076;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint64_t, 7, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2077;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<r2::Unsigned128, 7, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2085;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint32_t, 8, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2086;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint64_t, 8, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2095;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint32_t, 9, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2096;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint64_t, 9, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2115;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint32_t, 11, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2116;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint64_t, 11, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2117;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<r2::Unsigned128, 11, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2135;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint32_t, 13, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2136;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint64_t, 13, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2155;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint32_t, 15, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2156;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint64_t, 15, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2775;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint32_t, 0, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 2776;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BalancedRibbonFilter<uint64_t, 0, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }

    // StandardRibbon
    a = 3016;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<uint64_t, 1, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3017;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<r2::Unsigned128, 1, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3036;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<uint64_t, 3, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3037;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<r2::Unsigned128, 3, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3056;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<uint64_t, 5, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3057;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<r2::Unsigned128, 5, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3072;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<uint64_t, 7, 25>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3073;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<uint64_t, 7, 20>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3074;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<uint64_t, 7, 15>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3075;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<uint64_t, 7, 10>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3076;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<uint64_t, 7, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3077;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<r2::Unsigned128, 7, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3086;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<uint64_t, 7, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3087;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<r2::Unsigned128, 7, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3088;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<uint64_t, 7, 0, true>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3089;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf =
            FilterBenchmark<StandardRibbonFilter<r2::Unsigned128, 7, 0, true>>()(
                add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3096;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<uint64_t, 9, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3097;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<r2::Unsigned128, 9, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3116;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<uint64_t, 11, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3117;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<r2::Unsigned128, 11, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3136;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<uint64_t, 13, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3137;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<r2::Unsigned128, 13, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3156;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<uint64_t, 15, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3157;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<r2::Unsigned128, 15, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3776;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<uint64_t, 0, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 3777;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<StandardRibbonFilter<r2::Unsigned128, 0, 0>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }

#undef ADD
// bumping ribbon
#define ADD(mode, sparse, coeffbits, resbits, interleaved, cls)                  \
    a = 4000 + 1000 * interleaved + 2000 * cls +                                 \
        200 * static_cast<int>(mode) + 100 * sparse + coeffbits + resbits;       \
    if (coeffbits == 128)                                                        \
        a -= 32;                                                                 \
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) { \
        auto cf = FilterBenchmark<BumpRibbonFilter<3, mode, sparse, interleaved, \
                                                   cls, coeffbits, resbits>>()(  \
            add_count, to_add, intersectionsize, mixed_sets, true, seed);        \
        print(names[a], cf);                                                     \
    }

#define ADD_MODES(sparse, coeffbits, resbits, interleaved, cls)                    \
    ADD(ribbon::ThreshMode::normal, sparse, coeffbits, resbits, interleaved, cls); \
    ADD(ribbon::ThreshMode::onebit, sparse, coeffbits, resbits, interleaved, cls); \
    ADD(ribbon::ThreshMode::twobit, sparse, coeffbits, resbits, interleaved, cls);

// interleaved: different bit numbers; only non-sparse
#define ADD_INT(coeffbits)                                                     \
    ADD_MODES(false, coeffbits, 1, /* int */ true, /* cls */ false);           \
    ADD_MODES(false, coeffbits, 3, /* int */ true, /* cls */ false);           \
    ADD_MODES(false, coeffbits, 5, /* int */ true, /* cls */ false);           \
    ADD_MODES(false, coeffbits, 7, /* int */ true, /* cls */ false);           \
    ADD_MODES(false, coeffbits, 8, /* int */ true, /* cls */ false);           \
    ADD_MODES(false, coeffbits, 9, /* int */ true, /* cls */ false);           \
    ADD_MODES(false, coeffbits, 11, /* int */ true, /* cls */ false);          \
    ADD_MODES(false, coeffbits, 13, /* int */ true, /* cls */ false);          \
    ADD_MODES(false, coeffbits, 15, /* int */ true, /* cls */ false);          \
    ADD_MODES(false, coeffbits, 16, /* int */ true, /* cls */ false);

// full-byte configs
#define ADD_RBB(sparse, coeffbits, interleaved, cls)                           \
    ADD_MODES(sparse, coeffbits, 8, interleaved, cls);                         \
    ADD_MODES(sparse, coeffbits, 16, interleaved, cls);

    // only full-byte configs
#define ADD_SEQB(sparse, interleaved, cls)                                     \
    ADD_RBB(sparse, 32, interleaved, cls);                                     \
    ADD_RBB(sparse, 64, interleaved, cls);

    // Basic storage, sparse + dense.  Only run r=8 and r=16 because the others
    // are not implemented properly (wasted bits galore)
    ADD_SEQB(false, false, false);
    ADD_SEQB(true, false, false);

    // Interleaved storage, dense only, r=1,3,5,7,8,9,11,13,15,16
    ADD_INT(32);
    ADD_INT(64);
    ADD_INT(128);

    // Cache-Line Storage, sparse and dense. Only run r=8 and r=16 because the
    // others are not implemented properly (wasted bits galore)
    ADD_SEQB(false, false, true);
    // ADD_SEQB(true, false, true);

    // Stefan Walzer's things
    // TwoBlock
    a = 8000;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<TwoBlockRetrieval<10000>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8001;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<TwoBlockRetrieval<20000>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    // LMSS
    a = 8010;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<LMSSRetrieval<12, 900>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8011;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<LMSSRetrieval<150, 990>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8012;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<LMSSRetrieval<800, 997>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    // LMSS 8-bit
    a = 8013;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<LMSSRetrieval<12, 900, uint8_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8014;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<LMSSRetrieval<150, 990, uint8_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8015;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<LMSSRetrieval<800, 997, uint8_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    // LMSS 16-bit
    a = 8016;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<LMSSRetrieval<12, 900, uint16_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8017;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<LMSSRetrieval<150, 990, uint16_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8018;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<LMSSRetrieval<800, 997, uint16_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    // GOV
    a = 8020;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<GovRetrieval<3, 910, 10000, false>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8021;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<GovRetrieval<3, 900, 1000, true>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8022;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<GovRetrieval<4, 970, 10000, false>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8023;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<GovRetrieval<4, 960, 1000, true>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    // coupled
    /*
    a = 8030;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<CoupledRetrieval<3, 120, 905>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8031;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<CoupledRetrieval<4, 120, 960>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8032;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<CoupledRetrieval<7, 120, 979>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    // coupled r=8
    a = 8033;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<CoupledRetrieval<3, 120, 905, uint8_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8034;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<CoupledRetrieval<4, 120, 960, uint8_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8035;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<CoupledRetrieval<7, 120, 979, uint8_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    // coupled r=16
    a = 8036;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<CoupledRetrieval<3, 120, 905, uint16_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8037;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<CoupledRetrieval<4, 120, 960, uint16_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8038;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<CoupledRetrieval<7, 120, 979, uint16_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    // coupled z=60 (for n=10^6)
    a = 8040;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<CoupledRetrieval<3, 60, 885>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8041;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<CoupledRetrieval<4, 60, 940>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8042;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<CoupledRetrieval<7, 60, 959>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    // coupled r=8
    a = 8043;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<CoupledRetrieval<3, 60, 885, uint8_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8044;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<CoupledRetrieval<4, 60, 940, uint8_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8045;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<CoupledRetrieval<7, 60, 959, uint8_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    // coupled r=16
    a = 8046;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<CoupledRetrieval<3, 60, 885, uint16_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8047;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<CoupledRetrieval<4, 60, 940, uint16_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8048;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<CoupledRetrieval<7, 60, 959, uint16_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    */
    // Simple Coupled
    a = 8030;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<SimpleCoupledRetrieval<3, bool>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8031;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<SimpleCoupledRetrieval<4, bool>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8032;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<SimpleCoupledRetrieval<7, bool>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8033;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<SimpleCoupledRetrieval<3, uint8_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8034;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<SimpleCoupledRetrieval<4, uint8_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8035;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<SimpleCoupledRetrieval<7, uint8_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8036;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<SimpleCoupledRetrieval<3, uint16_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8037;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<SimpleCoupledRetrieval<4, uint16_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8038;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<SimpleCoupledRetrieval<7, uint16_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }


    // BPZ (basically, Xor as retrieval)
    a = 8040;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BPZRetrieval<810, bool>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8041;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BPZRetrieval<810, uint8_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 8042;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<BPZRetrieval<810, uint16_t>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }

    // Quotient filter
    a = 8500;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<QuotientFilter<7>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 8501;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<QuotientFilter<10>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }
    a = 8502;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<QuotientFilter<13>>()(
            add_count, to_add, intersectionsize, mixed_sets, false, seed);
        print(names[a], cf);
    }


    // Sort ----------------------------------------------------------
    a = 9000;
    if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
        const auto start_time = NowNanos();
        const auto size = sorter(to_add);
        const auto sort_time = NowNanos() - start_time;
        std::cout << "Sort time: " << sort_time * 1.0 / size << " ns/key"
                  << std::endl;
    }

    // 8-bit Morton filters
    /*
    a = 9800;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<MortonFilter<CompressedCuckoo::Morton1_8>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    */
    a = 9801;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<MortonFilter<CompressedCuckoo::Morton3_8>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 9802;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<MortonFilter<CompressedCuckoo::Morton7_8>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    /*
    a = 9803;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        // This is crashy! It segfaults every time.
        auto cf = FilterBenchmark<MortonFilter<CompressedCuckoo::Morton15_8>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    */

    // 6-bit Morton filters
    /*
    a = 9811;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<MortonFilter<CompressedCuckoo::Morton3_6>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }

    a = 9812;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<MortonFilter<CompressedCuckoo::Morton7_6>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    a = 9813;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<MortonFilter<CompressedCuckoo::Morton15_6>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    */

    // 12-bit Morton filters
    /*
    a = 9821;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<MortonFilter<CompressedCuckoo::Morton3_12>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    */
    a = 9822;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<MortonFilter<CompressedCuckoo::Morton7_12>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    /*
    a = 9823;
    if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<MortonFilter<CompressedCuckoo::Morton15_12>>()(
            add_count, to_add, intersectionsize, mixed_sets, true, seed);
        print(names[a], cf);
    }
    */

    outfile.close();
    return 0;
}
