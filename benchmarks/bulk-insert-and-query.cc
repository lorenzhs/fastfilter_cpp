// This benchmark reports on the bulk insert and bulk query rates. It is invoked as:
//
//     ./bulk-insert-and-query.exe 158000
//
// That invocation will test each probabilistic membership container type with 158000
// randomly generated items. It tests bulk Add() from empty to full and Contain() on
// filters with varying rates of expected success. For instance, at 75%, three out of
// every four values passed to Contain() were earlier Add()ed.
//
// Example usage:
//
// for alg in `seq 0 1 14`; do for num in `seq 10 10 200`; do ./bulk-insert-and-query.exe ${num}000000 ${alg}; done; done > results.txt

#include <climits>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <vector>
#include <random>
#include <set>
#include <memory>
#include <stdio.h>

// morton
#include "compressed_cuckoo_filter.h"
#include "morton_sample_configs.h"

#include "cuckoofilter.h"
#include "cuckoofilter_stable.h"
#include "xorfilter.h"
#include "xorfilter_10bit.h"
#include "xorfilter_13bit.h"
#include "xorfilter_10_666bit.h"
#include "xorfilter_2.h"
#include "xorfilter_2n.h"
#include "xorfilter_plus.h"
#include "xorfilter_plus2.h"
#include "xorfilter_singleheader.h"
#include "xor_fuse_filter.h"
#include "bloom.h"
#include "counting_bloom.h"
#include "gcs.h"
#ifdef __AVX2__
#include "gqf_cpp.h"
#include "simd-block.h"
#endif
#include "random.h"
#include "simd-block-fixed-fpp.h"
#include "timing.h"
#include "linux-perf-events.h"
#include "ribbon_impl.h"
#include "bloom_impl.h"

using namespace std;
using namespace hashing;
using namespace cuckoofilter;
using namespace xorfilter;
using namespace xorfilter2;
using namespace xorfilter2n;
using namespace xorfilter_plus;
using namespace xorfilter_plus2;
using namespace xorfusefilter;
using namespace bloomfilter;
using namespace counting_bloomfilter;
using namespace gcsfilter;
using namespace CompressedCuckoo; // Morton filter namespace
#ifdef __AVX2__
using namespace gqfilter;
#endif
using namespace ribbon;

// The number of items sampled when determining the lookup performance
const size_t MAX_SAMPLE_SIZE = 10 * 1000 * 1000;

// The statistics gathered for each table type:
struct Statistics {
  size_t add_count;
  double nanos_per_add;
  double nanos_per_remove;
  // key: percent of queries that were expected to be positive
  map<int, double> nanos_per_finds;
  double false_positive_probabilty;
  double bits_per_item;
};

// Inlining the "contains" which are executed within a tight loop can be both
// very detrimental or very beneficial, and which ways it goes depends on the
// compiler. It is unclear whether we want to benchmark the inlining of Contains,
// as it depends very much on how "contains" is used. So it is maybe reasonable
// to benchmark it without inlining.
//
#define CONTAIN_ATTRIBUTES  __attribute__ ((noinline))

// Output for the first row of the table of results. type_width is the maximum number of
// characters of the description of any table type, and find_percent_count is the number
// of different lookup statistics gathered for each table. This function assumes the
// lookup expected positive probabiilties are evenly distributed, with the first being 0%
// and the last 100%.
string StatisticsTableHeader(int type_width, const std::vector<double> &found_probabilities) {
  ostringstream os;

  os << string(type_width, ' ');
  os << setw(8) << right << "";
  os << setw(8) << right << "";
  for (size_t i = 0; i < found_probabilities.size(); ++i) {
    os << setw(8) << "find";
  }
  os << setw(8) << "1Xadd+";
  os << setw(8) << "" << setw(11) << "" << setw(11)
     << "optimal" << setw(8) << "wasted" << setw(8) << "million" << endl;

  os << string(type_width, ' ');
  os << setw(8) << right << "add";
  os << setw(8) << right << "remove";
  for (double prob : found_probabilities) {
    os << setw(8 - 1) << static_cast<int>(prob * 100.0) << '%';
  }
  os << setw(8 - 5) << found_probabilities.size() << "Xfind";
  os << setw(9) << "ε%" << setw(11) << "bits/item" << setw(11)
     << "bits/item" << setw(8) << "space%" << setw(8) << "keys";
  return os.str();
}

// Overloading the usual operator<< as used in "std::cout << foo", but for Statistics
template <class CharT, class Traits>
basic_ostream<CharT, Traits>& operator<<(
    basic_ostream<CharT, Traits>& os, const Statistics& stats) {
  os << fixed << setprecision(2) << setw(8) << right
     << stats.nanos_per_add;
  double add_and_find = stats.nanos_per_add;
  os << fixed << setprecision(2) << setw(8) << right
     << stats.nanos_per_remove;
  for (const auto& fps : stats.nanos_per_finds) {
    os << setw(8) << fps.second;
    add_and_find += fps.second;
  }
  os << setw(8) << add_and_find;

  // we get some nonsensical result for very small fpps
  if(stats.false_positive_probabilty > 0.0000001) {
    const auto minbits = log2(1 / stats.false_positive_probabilty);
    os << setw(8) << setprecision(4) << stats.false_positive_probabilty * 100
       << setw(11) << setprecision(2) << stats.bits_per_item << setw(11) << minbits
       << setw(8) << setprecision(1) << 100 * (stats.bits_per_item / minbits - 1)
       << " " << setw(7) << setprecision(3) << (stats.add_count / 1000000.);
  } else {
    os << setw(8) << setprecision(4) << stats.false_positive_probabilty * 100
       << setw(11) << setprecision(2) << stats.bits_per_item << setw(11) << 64
       << setw(8) << setprecision(1) << 0
       << " " << setw(7) << setprecision(3) << (stats.add_count / 1000000.);
  }
  return os;
}

template<typename Table>
struct FilterAPI {};

template <typename ItemType, size_t bits_per_item, template <size_t> class TableType, typename HashFamily>
struct FilterAPI<CuckooFilter<ItemType, bits_per_item, TableType, HashFamily>> {
  using Table = CuckooFilter<ItemType, bits_per_item, TableType, HashFamily>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table * table) {
    if (0 != table->Add(key)) {
      throw logic_error("The filter is too small to hold all of the elements");
    }
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
    throw std::runtime_error("Unsupported");
  }
  static void Remove(uint64_t key, Table * table) {
    table->Delete(key);
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
    return (0 == table->Contain(key));
  }
};

template <typename ItemType, size_t bits_per_item, template <size_t> class TableType, typename HashFamily>
struct FilterAPI<CuckooFilterStable<ItemType, bits_per_item, TableType, HashFamily>> {
  using Table = CuckooFilterStable<ItemType, bits_per_item, TableType, HashFamily>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table * table) {
    if (0 != table->Add(key)) {
      throw logic_error("The filter is too small to hold all of the elements");
    }
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
    throw std::runtime_error("Unsupported");
  }
  static void Remove(uint64_t key, Table * table) {
    table->Delete(key);
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
    return (0 == table->Contain(key));
  }
};

template <typename ItemType, size_t bits_per_item, size_t percent_extra_pad,
          template <size_t> class TableType = SingleTable,
          typename HashFamily = hashing::TwoIndependentMultiplyShift>
class CuckooFilterStablePad : public CuckooFilterStable<ItemType, bits_per_item, TableType, HashFamily> {
 public:
  explicit CuckooFilterStablePad(const size_t max_num_keys)
  : CuckooFilterStable<ItemType, bits_per_item, TableType, HashFamily>(max_num_keys + (percent_extra_pad * max_num_keys / 100)) {}
};

template <typename ItemType, size_t bits_per_item, size_t percent_extra_pad, template <size_t> class TableType, typename HashFamily>
struct FilterAPI<CuckooFilterStablePad<ItemType, bits_per_item, percent_extra_pad, TableType, HashFamily>> {
  using Table = CuckooFilterStablePad<ItemType, bits_per_item, percent_extra_pad, TableType, HashFamily>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table * table) {
    if (0 != table->Add(key)) {
      throw logic_error("The filter is too small to hold all of the elements");
    }
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
    throw std::runtime_error("Unsupported");
  }
  static void Remove(uint64_t key, Table * table) {
    table->Delete(key);
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
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
  static void Add(uint64_t key, Table* table) {
    table->Add(key);
  }
  static void AddAll(const vector<uint64_t> keys, const size_t start, const size_t end, Table* table) {
    throw std::runtime_error("Unsupported");
  }
  static void Remove(uint64_t key, Table * table) {
    throw std::runtime_error("Unsupported");
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
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
  static void Add(uint64_t key, Table* table) {
    table->Add(key);
  }
  static void AddAll(const vector<uint64_t> keys, const size_t start, const size_t end, Table* table) {
    throw std::runtime_error("Unsupported");
  }
  static void Remove(uint64_t key, Table * table) {
    throw std::runtime_error("Unsupported");
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
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
  static void Add(uint64_t key, Table* table) {
    table->Add(key);
  }
  static void AddAll(const vector<uint64_t> keys, const size_t start, const size_t end, Table* table) {
    throw std::runtime_error("Unsupported");
  }
  static void Remove(uint64_t key, Table * table) {
    throw std::runtime_error("Unsupported");
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
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
  static void Add(uint64_t key, Table* table) {
    table->Add(key);
  }
  static void AddAll(const vector<uint64_t> keys, const size_t start, const size_t end, Table* table) {
    throw std::runtime_error("Unsupported");
  }
  static void Remove(uint64_t key, Table * table) {
    throw std::runtime_error("Unsupported");
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
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
  static void Add(uint64_t key, Table* table) {
    table->Add(key);
  }
  static void AddAll(const vector<uint64_t> keys, const size_t start, const size_t end, Table* table) {
    table->AddAll(keys, start, end);
  }
  static void Remove(uint64_t key, Table * table) {
    throw std::runtime_error("Unsupported");
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
    return table->Find(key);
  }
};
#endif

template <typename ItemType, typename FingerprintType>
struct FilterAPI<XorFilter<ItemType, FingerprintType>> {
  using Table = XorFilter<ItemType, FingerprintType>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
    throw std::runtime_error("Unsupported");
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
    table->AddAll(keys, start, end);
  }
  static void Remove(uint64_t key, Table * table) {
    throw std::runtime_error("Unsupported");
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
    return (0 == table->Contain(key));
  }
};

template <typename ItemType, typename FingerprintType>
struct FilterAPI<XorFuseFilter<ItemType, FingerprintType>> {
  using Table = XorFuseFilter<ItemType, FingerprintType>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
    throw std::runtime_error("Unsupported");
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
    table->AddAll(keys, start, end);
  }
  static void Remove(uint64_t key, Table * table) {
    throw std::runtime_error("Unsupported");
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
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

  static Hash HashFn(const Hash& input, Seed raw_seed) {
    // No re-seeding for Homogeneous, because it can be skipped in practice
    return input;
  }
};

template <typename CoeffType, uint32_t kNumColumns, bool kSmash = false>
struct RibbonTsSeeded : public StandardRehasherAdapter<RibbonTsHomog<CoeffType, kNumColumns>> {
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
    double overhead = (4.0 + kFractionalCols * 0.25) / (8.0 * sizeof(CoeffType));
    return 1.0 + overhead;
  }

  HomogRibbonFilter(size_t add_count)
      : num_slots(InterleavedSoln::RoundUpNumSlots((size_t)(GetBestOverheadFactor() * add_count))),
        bytes(static_cast<size_t>((num_slots * kFractionalCols + 7) / 8)),
        ptr(new char[bytes]),
        soln(ptr.get(), bytes) {}

  void AddAll(const vector<uint64_t> keys, const size_t start, const size_t end) {
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
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
    throw std::runtime_error("Unsupported");
  }
  static void AddAll(const vector<uint64_t> keys, const size_t start, const size_t end, Table* table) {
    table->AddAll(keys, start, end);
  }
  static void Remove(uint64_t key, Table * table) {
    throw std::runtime_error("Unsupported");
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
    return table->Contain(key);
  }
};

template <typename CoeffType, uint32_t kNumColumns, uint32_t kMinPctOverhead, uint32_t kMilliBitsPerKey = 7700>
class BalancedRibbonFilter {
  using TS = RibbonTsSeeded<CoeffType, kNumColumns>;
  IMPORT_RIBBON_IMPL_TYPES(TS);
  static constexpr uint32_t kBitsPerVshard = 8;
  using BalancedBanding = ribbon::BalancedBanding<TS, kBitsPerVshard>;
  using BalancedHasher = ribbon::BalancedHasher<TS, kBitsPerVshard>;

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
    return InterleavedSoln::RoundUpNumSlots((size_t)(add_count + overhead * add_count + add_per_vshard / 5));
  }

  BalancedRibbonFilter(size_t add_count)
      : log2_vshards((uint32_t)FloorLog2((add_count + add_count / 3 + add_count / 5) / (128 * sizeof(CoeffType)))),
        num_slots(GetNumSlots(add_count, log2_vshards)),
        bytes(static_cast<size_t>((num_slots * kFractionalCols + 7) / 8)),
        ptr(new char[bytes]),
        soln(ptr.get(), bytes),
        meta_bytes(BalancedHasher(log2_vshards, nullptr).GetMetadataLength()),
        meta_ptr(new char[meta_bytes]),
        hasher(log2_vshards, meta_ptr.get()) {}

  void AddAll(const vector<uint64_t> keys, const size_t start, const size_t end) {
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
  using Table = BalancedRibbonFilter<CoeffType, kNumColumns, kMinPctOverhead, kMilliBitsPerKey>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
    throw std::runtime_error("Unsupported");
  }
  static void AddAll(const vector<uint64_t> keys, const size_t start, const size_t end, Table* table) {
    table->AddAll(keys, start, end);
  }
  static void Remove(uint64_t key, Table * table) {
    throw std::runtime_error("Unsupported");
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
    return table->Contain(key);
  }
};

template <typename CoeffType, uint32_t kNumColumns, uint32_t kMinPctOverhead, bool kUseSmash = false>
class StandardRibbonFilter {
  using TS = RibbonTsSeeded<CoeffType, kNumColumns, kUseSmash>;
  IMPORT_RIBBON_IMPL_TYPES(TS);

  size_t num_slots;

  size_t bytes;
  unique_ptr<char[]> ptr;
  InterleavedSoln soln;
  Hasher hasher;
public:
  static constexpr double kFractionalCols =
    kNumColumns == 0 ? 7.7 : kNumColumns;

  static double GetNumSlots(size_t add_count) {
    double overhead;
    if (sizeof(CoeffType) == 8) {
      overhead = -0.0251 + std::log(1.0 * add_count) * 1.4427 * 0.0083;
    } else {
      assert(sizeof(CoeffType) == 16);
      overhead = -0.0176 + std::log(1.0 * add_count) * 1.4427 * 0.0038;
    }
    overhead = std::max(overhead, 0.01 * kMinPctOverhead);
    return InterleavedSoln::RoundUpNumSlots((size_t)(add_count + overhead * add_count));
  }

  StandardRibbonFilter(size_t add_count)
      : num_slots(GetNumSlots(add_count)),
        bytes(static_cast<size_t>((num_slots * kFractionalCols + 7) / 8)),
        ptr(new char[bytes]),
        soln(ptr.get(), bytes)
        {}

  void AddAll(const vector<uint64_t> keys, const size_t start, const size_t end) {
    Banding b;
    if (b.ResetAndFindSeedToSolve(num_slots, keys.begin() + start, keys.begin() + end)) {
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
  using Table = StandardRibbonFilter<CoeffType, kNumColumns, kMinPctOverhead, kUseSmash>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
    throw std::runtime_error("Unsupported");
  }
  static void AddAll(const vector<uint64_t> keys, const size_t start, const size_t end, Table* table) {
    table->AddAll(keys, start, end);
  }
  static void Remove(uint64_t key, Table * table) {
    throw std::runtime_error("Unsupported");
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
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
      FastLocalBloomImpl::AddHash(a, b, bytes, probes, ptr.get());
      a *= kMixFactor;
      b *= kMixFactor;
    }
  }
  void AddAll(const vector<uint64_t> keys, const size_t start, const size_t end) {
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
      rv &= FastLocalBloomImpl::HashMayMatch(a, b, bytes, probes, ptr.get());
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
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
    table->Add(key);
  }
  static void AddAll(const vector<uint64_t> keys, const size_t start, const size_t end, Table* table) {
    table->AddAll(keys, start, end);
  }
  static void Remove(uint64_t key, Table * table) {
    throw std::runtime_error("Unsupported");
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
    return table->Contain(key);
  }
};

class MortonFilter {
    Morton3_8* filter;
    size_t size;
public:
    MortonFilter(const size_t size) {
        filter = new Morton3_8((size_t) (size / 0.95) + 64);
        // filter = new Morton3_8((size_t) (2.1 * size) + 64);
        this->size = size;
    }
    ~MortonFilter() {
        delete filter;
    }
    void Add(uint64_t key) {
        filter->insert(key);
    }
    void AddAll(const vector<uint64_t> keys, const size_t start, const size_t end) {
        size_t size = end - start;
        ::std::vector<uint64_t> k(size);
        ::std::vector<bool> status(size);
        for (size_t i = start; i < end; i++) {
            k[i - start] = keys[i];
        }
        // TODO return value and status is ignored currently
        filter->insert_many(k, status, size);
    }
    inline bool Contain(uint64_t &item) {
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

template<>
struct FilterAPI<MortonFilter> {
    using Table = MortonFilter;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table* table) {
        table->Add(key);
    }
    static void AddAll(const vector<uint64_t> keys, const size_t start, const size_t end, Table* table) {
        table->AddAll(keys, start, end);
    }
    static void Remove(uint64_t key, Table * table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, Table * table) {
        return table->Contain(key);
    }
};

class XorSingle {
public:
    xor8_s filter; // let us expose the struct. to avoid indirection
    explicit XorSingle(const size_t size) {
        if (!xor8_allocate(size, &filter)) {
            throw ::std::runtime_error("Allocation failed");
        }
    }
    ~XorSingle() {
        xor8_free(&filter);
    }
    bool AddAll(const uint64_t* data, const size_t start, const size_t end) {
        return xor8_buffered_populate(data + start, end - start, &filter);
    }
    inline bool Contain(uint64_t &item) const {
        return xor8_contain(item, &filter);
    }
    inline size_t SizeInBytes() const {
        return xor8_size_in_bytes(&filter);
    }
    XorSingle(XorSingle && o) : filter(o.filter)  {
        o.filter.fingerprints = nullptr; // we take ownership for the data
    }
private:
    XorSingle(const XorSingle & o) = delete;
};

template<>
struct FilterAPI<XorSingle> {
    using Table = XorSingle;
    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }
    static void Add(uint64_t key, Table* table) {
        throw std::runtime_error("Unsupported");
    }
    static void AddAll(const vector<uint64_t> keys, const size_t start, const size_t end, Table* table) {
        table->AddAll(keys.data(), start, end);
    }
    static void Remove(uint64_t key, Table * table) {
        throw std::runtime_error("Unsupported");
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
        // some compilers are not smart enough to do the inlining properly
        return xor8_contain(key, & table->filter);
    }
};

template<size_t blocksize, int k, typename HashFamily>
struct FilterAPI<SimpleBlockFilter<blocksize,k,HashFamily>> {
  using Table = SimpleBlockFilter<blocksize,k,HashFamily>;
  static Table ConstructFromAddCount(size_t add_count) {
    Table ans(ceil(add_count * 8.0 / CHAR_BIT));
    return ans;
  }
  static void Add(uint64_t key, Table* table) {
    table->Add(key);
  }
  static void AddAll(const vector<uint64_t> keys, const size_t start, const size_t end, Table* table) {
    throw std::runtime_error("Unsupported");
  }
  static void Remove(uint64_t key, Table * table) {
    throw std::runtime_error("Unsupported");
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
    return table->Find(key);
  }
};


template <typename ItemType, typename FingerprintType, typename HashFamily>
struct FilterAPI<XorFilter<ItemType, FingerprintType, HashFamily>> {
  using Table = XorFilter<ItemType, FingerprintType, HashFamily>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
    throw std::runtime_error("Unsupported");
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
    table->AddAll(keys, start, end);
  }
  static void Remove(uint64_t key, Table * table) {
    throw std::runtime_error("Unsupported");
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
    return (0 == table->Contain(key));
  }
};

template <typename ItemType, typename FingerprintType, typename FingerprintStorageType, typename HashFamily>
struct FilterAPI<XorFilter2<ItemType, FingerprintType, FingerprintStorageType, HashFamily>> {
  using Table = XorFilter2<ItemType, FingerprintType, FingerprintStorageType, HashFamily>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
    throw std::runtime_error("Unsupported");
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
    table->AddAll(keys, start, end);
  }
  static void Remove(uint64_t key, Table * table) {
    throw std::runtime_error("Unsupported");
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
    return (0 == table->Contain(key));
  }
};

template <typename ItemType, typename HashFamily>
struct FilterAPI<XorFilter10<ItemType, HashFamily>> {
  using Table = XorFilter10<ItemType, HashFamily>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
    throw std::runtime_error("Unsupported");
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
    table->AddAll(keys, start, end);
  }
  static void Remove(uint64_t key, Table * table) {
    throw std::runtime_error("Unsupported");
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
    return (0 == table->Contain(key));
  }
};

template <typename ItemType, typename HashFamily>
struct FilterAPI<XorFilter13<ItemType, HashFamily>> {
  using Table = XorFilter13<ItemType, HashFamily>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
    throw std::runtime_error("Unsupported");
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
    table->AddAll(keys, start, end);
  }
  static void Remove(uint64_t key, Table * table) {
    throw std::runtime_error("Unsupported");
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
    return (0 == table->Contain(key));
  }
};

template <typename ItemType, typename HashFamily>
struct FilterAPI<XorFilter10_666<ItemType, HashFamily>> {
  using Table = XorFilter10_666<ItemType, HashFamily>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
    throw std::runtime_error("Unsupported");
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
    table->AddAll(keys, start, end);
  }
  static void Remove(uint64_t key, Table * table) {
    throw std::runtime_error("Unsupported");
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
    return (0 == table->Contain(key));
  }
};

template <typename ItemType, typename FingerprintType, typename FingerprintStorageType, typename HashFamily>
struct FilterAPI<XorFilter2n<ItemType, FingerprintType, FingerprintStorageType, HashFamily>> {
  using Table = XorFilter2n<ItemType, FingerprintType, FingerprintStorageType, HashFamily>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
    throw std::runtime_error("Unsupported");
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
    table->AddAll(keys, start, end);
  }
  static void Remove(uint64_t key, Table * table) {
    throw std::runtime_error("Unsupported");
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
    return (0 == table->Contain(key));
  }
};

template <typename ItemType, typename FingerprintType, typename HashFamily>
struct FilterAPI<XorFilterPlus<ItemType, FingerprintType, HashFamily>> {
  using Table = XorFilterPlus<ItemType, FingerprintType, HashFamily>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
    throw std::runtime_error("Unsupported");
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
    table->AddAll(keys, start, end);
  }
  static void Remove(uint64_t key, Table * table) {
    throw std::runtime_error("Unsupported");
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
    return (0 == table->Contain(key));
  }
};

template <typename ItemType, typename FingerprintType, typename FingerprintStorageType, typename HashFamily>
struct FilterAPI<XorFilterPlus2<ItemType, FingerprintType, FingerprintStorageType, HashFamily>> {
  using Table = XorFilterPlus2<ItemType, FingerprintType, FingerprintStorageType, HashFamily>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
    throw std::runtime_error("Unsupported");
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
    table->AddAll(keys, start, end);
  }
  static void Remove(uint64_t key, Table * table) {
    throw std::runtime_error("Unsupported");
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
    return (0 == table->Contain(key));
  }
};

template <typename ItemType, size_t bits_per_item, typename HashFamily>
struct FilterAPI<GcsFilter<ItemType, bits_per_item, HashFamily>> {
  using Table = GcsFilter<ItemType, bits_per_item, HashFamily>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
    throw std::runtime_error("Unsupported");
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
    table->AddAll(keys, start, end);
  }
  static void Remove(uint64_t key, Table * table) {
    throw std::runtime_error("Unsupported");
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
    return (0 == table->Contain(key));
  }
};

#ifdef __AVX2__
template <typename ItemType, size_t bits_per_item, typename HashFamily>
struct FilterAPI<GQFilter<ItemType, bits_per_item, HashFamily>> {
  using Table = GQFilter<ItemType, bits_per_item, HashFamily>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
    table->Add(key);
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
    throw std::runtime_error("Unsupported");
  }
  static void Remove(uint64_t key, Table * table) {
    table->Remove(key);
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
    return (0 == table->Contain(key));
  }
};
#endif

template <typename ItemType, size_t bits_per_item, bool branchless, typename HashFamily>
struct FilterAPI<BloomFilter<ItemType, bits_per_item, branchless, HashFamily>> {
  using Table = BloomFilter<ItemType, bits_per_item, branchless, HashFamily>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
    table->Add(key);
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
    table->AddAll(keys, start, end);
  }
  static void Remove(uint64_t key, Table * table) {
    throw std::runtime_error("Unsupported");
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
    return (0 == table->Contain(key));
  }
};

template <typename ItemType, size_t bits_per_item, bool branchless, typename HashFamily>
struct FilterAPI<CountingBloomFilter<ItemType, bits_per_item, branchless, HashFamily>> {
  using Table = CountingBloomFilter<ItemType, bits_per_item, branchless, HashFamily>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
    table->Add(key);
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
    table->AddAll(keys, start, end);
  }
  static void Remove(uint64_t key, Table * table) {
    table->Remove(key);
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
    return (0 == table->Contain(key));
  }
};

template <typename ItemType, size_t bits_per_item, bool branchless, typename HashFamily>
struct FilterAPI<SuccinctCountingBloomFilter<ItemType, bits_per_item, branchless, HashFamily>> {
  using Table = SuccinctCountingBloomFilter<ItemType, bits_per_item, branchless, HashFamily>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
    table->Add(key);
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
    table->AddAll(keys, start, end);
  }
  static void Remove(uint64_t key, Table * table) {
    table->Remove(key);
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
    return (0 == table->Contain(key));
  }
};

template <typename ItemType, size_t bits_per_item, typename HashFamily>
struct FilterAPI<SuccinctCountingBlockedBloomFilter<ItemType, bits_per_item, HashFamily>> {
  using Table = SuccinctCountingBlockedBloomFilter<ItemType, bits_per_item, HashFamily>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
    table->Add(key);
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
    throw std::runtime_error("Unsupported");
    // table->AddAll(keys, start, end);
  }
  static void Remove(uint64_t key, Table * table) {
    table->Remove(key);
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
    return table->Contain(key);
  }
};

template <typename ItemType, size_t bits_per_item, typename HashFamily>
struct FilterAPI<SuccinctCountingBlockedBloomRankFilter<ItemType, bits_per_item, HashFamily>> {
  using Table = SuccinctCountingBlockedBloomRankFilter<ItemType, bits_per_item, HashFamily>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
    table->Add(key);
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
    throw std::runtime_error("Unsupported");
    // table->AddAll(keys, start, end);
  }
  static void Remove(uint64_t key, Table * table) {
    table->Remove(key);
  }
  CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table * table) {
    return table->Contain(key);
  }
};

// assuming that first1,last1 and first2, last2 are sorted,
// this tries to find out how many of first1,last1 can be
// found in first2, last2, this includes duplicates
template<class InputIt1, class InputIt2>
size_t match_size_iter(InputIt1 first1, InputIt1 last1,
                          InputIt2 first2, InputIt2 last2) {
    size_t answer = 0;
    while (first1 != last1 && first2 != last2) {
        if (*first1 < *first2) {
            ++first1;
        } else  if (*first2 < *first1) {
            ++first2;
        } else {
            answer ++;
            ++first1;
        }
    }
    return answer;
}

template<class InputIt>
size_t count_distinct(InputIt first, InputIt last) {
    if(last  == first) return 0;
    size_t answer = 1;
    auto val = *first;
    first++;

    while (first != last) {
      if(val != *first) ++answer;
      first++;
    }
    return answer;
}

size_t match_size(vector<uint64_t> a,  vector<uint64_t> b, size_t * distincta, size_t * distinctb) {
  // could obviously be accelerated with a Bloom filter
  // But this is surprisingly fast!
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  if(distincta != NULL) *distincta  = count_distinct(a.begin(), a.end());
  if(distinctb != NULL) *distinctb  = count_distinct(b.begin(), b.end());
  return match_size_iter(a.begin(), a.end(),b.begin(), b.end());
}

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

template <typename Table>
Statistics FilterBenchmark(
    size_t add_count, const vector<uint64_t>& to_add, size_t distinct_add, const vector<uint64_t>& to_lookup, size_t distinct_lookup,
    size_t intersectionsize, bool hasduplicates,
    const std::vector<samples_t> & mixed_sets, int seed, bool batchedadd = false, bool remove = false) {
  if (add_count > to_add.size()) {
    throw out_of_range("to_add must contain at least add_count values");
  }


  Table filter = FilterAPI<Table>::ConstructFromAddCount(add_count);
  Statistics result;
#ifdef WITH_LINUX_EVENTS
  vector<int> evts;
  evts.push_back(PERF_COUNT_HW_CPU_CYCLES);
  evts.push_back(PERF_COUNT_HW_INSTRUCTIONS);
  evts.push_back(PERF_COUNT_HW_CACHE_MISSES);
  evts.push_back(PERF_COUNT_HW_BRANCH_MISSES);
  LinuxEvents<PERF_TYPE_HARDWARE> unified(evts);
  vector<unsigned long long> results;
  results.resize(evts.size());
  cout << endl;
  unified.start();
#else
   std::cout << "-" << std::flush;
#endif

  // Add values until failure or until we run out of values to add:
  if(batchedadd) {
    std::cout << "batched add" << std::flush;
  } else {
    std::cout << "1-by-1 add" << std::flush;
  }
  auto start_time = NowNanos();
  if(batchedadd) {
    FilterAPI<Table>::AddAll(to_add, 0, add_count, &filter);
  } else {
    for (size_t added = 0; added < add_count; ++added) {
      FilterAPI<Table>::Add(to_add[added], &filter);
    }
  }
  auto time = NowNanos() - start_time;
  std::cout << "\r             \r" << std::flush;
#ifdef WITH_LINUX_EVENTS
  unified.end(results);
  printf("add    ");
  printf("cycles: %5.1f/key, instructions: (%5.1f/key, %4.2f/cycle) cache misses: %5.2f/key branch misses: %4.2f/key\n",
    results[0]*1.0/add_count,
    results[1]*1.0/add_count ,
    results[1]*1.0/results[0],
    results[2]*1.0/add_count,
    results[3]*1.0/add_count);
#else
  std::cout << "." << std::flush;
#endif

  // sanity check:
  for (size_t added = 0; added < add_count; ++added) {
    assert(FilterAPI<Table>::Contain(to_add[added], &filter) == 1);
  }

  result.add_count = add_count;
  result.nanos_per_add = static_cast<double>(time) / add_count;
  result.bits_per_item = static_cast<double>(CHAR_BIT * filter.SizeInBytes()) / add_count;
  size_t found_count = 0;

  for (auto t :  mixed_sets) {
    const double found_probability = t.found_probability;
    const auto to_lookup_mixed =  t.to_lookup_mixed ;
    size_t true_match = t.true_match ;

#ifdef WITH_LINUX_EVENTS
    unified.start();
#else
    std::cout << "-" << std::flush;
#endif
    const auto start_time = NowNanos();
    found_count = 0;
#ifndef NEW_CONTAINS_BENCHMARK
    for (const auto v : to_lookup_mixed) {
      found_count += FilterAPI<Table>::Contain(v, &filter);
    }
#else
    auto lower = to_lookup_mixed.begin();
    auto upper = to_lookup_mixed.end();
    while (lower != upper) {
      while (FilterAPI<Table>::Contain(*(lower++), &filter)) {
        ++found_count;
        if (lower == upper) {
          goto lower_neq_upper;
        }
      }
      if (lower == upper) {
        goto lower_neq_upper;
      }
      while (FilterAPI<Table>::Contain(*(--upper), &filter)) {
        ++found_count;
        if (lower == upper) {
          goto lower_neq_upper;
        }
      }
    }
    lower_neq_upper:
#endif

    const auto lookup_time = NowNanos() - start_time;
#ifdef WITH_LINUX_EVENTS
    unified.end(results);
    printf("%3.2f%%  ",found_probability);
    printf("cycles: %5.1f/key, instructions: (%5.1f/key, %4.2f/cycle) cache misses: %5.2f/key branch misses: %4.2f/key\n",
      results[0]*1.0/to_lookup_mixed.size(),
      results[1]*1.0/to_lookup_mixed.size(),
      results[1]*1.0/results[0],
      results[2]*1.0/to_lookup_mixed.size(),
      results[3] * 1.0/to_lookup_mixed.size());
#else
    std::cout << "." << std::flush;
#endif

    if (found_count < true_match) {
           cerr << "ERROR: Expected to find at least " << true_match << " found " << found_count << endl;
           cerr << "ERROR: This is a potential bug!" << endl;
           // Indicate failure
           result.add_count = 0;
    }
    result.nanos_per_finds[100 * found_probability] =
        static_cast<double>(lookup_time) / t.actual_sample_size;
    if (0.0 == found_probability) {
      ////////////////////////////
      // This is obviously technically wrong!!! The assumption is that there is no overlap between the random
      // queries and the random content. This is likely true if your 64-bit values were generated randomly,
      // but not true in general.
      // NOTE(PD): the above objection is only valid if hashes added are
      // already guaranteed unique (unusual).
      ///////////////////////////
      // result.false_positive_probabilty =
      //    found_count / static_cast<double>(to_lookup_mixed.size());
      if(t.to_lookup_mixed.size() == intersectionsize) {
        cerr << "WARNING: fpp is probably meaningless! " << endl;
      }
      uint64_t positives = found_count  - intersectionsize;
      uint64_t samples = to_lookup_mixed.size() - intersectionsize;

      if (positives * samples < 10000000000ULL) {
        //cerr << "NOTE: getting more samples for accurate FP rate" << endl;
        mt19937_64 rnd(start_time);
        while (positives * samples < 10000000000ULL) {
          // Need more samples for accurate FP rate
          positives += FilterAPI<Table>::Contain(rnd(), &filter);
          samples++;
        }
      }

      result.false_positive_probabilty = 1.0 * positives / samples;
    }
  }

  // Remove
  result.nanos_per_remove = 0;
  if (remove) {
    std::cout << "1-by-1 remove" << std::flush;
#ifdef WITH_LINUX_EVENTS
    unified.start();
#else
    std::cout << "-" << std::flush;
#endif
    start_time = NowNanos();
    for (size_t added = 0; added < add_count; ++added) {
      FilterAPI<Table>::Remove(to_add[added], &filter);
    }
    time = NowNanos() - start_time;
    result.nanos_per_remove = static_cast<double>(time) / add_count;
#ifdef WITH_LINUX_EVENTS
    unified.end(results);
    printf("remove ");
    printf("cycles: %5.1f/key, instructions: (%5.1f/key, %4.2f/cycle) cache misses: %5.2f/key branch misses: %4.2f/key\n",
      results[0]*1.0/add_count,
      results[1]*1.0/add_count ,
      results[1]*1.0/results[0],
      results[2]*1.0/add_count,
      results[3]*1.0/add_count);
#else
    std::cout << "." << std::flush;
#endif
  }

#ifndef WITH_LINUX_EVENTS
  std::cout << "\r             \r" << std::flush;
#endif

  return result;
}

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

void parse_comma_separated(char * c, std::set<int> & answer ) {
    std::stringstream ss(c);
    int i;
    while (ss >> i) {
        answer.insert(i);
        if (ss.peek() == ',')
            ss.ignore();
    }
}


int main(int argc, char * argv[]) {
  std::map<int,std::string> names = {
    // Xor
    {0, "Xor8"}, {1, "Xor12"}, {2, "Xor16"},
    {3, "XorPlus8"}, {4, "XorPlus16"},
    {5, "Xor10"}, {6, "Xor10.666"},
    {7, "Xor10(NBitArray)"}, {8, "Xor14(NBitArray)"}, {9, "XorPowTwo8"},
    // Cuckooo
    {10,"Cuckoo8"}, {11,"Cuckoo10"}, {12,"Cuckoo12"},
    {13,"Cuckoo14"}, {14,"Cuckoo16"},
    {15,"CuckooSemiSort13"},
    {16, "CuckooPowTwo8"}, {17, "CuckooPowTwo12"}, {18, "CuckooPowTwo16"},
    {19, "CuckooSemiSortPowTwo13"},
    // GCS
    {20,"GCS"},
#ifdef __AVX2__
    // CQF
    {30,"CQF"},
#endif
    // Bloom
    {40, "Bloom8"}, {41, "Bloom12" }, {42, "Bloom16"},
    {43, "Bloom8(addall)"}, {44, "Bloom12(addall)"}, {45, "Bloom16(addall)"},
    {46, "BranchlessBloom8(addall)"},
    {47, "BranchlessBloom12(addall)"},
    {48, "BranchlessBloom16(addall)"},
    // Blocked Bloom
    {50, "BlockedBloom(simple)"},
#ifdef __aarch64__
    {51, "BlockedBloom"},
    {52, "BlockedBloom(addall)"},
#elif defined( __AVX2__)
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

    {308,"Cuckoo8(Extra5Pct)"},
    {310,"Cuckoo10(Extra5Pct)"},
    {312,"Cuckoo12(Extra5Pct)"},
    {314,"Cuckoo14(Extra5Pct)"},
    {316,"Cuckoo16(Extra5Pct)"},

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
    {1155, "HomogRibbon32_15"},
    {1156, "HomogRibbon64_15"},
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

    // Sort
    {9000, "Sort"},

    // At the end because it tends to crash
    {9800, "Morton"},
  };

  // Parameter Parsing ----------------------------------------------------------

  if (argc < 2) {
    cout << "Usage: " << argv[0] << " <numberOfEntries> [<algorithmId> [<seed>]]" << endl;
    cout << " numberOfEntries: number of keys, we recommend at least 100000000" << endl;
    cout << " algorithmId: -1 for all default algos, or 0..n to only run this algorithm" << endl;
    cout << " algorithmId: can also be a comma-separated list of non-negative integers" << endl;
    for(auto i : names) {
      cout << "           "<< i.first << " : " << i.second << endl;
    }
    cout << " algorithmId: can also be set to the string 'all' if you want to run them all, including some that are excluded by default" << endl;
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
      if(strcmp(argv[2],"all") == 0) {
         for(auto i : names) {// we add all the named algos.
           algos.insert(i.first);
         }
      } else if(strstr(argv[2],",") != NULL) {
        // we have a list of algos
        algorithmId = 9999999; // disabling
        parse_comma_separated(argv[2], algos);
        if(algos.size() == 0) {
           cerr<< " no algo selected " << endl;
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
  int seed = -1;
  if (argc > 3) {
      stringstream input_string_3(argv[3]);
      input_string_3 >> seed;
      if (input_string_3.fail()) {
          cerr << "Invalid number: " << argv[3];
          return 2;
      }
  }
  size_t actual_sample_size = MAX_SAMPLE_SIZE;
  if (actual_sample_size > add_count) {
    actual_sample_size = add_count;
  } else if (actual_sample_size < 10000000) {
    actual_sample_size = 10000000;
  }

  // Generating Samples ----------------------------------------------------------

  vector<uint64_t> to_add = seed == -1 ?
      GenerateRandom64Fast(add_count, NowNanos()) :
      GenerateRandom64Fast(add_count, seed);
  vector<uint64_t> to_lookup = seed == -1 ?
      GenerateRandom64Fast(actual_sample_size, NowNanos()) :
      GenerateRandom64Fast(actual_sample_size, seed + add_count);

  if (seed >= 0 && seed < 64) {
    // 0-64 are special seeds
    uint rotate = seed;
    cout << "Using sequential ordering rotated by " << rotate << endl;
    for(uint64_t i = 0; i < to_add.size(); i++) {
        to_add[i] = xorfilter::rotl64(i, rotate);
    }
    for(uint64_t i = 0; i < to_lookup.size(); i++) {
        to_lookup[i] = xorfilter::rotl64(i + to_add.size(), rotate);
    }
  } else if (seed >= 64 && seed < 128) {
    // 64-127 are special seeds
    uint rotate = seed - 64;
    cout << "Using sequential ordering rotated by " << rotate << " and reversed bits " << endl;
    for(uint64_t i = 0; i < to_add.size(); i++) {
        to_add[i] = reverseBitsSlow(xorfilter::rotl64(i, rotate));
    }
    for(uint64_t i = 0; i < to_lookup.size(); i++) {
        to_lookup[i] = reverseBitsSlow(xorfilter::rotl64(i + to_add.size(), rotate));
    }
  }

  assert(to_lookup.size() == actual_sample_size);
  size_t distinct_lookup = to_lookup.size();
  size_t distinct_add = to_add.size();
  size_t intersectionsize = 0;
#ifdef CHECK_MATCH_SIZE // Can be really slow
  std::cout << "checking match size... " << std::flush;
  intersectionsize = match_size(to_lookup, to_add, &distinct_lookup, & distinct_add);
  std::cout << "\r                       \r" << std::flush;
#endif

  bool hasduplicates = false;
  if(intersectionsize > 0) {
    cout << "WARNING: Out of the lookup table, "<< intersectionsize<< " ("<<intersectionsize * 100.0 / to_lookup.size() << "%) of values are present in the filter." << endl;
    hasduplicates = true;
  }

  if(distinct_lookup != to_lookup.size()) {
    cout << "WARNING: Lookup contains "<< (to_lookup.size() - distinct_lookup)<<" duplicates." << endl;
    hasduplicates = true;
  }
  if(distinct_add != to_add.size()) {
    cout << "WARNING: Filter contains "<< (to_add.size() - distinct_add) << " duplicates." << endl;
    hasduplicates = true;
  }

  if (actual_sample_size > to_lookup.size()) {
    std::cerr << "actual_sample_size = "<< actual_sample_size << std::endl;
    throw out_of_range("to_lookup must contain at least actual_sample_size values");
  }

  std::vector<samples_t> mixed_sets;

  const std::vector<double> found_probabilities = { 0.0, 0.5, 1.0 };

  for (const double found_probability : found_probabilities) {
    std::cout << "generating samples with probability " << found_probability <<" ... " << std::flush;

    struct samples thisone;
    thisone.found_probability = found_probability;
    thisone.actual_sample_size = actual_sample_size;
    uint64_t mixingseed = seed == -1 ? random() : seed;
    thisone.to_lookup_mixed = DuplicateFreeMixIn(&to_lookup[0], &to_lookup[actual_sample_size], &to_add[0],
    &to_add[add_count], found_probability, mixingseed);
    assert(thisone.to_lookup_mixed.size() == actual_sample_size);
    thisone.true_match = match_size(thisone.to_lookup_mixed,to_add, NULL, NULL);
    double trueproba =  thisone.true_match /  static_cast<double>(actual_sample_size) ;
    double bestpossiblematch = fabs(round(found_probability * actual_sample_size) / static_cast<double>(actual_sample_size) - found_probability);
    double tolerance = bestpossiblematch > 0.01 ? bestpossiblematch : 0.01;
    double probadiff = fabs(trueproba - found_probability);
    if(probadiff >= tolerance) {
      cerr << "WARNING: You claim to have a find proba. of " << found_probability << " but actual is " << trueproba << endl;
      return EXIT_FAILURE;
    }
    mixed_sets.push_back(thisone);
    std::cout << "\r                                                                                         \r"  << std::flush;
  }
  constexpr int NAME_WIDTH = 32;
  cout << StatisticsTableHeader(NAME_WIDTH, found_probabilities) << endl;

  // Algorithms ----------------------------------------------------------
  int a;

  // Xor ----------------------------------------------------------
  a = 0;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilter<uint64_t, uint8_t, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilter2<uint64_t, uint32_t, UInt12Array, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilter<uint64_t, uint16_t, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilterPlus<uint64_t, uint8_t, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 4;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilterPlus<uint64_t, uint16_t, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 5;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilter10<uint64_t, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 6;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilter10_666<uint64_t, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 7;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilter2<uint64_t, uint16_t, NBitArray<uint16_t, 10>, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 8;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilter2<uint64_t, uint16_t, NBitArray<uint16_t, 14>, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 9;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilter2n<uint64_t, uint8_t, UIntArray<uint8_t>, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }

  // Cuckoo ----------------------------------------------------------
  a = 10;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          CuckooFilterStable<uint64_t, 8, SingleTable, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, false, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 11;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          CuckooFilterStable<uint64_t, 10, SingleTable, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, false, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 12;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          CuckooFilterStable<uint64_t, 12, SingleTable, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, false, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 13;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          CuckooFilterStable<uint64_t, 14, SingleTable, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, false, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 14;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          CuckooFilterStable<uint64_t, 16, SingleTable, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, false, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 15;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          CuckooFilterStable<uint64_t, 13, PackedTable, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, false, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 16;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          CuckooFilter<uint64_t, 8, SingleTable, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, false, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 17;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          CuckooFilter<uint64_t, 12, SingleTable, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, false, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 18;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          CuckooFilter<uint64_t, 16, SingleTable, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, false, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 19;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          CuckooFilter<uint64_t, 13, PackedTable, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, false, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }

  // GCS ----------------------------------------------------------
  a = 20;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          GcsFilter<uint64_t, 8, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }

  // CQF ----------------------------------------------------------
#ifdef __AVX2__
  a = 30;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          GQFilter<uint64_t, 8, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, false, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
#endif

  // Bloom ----------------------------------------------------------
  a = 40;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BloomFilter<uint64_t, 8, false, NoopHash>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 41;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BloomFilter<uint64_t, 12, false, NoopHash>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 42;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BloomFilter<uint64_t, 16, false, NoopHash>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 43;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BloomFilter<uint64_t, 8, false, NoopHash>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 44;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BloomFilter<uint64_t, 12, false, NoopHash>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 45;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BloomFilter<uint64_t, 16, false, NoopHash>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 46;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BloomFilter<uint64_t, 8, true, NoopHash>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 47;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BloomFilter<uint64_t, 12, true, NoopHash>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 48;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BloomFilter<uint64_t, 16, true, NoopHash>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }

  a = 48;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BloomFilter<uint64_t, 16, true, NoopHash>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }

  // Blocked Bloom ----------------------------------------------------------
  a = 50;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          SimpleBlockFilter<8, 8, NoopHash>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, false);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
#ifdef __aarch64__
  a = 51;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<SimdBlockFilterFixed<NoopHash>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 52;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<SimdBlockFilterFixed<NoopHash>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
#endif
#ifdef __AVX2__
  a = 51;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<SimdBlockFilterFixed<NoopHash>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 52;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<SimdBlockFilterFixed<NoopHash>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 53;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
        auto cf = FilterBenchmark<SimdBlockFilterFixed64<NoopHash>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
#endif
#ifdef __SSE41__
  a = 54;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<SimdBlockFilterFixed16<NoopHash>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
#endif

  // Counting Bloom ----------------------------------------------------------
  a = 60;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          CountingBloomFilter<uint64_t, 10, true, NoopHash>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 61;
  if (algorithmId == a  || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          SuccinctCountingBloomFilter<uint64_t, 10, true, NoopHash>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 62;
  if (algorithmId == a  || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          SuccinctCountingBlockedBloomFilter<uint64_t, 10, NoopHash>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, false, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 63;
  if (algorithmId == a  || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          SuccinctCountingBlockedBloomRankFilter<uint64_t, 10, NoopHash>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, false, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }

  a = 70;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorSingle>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }

  // Xor Fuse Filter ----------------------------------------------------------
  a = 90;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFuseFilter<uint64_t, uint8_t>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }


  // Specific Xor/XorPlus bit widths
  a = 101;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilter2<uint64_t, uint8_t, NBitArray<uint8_t, 1>, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 103;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilter2<uint64_t, uint8_t, NBitArray<uint8_t, 3>, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 105;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilter2<uint64_t, uint8_t, NBitArray<uint8_t, 5>, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 107;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilter2<uint64_t, uint8_t, NBitArray<uint8_t, 7>, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 109;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilter2<uint64_t, uint16_t, NBitArray<uint16_t, 9>, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 111;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilter2<uint64_t, uint16_t, NBitArray<uint16_t, 11>, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 113;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilter2<uint64_t, uint16_t, NBitArray<uint16_t, 13>, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 115;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilter2<uint64_t, uint16_t, NBitArray<uint16_t, 15>, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 205;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilterPlus2<uint64_t, uint8_t, NBitArray<uint8_t, 5>, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 207;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilterPlus2<uint64_t, uint8_t, NBitArray<uint8_t, 7>, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 209;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilterPlus2<uint64_t, uint16_t, NBitArray<uint16_t, 9>, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 211;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilterPlus2<uint64_t, uint16_t, NBitArray<uint16_t, 11>, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 213;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilterPlus2<uint64_t, uint16_t, NBitArray<uint16_t, 13>, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 215;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          XorFilterPlus2<uint64_t, uint16_t, NBitArray<uint16_t, 15>, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  // Cuckoo (Extra5Pct) --------------------------------------------------
  a = 308;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          CuckooFilterStablePad<uint64_t, 8, 5, SingleTable, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, false, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 310;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          CuckooFilterStablePad<uint64_t, 10, 5, SingleTable, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, false, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 312;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          CuckooFilterStablePad<uint64_t, 12, 5, SingleTable, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, false, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 314;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          CuckooFilterStablePad<uint64_t, 14, 5, SingleTable, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, false, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 316;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          CuckooFilterStablePad<uint64_t, 16, 5, SingleTable, SimpleXorMul>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, false, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }

  // TwoBlockBloom(Rocks)
#define ADD(k) \
  a = 800 + k; \
  if (algorithmId == a || (algos.find(a) != algos.end())) { \
      auto cf = FilterBenchmark<RocksBloomFilter<k, 2>>( \
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed); \
      cout << setw(NAME_WIDTH) << names[a] << cf << endl; \
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
#define ADD(k) \
  a = 900 + k; \
  if (algorithmId == a || (algos.find(a) != algos.end())) { \
      auto cf = FilterBenchmark<RocksBloomFilter<k, 1>>( \
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed); \
      cout << setw(NAME_WIDTH) << names[a] << cf << endl; \
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
      auto cf = FilterBenchmark<RocksBloomFilter<6, 1, 10240>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }

  // Homogeneous Ribbon
  a = 1014;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint16_t, 1>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1015;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint32_t, 1>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1016;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint64_t, 1>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1017;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<Unsigned128, 1>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1034;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint16_t, 3>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1035;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint32_t, 3>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1036;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint64_t, 3>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1037;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<Unsigned128, 3>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1054;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint16_t, 5>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1055;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint32_t, 5>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1056;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint64_t, 5>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1057;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<Unsigned128, 5>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1074;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint16_t, 7>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1075;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint32_t, 7>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1076;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint64_t, 7>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1077;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<Unsigned128, 7>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1084;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint16_t, 8>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1085;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint32_t, 8>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1086;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint64_t, 8>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1087;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<Unsigned128, 8>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1094;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint16_t, 9>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1095;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint32_t, 9>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1096;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint64_t, 9>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1097;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<Unsigned128, 9>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1114;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint16_t, 11>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1115;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint32_t, 11>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1116;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint64_t, 11>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1117;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<Unsigned128, 11>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1135;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint32_t, 13>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1136;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint64_t, 13>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1155;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint32_t, 15>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1156;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint64_t, 15>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1275;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint32_t, 0, 2700>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1276;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint64_t, 0, 2700>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1335;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint32_t, 0, 3300>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1336;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint64_t, 0, 3300>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1774;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint16_t, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1775;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint32_t, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1776;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<uint64_t, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 1777;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          HomogRibbonFilter<Unsigned128, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }

  // BalancedRibbon
  a = 2015;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint32_t, 1, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2016;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint64_t, 1, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2035;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint32_t, 3, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2036;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint64_t, 3, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2055;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint32_t, 5, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2056;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint64_t, 5, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2071;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint32_t, 7, 25>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2072;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint32_t, 7, 20>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2073;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint32_t, 7, 15>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2074;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint32_t, 7, 10>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2075;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint32_t, 7, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2076;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint64_t, 7, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2077;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<Unsigned128, 7, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2085;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint32_t, 8, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2086;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint64_t, 8, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2095;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint32_t, 9, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2096;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint64_t, 9, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2115;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint32_t, 11, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2116;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint64_t, 11, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2135;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint32_t, 13, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2136;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint64_t, 13, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2155;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint32_t, 15, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2156;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint64_t, 15, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2775;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint32_t, 0, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 2776;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          BalancedRibbonFilter<uint64_t, 0, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }

  // StandardRibbon
  a = 3016;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<uint64_t, 1, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3017;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<Unsigned128, 1, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3036;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<uint64_t, 3, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3037;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<Unsigned128, 3, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3056;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<uint64_t, 5, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3057;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<Unsigned128, 5, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3072;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<uint64_t, 7, 25>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3073;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<uint64_t, 7, 20>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3074;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<uint64_t, 7, 15>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3075;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<uint64_t, 7, 10>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3076;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<uint64_t, 7, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3077;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<Unsigned128, 7, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3086;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<uint64_t, 7, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3087;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<Unsigned128, 7, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3088;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<uint64_t, 7, 0, true>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3089;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<Unsigned128, 7, 0, true>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3096;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<uint64_t, 9, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3097;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<Unsigned128, 9, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3116;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<uint64_t, 11, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3117;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<Unsigned128, 11, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3136;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<uint64_t, 13, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3137;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<Unsigned128, 13, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3156;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<uint64_t, 15, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3157;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<Unsigned128, 15, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3776;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<uint64_t, 0, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }
  a = 3777;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          StandardRibbonFilter<Unsigned128, 0, 0>>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }

  // Sort ----------------------------------------------------------
  a = 9000;
  if (algorithmId == a || algorithmId < 0 || (algos.find(a) != algos.end())) {
      auto start_time = NowNanos();
      std::sort(to_add.begin(), to_add.end());
      const auto sort_time = NowNanos() - start_time;
      std::cout << "Sort time: " << sort_time / to_add.size() << " ns/key\n";
  }

  a = 9800;
  if (algorithmId == a || (algos.find(a) != algos.end())) {
      auto cf = FilterBenchmark<
          MortonFilter>(
          add_count, to_add, distinct_add, to_lookup, distinct_lookup, intersectionsize, hasduplicates, mixed_sets, seed, true);
      cout << setw(NAME_WIDTH) << names[a] << cf << endl;
  }

}
