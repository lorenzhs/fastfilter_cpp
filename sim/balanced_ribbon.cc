#include <vector>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <algorithm>
#include <unistd.h>

#include "../src/ribbon/ribbon_impl.h"

using namespace ribbon;

template <typename CoeffType, bool firstCoeffAlwaysOne>
struct RibbonTS {
  static constexpr bool kIsFilter = false;
  static constexpr bool kHomogeneous = false;
  static constexpr bool kFirstCoeffAlwaysOne = firstCoeffAlwaysOne;
  static constexpr bool kUseSmash = false;
  using CoeffRow = CoeffType;
  using Hash = uint64_t;
  using Key = uint64_t;
  using Seed = uint32_t;
  using Index = size_t;
  using ResultRow = uint64_t;
  static constexpr bool kAllowZeroStarts = false;
  static constexpr uint32_t kFixedNumColumns = 64;

  static Hash HashFn(const Hash& input, Seed raw_seed) {
    return input;
  }
};

template <class TypesAndSettings>
class CustomHasher : public StandardHasher<TypesAndSettings> {
 public:
  IMPORT_RIBBON_TYPES_AND_SETTINGS(TypesAndSettings);

  inline CoeffRow GetCoeffRow(Hash h0) const {
    // Use a stronger re-mix than a standard Ribbon implementation is
    // OK with.
    uint64_t h = h0;
    // murmur something
    h ^= h >> 33;
    h *= UINT64_C(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h *= UINT64_C(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;

    CoeffRow v = StandardHasher<TypesAndSettings>::GetCoeffRow((Hash)h);
    // Ensure non-zero
    if ((v & coeff_mask_) == 0) {
        v >>= 32;
        if ((v & coeff_mask_) == 0) {
            v = 1;
        }
    }
    return v & coeff_mask_;
  }

  CoeffRow coeff_mask_ = static_cast<CoeffRow>(-1);
};

static constexpr uint64_t kGR = 0x9e3779b97f4a7c13;

template <bool firstCoeffAlwaysOne>
int RunTest(char *argv[]) {
    using TS = RibbonTS<uint64_t, firstCoeffAlwaysOne>;
    IMPORT_RIBBON_TYPES_AND_SETTINGS(TS);
    using Hasher = CustomHasher<TS>;
    using Banding = ribbon::StandardBandingBase<Hasher>;

    int coeff_bits = std::atoi(argv[1]);
    if (coeff_bits < 0) {
        if (!firstCoeffAlwaysOne) {
            return 42;
        }
        coeff_bits = -coeff_bits;
    } else {
        if (firstCoeffAlwaysOne) {
            return 43;
        }
    }
    if (coeff_bits > (int)kCoeffBits || coeff_bits < 1) {
        return 1;
    }

    int buckets_log2 = std::atoi(argv[2]);
    if (buckets_log2 > 40 || buckets_log2 < 0) {
        return 1;
    }

    Index bucket_size = (Index)std::atoi(argv[3]);

    std::mt19937_64 rand(getpid());

    uint64_t sum_added = 0;
    size_t iteration = 1;

    Index num_starts = (bucket_size << buckets_log2);
    Index num_slots_physical = num_starts + kCoeffBits - 1;
    Index num_slots = num_starts + coeff_bits - 1;

    uint64_t increment = ((kGR >> 1 >> (63 - buckets_log2)) | uint64_t{1}) << 1 << (63 - buckets_log2);

    std::cout << "starts: " << num_starts << std::endl;

    for (;; ++iteration) {
        Banding banding;
        banding.Reset(num_slots_physical);
        banding.coeff_mask_ = static_cast<CoeffRow>(-1) >> (64 - coeff_bits);

        Index added = 0;

        for (uint64_t bucket_hash = 0;; bucket_hash += increment) {
            uint64_t hash = bucket_hash + (rand() >> buckets_log2);
            //std::cout << "bucket=" << (hash >> 1 >> (63 - buckets_log2)) << std::endl;
            if (!banding.Add(std::make_pair(hash, rand()))) {
                break;
            }
            ++added;
        }
        sum_added += added;

        std::cout << "total added (iteration " << iteration << "): " << added << std::endl;
        std::cout << "epsilon (slots) at first failure: " << (1.0 - 1.0 * added / num_slots) << std::endl;
        if (added >= num_starts) {
            std::cout << "OVERload entries at first failure: " << (added - num_starts) << std::endl;
        } else {
            std::cout << "UNDERload epsilon (starts) at first failure: " << (1.0 - 1.0 * added / num_starts) << std::endl;
        }

        std::cout << "AVERAGE epsilon (slots) at first failure: " << (1.0 - 1.0 * sum_added / num_slots / iteration) << std::endl;
        uint64_t sum_starts = uint64_t{num_starts} * iteration;
        if (sum_added >= sum_starts) {
            std::cout << "AVERAGE OVERload entries at first failure: " << (1.0 * (sum_added - sum_starts) / iteration) << std::endl;
        } else {
            std::cout << "AVERAGE UNDERload epsilon (starts) at first failure: " << (1.0 - 1.0 * sum_added / sum_starts) << std::endl;
        }
    }

    return 0;
}

int main(int argc, char *argv[]) {
    int coeff_bits = std::atoi(argv[1]);
    if (coeff_bits < 0) {
        return RunTest<true>(argv);
    } else {
        return RunTest<false>(argv);
    }
}
