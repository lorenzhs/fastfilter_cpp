#include <vector>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <algorithm>
#include <unistd.h>
#include <assert.h>

inline size_t fastrange64(uint64_t hash, size_t range) {
  __uint128_t wide = __uint128_t{range} * hash;
  return static_cast<size_t>(wide >> 64);
}

inline uint32_t fastrange32(uint32_t hash, uint32_t range) {
    uint64_t wide = uint64_t{hash} * range;
    return static_cast<uint32_t>(wide >> 32);
}

// Best is around 20/20, but this can make for slightly faster queries
static constexpr uint32_t front_smash = 32;
static constexpr uint32_t back_smash = 31;

struct GaussData {
    uint64_t row = 0;
    uint32_t start = 0;
    uint32_t pivot = 0;
    void Reset(uint64_t h, uint32_t len) {
        uint32_t addrs = len - 63 + front_smash + back_smash;
        start = fastrange32((uint32_t)(h >> 32), addrs);
        start = std::max(start, front_smash);
        start -= front_smash;
        start = std::min(start, len - 64);
        assert(start < len - 63);
        row = h + 0x9e3779b97f4a7c13 * 0x9e3779b97f4a7c13;
        row ^= h >> 32;
        //row |= 1;
        row |= (uint64_t{1} << 63);
        pivot = 0;
    }
};

static uint32_t peak_dynamic_contention = 0;

uint32_t run(GaussData *data, uint32_t nkeys, uint32_t len) {
    uint32_t failed_rows = 0;
    for (uint32_t i = 0; i < nkeys; ++i) {
        GaussData &di = data[i];
        if (di.row == 0) {
            ++failed_rows;
            continue;
        }
        int tz = __builtin_ctzl(di.row);
        di.pivot = di.start + tz;
        assert(di.pivot < len);
        uint32_t contention = 0;
        for (uint32_t j = i + 1; j < nkeys; ++j) {
            GaussData &dj = data[j];
            assert(dj.start >= di.start);
            if (di.pivot < dj.start) {
                break;
            }
            ++contention;
            if ((dj.row >> (di.pivot - dj.start)) & 1) {
                dj.row ^= (di.row >> (dj.start - di.start));
            }
        }
        peak_dynamic_contention = std::max(peak_dynamic_contention, contention);
    }
    return failed_rows;
}

int main(int argc, char *argv[]) {
    std::mt19937_64 rand(getpid());

    uint32_t nkeys = (uint32_t)std::atoi(argv[1]);
    double f = std::atof(argv[2]);
    uint32_t len = (uint32_t)(f * nkeys / 64 + 0.5) * 64;

    std::vector<uint64_t> orig;
    for (uint32_t i = 0; i < nkeys; ++i) {
        uint64_t h = (uint64_t)rand();
        orig.push_back(h);
    }

    std::vector<uint64_t> hashes = orig;

    std::sort(hashes.begin(), hashes.end());

    GaussData *data = new GaussData[nkeys];
    uint32_t prev_start = -1;
    uint32_t cur_same_start_count = 0;
    uint32_t max_same_start_count = 0;
    uint32_t contention_from = 0;
    uint32_t peak_static_contention = 0;
    uint32_t min_static_spread = 1000;
    for (uint32_t i = 0; i < nkeys; ++i) {
        data[i].Reset(hashes[i], len);
        if (data[i].start == prev_start) {
            ++cur_same_start_count;
            max_same_start_count = std::max(max_same_start_count, cur_same_start_count);
            peak_static_contention = std::max(peak_static_contention, i - contention_from);
        } else {
            prev_start = data[i].start;
            cur_same_start_count = 1;
            while (data[contention_from].start + 64 <= prev_start) {
                ++contention_from;
            }
        }
        if (i >= 80) {
            min_static_spread = std::min(min_static_spread, data[i].start - data[i - 80].start);
        }
    }

    uint32_t failed_rows = run(data, nkeys, len);
    std::cout << "max_same_start_count: " << max_same_start_count << std::endl;
    std::cout << "peak_static_contention: " << peak_static_contention << std::endl;
    std::cout << "min_static_spread: " << min_static_spread << std::endl;
    std::cout << "peak_dynamic_contention: " << peak_dynamic_contention << std::endl;
    std::cout << "tail_waste: " << (len - data[nkeys-1].pivot) << std::endl;
    std::cout << std::endl;
    std::cout << "keys2 " << nkeys << " over " << len << " (" << ((double)len / nkeys) << "x)" << std::endl;
    std::cout << "kicked: " << failed_rows << " (" << (100.0 * failed_rows / nkeys) << "%)" << std::endl;

    uint32_t retries = 0;
    uint64_t seed = 1;
    while (failed_rows > 0 && retries < 100) {
        ++retries;
        seed *= 0x9e3779b97f4a7c13;
        for (uint32_t i = 0; i < nkeys; ++i) {
            hashes[i] = orig[i];
            if (i < nkeys /** 6 / 32*/) {
                hashes[i] *= seed;
            }
        }
        std::sort(hashes.begin(), hashes.end());

        for (uint32_t i = 0; i < nkeys; ++i) {
            if (false/*(orig[i] & 63) == 0*/) {
                continue;
            } else {
                data[i].Reset(hashes[i], len);
            }
        }
        failed_rows = run(data, nkeys, len);
    }

    std::cout << std::endl;
    std::cout << "retries_to_success: " << retries << std::endl;
    return 0;
}
