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

struct GaussData {
    uint64_t row = 0;
    uint32_t start = 0;
    uint32_t pivot = 0;
    void Reset(uint64_t h, uint32_t addrs, uint64_t seed) {
        start = fastrange32((uint32_t)(h >> 32), addrs);
        row = h + seed * seed;
        row ^= h >> 32;
        //row |= (uint64_t{1} << 63);
        //row = h * addrs;
        pivot = 0;
    }
};

static uint32_t peak_dynamic_contention = 0;

uint32_t run(GaussData *data, uint32_t nkeys) {
    uint32_t failed_rows = 0;
    for (uint32_t i = 0; i < nkeys; ++i) {
        GaussData &di = data[i];
        if (di.row == 0) {
            ++failed_rows;
            continue;
        }
        int tz = __builtin_ctzl(di.row);
        di.pivot = di.start + tz;
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
    uint32_t addrs = len - 63;

    std::vector<uint64_t> orig;
    for (uint32_t i = 0; i < nkeys; ++i) {
        uint64_t h = (uint64_t)rand();
        orig.push_back(h);
    }
    std::sort(orig.begin(), orig.end());

    GaussData *data = new GaussData[nkeys];
    uint32_t prev_start = -1;
    uint32_t cur_same_start_count = 0;
    uint32_t max_same_start_count = 0;
    uint32_t contention_from = 0;
    uint32_t peak_static_contention = 0;
    uint32_t min_static_spread = 1000;
    for (uint32_t i = 0; i < nkeys; ++i) {
        data[i].Reset(orig[i], addrs, 0x9e3779b97f4a7c13);
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

    uint32_t failed_rows1 = run(data, nkeys);
    std::cout << "max_same_start_count: " << max_same_start_count << std::endl;
    std::cout << "peak_static_contention: " << peak_static_contention << std::endl;
    std::cout << "min_static_spread: " << min_static_spread << std::endl;
    std::cout << "peak_dynamic_contention: " << peak_dynamic_contention << std::endl;
    std::cout << std::endl;
    std::cout << "keys2 " << nkeys << " over " << len << " (" << ((double)len / nkeys) << "x)" << std::endl;
    std::cout << "kicked: " << failed_rows1 << " (" << (100.0 * failed_rows1 / nkeys) << "%)" << std::endl;

    uint32_t nkeys2 = 0;
    for (uint32_t i = 0; i < nkeys; ++i) {
        if (false/*(orig[i] & 63) == 0*/) {
            continue;
        } else {
            data[nkeys2].Reset(orig[nkeys2], addrs, 0x2be9387616572381);
            ++nkeys2;
        }
    }

    uint32_t failed_rows2 = run(data, nkeys2);

    std::cout << std::endl;
    std::cout << "keys2 " << nkeys2 << " over " << len << " (" << ((double)len / nkeys2) << "x)" << std::endl;
    std::cout << "kicked2: " << failed_rows2 << " (" << (100.0 * failed_rows2 / nkeys2) << "%)" << std::endl;
    return 0;
}
