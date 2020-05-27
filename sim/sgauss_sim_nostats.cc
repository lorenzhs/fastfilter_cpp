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
    void Reset(uint64_t h, uint32_t addrs, uint32_t len, uint64_t seed) {
        start = fastrange32((uint32_t)(h >> 32), addrs);
        /*
        start = fastrange32((uint32_t)(h >> 32), len);
        if (start > addrs + 1) {
            // XXX: cheating (out of bounds)
            start = 1 + addrs + (start - addrs) / 2;
        }*/
        row = h * seed;
        row |= (uint64_t{1} << 63);
        pivot = 0;
    }
};

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
    for (uint32_t i = 0; i < nkeys; ++i) {
        data[i].Reset(orig[i], addrs, len, 0x9e3779b97f4a7c13);
    }

    uint32_t failed_rows = 0;
    for (uint32_t i = 0; i < nkeys; ++i) {
        GaussData &di = data[i];
        if (di.row == 0) {
            ++failed_rows;
            continue;
        }
        int tz = __builtin_ctzl(di.row);
        di.pivot = di.start + tz;
        for (uint32_t j = i + 1; j < nkeys; ++j) {
            GaussData &dj = data[j];
            assert(dj.start >= di.start);
            if (di.pivot < dj.start) {
                break;
            }
            if ((dj.row >> (di.pivot - dj.start)) & 1) {
                dj.row ^= (di.row >> (dj.start - di.start));
            }
        }
    }

    std::cout << "keys " << nkeys << " over " << len << " (" << ((double)len / nkeys) << "x)" << std::endl;
    std::cout << "kicked: " << failed_rows << " (" << (100.0 * failed_rows / nkeys) << "%)" << std::endl;

    return 0;
}
