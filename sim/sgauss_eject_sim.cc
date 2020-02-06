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
    uint32_t section = 0;
    void Reset(uint64_t h, uint32_t addrs, uint32_t len, uint64_t seed) {
        //start = fastrange32((uint32_t)(h >> 32), addrs);
        start = fastrange32((uint32_t)(h >> 32), len);
        if (start > addrs + 1) {
            start = 1 + addrs + (start - addrs) / 2;
        }
        row = h * seed;
        row |= (uint64_t{1} << 63);
        pivot = 0;
        section = (row ^ (row >> 27) ^ (row >> 51)) & 63;
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
    uint64_t kept_sections = -1;
    uint64_t pinned_sections = 0;
    while (__builtin_popcountl(pinned_sections) < 3) {
        pinned_sections |= uint64_t{1} << (rand() & 63);
    }
    uint32_t failed_rows = 0;

    restart:
    for (uint32_t i = 0; i < nkeys; ++i) {
        data[i].Reset(orig[i], addrs, len, 0x9e3779b97f4a7c13);
    }

    for (uint32_t i = 0; i < nkeys; ++i) {
        GaussData &di = data[i];
        uint64_t section_bit = uint64_t{1} << di.section;
        if ((kept_sections & section_bit) == 0) {
            // unkept
            continue;
        }
        if (di.row == 0) {
            while (pinned_sections & section_bit) {
                if (i > 0) {
                    --i;
                    section_bit = uint64_t{1} << data[i].section;
                } else {
                    kept_sections = 0;
                    failed_rows = nkeys;
                    goto abort;
                }
            }
            kept_sections &= ~section_bit;
            ++failed_rows;
            goto restart;
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
                // TODO?: forward-looking check for 0
            }
        }
    }
    abort:

    uint32_t kept_keys = 0;
    for (uint32_t i = 0; i < nkeys; ++i) {
        uint64_t section_bit = uint64_t{1} << data[i].section;
        if ((kept_sections & section_bit) == section_bit) {
            ++kept_keys;
        }
    }

    std::cout << "keys " << nkeys << " over " << len << " (" << ((double)len / nkeys) << "x)" << std::endl;
    std::cout << "kept_keys " << kept_keys << " over " << len << " xratio: " << ((double)len / kept_keys) << std::endl;
    std::cout << "kicked: " << failed_rows << " (" << (100.0 * failed_rows / nkeys) << "%)" << std::endl;

    return 0;
}
