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
    void Reset(uint64_t h, uint32_t addrs, uint64_t seed) {
        start = fastrange32((uint32_t)(h >> 32), addrs);
        row = h * seed;
        row |= (uint64_t{1} << 63);
        pivot = 0;
        section = (row ^ (row >> 27) ^ (row >> 51)) & 63;
    }
};

int main(int argc, char *argv[]) {
    std::mt19937_64 rand(getpid());

    uint32_t nkeys = (uint32_t)std::atoi(argv[1]);
    uint32_t shard_size = (uint32_t)std::atoi(argv[2]);
    double f = std::atof(argv[3]);
    uint32_t nshards = ((uint32_t)(nkeys * 1.02 * f) + shard_size - 1) / shard_size;
    uint32_t naddrs = nshards * shard_size - (shard_size / 16);
    uint32_t shard_max_keys = (uint32_t)(shard_size / 1.02);

    uint64_t *shard_counts = new uint64_t[nshards];
    for (uint32_t i = 0; i < nshards; ++i) {
        shard_counts[i] = 0;
    }
    for (uint32_t i = 0; i < nkeys; ++i) {
        shard_counts[fastrange32((uint32_t)rand(), naddrs) % nshards]++;
    }
    shard_counts[nshards - 1] -= shard_size / 16;
    shard_counts[0] += shard_size / 16;

    uint32_t unpinned_denom = 8;
    uint32_t section_bits = 32;
    uint32_t section_size = shard_max_keys / unpinned_denom / section_bits;
    uint64_t fallback_count = 0;
    for (uint32_t i = 0; i < nshards - 1; ++i) {
        if (shard_counts[i] > shard_max_keys) {
            uint64_t overflow = shard_counts[i] - shard_max_keys;
            if (overflow * unpinned_denom > shard_max_keys) {
                ++fallback_count;
            } else {
                uint64_t rounded_up = (overflow + section_size - 1) / section_size * section_size;
                shard_counts[i+1] += rounded_up;
                shard_counts[i] -= rounded_up;
            }
        }
    }
    uint32_t kicked = 0;
    uint32_t margin = 0;
    if (shard_counts[nshards - 1] > shard_max_keys) {
        kicked = shard_counts[nshards - 1] - shard_max_keys;
        ++fallback_count;
    } else {
        margin = shard_max_keys - shard_counts[nshards - 1];
    }

    std::cout << "keys " << nkeys << " over " << nshards << " shards" << std::endl;
    std::cout << "shard_max_keys: " << shard_max_keys << " shard_size: " << shard_size << " (" << (100.0 * shard_max_keys / shard_size) << "%)" << std::endl;
    std::cout << "kicked: " << kicked << " margin: " << margin << std::endl;
    std::cout << "fallback_count: " << fallback_count << " pct " << (100.0 * fallback_count / nshards) << std::endl;

    return 0;
}
