#include <vector>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <array>
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

static constexpr uint32_t unpinned_numer = 8;
static constexpr uint32_t unpinned_denom = 32;
static constexpr uint32_t threshold_range = 64;

int main(int argc, char *argv[]) {
    std::mt19937_64 rand(getpid());

    uint32_t nkeys = (uint32_t)std::atoi(argv[1]);
    uint32_t shard_size = (uint32_t)std::atoi(argv[2]);
    double f = std::atof(argv[3]);
    uint32_t nshards = ((uint32_t)(nkeys * f) + shard_size - 1) / shard_size;
    uint32_t shard_max_keys = (uint32_t)(shard_size / 1.005);
    uint32_t last_shard_size = (uint32_t)(nkeys * f) - (nshards - 1) * shard_size;
    uint32_t last_shard_max_keys = (uint32_t)(last_shard_size / 1.005);
    uint32_t all_shards_size = shard_size * (nshards - 1) + last_shard_size;

    uint32_t sqrt_nshards = 1;
    while (sqrt_nshards * sqrt_nshards < nshards) {
        ++sqrt_nshards;
    }

    uint32_t *pinned_shard_counts = new uint32_t[nshards];
    uint32_t *inherited_shard_counts = new uint32_t[nshards];
    uint32_t *bumped_section_threshold = new uint32_t[nshards];
    std::array<uint32_t,threshold_range> *shard_section_counts = new std::array<uint32_t,threshold_range>[nshards];
    for (uint32_t i = 0; i < nshards; ++i) {
        pinned_shard_counts[i] = 0;
        inherited_shard_counts[i] = 0;
        bumped_section_threshold[i] = 0;
        for (uint32_t j = 0; j < threshold_range; ++j) {
            shard_section_counts[i][j] = 0;
        }
    }
    for (uint32_t i = 0; i < nkeys; ++i) {
        uint32_t shard = fastrange32((uint32_t)rand(), all_shards_size) / shard_size;
        if (shard == nshards - 1 || ((uint32_t)rand() % unpinned_denom) < unpinned_numer) {
            uint32_t section = (uint32_t)rand() % threshold_range;
            shard_section_counts[shard][section]++;
        } else {
            pinned_shard_counts[shard]++;
        }
    }

    uint32_t iterations = 0;
    bool change;
    do {
        change = false;
        for (uint32_t i = 0; i < nshards; ++i) {
            uint32_t current = pinned_shard_counts[i] + inherited_shard_counts[i];
            uint32_t cur_shard_max_keys = (i < nshards - 1) ? shard_max_keys : last_shard_max_keys;
            if (current > cur_shard_max_keys) {
                // fallback
                continue;
            }
            uint32_t kept_sections = threshold_range - bumped_section_threshold[i];
            for (uint32_t j = 0; j < kept_sections; ++j) {
                current += shard_section_counts[i][j];
            }
            while (current > cur_shard_max_keys) {
                assert(kept_sections > 0);
                --kept_sections;
                bumped_section_threshold[i]++;
                uint32_t to_bump = shard_section_counts[i][kept_sections];
                current -= to_bump;
                for (uint32_t j = 0; j < to_bump; ++j) {
                    inherited_shard_counts[((i + 1) * shard_size + fastrange32((uint32_t)rand(), sqrt_nshards * shard_size)) % all_shards_size / shard_size]++;
                }
                change = true;
            }
        }
        ++iterations;
    } while (change);


    uint64_t fallback_count = 0;
    for (uint32_t i = 0; i < nshards; ++i) {
        uint32_t current = pinned_shard_counts[i] + inherited_shard_counts[i];
        uint32_t cur_shard_max_keys = (i < nshards - 1) ? shard_max_keys : last_shard_max_keys;
        if (current > cur_shard_max_keys) {
            ++fallback_count;
        } else {
            uint32_t kept_sections = threshold_range - bumped_section_threshold[i];
            uint32_t kept = 0;
            for (uint32_t j = 0; j < kept_sections; ++j) {
                kept += shard_section_counts[i][j];
            }
            assert (current + kept <= shard_max_keys);
        }
    }

    std::cout << "keys " << nkeys << " over " << nshards << " shards, unpinned_denom " << unpinned_denom << " threshold_range " << threshold_range << std::endl;
    std::cout << "shard_max_keys: " << shard_max_keys << " shard_size: " << shard_size << " (" << (100.0 * shard_max_keys / shard_size) << "%)" << " last_shard: " << last_shard_max_keys << " / " << last_shard_size << std::endl;
    std::cout << "utilization: " << (100.0 * nkeys / all_shards_size) << "%)" << std::endl;
    std::cout << "fallback_count: " << fallback_count << " pct " << (100.0 * fallback_count / nshards) << std::endl;
    std::cout << "iterations: " << iterations << std::endl;

    for (uint32_t i = 0; i < nshards && i < 20; ++i) {
        uint32_t kept_sections = threshold_range - bumped_section_threshold[i];
        uint32_t kept = 0;
        for (uint32_t j = 0; j < kept_sections; ++j) {
            kept += shard_section_counts[i][j];
        }
        std::cout << "@" << i << " " << pinned_shard_counts[i] << " + " << kept << " (" << kept_sections << "/" << threshold_range << ") + " << inherited_shard_counts[i] << " = " << (pinned_shard_counts[i] + kept + inherited_shard_counts[i]) << std::endl;
    }

    return 0;
}
