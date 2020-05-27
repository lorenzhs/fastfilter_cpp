#include <array>
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
        // Without grouping by 16, ~ 1.0032
        // With grouping by 16, ~ 1.0042
        start &= ~uint32_t{15};
        assert(start < len - 63);
        row = (h + 0x9e3779b97f4a7c13) * 0x9e3779b97f4a7c13;
        row ^= h >> 32;
        row |= (uint64_t{1} << 63);
        pivot = 0;
    }
};

static inline uint32_t getShard(uint64_t h, uint32_t shards) {
    return fastrange32((uint32_t)(h >> 32), shards);
}

static inline uint32_t getSection(uint64_t h) {
    uint32_t v = h & 1023;
    if (v < 300) {
        return v / 3;
    } else if (v < 428) {
        return v - 200;
    } else if (v < 512) {
        return (v + 256) / 3;
//    } else if (v < 532) {
//        return (v + 1516) / 8;
    } else {
        return 0;
    }
}

static inline uint64_t rot64(uint64_t h, int count) {
    return (h << count) | (h >> (64 - count));
}

int main(int argc, char *argv[]) {
    std::mt19937_64 rand(getpid());

    uint32_t nkeys = (uint32_t)std::atoi(argv[1]);
    double f = std::atof(argv[2]);
    uint32_t lenish = (uint32_t)(f * nkeys + 0.5);
    uint32_t shards = 1;
    while (lenish / shards > 1414) {
        shards *= 2;
    }
    uint32_t avg_len_per_shard = (lenish + shards / 2) / shards;
    uint32_t min_len_per_shard = avg_len_per_shard & ~uint32_t{63};
    uint32_t max_len_per_shard = (avg_len_per_shard + 63) & ~uint32_t{63};

    std::array<std::vector<uint64_t>, 256> *hashes = new std::array<std::vector<uint64_t>, 256>[shards];
    for (uint32_t i = 0; i < nkeys; ++i) {
        uint64_t h = (uint64_t)rand();
        if ((h & uint64_t{0x8000000000000380}) == uint64_t{0x8000000000000380}) {
            h -= uint64_t{0x8000000000000000};
        }
        hashes[getShard(h, shards)][getSection(h)].push_back(h);
    }

    GaussData *data = new GaussData[max_len_per_shard];
    std::vector<uint64_t> shard_hashes;
    std::vector<uint64_t> *bumped = new std::vector<uint64_t>[shards];

    for (uint32_t shard = 0; shard < shards; ++shard) {
        uint32_t len_this_shard = ((shard * avg_len_per_shard + 63 + avg_len_per_shard) & ~uint32_t{63}) - ((shard * avg_len_per_shard + 63) & ~uint32_t{63});
        assert(len_this_shard == min_len_per_shard || len_this_shard == max_len_per_shard);

        uint32_t last_section = 0;
        size_t kept_count = hashes[shard][last_section].size() + bumped[shard].size();
        for (; last_section < 255; ++last_section) {
            size_t next_count = hashes[shard][last_section + 1].size();
            if (kept_count + next_count > len_this_shard) {
                break;
            }
            kept_count += next_count;
        }
        std::cout << "pre-kept@" << shard << " = " << kept_count << " / " << len_this_shard << " (" << (1.0 * kept_count / len_this_shard) << ") (last=" << last_section << ")" << std::endl;
        if (shard == shards - 1) {
            // no more bumps
            if (last_section < 255) {
                uint32_t overflow_count = 0;
                for (uint32_t i = last_section + 1; i < 256; ++i) {
                    overflow_count += hashes[shard][i].size();
                }
                std::cout << "overflow! " << overflow_count << std::endl;
                return 1;
            }
        } else {
            if (kept_count > len_this_shard) {
                std::cout << "early overflow!" << std::endl;
                return 1;
            }
        }

        retry:
        uint64_t seed = rot64(uint64_t{0x9e3779b97f4a7c13}, (last_section * 13) & 63);
        for (uint64_t h : bumped[shard]) {
            shard_hashes.push_back(h /** seed */);
        }
        for (uint32_t i = 0; i <= last_section; ++i) {
            for (uint64_t h : hashes[shard][i]) {
                shard_hashes.push_back(rot64(h, (last_section * 39) & 63) * 0x9e3779b97f4a7c13);
                //shard_hashes.push_back(h * seed);
            }
        }
        assert(kept_count == shard_hashes.size());
        std::sort(shard_hashes.begin(), shard_hashes.end());
        for (uint64_t i = 0; i < kept_count; ++i) {
            data[i].Reset(shard_hashes[i], len_this_shard);
        }
        shard_hashes.clear();
        for (uint32_t i = 0; i < kept_count; ++i) {
            GaussData &di = data[i];
            if (di.row == 0) {
                if (last_section == 0) {
                    std::cout << "early2 overflow!" << std::endl;
                    return 1;
                }
                kept_count -= hashes[shard][last_section].size();
                --last_section;
                goto retry;
            }
            int tz = __builtin_ctzl(di.row);
            di.pivot = di.start + tz;
            for (uint32_t j = i + 1; j < kept_count; ++j) {
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
        // OK
        std::cout << "kept@" << shard << " = " << kept_count << " / " << len_this_shard << " (" << (1.0 * kept_count / len_this_shard) << ") (last=" << last_section << ")" << std::endl;
        if (shard < shards - 1) {
            for (uint32_t i = last_section + 1; i < 256; ++i) {
                // bump
                uint64_t keep_mask = shards / 2;
                if (keep_mask > 0) {
                    while ((shard & keep_mask) == keep_mask && (keep_mask & 1) == 0) {
                        keep_mask |= keep_mask / 2;
                    }
                    while (keep_mask < uint64_t{0x8000000000000000}) {
                        keep_mask <<= 1;
                    }
                }
                uint64_t other_mask = ~keep_mask >> 1;
                for (uint64_t h : hashes[shard][i]) {
                    uint64_t rot_h = (h >> 32) | (h << 32);
                    uint64_t alt_h = (uint64_t{0x8000000000000000} | (h >> 1)) ^ (rot_h & other_mask);
                    uint32_t new_shard = getShard(alt_h, shards);
                    assert(new_shard > shard);
                    bumped[new_shard].push_back(h * seed);
                }
            }
        }
    }

    return 0;
}
