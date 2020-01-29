#include <vector>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <algorithm>
#include <unistd.h>

inline size_t fastrange64(uint64_t hash, size_t range) {
  __uint128_t wide = __uint128_t{range} * hash;
  return static_cast<size_t>(wide >> 64);
}

inline uint32_t fastrange32(uint32_t hash, uint32_t range) {
    uint64_t wide = uint64_t{hash} * range;
    return static_cast<uint32_t>(wide >> 32);
}

static size_t seg_len = 0;
static size_t segs_base = 0;
static bool segs_not_shards = false;

size_t r0_3(uint64_t h) {
    size_t rv = (h % segs_base) * seg_len;
    h /= segs_base;
    return rv + (h % seg_len);
}
size_t r1_3(uint64_t h) {
    size_t rv = ((h % segs_base) + (1 * segs_not_shards)) * seg_len;
    h /= segs_base;
    h /= seg_len;
    return rv + (h % seg_len);
}
size_t r2_3(uint64_t h) {
    size_t rv = (((h % segs_base) + (2 * segs_not_shards)) % (segs_base + 1)) * seg_len;
    h /= segs_base;
    h /= seg_len;
    h /= seg_len;
    return rv + h;
}

void remove(std::vector<uint64_t>& v, uint64_t e) {
    v.erase(std::find(v.begin(), v.end(), e));
}

void insert(std::vector<uint64_t>& v, uint64_t e) {
    if (std::find(v.begin(), v.end(), e) == v.end()) {
        v.push_back(e);
    }
}

int main(int argc, char *argv[]) {
    std::mt19937_64 rand(getpid());

    size_t nkeys = (size_t)std::atoi(argv[1]);
    size_t len = (size_t)std::atoi(argv[2]);
    int segs_or_shards = std::atoi(argv[3]);
    if (segs_or_shards < 0) {
        // negative -> shards  (-1 -> Xor filter)
        segs_base = (size_t)-segs_or_shards;
        segs_not_shards = false;
        seg_len = len / segs_base;
        len = seg_len * segs_base;
    } else if (segs_or_shards > 0) {
        // positive -> segments
        segs_base = (size_t)segs_or_shards;
        segs_not_shards = true;
        seg_len = len / (segs_base + 1);
        len = seg_len * (segs_base + 1);
    } else {
        return 1;
    }

    std::vector<uint64_t> *arr = new std::vector<uint64_t>[len];

    size_t collision2 = 0;
    size_t collision3 = 0;
    size_t good_collision = 0;
    uint64_t mod = seg_len * seg_len * seg_len * segs_base;
    for (size_t i = 0; i < nkeys; ++i) {
        uint64_t h = (uint64_t)rand() % mod;
        size_t h0 = r0_3(h);
        if (std::find(arr[h0].begin(), arr[h0].end(), h) != arr[h0].end()) {
            good_collision++;
        } else {
            arr[h0].push_back(h);
            size_t h1 = r1_3(h);
            arr[h1].push_back(h);
            size_t h2 = r2_3(h);
            arr[h2].push_back(h);
            if (h0 == h1 || h1 == h2 || h0 == h2) {
                collision3++;
            }
        }
    }

    size_t initial_unmapped = 0;
    size_t max_overlap = 0;

    for (size_t i = 0; i < len; ++i) {
        if (arr[i].empty()) {
            initial_unmapped++;
        }
        max_overlap = std::max(max_overlap, arr[i].size());
    }

    size_t initial_run = 0;
    size_t kicked = 0;
    size_t later_mapped = 0;

    bool more_todo;
    do {
        more_todo = false;
        bool processed_single = false;
        for (size_t i = 0; i < len; ++i) {
            size_t count = arr[i].size();
            if (count == 0) {
                continue;
            } else if (count == 1) {
                processed_single = true;
                if (kicked == 0) {
                    initial_run++;
                } else {
                    later_mapped++;
                }
                uint64_t h = arr[i][0];
                for (size_t j : {r0_3(h), r1_3(h), r2_3(h)}) {
                    remove(arr[j], h);
                }
            } else {
                more_todo = true;
            }
        }
        if (!processed_single && more_todo) {
            bool good_kick = false;
            for (size_t i = 0; i < len; ++i) {
                size_t count = arr[i].size();
                if (count == 2) {
                    kicked++;
                    uint64_t h = arr[i][0];
                    for (size_t j : {r0_3(h), r1_3(h), r2_3(h)}) {
                        remove(arr[j], h);
                    }
                    good_kick = true;
                    break;
                }
            }
            if (!good_kick) {
                for (size_t i = 0; i < len; ++i) {
                    size_t count = arr[i].size();
                    if (count > 1) {
                        kicked++;
                        uint64_t h = arr[i][0];
                        for (size_t j : {r0_3(h), r1_3(h), r2_3(h)}) {
                            remove(arr[j], h);
                        }
                        break;
                    }
                }
            }
        }
    } while (more_todo);

    std::cout << "3x" << nkeys << " over " << len << ":" << std::endl;
    std::cout << "good collision " << good_collision << ", collision2 " << collision2 << ", collision3 " << collision3 << std::endl;
    std::cout << "initial_unmapped: " << initial_unmapped << " (" << (100.0 * initial_unmapped / len) << "%)" << std::endl;
    std::cout << "max_overlap: " << max_overlap << std::endl;
    std::cout << "initial_run: " << initial_run << " (" << (100.0 * initial_run / len) << "%)" << std::endl;
    std::cout << "later_mapped: " << later_mapped << " (" << (100.0 * later_mapped / len) << "%)" << std::endl;
    std::cout << "kicked: " << kicked << " (" << (100.0 * kicked / len) << "%)" << std::endl;
    return 0;
}
