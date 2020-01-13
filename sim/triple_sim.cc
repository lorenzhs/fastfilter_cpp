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

size_t r0(uint64_t h, size_t len) {
    return fastrange64(h, len / 3);
}
size_t r1(uint64_t h, size_t len) {
    return fastrange64((h >> 21) | (h << 43), len / 3) + len / 3;
}
size_t r2(uint64_t h, size_t len) {
    return fastrange64((h >> 42) | (h << 22), len / 3) + 2 * (len / 3);
}

void remove(std::vector<uint64_t>& v, uint64_t e) {
    v.erase(std::find(v.begin(), v.end(), e));
}

int main(int argc, char *argv[]) {
    std::mt19937_64 rand(getpid());

    size_t nkeys = (size_t)std::atoi(argv[1]);
    size_t len = (size_t)std::atoi(argv[2]);

    std::vector<uint64_t> *arr = new std::vector<uint64_t>[len];

    for (size_t i = 0; i < nkeys; ++i) {
        uint64_t h = (uint64_t)rand();
        arr[r0(h, len)].push_back(h);
        arr[r1(h, len)].push_back(h);
        arr[r2(h, len)].push_back(h);
    }

    size_t initial_unmapped = 0;

    for (size_t i = 0; i < len; ++i) {
        if (arr[i].empty()) {
            initial_unmapped++;
        }
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
                for (size_t j : {r0(h, len), r1(h, len), r2(h, len)}) {
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
                    for (size_t j : {r0(h, len), r1(h, len), r2(h, len)}) {
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
                        for (size_t j : {r0(h, len), r1(h, len), r2(h, len)}) {
                            remove(arr[j], h);
                        }
                        break;
                    }
                }
            }
        }
    } while (more_todo);

    std::cout << "3x" << nkeys << " over " << len << ":" << std::endl;
    std::cout << "initial_unmapped: " << initial_unmapped << " (" << (100.0 * initial_unmapped / len) << "%)" << std::endl;
    std::cout << "initial_run: " << initial_run << " (" << (100.0 * initial_run / len) << "%)" << std::endl;
    std::cout << "later_mapped: " << later_mapped << " (" << (100.0 * later_mapped / len) << "%)" << std::endl;
    std::cout << "kicked: " << kicked << " (" << (100.0 * kicked / len) << "%)" << std::endl;
    return 0;
}
