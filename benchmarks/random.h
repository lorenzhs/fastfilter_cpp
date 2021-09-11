// Generating random data

#pragma once

// sorry about the include path...
#include <bumpribbon/pcg-cpp/include/pcg_random.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <random>
#include <stdexcept>
#include <vector>

std::vector<uint64_t> GenerateRandom64Fast(size_t count, uint64_t start) {
    std::vector<uint64_t> result(count);
    pcg64 rng(start);
    std::generate(result.begin(), result.end(), [&rng]() { return rng(); });
    return result;
}


static inline uint64_t biased_random_bounded(uint64_t range, __uint128_t* seed) {
    __uint128_t random64bit, multiresult;
    *seed *= UINT64_C(0xda942042e4dd58b5);
    random64bit = *seed >> 64;
    multiresult = random64bit * range;
    return multiresult >> 64; // [0, range)
}

static inline uint64_t random_bounded(uint64_t range, __uint128_t* seed) {
    __uint128_t random64bit, multiresult;
    uint64_t leftover;
    uint64_t threshold;
    *seed *= UINT64_C(0xda942042e4dd58b5);
    random64bit = *seed >> 64;
    multiresult = random64bit * range;
    leftover = (uint64_t)multiresult;
    if (leftover < range) {
        threshold = -range % range;
        while (leftover < threshold) {
            *seed *= UINT64_C(0xda942042e4dd58b5);
            random64bit = *seed >> 64;
            multiresult = random64bit * range;
            leftover = (uint64_t)multiresult;
        }
    }
    return multiresult >> 64; // [0, range)
}

template <typename It, typename OutIt, typename RNG>
// pick capacity elements form x_begin, x_end, write them to storage
void reservoirsampling(OutIt storage, uint32_t capacity, const It x_begin,
                       const It x_end, RNG& rng) {
    if (capacity == 0)
        return;
    size_t size = x_end - x_begin;
    if (size < capacity) {
        throw std::logic_error("I cannot sample the requested number. This "
                               "is not going to end well.");
    }
    size_t i;
    for (i = 0; i < capacity; i++) {
        *(storage + i) = *(x_begin + i);
    }
    while (i < size) {
        size_t nextpos = rng(i);
        if (nextpos < capacity) {
            *(storage + nextpos) = *(x_begin + i);
        }
        i++;
    }
}

// Using two pointer ranges for sequences x and y, create a vector clone of x
// but for y_probability y's mixed in.
template <typename It, typename T = typename std::iterator_traits<It>::value_type>
std::vector<T> DuplicateFreeMixIn(const It x_begin, const It x_end,
                                  const It y_begin, const It y_end,
                                  double y_probability, uint64_t seed,
                                  uint64_t result_size = 0) {
    const size_t x_size = x_end - x_begin; //, y_size = y_end - y_begin;
    // use size of x if size parameter is 0
    const size_t size = result_size == 0 ? x_size : result_size;

    std::vector<T> result(size);
    size_t howmanyy = round(size * y_probability);
    size_t howmanyx = size - howmanyy;
    pcg64 rng(seed);
    reservoirsampling(result.data(), howmanyx, x_begin, x_end, rng);
    reservoirsampling(result.data() + howmanyx, howmanyy, y_begin, y_end, rng);
    for (size_t i = 0; i + 1 < size; ++i) {
        std::swap(result[i], result[i + 1 + rng(size - i - 1)]);
    }
    return result;
}
