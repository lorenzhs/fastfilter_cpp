// This benchmark reports on the bulk insert and bulk query rates. It is invoked
// as:
//
//     ./bulk-insert-and-query.exe 158000
//
// That invocation will test each probabilistic membership container type with
// 158000 randomly generated items. It tests bulk Add() from empty to full and
// Contain() on filters with varying rates of expected success. For instance, at
// 75%, three out of every four values passed to Contain() were earlier Add()ed.
//
// Example usage:
//
// for alg in `seq 0 1 14`; do for num in `seq 10 10 200`; do
// ./bulk-insert-and-query.exe ${num}000000 ${alg}; done; done > results.txt

#include "bench_common.hpp"

using std::cerr;
using std::cout;
using std::endl;

auto SeqInputGen(size_t add_count, uint64_t seed, size_t actual_sample_size) {
    std::cout << "Using add_count = " << add_count << " = " << add_count / 1e6
              << " million input items and actual_sample_size = "
              << actual_sample_size << " = " << actual_sample_size / 1e6
              << " million queries" << std::endl;

    std::vector<uint64_t> to_add = GenerateRandom64Fast(add_count, seed);
    std::vector<uint64_t> to_lookup =
        GenerateRandom64Fast(actual_sample_size, seed + add_count);

    assert(to_lookup.size() == actual_sample_size);

    size_t distinct_lookup, distinct_add;
    std::cout << "checking match size... " << std::flush;
    size_t intersectionsize =
        match_size(to_lookup, to_add, &distinct_lookup, &distinct_add);
    std::cout << "\r                       \r" << std::flush;

    if (intersectionsize > 0) {
        cout << "WARNING: Out of the lookup table, " << intersectionsize << " ("
             << intersectionsize * 100.0 / to_lookup.size()
             << "%) of values are present in the filter." << endl;
    }

    if (distinct_lookup != to_lookup.size()) {
        cout << "WARNING: Lookup contains "
             << (to_lookup.size() - distinct_lookup) << " duplicates." << endl;
    }
    if (distinct_add != to_add.size()) {
        cout << "WARNING: Filter contains " << (to_add.size() - distinct_add)
             << " duplicates." << endl;
    }

    std::vector<samples_t> mixed_sets;

    const std::vector<double> found_probabilities = {0.0, 0.5, 1.0};

    for (const double found_probability : found_probabilities) {
        std::cout << "generating samples with probability " << found_probability
                  << " ... " << std::flush;

        struct samples thisone;
        thisone.found_probability = found_probability;
        thisone.actual_sample_size = actual_sample_size;
        uint64_t mixingseed = seed + 12345 * found_probability + 42;
        thisone.to_lookup_mixed = DuplicateFreeMixIn(
            &to_lookup[0], &to_lookup[actual_sample_size], &to_add[0],
            &to_add[add_count], found_probability, mixingseed);
        assert(thisone.to_lookup_mixed.size() == actual_sample_size);
        thisone.true_match =
            match_size(thisone.to_lookup_mixed, to_add, NULL, NULL);
        const double trueproba = thisone.true_match * 1.0 / actual_sample_size,
                     probadiff = fabs(trueproba - found_probability);
        if (probadiff >= 0.001) {
            cerr << "WARNING: You claim to have a find probability of "
                 << found_probability << " but actual is " << trueproba << endl;
            if (probadiff >= 0.01)
                exit(EXIT_FAILURE);
            else
                cerr << "(ignoring, below 0.01 is acceptable)" << endl;
        }
        mixed_sets.push_back(thisone);
        std::cout
            << "\r                                                           "
               "                              \r"
            << std::flush;
    }
    return std::make_tuple(to_add, mixed_sets, found_probabilities,
                           intersectionsize);
}

template <typename Table>
Statistics SeqFilterBenchmark(size_t add_count, const std::vector<uint64_t> &to_add,
                              size_t intersectionsize,
                              const std::vector<samples_t> &mixed_sets,
                              bool batchedadd, uint64_t seed) {
    if (add_count > to_add.size()) {
        throw out_of_range("to_add must contain at least add_count values");
    }

    Table filter = FilterAPI<Table>::ConstructFromAddCount(add_count);
    Statistics result;
    std::cout << "-" << std::flush;

    // Add values until failure or until we run out of values to add:
    if (batchedadd) {
        std::cout << "batched add" << std::flush;
    } else {
        std::cout << "1-by-1 add" << std::flush;
    }
    auto start_time = NowNanos();

    try {
        if (batchedadd) {
            FilterAPI<Table>::AddAll(to_add, 0, add_count, &filter);
        } else {
            for (size_t added = 0; added < add_count; ++added) {
                FilterAPI<Table>::Add(to_add[added], &filter);
            }
        }
    } catch (const std::runtime_error &e) {
        std::cerr << "Failed to construct filter: " << e.what() << std::endl;
        result.add_count = 0;
        result.nanos_per_add = 0;
        result.nanos_per_remove = 0;
        for (const auto &t : mixed_sets) {
            result.nanos_per_finds[100 * t.found_probability] = 0;
        }
        result.false_positive_probabilty = 1;
        result.bits_per_item = 1;
        return result;
    }

    auto time = NowNanos() - start_time;
    std::cout << "\r             \r" << std::flush;
    std::cout << "." << std::flush;

#ifndef NDEBUG
    // sanity check:
    for (size_t added = 0; added < add_count; ++added) {
        assert(FilterAPI<Table>::Contain(to_add[added], &filter) == 1);
    }
#endif

    result.add_count = add_count;
    result.nanos_per_add = static_cast<double>(time) / add_count;
    result.bits_per_item =
        static_cast<double>(CHAR_BIT * filter.SizeInBytes()) / add_count;

    // ensure at least MIN_SAMPLE_SIZE queries are performed
    const size_t numqueries = mixed_sets[0].actual_sample_size,
                 rounds = (MIN_SAMPLE_SIZE + numqueries - 1) / numqueries;
    for (auto t : mixed_sets) {
        const double found_probability = t.found_probability;
        const auto to_lookup_mixed = t.to_lookup_mixed;
        size_t true_match = t.true_match;
        std::cout << "-" << std::flush;

        const auto start_time = NowNanos();
        size_t found_count = 0;
        for (size_t round = 0; round < rounds; ++round) {
            for (const auto v : to_lookup_mixed) {
                found_count += FilterAPI<Table>::Contain(v, &filter);
            }
        }
        // now restore fprate calculations by counting only 1 round
        found_count /= rounds;

        const auto lookup_time = NowNanos() - start_time;
        std::cout << "." << std::flush;

        if (found_count < true_match) {
            cerr << "ERROR: Expected to find at least " << true_match
                 << " found " << found_count << endl;
            cerr << "ERROR: This is a potential bug!" << endl;
            // Indicate failure
            result.add_count = 0;
        }
        result.nanos_per_finds[100 * found_probability] =
            static_cast<double>(lookup_time) / (rounds * t.actual_sample_size);
        if (0.0 == found_probability) {
            ////////////////////////////
            // This is obviously technically wrong!!! The assumption is that there is
            // no overlap between the random queries and the random content. This is
            // likely true if your 64-bit values were generated randomly, but not true
            // in general. NOTE(PD): the above objection is only valid if hashes added
            // are already guaranteed unique (unusual).
            ///////////////////////////
            // result.false_positive_probabilty =
            //    found_count / static_cast<double>(to_lookup_mixed.size());
            if (t.to_lookup_mixed.size() == intersectionsize) {
                cerr << "WARNING: fpp is probably meaningless! " << endl;
            }
            uint64_t positives = found_count - intersectionsize;
            uint64_t samples = to_lookup_mixed.size() - intersectionsize;

            if (positives * samples < 10000000000ULL) {
                // cerr << "NOTE: getting more samples for accurate FP rate" << endl;
                pcg64 rnd(seed);
                while (positives * samples < 10000000000ULL) {
                    // Need more samples for accurate FP rate
                    positives += FilterAPI<Table>::Contain(rnd(), &filter);
                    samples++;
                }
            }

            result.false_positive_probabilty = 1.0 * positives / samples;
        }
    }

    // Not testing remove
    result.nanos_per_remove = 0;
    std::cout << "\r             \r" << std::flush;

    return result;
}

template <typename Filter>
struct SeqFilterBenchmarkRunner {
    SeqFilterBenchmarkRunner() = default;

    template <typename... Args>
    Statistics operator()(Args &&...args) {
        return SeqFilterBenchmark<Filter>(std::forward<Args>(args)...);
    }
};

int main(int argc, char *argv[]) {
    auto sorter = [](auto &vec) {
        std::sort(vec.begin(), vec.end());
        return vec.size();
    };
    return do_main<SeqFilterBenchmarkRunner>(SeqInputGen, sorter, argc, argv);
}
