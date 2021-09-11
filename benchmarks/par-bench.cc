// This benchmark reports on the bulk insert and bulk query rates in a parallel
// setting. It is invoked as:
//
//     ./par-bench.exe 16 1024 158000
//
// That invocation will use 16 threads and 1024 filters. It will test each
// probabilistic membership container type with 158000 randomly generated
// items. It tests bulk Add() from empty to full and Contain() on filters with
// varying rates of expected success. For instance, at 75%, three out of every
// four values passed to Contain() were earlier Add()ed. For each query, a
// random filter is chosen. This puts high load on the memory system.
//
// Example usage:
//
// for alg in `seq 0 1 14`; do for num in `seq 10 10 200`; do
// ./par-bench.exe ${num}000000 ${alg}; done; done > results.txt

#include "../../ribbon/pcg-cpp/include/pcg_random.hpp"
#include "bench_common.hpp"

#include <tlx/thread_pool.hpp>

#include <numeric>
#include <thread>

static size_t num_threads = 2, num_filters = 8;

// add_count items *per filter*
// actual_sample_size queries *per thread*
auto ParInputGen(size_t add_count, uint64_t seed, size_t actual_sample_size) {
    std::cout << "Using add_count = " << add_count << " = " << add_count / 1e6
              << " million input items per filter and actual_sample_size = "
              << actual_sample_size << " = " << actual_sample_size / 1e6
              << " million queries per thread" << std::endl;

    // generate seeds
    std::vector<uint32_t> seeds32(4 * num_filters + 2 * num_threads + 2);
    std::vector<uint64_t> seeds(2 * num_filters + num_threads + 1);
    std::seed_seq seq{seed, seed >> 32};
    seq.generate(seeds32.begin(), seeds32.end());
    for (size_t i = 0; i < seeds.size(); i++) {
        // combine two 32-bit seeds
        seeds[i] = static_cast<uint64_t>(seeds32[2 * i]) |
                   (static_cast<uint64_t>(seeds32[2 * i + 1]) << 32);
    }

    std::cout << "Generating input data..." << std::endl;

    std::vector<std::vector<uint64_t>> to_add(num_filters);
    std::vector<size_t> distinct_lookup(num_filters), distinct_add(num_filters),
        intersectionsize(num_filters);

    // pairs (filter, key)
    const size_t bytes_per_item = sizeof(std::pair<uint64_t, uint64_t>);
    std::cout << "Allocating " << bytes_per_item * num_filters * add_count
              << " Bytes for inputs and "
              << bytes_per_item * num_filters * actual_sample_size
              << " for queries" << std::endl;
    std::vector<std::pair<uint64_t, uint64_t>> add_pairs(num_filters * add_count),
        lookup_pairs(num_filters * actual_sample_size);

    tlx::ThreadPool pool(num_threads);
    for (size_t filter_id = 0; filter_id < num_filters; filter_id++) {
        pool.enqueue([&, filter_id]() {
            // Generate items to add and items that aren't added for this filter
            to_add[filter_id] =
                GenerateRandom64Fast(add_count, seeds[2 * filter_id]);
            auto to_lookup = GenerateRandom64Fast(actual_sample_size,
                                                  seeds[2 * filter_id + 1]);
            assert(to_lookup.size() == actual_sample_size);

            // Check for any items that happen to be in both arrays or
            // repeatedly in either of them
            intersectionsize[filter_id] =
                match_size(to_lookup, to_add[filter_id],
                           &distinct_lookup[filter_id], &distinct_add[filter_id]);

            if (intersectionsize[filter_id] > 0) {
                cout << "WARNING: Out of the lookup table of filter "
                     << filter_id << ", " << intersectionsize[filter_id] << " ("
                     << intersectionsize[filter_id] * 100.0 / to_lookup.size()
                     << "%) of values are present in the filter." << endl;
            }
            if (distinct_lookup[filter_id] != to_lookup.size()) {
                cout << "WARNING: Lookup for filter " << filter_id << " contains "
                     << (to_lookup.size() - distinct_lookup[filter_id])
                     << " duplicates." << endl;
            }
            if (distinct_add[filter_id] != to_add[filter_id].size()) {
                cout << "WARNING: Filter " << filter_id << " contains "
                     << (to_add[filter_id].size() - distinct_add[filter_id])
                     << " duplicates." << endl;
            }

            // Insert them into the global query arrays as pairs of (filter_id, key)
            std::transform(to_lookup.begin(), to_lookup.end(),
                           lookup_pairs.begin() + filter_id * actual_sample_size,
                           [&filter_id](const auto &x) {
                               return std::make_pair(filter_id, x);
                           });

            std::transform(to_add[filter_id].begin(), to_add[filter_id].end(),
                           add_pairs.begin() + filter_id * add_count,
                           [&filter_id](const auto &x) {
                               return std::make_pair(filter_id, x);
                           });

            // Sort local part of add_pairs so that the whole thing is
            // lexicographically sorted at the end (important for match
            // determination). Can't sort before because to_add shouldn't be
            // sorted.
            std::sort(add_pairs.begin() + filter_id * add_count,
                      add_pairs.begin() + (filter_id + 1) * add_count,
                      [](const auto &x, const auto &y) {
                          assert(x.first == y.first);
                          return x.second < y.second;
                      });
        });
    }
    pool.loop_until_empty();

    // now shuffle the lookup (negative query) keys
    std::cout << "Shuffling lookup pairs..." << std::endl;
    pcg64 rng(seeds.back());
    std::shuffle(lookup_pairs.begin(), lookup_pairs.end(), rng);

    std::cout << "Generating query sets..." << std::endl;
    std::vector<std::vector<psamples_t>> mixed_sets(num_threads);

    const std::vector<double> found_probabilities = {0.0, 0.5, 1.0};

    for (size_t thread_id = 0; thread_id < num_threads; thread_id++) {
        pool.enqueue([&, thread_id]() {
            for (const double found_probability : found_probabilities) {
                struct psamples thisone;
                thisone.found_probability = found_probability;
                thisone.actual_sample_size = actual_sample_size;
                uint64_t mixingseed = seeds[2 * num_filters + thread_id];

                thisone.to_lookup_mixed = DuplicateFreeMixIn(
                    lookup_pairs.begin(), lookup_pairs.end(), add_pairs.begin(),
                    add_pairs.end(), found_probability, mixingseed,
                    actual_sample_size);
                assert(thisone.to_lookup_mixed.size() == actual_sample_size);
                thisone.true_match =
                    match_size_pairs(thisone.to_lookup_mixed, add_pairs);
                double trueproba = thisone.true_match * 1.0 / actual_sample_size;
                const double probadiff = fabs(trueproba - found_probability);
                // tolerate 1% accidentally positive queries in the negative query set
                if (probadiff >= 0.01) {
                    cerr << "WARNING: You claim to have a find probability of "
                         << found_probability << " but actual is " << trueproba
                         << " for thread " << thread_id << endl;
                    // exit(EXIT_FAILURE);
                }
                mixed_sets[thread_id].push_back(std::move(thisone));
            }
        });
    }
    pool.loop_until_empty();
    return std::make_tuple(to_add, mixed_sets, found_probabilities,
                           intersectionsize);
}

// wrapper to work around double-frees when creating a vector<Table> because of
// the absolutely brain-dead API used here
template <typename Table>
struct wrapper {
    Table t;
    explicit wrapper(size_t size)
        : t(FilterAPI<Table>::ConstructFromAddCount(size)) {}
    //! non-copyable: delete copy-constructor
    wrapper(const wrapper &) = delete;
    //! non-copyable: delete assignment operator
    wrapper &operator=(const wrapper &) = delete;
    //! move-constructor: default
    wrapper(wrapper &&) = default;
    //! move-assignment operator: default
    wrapper &operator=(wrapper &&) = default;
};

template <typename Table>
Statistics
ParFilterBenchmark(size_t add_count, const vector<vector<uint64_t>> &to_add,
                   std::vector<size_t> intersectionsize,
                   const std::vector<std::vector<psamples_t>> &mixed_sets,
                   bool batchedadd, uint64_t /* seed_ignored */) {
    for (size_t i = 0; i < to_add.size(); i++) {
        if (add_count > to_add[i].size()) {
            throw out_of_range("to_add[" + std::to_string(i) +
                               "] must contain at least add_count values");
        }
    }

    std::vector<wrapper<Table>> filters;
    filters.reserve(num_filters);
    for (size_t i = 0; i < num_filters; i++) {
        filters.emplace_back(add_count);
    }
    Statistics result;

    std::atomic<bool> ok = true;
    tlx::ThreadPool pool(num_threads);
    const auto start_time = NowNanos();
    for (size_t filter_id = 0; filter_id < num_filters; filter_id++) {
        pool.enqueue([&, filter_id]() {
            try {
                // Add values until failure or until we run out of values to add:
                if (batchedadd) {
                    FilterAPI<Table>::AddAll(to_add[filter_id], 0, add_count,
                                             &(filters[filter_id].t));
                } else {
                    for (size_t added = 0; added < add_count; ++added) {
                        FilterAPI<Table>::Add(to_add[filter_id][added],
                                              &(filters[filter_id].t));
                    }
                }
            } catch (const std::runtime_error &e) {
                std::cerr << "Failed to construct filter " << filter_id << ": "
                          << e.what() << std::endl;
                ok = false;
            }
        });
    }
    pool.loop_until_empty();

    if (!ok) {
        // some filter failed to construct
        result.add_count = 0;
        result.nanos_per_add = 0;
        result.nanos_per_remove = 0;
        for (const auto &t : mixed_sets[0]) {
            result.nanos_per_finds[100 * t.found_probability] = 0;
        }
        result.false_positive_probabilty = 1;
        result.bits_per_item = 1;
        return result;
    }

    auto time = NowNanos() - start_time;

    // sanity check:
#ifndef NDEBUG
    for (size_t filter_id = 0; filter_id < num_filters; filter_id++) {
        pool.enqueue([&, filter_id]() {
            for (size_t added = 0; added < add_count; ++added) {
                assert(FilterAPI<Table>::Contain(to_add[filter_id][added],
                                                 &(filters[filter_id].t)) == 1);
            }
        });
    }
    pool.loop_until_empty();
#endif

    result.add_count = add_count;
    result.nanos_per_add =
        static_cast<double>(time) / (add_count * num_filters / num_threads);

    size_t bytes = 0;
    for (size_t filter_id = 0; filter_id < num_filters; filter_id++) {
        bytes += filters[filter_id].t.SizeInBytes();
    }
    result.bits_per_item =
        static_cast<double>(CHAR_BIT * bytes) / (num_filters * add_count);
    std::atomic<size_t> found_count;

    /**************************************************************************/
    // Queries.
    // Ensure at least MIN_SAMPLE_SIZE queries are performed
    const size_t numqueries = mixed_sets[0][0].actual_sample_size,
                 rounds = (MIN_SAMPLE_SIZE + numqueries - 1) / numqueries;
    for (size_t i = 0; i < mixed_sets[0].size(); i++) {
        const auto start_time = NowNanos();

        found_count = 0;
        for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
            pool.enqueue([&, thread_id]() {
                size_t my_found_count = 0;

                for (size_t round = 0; round < rounds; ++round) {
                    for (const auto& [filter, key] :
                         mixed_sets[thread_id][i].to_lookup_mixed) {
                        my_found_count +=
                            FilterAPI<Table>::Contain(key, &(filters[filter].t));
                    }
                }
                // now restore fprate calculations by counting only 1 round
                my_found_count /= rounds;
                found_count.fetch_add(my_found_count);
            });
        }
        pool.loop_until_empty();

        const auto &t = mixed_sets[0][i];
        const size_t size = t.to_lookup_mixed.size();

        const auto lookup_time = NowNanos() - start_time;

        size_t true_match = 0;
        for (size_t t = 0; t < num_threads; t++) {
            true_match += mixed_sets[t][i].true_match;
        }
        if (found_count < true_match) {
            cerr << "ERROR: Expected to find at least " << true_match
                 << " found " << found_count << endl;
            cerr << "ERROR: This is a potential bug!" << endl;
            // Indicate failure
            result.add_count = 0;
        }
        result.nanos_per_finds[100 * t.found_probability] =
            static_cast<double>(lookup_time) / (rounds * t.actual_sample_size);
        if (0.0 == t.found_probability) {
            size_t intersection = std::accumulate(intersectionsize.begin(),
                                                  intersectionsize.end(), 0);
            if (size == intersection) {
                cerr << "WARNING: fpp is probably meaningless! " << endl;
            }
            uint64_t positives = found_count - intersection;
            uint64_t samples =
                num_threads * t.to_lookup_mixed.size() - intersection;

            result.false_positive_probabilty = 1.0 * positives / samples;
        }
    }

    // Not testing remove
    result.nanos_per_remove = 0;

    return result;
}

template <typename Filter>
struct ParFilterBenchmarkRunner {
    ParFilterBenchmarkRunner() = default;

    template <typename... Args>
    Statistics operator()(Args &&...args) {
        return ParFilterBenchmark<Filter>(std::forward<Args>(args)...);
    }
};

int main(int argc, char *argv[]) {
    if (argc < 4) {
        cout << "Usage: "
             << argv[0] << " <numthreads> <numfilters> <numentries> [<algorithmId> [<seed>]]"
             << endl;
    }
    if (argc > 1)
        num_threads = atoi(argv[1]);
    if (argc > 2)
        num_filters = atoi(argv[2]);
    if (num_threads < 2 || num_filters < num_threads) {
        std::cerr << "must have at least 2 threads and as many filters, have: "
                  << num_threads << " threads and " << num_filters << " filters"
                  << std::endl;
    }
    if (num_filters % num_threads != 0) {
        std::cerr
            << "num_threads must divide num_filters cleanly, have remainder "
            << num_filters % num_threads << std::endl;
    }
    std::cout << "Running with " << num_threads << " threads and "
              << num_filters << " filters" << std::endl;

    auto sorter = [&](auto &vec) {
        tlx::ThreadPool pool(num_threads);
        for (size_t id = 0; id < vec.size(); id++) {
            pool.enqueue(
                [&vec, id]() { std::sort(vec[id].begin(), vec[id].end()); });
        }
        size_t size = 0;
        for (const auto &sub : vec) {
            size += sub.size();
        }
        pool.loop_until_empty();
        // for computing time per item and thread
        return size / num_threads;
    };
    std::string outfn = "results_par_" + std::to_string(num_threads) + "t_" +
                        std::to_string(num_filters) + "f_";
    return do_main<ParFilterBenchmarkRunner>(ParInputGen, sorter, argc - 2,
                                             argv + 2, outfn);
}
