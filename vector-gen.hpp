#ifndef __CS267_VECTOR_GEN_HPP__
#define __CS267_VECTOR_GEN_HPP__


#include <algorithm>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

#include <utility>


using ValueType = int;
using IndexValue = std::pair<size_t, ValueType>;
using SparseVector = std::vector<IndexValue>;
using DenseVector = std::vector<ValueType>;

// ----------------
// Helper functions
// ----------------


ValueType generate_value(std::uniform_real_distribution<> &dist, std::mt19937 &rng) {
    // return dist(rng);

    // just return 1 for now for testing purposes
    return 1.0;
}

bool contains_index(const SparseVector& v, size_t index) {
    for (const auto& pair : v) {
        if (pair.first == index) {
            return true;
        }
    }
    return false;
}


/* 
* This code generates sparse vectors (represented as a vector of index->value) of random real numbers in [0, 1].
* The vectors can be generated using different distributions (uniform, geometric, Poisson).
* The desired density of the vectors can also be specified.

* multiplier is the factor to scale the generated index to have an expected value of length/2 (analogous to uniform distribution)
*/
template <typename Distribution>
SparseVector sparse_vector(long seed, size_t length, Distribution& dist, double goal_density, double multiplier) {
    std::mt19937 rng;
    rng.seed(seed);
    std::uniform_real_distribution<> uniform_dist(0., 1.);

    SparseVector result;

    size_t count = 0;

    do {
        size_t index = multiplier * dist(rng);
        if (index < length) {
            // check if the index isn't already filled
            if (!contains_index(result, index)) {
                ValueType value = generate_value(uniform_dist, rng);
                result.emplace_back(index, value);
                ++count;
            }
        }
    } while ((static_cast<double>(count) / length) < goal_density);

    // sort the vector, default comparison is by the first element of the pair, which is the index, which is what we want
    std::sort(result.begin(), result.end());

    return result;
}

DenseVector convert_to_dense(const SparseVector& vec, size_t length) {
    DenseVector dense_vector(length);
    for (const auto& pair: vec) {
        dense_vector[pair.first] = pair.second;
    }
    return dense_vector;
}

SparseVector sparse_uniform_vector(long seed, size_t length, double density) {
    std::uniform_real_distribution<> dist(0, 1);
    return sparse_vector(seed, length, dist, density, length);
}

SparseVector sparse_exponential_vector(long seed, size_t length, double lambda, double density) {
    std::exponential_distribution<> dist(lambda);
    return sparse_vector(seed, length, dist, density, length * lambda / 2);
}

SparseVector sparse_poisson_vector(long seed, size_t length, double lambda, double density) {
    std::poisson_distribution<> dist(lambda);
    return sparse_vector(seed, length, dist, density, length * 1 / (2 * lambda));
}

#endif
