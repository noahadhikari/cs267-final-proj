#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

#include <utility>


using ValueType = double;
using IndexValue = std::pair<size_t, ValueType>;
using SparseVector = std::vector<IndexValue>;

std::vector<ValueType> convert_to_dense(const SparseVector& vec, size_t length) {
    std::vector<ValueType> dense_vector(length);
    for (const auto& [index, value] : vec) {
        dense_vector[index] = value;
    }
    return dense_vector;
}

SparseVector sparse_uniform_vector(size_t length, double density) {
    std::uniform_real_distribution<> dist(0, 1);
    return sparse_vector(length, dist, density);
}

SparseVector sparse_geometric_vector(size_t length, double p, double density) {
    std::geometric_distribution<> dist(p);
    return sparse_vector(length, dist, density);
}

SparseVector sparse_poisson_vector(size_t length, double lambda, double density) {
    std::poisson_distribution<> dist(lambda);
    return sparse_vector(length, dist, density);
}

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
* This code generates dense vectors of random real numbers in [0, 1].
* The vectors can be generated using different distributions (uniform, geometric, Poisson).
* The desired density of the vectors can also be specified.
*/
template <typename Distribution>
std::vector<ValueType> dense_vector(size_t length, Distribution& dist, double goal_density) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<> uniform_dist(0., 1.);
    std::vector<ValueType> result(length);
    size_t count = 0;
    
    do {
        size_t index = length * dist(rng);
        if (index < length) {
            // check if the index isn't already filled
            if (result[index] == 0.0) {
                ValueType value = generate_value(uniform_dist, rng);
                result[index] = value;
                ++count;
            }
        }
    } while (static_cast<double>(count) / length < goal_density);

    return result;
}


/* 
* This code generates sparse vectors (represented as a vector of index->value) of random real numbers in [0, 1].
* The vectors can be generated using different distributions (uniform, geometric, Poisson).
* The desired density of the vectors can also be specified.
*/
template <typename Distribution>
SparseVector sparse_vector(size_t length, Distribution& dist, double goal_density) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<> uniform_dist(0., 1.);

    SparseVector result;

    size_t count = 0;

    do {
        size_t index = length * dist(rng);
        if (index < length) {
            // check if the index isn't already filled
            if (!contains_index(result, index)) {
                ValueType value = generate_value(uniform_dist, rng);
                result.emplace_back(index, value);
                ++count;
            }
        }
    } while (static_cast<double>(count) / length < goal_density);

    // sort the vector, default comparison is by the first element of the pair, which is the index, which is what we want
    std::sort(result.begin(), result.end());

    return result;
}