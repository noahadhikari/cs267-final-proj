#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

/* 
* This code generates dense vectors of random real numbers in [0, 1].
* The vectors can be generated using different distributions (uniform, geometric, Poisson).
* The desired density of the vectors can also be specified.
*/
template <typename Distribution>
std::vector<double> dense_vector(size_t length, Distribution& dist, double goal_density) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<> uniform_dist(0., 1.);
    std::vector<double> result(length, 0.0);
    size_t count = 0;
    
    do {
        size_t index = length * dist(rng);
        if (index < length) {
            // check if the index isn't already filled
            if (result[index] == 0.0) {
                double value = uniform_dist(rng);
                result[index] = value;
                ++count;
            }
        }
    } while (static_cast<double>(count) / length < goal_density);

    for (size_t i = 0; i < length; ++i) {
        result.push_back(static_cast<double>(dist(rng)));
    }

    return result;
}


/* 
* This code generates sparse vectors (represented as a map of index->value) of random real numbers in [0, 1].
* The vectors can be generated using different distributions (uniform, geometric, Poisson).
* The desired density of the vectors can also be specified.
*/
template <typename Distribution>
std::unordered_map<size_t, double> sparse_vector(size_t length, Distribution& dist, double goal_density) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<> uniform_dist(0., 1.);
    std::vector<double> result(length, 0.0);
    size_t count = 0;

    do {
        size_t index = length * dist(rng);
        if (index < length) {
            // check if the index isn't already filled
            if (result.find(index) == result.end()) {
                double value = uniform_dist(rng);
                result[index] = value;
                ++count;
            }
        }
    } while (static_cast<double>(count) / length < goal_density);

    return result;
}

std::vector<double> dense_uniform_vector(size_t length, double goal_density) {
    std::uniform_real_distribution<> dist(0, 1);
    return dense_vector(length, dist, goal_density);
}

std::vector<double> dense_geometric_vector(size_t length, double p, double goal_density) {
    std::geometric_distribution<> dist(p);
    return dense_vector(length, dist, goal_density);
}

std::vector<double> dense_poisson_vector(size_t length, double lambda, double goal_density) {
    std::poisson_distribution<> dist(lambda);
    return dense_vector(length, dist, goal_density);
}

std::unordered_map<size_t, double> sparse_uniform_vector(size_t length, double density) {
    std::uniform_real_distribution<> dist(0, 1);
    return sparse_vector(length, dist, density);
}

std::unordered_map<size_t, double> sparse_geometric_vector(size_t length, double p, double density) {
    std::geometric_distribution<> dist(p);
    return sparse_vector(length, dist, density);
}

std::unordered_map<size_t, double> sparse_poisson_vector(size_t length, double lambda, double density) {
    std::poisson_distribution<> dist(lambda);
    return sparse_vector(length, dist, density);
}
