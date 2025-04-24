#ifndef _MERGING_HPP
#define _MERGING_HPP

#include "vector-gen.hpp"
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <vector>

// assumes a and b are the same size
DenseVector two_way_merge(const DenseVector& a, const DenseVector& b) {
    DenseVector result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

// assumes all vecs are the same size
DenseVector k_way_merge(const std::vector<DenseVector>& vecs) {
    DenseVector result(vecs[0].size());
    for (size_t i = 0; i < vecs.size(); ++i) {
        for (size_t j = 0; j < vecs[0].size(); ++j) {
            result[j] += vecs[i][j];
        }
    }
    return result;
}

SparseVector two_way_merge(const SparseVector& a, const SparseVector& b) {
    SparseVector result;
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i].first == b[j].first) {
            result.emplace_back(a[i].first, a[i].second + b[j].second);
            ++i;
            ++j;
        } else if (a[i].first < b[j].first) {
            result.push_back(a[i++]);
        } else {
            result.push_back(b[j++]);
        }
    }
    while (i < a.size())
        result.push_back(a[i++]);
    while (j < b.size())
        result.push_back(b[j++]);
    return result;
}

SparseVector k_way_merge(const std::vector<SparseVector>& vecs) {

    return SparseVector();


    // TODO implement or don't
    // (specific item in vec, jth vector in vecs)
    // using PQItem = std::pair<IndexValue, size_t>;

    // std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> min_heap;

    // std::vector<IndexValue> merged_vec;

    // auto stop_cond = [&]() {
    //     return std::all_of(indices.begin(), indices.end(),
    //                        [&](size_t idx) { return idx >= vecs[idx].size(); });
    // };

    // while (!stop_cond()) {

    // }


}

SparseVector hash_merge(const std::vector<SparseVector>& vecs) {
    std::unordered_map<size_t, ValueType> merged_map;
    for (const auto& vec : vecs) {
        for (const auto& pair : vec) {
            const auto& index = pair.first;
            const auto& value = pair.second;
            if (merged_map.find(index) == merged_map.end()) {
                merged_map[index] = value;
            } else {
                merged_map[index] += value;
            }
        }
    }
    SparseVector merged_vec;
    for (const auto& indexValue : merged_map) {
        merged_vec.push_back(indexValue);
    }

    std::sort(merged_vec.begin(), merged_vec.end());

    return merged_vec;
}

#endif