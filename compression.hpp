#include "vector-gen.hpp"
#include <bitset>

struct DeltaEncodedVector {
    std::vector<ValueType> values;
    std::vector<uint8_t> delta_data;
    size_t bytes_per_delta;
};

void compress_deltas(DeltaEncodedVector& dev, const std::vector<size_t>& deltas, size_t max_delta) {
    if (max_delta == 0) {
        throw std::invalid_argument("max_delta cannot be zero");
    } 
    size_t num_bits = static_cast<size_t>(std::ceil(std::log2(max_delta + 1)));
    size_t bytes_per_delta = (num_bits + 7) / 8;
    dev.bytes_per_delta = bytes_per_delta;

    dev.delta_data.resize(bytes_per_delta * deltas.size());

    for (size_t i = 0; i < deltas.size(); ++i) {
        size_t delta = deltas[i];
        for (size_t byte = 0; byte < bytes_per_delta; ++byte) {
            dev.delta_data[i * bytes_per_delta + byte] = static_cast<uint8_t>((delta >> (8 * byte)) & 0xFF);
        }
    }
}

DeltaEncodedVector delta_encode(const SparseVector& sparse_vector) {
    DeltaEncodedVector result;
    result.values.reserve(sparse_vector.size());

    // run through the vector indices to get the deltas and copy the values
    std::vector<size_t> deltas;
    size_t max_delta = 0;
    deltas.reserve(sparse_vector.size());
    size_t last_index = 0;
    for (const IndexValue& iv : sparse_vector) {
        result.values.emplace_back(iv.second);
        size_t index = iv.first;
        size_t delta = index - last_index;
        max_delta = std::max(max_delta, delta);
        deltas.push_back(delta);
        last_index = index;
    }
    compress_deltas(result, deltas, max_delta);
    return result;
}

SparseVector delta_decode(const DeltaEncodedVector& dev) {
    SparseVector result;
    size_t current_index = 0;
    size_t num_values = dev.values.size();
    size_t bpd = dev.bytes_per_delta;

    result.reserve(num_values);

    for (size_t i = 0; i < num_values; ++i) {
        size_t delta = 0;
        for (size_t byte = 0; byte < bpd; ++byte) {
            delta |= static_cast<size_t>(dev.delta_data[i * bpd + byte]) << (8 * byte);
        }

        current_index += delta;
        result.emplace_back(current_index, dev.values[i]);
    }

    return result;
}





// number of bits in mask. need to know at compile-time for bitmask compression so can't do as command line arg
constexpr size_t VECTOR_LENGTH = 1000000;
struct BitmaskEncodedVector {
    std::bitset<VECTOR_LENGTH> mask;
    std::vector<ValueType> values;
};

BitmaskEncodedVector bitmask_encode(const SparseVector& sparse_vector) {
    BitmaskEncodedVector result;
    result.values.reserve(sparse_vector.size());

    // run through the vector indices to get nonzero bits and copy the values
    for (const IndexValue& iv : sparse_vector) {
        result.values.push_back(iv.second);
        size_t index = iv.first;
        result.mask.set(index);
    }
    return result;
}

SparseVector bitmask_decode(const BitmaskEncodedVector& bmv) {
    SparseVector result;
    size_t value_index = 0;

    for (size_t i = 0; i < VECTOR_LENGTH; ++i) {
        if (bmv.mask.test(i)) {
            result.emplace_back(i, bmv.values[value_index]);
            ++value_index;
        }
    }

    return result;
}


std::vector<SparseVector> delta_decode_all(const std::vector<DeltaEncodedVector>& devs) {
    std::vector<SparseVector> result;
    result.reserve(devs.size());

    for (const DeltaEncodedVector& dev : devs) {
        result.push_back(delta_decode(dev));
    }

    return result;
}

std::vector<SparseVector> bitmask_decode_all(const std::vector<BitmaskEncodedVector>& bmvs) {
    std::vector<SparseVector> result;
    result.reserve(bmvs.size());

    for (const BitmaskEncodedVector& bmv : bmvs) {
        result.push_back(bitmask_decode(bmv));
    }

    return result;
}
