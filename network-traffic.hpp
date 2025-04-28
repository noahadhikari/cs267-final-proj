#ifndef NETWORK_TRAFFIC_HPP
#define NETWORK_TRAFFIC_HPP

#include "common.h"
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <queue>
#include <cstring>
#include <mpi.h>

#include "compression.hpp"

MPI_Datatype MPI_INDEX_VALUE_TYPE;

enum class CompressionType {
    NONE,
    DELTA,
    BITMASK,
};

std::vector<DeltaEncodedVector> alltoallv_comm_sparse_delta(const DeltaEncodedVector& vec, int rank, int num_procs) {
    // Send vector lengths
    int vec_len = vec.values.size();
    std::vector<int> all_vector_lengths(num_procs);
    MPI_Allgather(&vec_len, 1, MPI_INT, all_vector_lengths.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Send bytes_per_delta
    int bytes_per_delta = vec.bytes_per_delta;
    std::vector<int> all_bytes_per_delta(num_procs);
    MPI_Allgather(&bytes_per_delta, 1, MPI_INT, all_bytes_per_delta.data(), 1, MPI_INT, MPI_COMM_WORLD);

    //
    // Send delta_data
    //

    int delta_len = vec.delta_data.size();
    std::vector<int> all_delta_lengths(num_procs);
    MPI_Allgather(&delta_len, 1, MPI_INT, all_delta_lengths.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> sendcounts_delta(num_procs, delta_len);
    std::vector<int> sdispls_delta(num_procs, 0);
    std::vector<int> rdispls_delta(num_procs, 0);

    int total_recv_delta = 0;
    for (int i = 0; i < num_procs; i++) {
        rdispls_delta[i] = total_recv_delta;
        total_recv_delta += all_delta_lengths[i];
    }

    std::vector<uint8_t> all_delta_data(total_recv_delta);

    MPI_Alltoallv(vec.delta_data.data(), sendcounts_delta.data(), sdispls_delta.data(), MPI_UINT8_T,
                  all_delta_data.data(), all_delta_lengths.data(), rdispls_delta.data(), MPI_UINT8_T,
                  MPI_COMM_WORLD);

    //
    // Send values
    //

    std::vector<int> sendcounts_val(num_procs, vec_len);
    std::vector<int> sdispls_val(num_procs, 0);
    std::vector<int> rdispls_val(num_procs, 0);

    int total_recv_val = 0;
    for (int i = 0; i < num_procs; i++) {
        rdispls_val[i] = total_recv_val;
        total_recv_val += all_vector_lengths[i];
    }

    std::vector<ValueType> all_values(total_recv_val);

    MPI_Alltoallv(vec.values.data(), sendcounts_val.data(), sdispls_val.data(), MPI_INT,
                  all_values.data(), all_vector_lengths.data(), rdispls_val.data(), MPI_INT,
                  MPI_COMM_WORLD);

    //
    // Rebuild DeltaEncodedVectors
    //

    std::vector<DeltaEncodedVector> results(num_procs);

    for (int src = 0; src < num_procs; src++) {
        // Extract slices
        std::vector<ValueType> values(
            all_values.begin() + rdispls_val[src],
            all_values.begin() + rdispls_val[src] + all_vector_lengths[src]
        );

        std::vector<uint8_t> delta_data(
            all_delta_data.begin() + rdispls_delta[src],
            all_delta_data.begin() + rdispls_delta[src] + all_delta_lengths[src]
        );

        DeltaEncodedVector reconstructed;
        reconstructed.values = std::move(values);
        reconstructed.delta_data = std::move(delta_data);
        reconstructed.bytes_per_delta = all_bytes_per_delta[src];

        results[src] = std::move(reconstructed);
    }

    return results;
}

std::vector<BitmaskEncodedVector> alltoallv_comm_sparse_bitmask(const BitmaskEncodedVector& vec, int rank, int num_procs) {
    // --- Step 1: Send bitsets ---
    constexpr int num_words = (VECTOR_LENGTH + 63) / 64;

    std::vector<uint64_t> send_buffer(num_words, 0);
    for (int i = 0; i < VECTOR_LENGTH; ++i) {
        if (vec.mask[i]) {
            send_buffer[i / 64] |= (uint64_t(1) << (i % 64));
        }
    }

    std::vector<uint64_t> recv_buffer(num_words * num_procs);

    MPI_Allgather(
        send_buffer.data(), num_words, MPI_UINT64_T,
        recv_buffer.data(), num_words, MPI_UINT64_T,
        MPI_COMM_WORLD
    );

    // --- Step 2: Send values ---
    int local_num_values = vec.values.size();
    std::vector<int> all_num_values(num_procs);
    MPI_Allgather(
        &local_num_values, 1, MPI_INT,
        all_num_values.data(), 1, MPI_INT,
        MPI_COMM_WORLD
    );

    std::vector<int> displs(num_procs, 0);
    int total_recv_values = 0;
    for (int i = 0; i < num_procs; ++i) {
        displs[i] = total_recv_values;
        total_recv_values += all_num_values[i];
    }

    std::vector<ValueType> all_values(total_recv_values);

    MPI_Allgatherv(
        vec.values.data(), local_num_values, MPI_INT,
        all_values.data(), all_num_values.data(), displs.data(), MPI_INT,
        MPI_COMM_WORLD
    );

    // --- Step 3: Rebuild ---
    std::vector<BitmaskEncodedVector> result(num_procs);
    int values_cursor = 0;

    for (int proc = 0; proc < num_procs; ++proc) {
        BitmaskEncodedVector tmp;
        tmp.mask.reset();

        for (int i = 0; i < VECTOR_LENGTH; ++i) {
            if (recv_buffer[proc * num_words + (i / 64)] & (uint64_t(1) << (i % 64))) {
                tmp.mask.set(i);
            }
        }

        int num_vals = all_num_values[proc];
        tmp.values.reserve(num_vals);
        for (int k = 0; k < num_vals; ++k) {
            tmp.values.push_back(all_values[values_cursor++]);
        }

        result[proc] = std::move(tmp);
    }

    return result;
}




std::vector<SparseVector> alltoallv_comm_sparse_nocompress(const SparseVector& vec, int rank, int num_procs) {
    int vec_len = vec.size();
    int* all_vector_lengths = (int*) malloc(num_procs * sizeof(int));

    MPI_Allgather(&vec_len, 1, MPI_INT, all_vector_lengths, 1, MPI_INT, MPI_COMM_WORLD);

    int* sendcounts = (int*) malloc(num_procs * sizeof(int));
    int* sdispls = (int*) malloc(num_procs * sizeof(int));
    int* rdispls = (int*) malloc(num_procs * sizeof(int));

    int total_recv_count = 0;
    for (int i = 0; i < num_procs; i++) {
        sendcounts[i] = vec_len;
        sdispls[i] = 0;
        
        rdispls[i] = total_recv_count;
        total_recv_count += all_vector_lengths[i];
    }

    IndexValue* all_vectors = (IndexValue*) malloc(total_recv_count * sizeof(IndexValue));
    
    MPI_Alltoallv(vec.data(), sendcounts, sdispls, MPI_INDEX_VALUE_TYPE,
                 all_vectors, all_vector_lengths, rdispls, MPI_INDEX_VALUE_TYPE,
                 MPI_COMM_WORLD);

    free(sendcounts);
    free(sdispls);
    
    std::vector<SparseVector> result(num_procs);
    for (int i = 0; i < num_procs; i++) {
        result[i].resize(all_vector_lengths[i]);
        memcpy(result[i].data(), all_vectors + rdispls[i], all_vector_lengths[i] * sizeof(IndexValue));
    }
    
    free(all_vector_lengths);
    free(rdispls);
    free(all_vectors);
    
    return result;
}



SparseVector alltoallv_comm_sparse(const SparseVector& vec, int rank, int num_procs, CompressionType compression_type) {

    switch (compression_type) {
        case CompressionType::NONE: {
            return hash_merge(alltoallv_comm_sparse_nocompress(vec, rank, num_procs));
        }
        case CompressionType::DELTA: {
            auto delta_encoded = delta_encode(vec);
            auto delta_collected = alltoallv_comm_sparse_delta(delta_encoded, rank, num_procs);
            auto delta_decoded = delta_decode_all(delta_collected);
            return hash_merge(delta_decoded);
        }
        case CompressionType::BITMASK: {   
            auto bitmask_encoded = bitmask_encode(vec);
            auto bitmask_collected = alltoallv_comm_sparse_bitmask(bitmask_encoded, rank, num_procs);
            auto bitmask_decoded = bitmask_decode_all(bitmask_collected);
            return hash_merge(bitmask_decoded);
        }
        default:
            throw std::invalid_argument("Unknown compression type");
    }
}

// std::vector<DenseVector> all_to_all_comm_dense(const DenseVector& vec, int rank, int num_procs) {
//     int vec_len = vec.size();
//     std::vector<double> recvbuf(vec_len * num_procs);

//     MPI_Alltoall(vec.data(), vec_len, MPI_DOUBLE, recvbuf.data(), vec_len, MPI_DOUBLE, MPI_COMM_WORLD);

//     std::vector<DenseVector> result(num_procs, DenseVector(vec_len));
//     for (int i = 0; i < num_procs; ++i) {
//         std::memcpy(result[i].data(), recvbuf.data() + i * vec_len, vec_len * sizeof(ValueType));
//     }
//     return result;
// }

// assumes ValueType = int
std::vector<ValueType> all_reduce_sum_dense(const DenseVector& vec, int rank, int num_procs) {
    int vec_len = vec.size();
    std::vector<ValueType> sendbuf(vec.begin(), vec.end()); // copy vec to contiguous memory
    std::vector<ValueType> recvbuf(vec_len);

    MPI_Allreduce(sendbuf.data(), recvbuf.data(), vec_len, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    return recvbuf;
}

/* 
* Butterfly topology should only be used for the number of processes that have a power of 2
*/
SparseVector butterfly_reduce_sparse(const SparseVector& vec, int rank, int num_procs) {
    SparseVector result = vec;

    for (int i = 0; (1 << i) < num_procs; ++i) {
        int partner = rank ^ (1 << i);
        // std::cout << "Rank" << rank << ": ";
        // std::cout << "Step" << i;
        // std::cout << "Partner: " << partner << std::endl;

        int send_count = result.size();
        int recv_count = 0;

        // exchange sizes
        MPI_Sendrecv(&send_count, 1, MPI_INT, partner, 0, 
            &recv_count, 1, MPI_INT, partner, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<IndexValue> recv_buffer(recv_count);

        // exchange index-value pairs
        MPI_Sendrecv(result.data(), send_count, MPI_INDEX_VALUE_TYPE, partner, 1,
            recv_buffer.data(), recv_count, MPI_INDEX_VALUE_TYPE, partner, 1,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        SparseVector partner_vec(recv_buffer.begin(), recv_buffer.end());
        result = two_way_merge(result, partner_vec);
    }
    return result;
}

/* 
* Binary tree topology
*/
SparseVector tree_reduce_sparse(const SparseVector& vec, int rank, int num_procs) {
    SparseVector result = vec;

    int parent = (rank - 1) / 2;
    int left_child = 2 * rank + 1;
    int right_child = 2 * rank + 2;

    // reduce upwards and receive from both left and right child
    // std::vector<SparseVector> vecs;
    // vecs.push_back(result);
    if (left_child < num_procs) {
        int recv_count;

        // exchange sizes
        MPI_Recv(&recv_count, 1, MPI_INT, left_child, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::vector<IndexValue> recv_buffer(recv_count);
        MPI_Recv(recv_buffer.data(), recv_count, MPI_INDEX_VALUE_TYPE, left_child, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        SparseVector left_vec(recv_buffer.begin(), recv_buffer.end());
        // vecs.push_back(left_vec);
        result = two_way_merge(result, left_vec);
    }

    if (right_child < num_procs) {
        int recv_count;

        // exchange sizes
        MPI_Recv(&recv_count, 1, MPI_INT, right_child, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::vector<IndexValue> recv_buffer(recv_count);
        MPI_Recv(recv_buffer.data(), recv_count, MPI_INDEX_VALUE_TYPE, right_child, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        SparseVector right_vec(recv_buffer.begin(), recv_buffer.end());
        // vecs.push_back(right_vec);
        result = two_way_merge(result, right_vec);
    }

    // merging
    // result = hash_merge(vecs);

    // send merged result to parent
    if (rank != 0) {
        int send_count = result.size();
        MPI_Send(&send_count, 1, MPI_INT, parent, 0, MPI_COMM_WORLD);
        MPI_Send(result.data(), send_count, MPI_INDEX_VALUE_TYPE, parent, 1, MPI_COMM_WORLD);
        result.clear();
    }

    // broadcast the final result to everyone
    // if (rank == 0) {
    //     // root node sends to children
    //     if (left_child < num_procs) {
    //         int send_count = result.size();
    //         MPI_Send(&send_count, 1, MPI_INT, left_child, 2, MPI_COMM_WORLD);
    //         MPI_Send(result.data(), send_count, MPI_INDEX_VALUE_TYPE, left_child, 3, MPI_COMM_WORLD);
    //     }
    //     if (right_child < num_procs) {
    //         int send_count = result.size();
    //         MPI_Send(&send_count, 1, MPI_INT, right_child, 2, MPI_COMM_WORLD);
    //         MPI_Send(result.data(), send_count, MPI_INDEX_VALUE_TYPE, right_child, 3, MPI_COMM_WORLD);
    //     }
    // } else {
    //     // non root node will receive
    //     int recv_count;
    //     MPI_Recv(&recv_count, 1, MPI_INT, parent, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    //     std::vector<IndexValue> recv_buffer(recv_count);
    //     MPI_Recv(recv_buffer.data(), recv_count, MPI_INDEX_VALUE_TYPE, parent, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    //     result.assign(recv_buffer.begin(), recv_buffer.end());

    //     if (left_child < num_procs) {
    //         MPI_Send(&recv_count, 1, MPI_INT, left_child, 2, MPI_COMM_WORLD);
    //         MPI_Send(result.data(), recv_count, MPI_INDEX_VALUE_TYPE, left_child, 3, MPI_COMM_WORLD);
    //     }
    //     if (right_child < num_procs) {
    //         MPI_Send(&recv_count, 1, MPI_INT, right_child, 2, MPI_COMM_WORLD);
    //         MPI_Send(result.data(), recv_count, MPI_INDEX_VALUE_TYPE, right_child, 3, MPI_COMM_WORLD);
    //     }
    // }

    // broadcast
    int final_size = result.size();
    MPI_Bcast(&final_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        result.resize(final_size);
    }
    MPI_Bcast(result.data(), final_size, MPI_INDEX_VALUE_TYPE, 0, MPI_COMM_WORLD);

    return result;
}


SparseVector ring_reduce_sparse(const SparseVector& vec, int rank, int num_procs) {
    SparseVector result = vec;
    
    if (num_procs == 1) {
        return result;
    }
    
    if (rank == 0) {
        // only send data to the next process
        int send_count = result.size();
        MPI_Send(&send_count, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Send(result.data(), send_count, MPI_INDEX_VALUE_TYPE, 1, 1, MPI_COMM_WORLD);
    }
    else if (rank < num_procs - 1) {
        // middle processes -- receive, merge, and forward
        int recv_count;
        MPI_Recv(&recv_count, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::vector<IndexValue> recv_buffer(recv_count);
        MPI_Recv(recv_buffer.data(), recv_count, MPI_INDEX_VALUE_TYPE, rank - 1, 1, 
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // merge received data with local data
        SparseVector recv_vec(recv_buffer.begin(), recv_buffer.end());
        result = two_way_merge(result, recv_vec);
        
        // send to next process
        int send_count = result.size();
        MPI_Send(&send_count, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        MPI_Send(result.data(), send_count, MPI_INDEX_VALUE_TYPE, rank + 1, 1, MPI_COMM_WORLD);
    }
    else {
        // only last process -- receive, merge, and broadcast the result
        int recv_count;
        MPI_Recv(&recv_count, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::vector<IndexValue> recv_buffer(recv_count);
        MPI_Recv(recv_buffer.data(), recv_count, MPI_INDEX_VALUE_TYPE, rank - 1, 1, 
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // merge received data with local data
        SparseVector recv_vec(recv_buffer.begin(), recv_buffer.end());
        result = two_way_merge(result, recv_vec);
    }
    
    // broadcast the final size and result from the last process to all processes
    int final_size = 0;
    if (rank == num_procs - 1) {
        final_size = result.size();
    }
    
    // num_procs -1 broadcasts the size
    MPI_Bcast(&final_size, 1, MPI_INT, num_procs - 1, MPI_COMM_WORLD);
    
    // allocate space for the final result on non-last processes
    if (rank != num_procs - 1) {
        result.resize(final_size);
    }
    
    // broadcast the data from the last process
    MPI_Bcast(result.data(), final_size, MPI_INDEX_VALUE_TYPE, num_procs - 1, MPI_COMM_WORLD);
    
    return result;
}


#endif