#ifndef NETWORK_TRAFFIC_HPP
#define NETWORK_TRAFFIC_HPP

#include "common.h"
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <queue>
#include <cstring>
#include <mpi.h>

MPI_Datatype MPI_INDEX_VALUE_TYPE;

std::vector<SparseVector> all_to_all_comm_sparse(const SparseVector& vec, int rank, int num_procs){
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
    std::vector<SparseVector> vecs;
    vecs.push_back(result);
    if (left_child < num_procs) {
        int recv_count;

        // exchange sizes
        MPI_Recv(&recv_count, 1, MPI_INT, left_child, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::vector<IndexValue> recv_buffer(recv_count);
        MPI_Recv(recv_buffer.data(), recv_count, MPI_INDEX_VALUE_TYPE, left_child, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        SparseVector left_vec(recv_buffer.begin(), recv_buffer.end());
        vecs.push_back(left_vec);
    }

    if (right_child < num_procs) {
        int recv_count;

        // exchange sizes
        MPI_Recv(&recv_count, 1, MPI_INT, right_child, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::vector<IndexValue> recv_buffer(recv_count);
        MPI_Recv(recv_buffer.data(), recv_count, MPI_INDEX_VALUE_TYPE, right_child, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        SparseVector right_vec(recv_buffer.begin(), recv_buffer.end());
        vecs.push_back(right_vec);
    }

    // merging
    result = hash_merge(vecs);

    // send merged result to parent
    if (rank != 0) {
        int send_count = result.size();
        MPI_Send(&send_count, 1, MPI_INT, parent, 0, MPI_COMM_WORLD);
        MPI_Send(result.data(), send_count, MPI_INDEX_VALUE_TYPE, parent, 1, MPI_COMM_WORLD);
        result.clear();
    }

    // broadcast the final result to everyone
    if (rank == 0) {
        // root node sends to children
        if (left_child < num_procs) {
            int send_count = result.size();
            MPI_Send(&send_count, 1, MPI_INT, left_child, 2, MPI_COMM_WORLD);
            MPI_Send(result.data(), send_count, MPI_INDEX_VALUE_TYPE, left_child, 3, MPI_COMM_WORLD);
        }
        if (right_child < num_procs) {
            int send_count = result.size();
            MPI_Send(&send_count, 1, MPI_INT, right_child, 2, MPI_COMM_WORLD);
            MPI_Send(result.data(), send_count, MPI_INDEX_VALUE_TYPE, right_child, 3, MPI_COMM_WORLD);
        }
    } else {
        // non root node will receive
        int recv_count;
        MPI_Recv(&recv_count, 1, MPI_INT, parent, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<IndexValue> recv_buffer(recv_count);
        MPI_Recv(recv_buffer.data(), recv_count, MPI_INDEX_VALUE_TYPE, parent, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        result.assign(recv_buffer.begin(), recv_buffer.end());

        if (left_child < num_procs) {
            MPI_Send(&recv_count, 1, MPI_INT, left_child, 2, MPI_COMM_WORLD);
            MPI_Send(result.data(), recv_count, MPI_INDEX_VALUE_TYPE, left_child, 3, MPI_COMM_WORLD);
        }
        if (right_child < num_procs) {
            MPI_Send(&recv_count, 1, MPI_INT, right_child, 2, MPI_COMM_WORLD);
            MPI_Send(result.data(), recv_count, MPI_INDEX_VALUE_TYPE, right_child, 3, MPI_COMM_WORLD);
        }
    }
    return result;
}


#endif