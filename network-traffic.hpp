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

std::vector<DenseVector> all_to_all_comm_dense(const DenseVector& vec, int rank, int num_procs) {
    int vec_len = vec.size();
    std::vector<double> recvbuf(vec_len * num_procs);

    MPI_Alltoall(vec.data(), vec_len, MPI_DOUBLE, recvbuf.data(), vec_len, MPI_DOUBLE, MPI_COMM_WORLD);

    std::vector<DenseVector> result(num_procs, DenseVector(vec_len));
    for (int i = 0; i < num_procs; ++i) {
        std::memcpy(result[i].data(), recvbuf.data() + i * vec_len, vec_len * sizeof(ValueType));
    }
    return result;
}


#endif