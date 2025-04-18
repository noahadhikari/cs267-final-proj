#include "vector-gen.hpp"

#include <algorithm>
#include <unordered_map>
#include <vector>
#include <queue>


void all_to_all_comm(SparseVector vec, int rank, int num_procs){
    int vec_len = size(vec)
    int *all_vector_lengths = malloc(num_procs * sizeof(int));
    
    MPI_Allgather(&vec_len, 1, MPI_INT, all_vector_lengths, MPI_INT, MPI_COMM_WORLD);

    int *rdispls = malloc(num_procs * sizeof(int));
    int total_recv_count = 0;
    for (int i = 0; i < num_procs; i++) {
        rdispls[i] = total_recv_count;
        total_recv_count += recvcounts[i];
    }
    
    double *recvbuf = malloc(total_recv_count * sizeof(IndexValue));
    
    // Perform the variable-length vector exchange
    MPI_Alltoallv(sendbuf, sendcounts, sdispls, mpi_pair_type,
                 recvbuf, recvcounts, rdispls, mpi_pair_type,
                 MPI_COMM_WORLD);
    
    return recvbuf;
}