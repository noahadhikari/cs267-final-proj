#include "common.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <random>
#include <vector>
#include <stddef.h>

// =================
// Helper Functions
// =================

// I/O routines
void save() {
    // static bool first = true;

    // if (first) {
    //     fsave << num_parts << " " << size << "\n";
    //     first = false;
    // }

    // for (int i = 0; i < num_parts; ++i) {
    //     fsave << parts[i].x << " " << parts[i].y << "\n";
    // }

    // fsave << std::endl;
}

void print_dense_vector(const DenseVector& vec) {
    std::cout << "[ ";
    for (const auto& val : vec) {
        std::cout << val << " ";
    }
    std::cout << "]\n";
}

void print_sparse_vector(const SparseVector& vec) {
    std::cout << "[ ";
    for (const auto& pair : vec) {
        std::cout << "(" << pair.first << ", " << pair.second << ") ";
    }
    std::cout << "]\n";
}

int main(int argc, char** argv) {
    // Open Output File
    // char* savename = find_string_option(argc, argv, "-o", nullptr);
    // std::ofstream fsave(savename);

    // Init MPI
    // TODO: initialize



    int num_procs, rank; // TODO: number of processes is defined when running mpirun or smth
    
    MPI_Init(&argc, &argv); 
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cout << "Running main\n" << std::endl; 
    }

    int blocklengths[2] = {1, 1};
    MPI_Aint offsets[2];
    MPI_Datatype types[2] = {MPI_UNSIGNED_LONG, MPI_DOUBLE}; //TODO: NOT SURE IF SIZE_T IS UNSIGNED INT OR LONG
    
    offsets[0] = offsetof(IndexValue, first);
    offsets[1] = offsetof(IndexValue, second);
    
    MPI_Type_create_struct(2, blocklengths, offsets, types, &MPI_INDEX_VALUE_TYPE);
    MPI_Type_commit(&MPI_INDEX_VALUE_TYPE);

    int length = 1000;
    int baseline = 1; // default: naive
    int distribution = 1; // default: uniform
    double density = 0.1;
    double param = 0.1;
    long seed = 0;

    SparseVector vec;
    // Parse Args
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-l") == 0 && i + 1 < argc) {
            length = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            density = std::stod(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            // indicate which baseline to run, 1 for dense, 2 for sparse
            baseline = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0 && i + 1 < argc) {
            // indicate which vector distribution to generate, 1 for uniform, 2 for exponential, 3 for poisson
            distribution = std::stoi(argv[++i]);
            if (distribution == 3) {
                param = length / 2;
            }
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            // indicate param for geometric and poisson distribution
            param = std::stod(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            // indicate seed for random number generation
            seed = std::stol(argv[++i]);
        }
    }

    seed += rank; // make the seed unique for each process

    // generate vector distribution
    switch (distribution) {
        // uniform
        case 1: {
            vec = sparse_uniform_vector(seed, length, density);
            break;
        }
        // exponential
        case 2: {
            vec = sparse_exponential_vector(seed, length, param, density);
            break;
        }
        // poisson
        case 3: {
            vec = sparse_poisson_vector(seed, length, param, density);
            break;
        }
        default:
            std::cerr << "Unknown distribution " << baseline << "\n";
            return 1;
    }

    auto start_time = std::chrono::steady_clock::now();

    switch (baseline) {
        // dense vector baseline
        case 1: {
            DenseVector dense_vec = convert_to_dense(vec, length);
            start_time = std::chrono::steady_clock::now();

            std::cout << "Rank" << rank << ": ";
            print_dense_vector(dense_vec);

            std::vector<ValueType> reduced_vec = all_reduce_sum_dense(dense_vec, rank, num_procs);

            std::cout << "Rank" << rank << ": ";
            print_dense_vector(reduced_vec);
            break;
        }
        // sparse vector baseline
        case 2: {
            // std::cout << "Rank" << rank << ": ";
            // print_sparse_vector(vec);
            SparseVector result = alltoallv_comm_sparse(vec, rank, num_procs, CompressionType::NONE);

            // std::cout << "Rank" << rank << ": ";
            // print_sparse_vector(result);
            break;
        }
        // sparse vector + butterfly
        case 3: {
            // std::cout << "Rank" << rank << ": ";
            // print_sparse_vector(vec);
            SparseVector result = butterfly_reduce_sparse(vec, rank, num_procs);
            // std::cout << "Rank" << rank << ": ";
            // print_sparse_vector(result);
            break;
        }
        // sparse vector + tree
        case 4: {
            // std::cout << "Rank" << rank << ": ";
            // print_sparse_vector(vec);
            SparseVector result = tree_reduce_sparse(vec, rank, num_procs);
            // std::cout << "Rank" << rank << ": ";
            // print_sparse_vector(result);
            break;
        }
        // sparse vector + ring
        case 5: {
            SparseVector result = ring_reduce_sparse(vec, rank, num_procs);
            break;
        }
        // uncompressed alltoall
        case 11: {
            // std::cout << "Rank" << rank << ": ";
            // print_sparse_vector(vec);
            SparseVector result = alltoallv_comm_sparse(vec, rank, num_procs, CompressionType::NONE);
            // std::cout << "Rank" << rank << ": ";
            // print_sparse_vector(result);
            break;
        }
        // delta alltoall
        case 12: {
            // std::cout << "Rank" << rank << ": ";
            // print_sparse_vector(vec);
            SparseVector result = alltoallv_comm_sparse(vec, rank, num_procs, CompressionType::DELTA);
            // std::cout << "Rank" << rank << ": ";
            // print_sparse_vector(result);
            break;

        }
        // bitmask alltoall
        case 13: {
            // check if length matches the constexpr VECTOR_LENGTH
            if (length != VECTOR_LENGTH) {
                std::cerr << "VECTOR_LENGTH does not match -l=" << length << ". VECTOR_LENGTH is currently " << VECTOR_LENGTH << "\n"
                          << "Change the VECTOR_LENGTH in compression.hpp to match the length of the vector in -l=" << length << "\n";
                return 1;
            }
            start_time = std::chrono::steady_clock::now();
            // std::cout << "Rank" << rank << ": ";
            // print_sparse_vector(vec);
            SparseVector result = alltoallv_comm_sparse(vec, rank, num_procs, CompressionType::BITMASK);

            // std::cout << "Rank" << rank << ": ";
            // print_sparse_vector(result);
            break;
        }
        default:
            std::cerr << "Unknown baseline " << baseline << "\n";
            return 1;
    }

    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    // Finalize
    if (rank == 0) {
        std::cout << "Time = " << seconds << " seconds." << "\n";
    }
    // if (fsave) {
    //     fsave.close();
    // }

    MPI_Type_free(&MPI_INDEX_VALUE_TYPE);
    MPI_Finalize();
    return 0;
}