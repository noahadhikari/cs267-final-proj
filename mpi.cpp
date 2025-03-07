#include "common.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <mpi.h>
#include <vector>
#include <iostream>

static double global_size;
static double bin_size = cutoff + 1e-5;
static int num_bins;
static int num_processors;

// Define a simple struct to hold the global bin range for each rank.
// this bin range is inclusive.
// out-of-bounds bins are ok - we will just not have any particles in those bins when distributing,
// make sure to handle this when iterating over the bins, check if indices are valid
struct BinRange {

    // constructor
    BinRange(int f, int l)
        : ghost_first(f - num_bins), rank_first(f), rank_last(l), ghost_last(l + num_bins) {}

    int ghost_first;
    int rank_first;
    int rank_last;
    int ghost_last;

    friend std::ostream& operator<<(std::ostream& os, const BinRange& br) {
        os << "ghost_first: " << br.ghost_first << ", rank_first: " << br.rank_first
           << ", rank_last: " << br.rank_last << ", ghost_last: " << br.ghost_last;
        return os;
    }
};

static std::vector<std::vector<int>>
    bins; // a list of bins, where each bin is a list of the particle indices

// Get bin given the particle's x and y coordinates
inline int get_bin(double x, double y) {
    int bin_x = (int)(x / bin_size);
    int bin_y = (int)(y / bin_size);
    return bin_y * num_bins + bin_x;
}

// Apply the force from neighbor to particle
void apply_force(double& particle_ax, double& particle_ay, const particle_t& particle,
                 const particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;
    particle_ax += coef * dx;
    particle_ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }
    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }


    // reset accelerations
    p.ax = 0.0;
    p.ay = 0.0;
}

// does not include ghost bins
BinRange get_rank_bin_range(int rank) {
    int num_rows = std::ceil(global_size / bin_size);
    int rows_per_proc = num_rows / num_processors;
    int remainder_rows = num_rows % num_processors;

    int first_row = rank * rows_per_proc + std::min(rank, remainder_rows);
    int last_row = first_row + rows_per_proc - 1;

    if (rank < remainder_rows) {
        last_row++;
    }

    return BinRange(first_row * num_bins, (last_row + 1) * num_bins - 1);
}

// Distribute particles into real and ghost bins.
void bin_particles(int rank, particle_t* parts, int num_parts, particle_t* full_parts) {
    BinRange bin_range = get_rank_bin_range(rank);

    for (int i = 0; i < num_parts; i++) {
        int part_bin = get_bin(parts[i].x, parts[i].y);
        int local_bin_idx = part_bin - bin_range.ghost_first;

        if (0 <= local_bin_idx && local_bin_idx < bins.size()) {
            bins[local_bin_idx].push_back(parts[i].id - 1);
        }

        if (full_parts != nullptr) {
            full_parts[parts[i].id - 1] = parts[i];
        }
    }
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Set bin size and compute number of bins per side
    global_size = size;
    num_bins = std::ceil(size / bin_size);
    num_processors = num_procs;

    // Set the range for the current processor from the data structure.
    BinRange rank_bin_range = get_rank_bin_range(rank);

    // Resize bins to cover ghost bins above, my bins, ghost bins below. ok if we have some unused
    // bins in the first/last rows
    bins.resize(rank_bin_range.ghost_last - rank_bin_range.ghost_first + 1);

    bin_particles(rank, parts, num_parts, nullptr);
}

int count_local_particles(int rank) {
    int count = 0;
    for (int i = num_bins; i < bins.size() - num_bins; i++) {
        count += bins.at(i).size();
    }
    return count;
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    /*
    Apply force
    Move
    Send full particles that were originally in this row to relevant neighbors
    recieve full particles and reset the bins
    */

    // num_bins = 3

    // processor 1
    // 0 1 2 <-- (first - num_bins, first - 1)
    // 3 4 5 <-- current rank bins, get_rank_bin_range(rank) = (3, 5) = (first, last)
    // 6 7 8 <-- (last + 1, last + num_bins)

    // processor 1 bins: [0 1 2 3 4 5 6 7 8]

    // processor 2:
    // 3 4 5
    // 6 7 8
    // 9 10 11

    // processor 2 bins: [0 1 2 3 4 5 6 7 8]
    // to get the local bin corresponding to a global particle bin, take (part_bin - (first -
    // num_bins)) (subtract off the top-left corner)

    BinRange rank_bin_range = get_rank_bin_range(rank);
    int local_bin_start = rank_bin_range.rank_first - rank_bin_range.ghost_first;
    int local_bin_end = rank_bin_range.rank_last - rank_bin_range.ghost_first;

    // Loop over all local bins
    for (int local_bin = local_bin_start; local_bin <= local_bin_end; local_bin++) {
        int local_bin_y = local_bin / num_bins;
        int local_bin_x = local_bin % num_bins;

        int y_start = local_bin_y - 1;
        int y_end = local_bin_y + 1;
        int x_start = std::max(0, local_bin_x - 1);
        int x_end = std::min(num_bins - 1, local_bin_x + 1);

        for (int p_i : bins[local_bin]) {
            for (int neighbor_y = y_start; neighbor_y <= y_end; neighbor_y++) {
                for (int neighbor_x = x_start; neighbor_x <= x_end; neighbor_x++) {
                    int curr_neighbor_bin = neighbor_y * num_bins + neighbor_x;
                    for (int p_j : bins[curr_neighbor_bin]) {
                        if (p_i != p_j) {
                            apply_force(parts[p_i].ax, parts[p_i].ay, parts[p_i], parts[p_j]);
                        }
                    }
                }
            }
        }
    }

    /*
    Move particles and keep track of which particles need to be sent where

    Each rank needs to know from its neighbors which particles moved into its region
    and it also needs to know which particles are in the ghost regions
    */
    std::vector<particle_t> prev_rank_send_particles;
    std::vector<particle_t> next_rank_send_particles;
    std::vector<particle_t> local_particles;

    // only move particles that are in this rank's bins, don't move ghost particles
    for (int curr_bin = local_bin_start; curr_bin <= local_bin_end; curr_bin++) {
        for (auto p_i : bins[curr_bin]) {
            move(parts[p_i], size);

            int new_bin = get_bin(parts[p_i].x, parts[p_i].y);

            // rows above and also the first row of this rank
            // no lower bound since it may jump more than one row, but still probably in the previous processor
            if (new_bin <= rank_bin_range.rank_first + num_bins - 1) {
                prev_rank_send_particles.push_back(parts[p_i]);

            // last row of this rank and also the rows below
            // no upper bound since it may jump more than one row, but still probably in the next processor
            } else if (rank_bin_range.rank_last - num_bins + 1 <= new_bin) {
                next_rank_send_particles.push_back(parts[p_i]);
            }

            // If the particle remains in this rank's rowspan, then it needs to be rebinned
            if (rank_bin_range.ghost_first <= new_bin && new_bin <= rank_bin_range.ghost_last) {
                local_particles.push_back(parts[p_i]);
            }
        }
    }

    // rank 0 sends first then receives, everything else receives from prev, sends to prev, sends to next, receives from next

    int prev_rank_send_count = prev_rank_send_particles.size();
    int next_rank_send_count = next_rank_send_particles.size();

    int prev_rank_recv_count = 0;
    int next_rank_recv_count = 0;

    std::vector<particle_t> prev_rank_recv_particles;
    std::vector<particle_t> next_rank_recv_particles;


    if (rank == 0) {
        // rank 0 doesn't have any prev rank

        // rank 0 sends first, then receives

        if (num_processors > 1) {
            MPI_Send(&next_rank_send_count, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Send(next_rank_send_particles.data(), next_rank_send_count, PARTICLE, rank + 1, 0,
                    MPI_COMM_WORLD);

            MPI_Recv(&next_rank_recv_count, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            next_rank_recv_particles.resize(next_rank_recv_count);
            MPI_Recv(next_rank_recv_particles.data(), next_rank_recv_count, PARTICLE, rank + 1, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

    } else {
        // every other rank receives from prev, sends to prev, sends to next, receives from next

        // receive from prev, then send to prev

        MPI_Recv(&prev_rank_recv_count, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        prev_rank_recv_particles.resize(prev_rank_recv_count);
        MPI_Recv(prev_rank_recv_particles.data(), prev_rank_recv_count, PARTICLE, rank - 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Send(&prev_rank_send_count, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
        MPI_Send(prev_rank_send_particles.data(), prev_rank_send_count, PARTICLE, rank - 1, 0,
                 MPI_COMM_WORLD);

        

        // send to next, then receive to next. the last processor doesn't have a next rank

        if (rank < num_processors - 1) {
            MPI_Send(&next_rank_send_count, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Send(next_rank_send_particles.data(), next_rank_send_count, PARTICLE, rank + 1, 0,
                    MPI_COMM_WORLD);

            MPI_Recv(&next_rank_recv_count, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            next_rank_recv_particles.resize(next_rank_recv_count);
            MPI_Recv(next_rank_recv_particles.data(), next_rank_recv_count, PARTICLE, rank + 1, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    for (std::vector<int>& bin : bins) {
        bin.clear();
    }

    bin_particles(rank, prev_rank_recv_particles.data(), prev_rank_recv_particles.size(), parts);

    bin_particles(rank, local_particles.data(), local_particles.size(), parts);

    bin_particles(rank, next_rank_recv_particles.data(), next_rank_recv_particles.size(), parts);

    std::cout << count_local_particles(rank) << " particles in rank " << rank << std::endl;
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.

    // First do a Gather to get particle counts

    // Then do Gatherv

    // Build a local vector of particles owned by this rank.
    // Here we assume that if a particle's bin (computed from its x and y) falls
    // in [first_bin_global_ind, last_bin_global_ind], then this rank owns it.
    std::vector<particle_t> local_particles;

    BinRange rank_bin_range = get_rank_bin_range(rank);

    for (int i = 0; i < num_parts; i++) {
        int part_bin = get_bin(parts[i].x, parts[i].y);
        if (part_bin >= rank_bin_range.rank_first && part_bin <= rank_bin_range.rank_last) {
            local_particles.push_back(parts[i]);
        }
    }
    int local_count = local_particles.size();

    std::vector<int> counts;
    if (rank == 0) {
        counts.resize(num_procs);
    }

    MPI_Gather(&local_count, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0,
               MPI_COMM_WORLD);

    // Compute displacements on rank 0.
    std::vector<int> displs;
    int total_count = 0;
    if (rank == 0) {
        displs.resize(num_procs);
        displs[0] = 0;
        total_count = counts[0];
        for (int i = 1; i < num_procs; i++) {
            displs[i] = displs[i - 1] + counts[i - 1];
            total_count += counts[i];
        }
    }

    // Prepare a buffer on rank 0 to receive all particles.
    std::vector<particle_t> gathered;
    if (rank == 0) {
        gathered.resize(total_count);
    }

    // Gather the particles using MPI_Gatherv.
    MPI_Gatherv(local_particles.data(), local_count, PARTICLE,
                rank == 0 ? gathered.data() : nullptr, rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr, PARTICLE, 0, MPI_COMM_WORLD);

    // On rank 0, sort the gathered particles by particle id.
    if (rank == 0) {
        for (const auto& particle : gathered) {
            parts[particle.id - 1] = particle;
        }
    }
}

