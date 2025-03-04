#include "common.h"
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cmath>

// Define a simple struct to hold the global bin range for each rank.
struct BinRange {
    int first_global_bin_ind;
    int last_global_bin_ind;
};

// Global vector storing the bin ranges for all ranks.
static std::vector<BinRange> rank_bin_ranges;

static std::vector<particle_t> real_particles; // particles strictly in current processor area
static std::vector<particle_t> ghost_particles; // particles in ghost region of current processor area

static double bin_size;
static int num_bins;

static int first_bin_global_ind; // first bin index for current processor (not including ghost bins)
static int last_bin_global_ind;  // last bin index for current processor

static int first_ghost_bin_global_ind;
static int last_ghost_bin_global_ind;

static std::vector<std::vector<int>> bins; // a list of bins, where each bin is a list of the particle indices

// Get bin given the particle's x and y coordinates
inline int get_bin(double x, double y) {
    int bin_x = (int)(x / bin_size);
    int bin_y = (int)(y / bin_size);
    return bin_y * num_bins + bin_x;
}

// Return true if the given particle is a ghost particle or not
inline bool is_ghost(double x, double y, int rank) {
    int bin_index = get_bin(x, y);
    // Check if this particle's bin is in the bin row right above or right below
    if ((bin_index >= first_bin_global_ind - num_bins && bin_index < first_bin_global_ind) ||
        (bin_index <= last_bin_global_ind + num_bins && bin_index > last_bin_global_ind)) {
        return true;
    }
    return false;
}

inline int get_rank_from_bin(int global_bin, int num_procs) { 
    int bin_row = global_bin / num_bins;
    int bins_rows_per_proc = num_bins / num_procs;
    int remainder_bin_rows = num_bins % num_procs;
    if (bin_row < (bins_rows_per_proc + 1) * remainder_bin_rows) {
        return bin_row / (bins_rows_per_proc + 1);
    } else {
        return remainder_bin_rows +
            (bin_row - (bins_rows_per_proc + 1) * remainder_bin_rows) / bins_rows_per_proc;
    }
}

// Apply the force from neighbor to particle
void apply_force(double& particle_ax, double& particle_ay, const particle_t& particle, const particle_t& neighbor) {
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
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Set bin size and compute number of bins per side
    bin_size = cutoff + 1e-5;
    num_bins = std::ceil(size / bin_size);

    // Compute how many bin rows each processor gets.
    int bins_rows_per_proc = num_bins / num_procs;
    int remainder_bin_rows = num_bins % num_procs;

    // Create a vector to store the bin range for each rank.
    rank_bin_ranges.resize(num_procs);
    for (int r = 0; r < num_procs; r++) {
        int first_bin_row = r * bins_rows_per_proc + std::min(r, remainder_bin_rows);
        int last_bin_row = first_bin_row + bins_rows_per_proc - 1;
        if (r < remainder_bin_rows) {
            last_bin_row++;  // Extra row for first few ranks.
        }
        rank_bin_ranges[r].first_global_bin_ind = first_bin_row * num_bins;
        rank_bin_ranges[r].last_global_bin_ind  = (last_bin_row + 1) * num_bins - 1;
    }

    // Set the range for the current processor from the data structure.
    first_bin_global_ind = rank_bin_ranges[rank].first_global_bin_ind;
    last_bin_global_ind = rank_bin_ranges[rank].last_global_bin_ind;

    // Calculate ghost bin indices.
    first_ghost_bin_global_ind = std::max(0, first_bin_global_ind - num_bins);
    last_ghost_bin_global_ind  = std::min(num_bins * num_bins - 1, last_bin_global_ind + num_bins);

    // Resize bins to cover all ghost bins.
    bins.resize(last_ghost_bin_global_ind - first_ghost_bin_global_ind + 1);

    // Distribute particles into real and ghost bins.
    for (int i = 0; i < num_parts; i++) {
        int part_bin = get_bin(parts[i].x, parts[i].y);
        if (part_bin >= first_bin_global_ind - num_bins && part_bin <= last_bin_global_ind + num_bins) {
            if (part_bin >= first_bin_global_ind && part_bin <= last_bin_global_ind) {
                real_particles.push_back(parts[i]);
            } else {
                ghost_particles.push_back(parts[i]);
            }
            bins[part_bin - first_ghost_bin_global_ind].push_back(i);
        }
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    int recvbuf;

    //assume we can only receive particles from adjacent processors
    for (int n = rank - 1; n <= std::min(rank + 1, num_procs - 1); n++) {
        MPI_Recv(recvbuf, 1, MPI_INT, n, 0, MPI_COMM_WORLD);
        int global_bin = get_bin(parts[recvbuf].x, parts[recvbuf].y);
        int local_bin_ind = global_bin - first_ghost_bin_global_ind;
        bins[local_bin_ind].push_back(recvbuf);
    }
    
    // Write this function
    int start_bin_local_ind = first_bin_global_ind - first_ghost_bin_global_ind;
    int last_bin_local_ind = last_bin_global_ind - first_ghost_bin_global_ind;

    for (int i = start_bin_local_ind; i < last_bin_local_ind; i++) {
        int curr_bin = i;
        for (int p_i: bins[curr_bin]) {
            for (int y_offset = -1; y_offset < 2; y_offset++) {
                for (int x_offset = -1; x_offset < 2; x_offset++) {
                    int curr_neighbor_bin = i + num_bins * y_offset + x_offset;
                    if (curr_neighbor_bin < 0 || curr_neighbor_bin >= num_bins * num_bins) {
                        continue;
                    }
                    for (int p_j: bins[curr_neighbor_bin]) {
                        if (p_i != p_j) {
                            apply_force(parts[p_i], parts[p_j]);
                        }
                    }
                }
            }
        }
    }

    for (int i = start_bin_local_ind; i < last_bin_local_ind; i++) {
        for (int j = 0; j < bins[i].size; j++) {
            move(parts[j], size);
            //get new bin of parts[j]
            int new_bin = get_bin(parts[j]);
            new_bin_local_ind = new_bin - first_ghost_bin_global_ind;
            if (new_bin_local_ind >= 0 && new_bin_local_ind < bins.size()) {
                if (new_bin_local_ind != i) {
                    //remove bins[i][j] from bins[i] and move it to bins[new_bin_local_ind]
                    bins[new_bin_local_ind].push_back(bins[i][j]);
                
                    // Remove the particle index from the current bin
                    bins[i].erase(bins[i].begin() + j);
                }


            } else {
                //get the global bin (new_bin)
                int new_rank = get_rank_from_bin(new_bin, num_procs);

                int sendbuf = bins[i][j] //sending the particle index
                MPI_Send(&sendbuf, 1, MPI_INT, new_rank, 0, MPI_COMM_WORLD);

                if (new_rank > 0 && new_bin < rank_bin_ranges[new_rank].first_bin_global_ind + num_bins) {
                    MPI_Send(&sendbuf, 1, MPI_INT, new_rank - 1, 0, MPI_COMM_WORLD);
                } else if (new_rank < num_procs - 1 && new_bin > rank_bin_ranges[new_rank].last_bin_global_ind - num_bins) {
                    MPI_Send(&sendbuf, 1, MPI_INT, new_rank + 1, 0, MPI_COMM_WORLD);
                }

                bins[i].erase(bins[i].begin() + j);
            }
        }
    }

}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.

    //First do a Gather to get particle counts

    //Then do Gatherv
}