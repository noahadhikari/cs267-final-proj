#!/bin/bash


# if the sparse.csv is empty, add a header

if [ ! -s sparse.csv ]; then
    echo "Length,Density,Baseline,TasksPerNode,Distribution,Time" > sparse.csv
fi

densities=("0.01" "0.03" "0.05" "0.1" "0.2")
# densities=("0.1")

baselines=("5")

tasks_per_node=("2" "4" "8" "16" "32" "64")

lengths=("1000000")

distributions=("1" "2" "3")

echo "Running sparse execution tests..."

for l in "${lengths[@]}"; do

    # Replace VECTOR_LENGTH line to match for bitmask compression
    sed -i "s/constexpr size_t VECTOR_LENGTH = [0-9]\+;/constexpr size_t VECTOR_LENGTH = $l;/" ../compression.hpp
    cmake -DCMAKE_BUILD_TYPE=Release ..; make;

    for b in "${baselines[@]}"; do
        # for v in "${distributions[@]}"; do
        for d in "${densities[@]}"; do
            for tpn in "${tasks_per_node[@]}"; do
                echo "Running with length=$l, distr=$v, density=$d, baseline=$b, tasks-per-node=$tpn..."
                output=$(srun -N 1 --ntasks-per-node=$tpn ./sparse -l $l -d $d -b $b)

                simulation_time=$(echo "$output" | grep -oP 'Time = \K[0-9.]+')
                echo "$l,$d,$b,$tpn,$v,$simulation_time" >> sparse.csv
            done
            echo "Running with length=$l, distr=$v, density=$d, baseline=$b, tasks-per-node=2x64..."
            output=$(srun -N 2 --ntasks-per-node=64 ./sparse -l $l -d $d -b $b)

            simulation_time=$(echo "$output" | grep -oP 'Time = \K[0-9.]+')
            echo "$l,$d,$b,128,$v,$simulation_time" >> sparse.csv
        done
        # done
    done
done

echo "Sparse execution tests completed."
