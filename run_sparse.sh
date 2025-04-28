#!/bin/bash

# echo "Density,Baseline,TasksPerNode,Time" > sparse.csv

densities=("0.01" "0.03" "0.05" "0.1" "0.2")

baselines=("11" "12" "13")

tasks_per_node=("2" "4" "8" "16" "32" "64")

length=1000000

echo "Running sparse execution tests..."

for b in "${baselines[@]}"; do
    for d in "${densities[@]}"; do
        for tpn in "${tasks_per_node[@]}"; do
            echo "Running with density=$d, baseline=$b, tasks-per-node=$tpn..."
            output=$(srun -N 1 --ntasks-per-node=$tpn ./sparse -l $length -d $d -b $b)

            simulation_time=$(echo "$output" | grep -oP 'Time = \K[0-9.]+')
            echo "$d,$b,$tpn,$simulation_time" >> sparse.csv
        done
        echo "Running with density=$d, baseline=$b, tasks-per-node=128..."
        output=$(srun -N 2 --ntasks-per-node=64 ./sparse -l $length -d $d -b $b)

        simulation_time=$(echo "$output" | grep -oP 'Time = \K[0-9.]+')
        echo "$d,$b,128,$simulation_time" >> sparse.csv
    done
done

echo "Sparse execution tests completed."
