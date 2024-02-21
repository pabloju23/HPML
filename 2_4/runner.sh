#!/bin/bash

# Define the range of threads and blocks
threads=(256 512 1024)
blocks=(1 2 4 16 128 512)

# Define the number of rectangles
rectangles=(10000000 50000000 100000000 500000000)

# Loop through each combination
for th in "${threads[@]}"; do
    for bl in "${blocks[@]}"; do
        for rec in "${rectangles[@]}"; do
            echo "Running with Threads: $th, Blocks: $bl, Rectangles: $rec"
            patan-run cuda pi_par_loop.cu $th $bl $rec
            echo ""
        done
    done
done
