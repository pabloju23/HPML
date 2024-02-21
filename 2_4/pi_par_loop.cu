#include <stdio.h>
#include <stdlib.h>

// CUDA header for handling CUDA events
#include <cuda_runtime.h>

// CUDA kernel definition
__global__ void calculatePi(float step, int num_steps, float* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x;
    float sum = 0.0;
    for (int i = idx; i < num_steps; i += blockDim.x * gridDim.x) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    atomicAdd(result, step * sum); // Use atomic operation to accumulate local result
}

int main(int argc, char* argv[]) {
    float t1, t2, t_seq, t_par, sp, ep;

    // Adjust the number of rectangles, threads, and blocks
    int num_steps = 100000;
    int num_threads = 256;
    int num_blocks = 1;

    if (argc == 4) {
        num_steps = atoi(argv[1]);
        num_threads = atoi(argv[2]);
        num_blocks = atoi(argv[3]);
    } else if (argc != 1) {
        printf("Wrong number of parameters\n");
        printf("./a.out [ num_steps num_threads num_blocks ]\n");
        exit(-1);
    }

    /*************************************/
    /******** Computation of pi **********/
    /*************************************/

    int i;
    double step = 1.0 / (double)num_steps;
    float pi = 0.0;

    //
    // Sequential implementation
    //
    double x, sum = 0.0;
    t1 = clock();
    step = 1.0 / (double)num_steps;
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    t2 = clock();
    t_seq = (t2 - t1) / (double)CLOCKS_PER_SEC;

    printf(" pi_seq = %20.15f\n", pi);
    printf(" time_seq = %20.15f\n", t_seq);

    //
    // Parallel implementation
    //

    // Allocate memory for the result on the device
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));

    // Initialize result to 0
    cudaMemset(d_result, 0, sizeof(double));

    // Start timing parallel computation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Call the CUDA kernel
    calculatePi<<<num_blocks, num_threads>>>(step, num_steps, d_result);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_par, start, stop);
    // Calculate speedup and efficiency
    sp = t_seq / t_par;
    ep = sp / num_threads;

    // Copy result from device to host
    cudaMemcpy(&pi, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated memory
    cudaFree(d_result);

    printf(" pi_par = %20.15f\n", pi);
    printf(" time_par = %20.15f, Sp = %20.15f , Ep = %20.15f\n", t_par, sp, ep);

    return 0;
}