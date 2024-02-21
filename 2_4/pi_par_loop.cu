#include <stdio.h>
#include <stdlib.h>

// CUDA header for handling CUDA events
#include <cuda_runtime.h>

__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// CUDA kernel definition
__global__ void calculatePi(double step, int num_steps, double* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double x;
    double sum = 0.0;
    for (int i = idx; i < num_steps; i += blockDim.x * gridDim.x) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    atomicAddDouble(result, step * sum); // Use custom atomic operation to accumulate local result
}

int main(int argc, char* argv[]) {
    double t_seq, sp, ep;
    float t_seq_float, t_par_float;

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
    double pi = 0.0;

    //
    // Sequential implementation
    //
    double x, sum = 0.0;
    cudaEvent_t start_seq, stop_seq;
    cudaEventCreate(&start_seq);
    cudaEventCreate(&stop_seq);
    cudaEventRecord(start_seq);
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    cudaEventRecord(stop_seq);
    cudaEventSynchronize(stop_seq);
    cudaEventElapsedTime(&t_seq_float, start_seq, stop_seq);
    t_seq = static_cast<double>(t_seq_float) / 1000.0; // Convert from ms to s

    printf(" pi_seq = %20.15f\n", pi);
    printf(" time_seq = %20.15f\n", t_seq);

    //
    // Parallel implementation
    //

    // Allocate memory for the result on the device
    double* d_result;
    cudaMalloc(&d_result, sizeof(double));

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
    cudaEventElapsedTime(&t_par_float, start, stop);
    double t_par = static_cast<double>(t_par_float) / 1000.0; // Convert from ms to s
    // Calculate speedup and efficiency
    sp = t_seq / t_par;
    ep = sp / num_threads;

    // Copy result from device to host
    cudaMemcpy(&pi, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    // Free allocated memory
    cudaFree(d_result);

    printf(" pi_par = %20.15f\n", pi);
    printf(" time_par = %20.15f, Sp = %20.15f , Ep = %20.15f\n", t_par, sp, ep);

    return 0;
}