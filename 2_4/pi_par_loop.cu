#include <stdio.h>
#include <stdlib.h>
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
    int window = num_steps / (gridDim.x * blockDim.x);
    int start = idx * window;
    int end = (idx + 1) * window;
    if (idx == (gridDim.x * blockDim.x) -1) end = num_steps;
    double sum = 0.0;
    for (int i = start; i < end; i ++) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    atomicAddDouble(result, sum); // Use custom atomic operation to accumulate local result
}

int main(int argc, char* argv[]) {
    float t_seq, t_par, sp, ep;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Adjust the number of rectangles, threads, and blocks
    int num_steps = 100000;
    int num_threads = 256;
    int num_blocks = 1;

    if (argc == 4) {
        num_steps = atoi(argv[3]);
        num_threads = atoi(argv[1]);
        num_blocks = atoi(argv[2]);
        printf("Using %d threads, %d blocks and %d steps\n", num_threads, num_blocks, num_steps);
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
    cudaEventRecord(start);
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    t_seq = 0.0;
    cudaEventElapsedTime(&t_seq, start, stop);
    
    printf(" pi_seq = %20.15f\n", pi);
    printf(" time_seq = %20.15f\n", t_seq);

    //
    // Parallel implementation
    //

    int size = num_blocks * num_threads;
    sum = 0.0;
    cudaEventRecord(start);
    // Call the CUDA
    // Allocate memory for the result on the device
    double* d_result;
    cudaMalloc((void**)&d_result, sizeof(double));

    // Initialize result to 0
    cudaMemset(d_result, 0, sizeof(double));

    // Call the CUDA kernel
    calculatePi<<<num_blocks, num_threads>>>(step, num_steps, d_result);

    // Copy result from device to host
    cudaMemcpy(&sum, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    // Free the memory
    cudaFree(d_result);
    pi = step * sum;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    t_par = 0.0;
    cudaEventElapsedTime(&t_par, start, stop);
    
    // Calculate speedup and efficiency
    sp = t_seq / t_par;
    ep = sp / size;

    printf(" pi_par = %20.15f\n", pi);
    printf(" time_par = %20.15f, Sp = %20.15f , Ep = %20.15f\n", t_par, sp, ep);
}