#include <stdio.h>

__global__ void mykernel( void ) {
    // Kernel code goes here
    printf("Hello, world from GPU!\n");
}

int main( int argc, char *argv[] ) {
    // Launch kernel on GPU
    mykernel<<<1, 1>>>();
    cudaDeviceSynchronize(); // Ensures all kernel executions are completed before continuing

    // Print "Hello, world!" from CPU
    printf("Hello, world from CPU!\n");

    return 0;
}
