#include <stdio.h>
#include <stdlib.h>

#define N 256

.... void add_1_1 (int *a, int *b, int *c) {
  ...
}

.... void add_N_1 (int *a, int *b, int *c) {
  ...
}

.... void add_1_N (int *a, int *b, int *c) {
  ...
}

void random_ints (int *v, int dim) {
  int i;
  
  for (i=0; i<dim; i++) {
    v[i] = rand() % 100 - 50;
  }
}

int main(void) {
  int i, j, dim;
  int *a, *b, *c;       // host copies of a, b, c
  int *d_a, *d_b, *d_c; // device copies of a, b, c
  int size = N * sizeof(int);
  int sum_seq = 0, sum_cpu = 0, sum_gpu = 0;

  // Alloc space for host copies of a, b, c and setup input values
  a = (int *)malloc(size); random_ints(a, N);
  b = (int *)malloc(size); random_ints(b, N);
  c = (int *)malloc(size); 
  for (i=0; i<N; i++) {
    sum_seq += (a[i] + b[i]);
  }

  // Alloc space for device copies of a, b, c
  // ... for d_a
  // ... for d_b
  // ... for d_c

  // Copy inputs to device
  // ... a -> d_a
  // ... b -> d_b

  for (j=0; j<3; j++) {
    sum_cpu = sum_seq; dim = N;
    random_ints(c, dim);
    printf ("(A) First Position in each vector => (%d, %d, %d)\n", *a, *b, *c);
    printf ("(A) Last  Position in each vector => (%d, %d, %d)\n", 
                                                       a[N-1], b[N-1], c[N-1]);

    // Launch add() kernel on GPU
    switch (j) {
      case 0:
          sum_cpu = a[0] + b[0]; dim = 1;
          add_1_1 ... (d_a, d_b, d_c); 
          break;
      case 1:
          add_N_1 ... (d_a, d_b, d_c);
          break;
      case 2:
          add_1_N ... (d_a, d_b, d_c);
          break;
      default:
          break;
    }

    // Copy result back to host
    // ... c <- d_c
    printf ("(B) First Position in each vector => (%d, %d, %d)\n", *a, *b, *c);
    printf ("(B) Last  Position in each vector => (%d, %d, %d)\n", 
                                                       a[N-1], b[N-1], c[N-1]);
    sum_gpu = 0;
    for (i=0; i<dim; i++) {
      sum_gpu += c[i];
    }
    printf ("sum_cpu = %d , sum_gpu = %d , diff = %d\n", sum_cpu, sum_gpu, 
                                                          sum_cpu-sum_gpu);
  }

  // Cleanup
  // .. freeing d_a, d_b, d_c
  free(a); free(b); free(c);

  return 0;
}
