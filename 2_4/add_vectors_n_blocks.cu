#include <stdio.h>
#include <stdlib.h>

#define N (2048*2048)
#define THREADS_PER_BLOCK 512

... void add_N_M (int *a, int *b, int *c, int dim) {
  // ...
}

void random_ints (int *v, int dim) {
  int i;
  
  for (i=0; i<dim; i++) {
    v[i] = rand() % 100 - 50;
  }
}

int main(void) {
  int *a, *b, *c;       // host copies of a, b, c
  int *d_a, *d_b, *d_c; // device copies of a, b, c
  int size = N * sizeof(int);
  int i, sum_cpu = 0, sum_gpu = 0;

  // Alloc space for host copies of a, b, c and setup input values
  a = (int *)malloc(size); random_ints(a, N);
  b = (int *)malloc(size); random_ints(b, N);
  c = (int *)malloc(size); random_ints(c, N);
  for (i=0; i<N; i++) {
    sum_cpu += (a[i] + b[i]);
  }

  // Alloc space for device copies of a, b, c
  // ... for d_a
  // ... for d_b
  // ... for d_c

  // Copy inputs to device
  // ... a -> d_a
  // ... b -> d_b

  // Launch add() kernel on GPU
  printf ("(A) First Position in each vector => (%d, %d, %d)\n", *a, *b, *c);
  printf ("(A) Last  Position in each vector => (%d, %d, %d)\n", 
                                                     a[N-1], b[N-1], c[N-1]);
  int numBlocks = (N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
  printf ("numBlocks => %d\n", numBlocks);
  add_N_M ... (d_a, d_b, d_c, N);

  // Copy result back to host
  // ... c <- d_c
  printf ("(B) First Position in each vector => (%d, %d, %d)\n", *a, *b, *c);
  printf ("(B) Last  Position in each vector => (%d, %d, %d)\n", 
                                                     a[N-1], b[N-1], c[N-1]);
  sum_gpu = 0;
  for (i=0; i<N; i++) {
    sum_gpu += c[i];
  }
  printf ("sum_cpu = %d , sum_gpu = %d , diff = %d\n", sum_cpu, sum_gpu, 
                                                        sum_cpu-sum_gpu);

  // Cleanup
  // .. freeing d_a, d_b, d_c
  free(a); free(b); free(c);

  return 0;
}
