#include <stdio.h>
#include <stdlib.h>

// Definition of CUDA kernels
// ... 

int main(int argc, char *argv[]) {
  double t1, t2, t_seq, t_par, sp, ep;

  // Adjust the number of rectangules
  int num_steps = 100000;
  if (argc == 2) {
    num_steps = atoi(argv[1]);
  } else if (argc > 2) {
    printf("Wrong number of parameters\n");
    printf("./a.out [ num_steps ]\n");
    exit(-1);
  }

/*************************************/
/******** Computation of pi **********/
/*************************************/

  int i;
  double step = 1.0 / (double) num_steps;  
  double pi = 0.0;

  //
  // Sequential implementation
  //
  double x, sum = 0.0;
  t1 = ....
  step = 1.0 / (double) num_steps;  
  for (i=0; i<num_steps; i++){
     x = (i+0.5)*step;
     sum = sum + 4.0/(1.0+x*x);
  }
  pi = step * sum;
  t2 = ...
  t_seq = (t2-t1);

  printf(" pi_seq = %20.15f\n", pi);
  printf(" time_seq = %20.15f\n", t_seq);

  //
  // Parallel implementation
  //
  
  // Defining the number of active threads
  int size = 0;
  
  t1 = ...
  // Call to the CUDA 
  // ...
  t2 = ...
  t_par = ...
  sp = ...
  ep = ...

  printf(" pi_par = %20.15f\n", pi);
  printf(" time_par = %20.15f, Sp = %20.15f , Ep = %20.15f\n", t_par, sp, ep);
}
