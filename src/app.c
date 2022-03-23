#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"
#include <math.h>




int main()
{


  srand(time(NULL)); 

  RNN *rnn = malloc(sizeof(RNN));
  int intputs = 2 , hidden = 1 , output = 2;
  initialize_rnn(rnn, intputs, hidden, output);
  //display_matrix(rnn->Wxh, rnn->input_size, rnn->hidden_size);

  double **intput = allocate_dynamic_float_matrix(2,2);
  double m1[] = {1,0};
  double m2[] = {0,1};
  intput[0] = m1;
  intput[1] = m2;

  // the target
  int target  = 1;

  
  
  //printf("\n-----tout est oki-----\n");

  for (int i = 0; i < 30; i++)
  {
    //forward
    double **last_h = forward(rnn, intput, 2);

    for (int j = 0; j < rnn->output_size; j++)
    {
      printf("\n %d : %lf", j, rnn->y[j]);
    }

    double loss = (-1)*log(rnn->y[target]);


    printf("\n log error : %lf", loss);

    //# Build dL/dy
    double *dl_dy = malloc(sizeof(double)*rnn->output_size);
    copy_vect(dl_dy, rnn->y, rnn->output_size);
    dl_dy[target] = dl_dy[target] - 1 ;

    //backforward
    backforward(rnn, dl_dy, last_h, 2);



    
  }
  
  printf("\n-----tout est oki-----\n");

	return 0 ;

}
