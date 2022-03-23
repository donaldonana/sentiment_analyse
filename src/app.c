#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"



int main()
{


  srand(time(NULL)); 

  RNN *rnn = malloc(sizeof(RNN));
  int intputs = 2 , hidden = 3 , output = 2;
  initialize_rnn(rnn, intputs, hidden, output);
  //display_matrix(rnn->Wxh, rnn->input_size, rnn->hidden_size);

  double **intput = allocate_dynamic_float_matrix(2,2);
  double m1[] = {1,0};
  double m2[] = {0,1};
  intput[0] = m1;
  intput[1] = m2;

  forward(rnn, intput, 2);

  for (int i = 0; i < rnn->output_size; i++)
  {
    printf("\n %lf", rnn->y[i]);
  }
  
  
  printf("\n-----tout est oki-----");
	return 0 ;

}
