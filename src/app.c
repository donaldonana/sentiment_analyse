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
  display_matrix(rnn->Wxh, rnn->input_size, rnn->hidden_size);
  printf("\n-----tout est oki-----");
	return 0 ;

}
