#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include <time.h>









void initialize_rnn(RNN *rnn, int input_size, int hidden_size, int output_size)
{

	rnn->input_size = input_size;
	rnn->hidden_size = hidden_size;
	rnn->output_size = output_size;
	rnn->Wxh = allocate_dynamic_float_matrix(rnn->input_size, rnn->hidden_size);
	randomly_initalialize_mat(rnn->Wxh, rnn->input_size, rnn->hidden_size);
	rnn->Whh = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->hidden_size);
	randomly_initalialize_mat(rnn->Whh, rnn->hidden_size, rnn->hidden_size);
	rnn->Wyh = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->output_size);
	randomly_initalialize_mat(rnn->Wyh, rnn->hidden_size, rnn->output_size);
	rnn->bh = malloc(sizeof(double)*rnn->hidden_size);
	initialize_vect_zero(rnn->bh, rnn->hidden_size);
	rnn->by = malloc(sizeof(double)*rnn->output_size);
	initialize_vect_zero(rnn->by, rnn->output_size);
	rnn->y = malloc(sizeof(double)*rnn->output_size);


}


void randomly_initalialize_mat(double **a, int row, int col)
{

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			a[i][j] = (double)rand()/(double)(RAND_MAX/1);
		}
		
	}
	
}

void initialize_vect_zero(double *a, int n)
{
	for (int i = 0; i < n; i++)
	{
		a[i] = 0 ;
	}
	
}






/**
* MSE (Mean Square Error) function
* @param y_pred : The predicted values
* @param y : Real values
* @param n : y_pred and y length
* @description : This function compute the MSE beetwen the real and  predicted values
* @return: The MSE beetwen y_pred, y
**/
double MSE(double *y_pred , double *y, double n) {

	double diff, sum_sq = 0.00 ;

	for (int i = 0; i < n; ++i)
	{
		diff = y[i] - y_pred[i] ;
		sum_sq += pow(diff,2) ;

	}

	return (sum_sq / n );
}





//====================================================



void randomize(int *array, int n) {
    int i;
    for(i = n-1; i > 0; i--) {
        int j = rand() % (i+1);
        int t = array[j];
        array[j] = array[i];
        array[i] = t;
    }
}


void plot_error_iter(double *e)
{
	char * commandsForGnuplot[] = {"set title \"TITLE\"", "plot 'data.temp' w l"};
	FILE * temp = fopen("data.temp", "w");
	FILE * gnuplotPipe = popen ("gnuplot -persistent", "w");

	for (int i=0; i < EPOCHS; i++)
	{
		fprintf(temp, "%lf %lf \n", (double)i , e[i]); //Write the data to a temporary file
	}

	for (int i=0; i < NUM_COMMANDS; i++)
	{
		fprintf(gnuplotPipe, "%s \n", commandsForGnuplot[i]); //Send commands to gnuplot one by one.
	}
}



double **allocate_dynamic_float_matrix(int row, int col)
{
    double **ret_val;
    int i;

    ret_val = malloc(sizeof(double *) * row);
    if (ret_val == NULL)
    {
        perror("memory allocation failure");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < row; ++i)
    {
        ret_val[i] = malloc(sizeof(double) * col);
        if (ret_val[i] == NULL)
        {
            perror("memory allocation failure");
            exit(EXIT_FAILURE);
        }
    }

    return ret_val;
}

void deallocate_dynamic_float_matrix(float **matrix, int row)
{
    int i;

    for (i = 0; i < row; ++i)
    {
        free(matrix[i]);
    }
    free(matrix);
}

void display_matrix(double **a, int row, int col)
{
	for (int i = 0; i < row; i++)
	{
		printf("\n");
		for (int j = 0; j < col; j++)
		{
			printf(" %lf \t", a[i][j]);
		}
		
	}
	
}
