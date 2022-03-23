#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include <time.h>




//tp: taille de la phrase (nombre de mots)
void forward(RNN *rnn, double **x, int t_p)
{
	//self.last_inputs = inputs
	double **hs = malloc(sizeof(double *)*100);
	double *h = malloc(sizeof(double)*rnn->hidden_size);
	initialize_vect_zero(h, rnn->hidden_size);
	hs[0] = h ;


	for (int i = 0; i < t_p; i++)
	{
		//h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
		double *temp1 = mat_mul(x[i], rnn->Wxh, rnn->input_size, rnn->hidden_size);
		double *temp2 = mat_mul(hs[i], rnn->Whh, rnn->hidden_size, rnn->hidden_size);
		double *temp3 = add_vect(temp1, temp2, rnn->hidden_size);
		double *temp4 = add_vect(temp3, rnn->bh, rnn->hidden_size);
		h = tan_h(rnn->hidden_size, temp4);
		hs[i+1] = h ;
	}

	double *temp5 = mat_mul(h, rnn->Wyh, rnn->hidden_size, rnn->output_size);
	double *temp6 = add_vect(temp5, rnn->by, rnn->output_size);
	rnn->y = softmax(rnn->output_size, temp6);
	
}

double *mat_mul(double* a, double** b, int n, int p) {
    // matrix a of size 1 x n (array)
    // matrix b of size n x p
    // matrix result of size 1 x p (array)
    // result = a * b
	double *result = malloc(sizeof(double)*p);
    int j, k;
    for (j = 0; j < p; j++) {
        result[j] = 0.0;
        for (k = 0; k < n; k++)
            result[j] += (a[k] * b[k][j]);
    }

	return result;
}

double *add_vect(double *a, double *b, int n)
{
	double *result = malloc(sizeof(double)*n);
	for (int i = 0; i < n; i++)
	{
		result[i] = a[i] + b[i] ;
	}

	return result ;
	
}


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


double *tan_h(int n, double* input) {
    //output[0] = 1; // Bias term

	double *output = malloc(sizeof(double)*n);

    int i;
    for (i = 0; i < n; i++) 
	{
        output[i] = tanh(input[i]); // tanh function

	}

	return output ; 
}


double *softmax(int n, double* input) {

	double *output = malloc(sizeof(double)*n);
    //output[0] = 1; // Bias term

    int i;
    double sum = 0.0;
    for (i = 0; i < n; i++)
	{
        sum += exp(input[i]);

	}

    for (i = 0; i < n; i++) 
	{
        output[i] = exp(input[i]) / sum; // Softmax function

	}

	return output;
}




void identity(int n, double* input, double* output) {
    output[0] = 1; // Bias term

    int i;
    for (i = 0; i < n; i++) 
        output[i+1] = input[i]; // Identity function
}

void sigmoid(int n, double* input, double* output) {
    output[0] = 1; // Bias term

    int i;
    for (i = 0; i < n; i++) 
        output[i+1] = 1.0 / (1.0 + exp(-input[i])); // Sigmoid function
}


void relu(int n, double* input, double* output) {

    output[0] = 1; // Bias term

    int i;
    for (i = 0; i < n; i++) 
        output[i+1] = MAX(0.0, input[i]); // ReLU function
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
