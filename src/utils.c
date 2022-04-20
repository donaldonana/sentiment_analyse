#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include <time.h>
#include <string.h>
#include <pthread.h>
#define MAX_STRING 100
#define TAILLE_MAX 1000000


//tp: taille de la phrase (nombre de mots)
double **forward(RNN *rnn, double **x, int t_p)
{
	//self.last_inputs = inputs
	double **hs = malloc(sizeof(double *)*t_p);
	double *h = malloc(sizeof(double)*rnn->hidden_size);
	initialize_vect_zero(h, rnn->hidden_size);
	hs[0] = h ;
	rnn->last_intput = x;

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

	return hs;
	
}

void backforward(RNN *rnn, double *d_y, double **last_h, int t_p)
{

	// Calculate dL/dWhy and dL/dby.
	double *d_by = d_y;
	double **d_Why = vect_mult(last_h[t_p-1], d_y, rnn->hidden_size, rnn->output_size);
	// Initialize dL/dWhh, dL/dWxh, and dL/dbh to zero.
	double **d_Whh = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->hidden_size);
	initialize_mat_zero(d_Whh, rnn->hidden_size, rnn->hidden_size);
	double **d_Wxh = allocate_dynamic_float_matrix(rnn->input_size, rnn->hidden_size);
	initialize_mat_zero(d_Wxh, rnn->input_size, rnn->hidden_size);
	double *d_bh = malloc(sizeof(double)*rnn->hidden_size);
	initialize_vect_zero(d_bh, rnn->hidden_size);

	//Calculate dL/dh for the last h.
	double *d_h = mat_mul(d_y, trans_mat(rnn->Wyh, rnn->hidden_size, rnn->output_size),  rnn->output_size, rnn->hidden_size);


	for (int i = t_p; i > 0; i--)
	{
		double *temp1 = vect_pow_2(last_h[i], rnn->hidden_size);
		double *temp2 = one_minus_vect(temp1, rnn->hidden_size);
		double *temp = hadamar_vect(d_h, temp2, rnn->hidden_size);


		// dL/db = dL/dh * (1 - h^2)
		d_bh = add_vect(d_bh, temp, rnn->hidden_size);

		//dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
		d_Whh = add_matrix(d_Whh, vect_mult(temp, last_h[i-1], rnn->hidden_size, rnn->hidden_size), 
		rnn->hidden_size, rnn->hidden_size);

		//dL/dWxh = dL/dh * (1 - h^2) * x
		d_Wxh = add_matrix(d_Wxh, vect_mult(temp, rnn->last_intput[i-1], rnn->input_size, rnn->hidden_size), 
		rnn->input_size, rnn->hidden_size);

		//Update weights and biases using gradient descent.

		rnn->Whh = minus_matrix(rnn->Whh, scal_mult_mat(d_Whh, 0.05, rnn->hidden_size, rnn->hidden_size),
							rnn->hidden_size, rnn->hidden_size);

		
		double **donald = scal_mult_mat(d_Wxh, 0.05, rnn->input_size, rnn->hidden_size);


		rnn->Wxh = minus_matrix(rnn->Wxh, donald,
							rnn->input_size, rnn->hidden_size);

		rnn->Wyh = minus_matrix(rnn->Wyh, scal_mult_mat(d_Why, 0.05, rnn->hidden_size, rnn->output_size),
							rnn->hidden_size, rnn->output_size);
		rnn->bh = minus_vect(rnn->bh, scal_mult_vect(d_bh, 0.05, rnn->hidden_size), rnn->hidden_size);
		rnn->by = minus_vect(rnn->by, scal_mult_vect(d_by, 0.05, rnn->output_size), rnn->output_size);





	}

	//printf("\n ----tout est ok back----- \n");
	


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

double **vect_mult(double *a , double *b, int n , int m)
{
	// matrix a of size n x 1 (array)
    // matrix b of size 1 x m
    // matrix result of size n x m (array)
    // result = a * b

	double **result = allocate_dynamic_float_matrix(n,m);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			result[i][j] = a[i]*b[j]; 
		}
		
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

double *minus_vect(double *a, double *b, int n)
{
	double *result = malloc(sizeof(double)*n);
	for (int i = 0; i < n; i++)
	{
		result[i] = a[i] - b[i] ;
	}

	return result ;
	
}

double *scal_mult_vect(double *a, double scal, int n)
{
	double *result = malloc(sizeof(double)*n);
	for (int i = 0; i < n; i++)
	{
		result[i] = a[i] * scal ;
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

void initialize_mat_zero(double **a, int row, int col)
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			a[i][j] = 0;
		}
		
	}
	
}

double **scal_mult_mat(double **a, double scal, int row, int col)
{
	double **result = allocate_dynamic_float_matrix(row, col);

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{

			result[i][j] = scal*a[i][j];

		}
		
	}

	return result;
	
}

double **add_matrix(double **a , double **b, int row, int col)
{

	double **result = allocate_dynamic_float_matrix(row, col);

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			result[i][j] = a[i][j] + b[i][j];
		}
		
	}

	return result;
}

double **minus_matrix(double **a , double **b, int row, int col)
{

	double **result = allocate_dynamic_float_matrix(row, col);

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			result[i][j] = a[i][j] - b[i][j];
		}
		
	}

	return result;
}

double **trans_mat(double **a, int row , int col)
{
	double **result = allocate_dynamic_float_matrix(col, row);
	
	for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            result[j][i] = a[i][j];
        }
    }

	return result ;
}

double *vect_pow_2(double *a, int n)
{
	double *result = malloc(sizeof(double)*n);
	for (int i = 0; i < n; i++)
	{
		result[i] = pow(a[i], 2);
	}

	return result;
	
}

double *one_minus_vect(double *a , int n)
{
	double *result = malloc(sizeof(double)*n);
	for (int i = 0; i < n; i++)
	{
		result[i] = 1 - a[i];
	}

	return result;

}

double *hadamar_vect(double *a, double *b, int n)
{
	double *result = malloc(sizeof(double)*n);
	for (int i = 0; i < n; i++)
	{
		result[i] = b[i] * a[i];
	}

	return result;
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

void copy_vect(double *a, double *b , int n)
{
	for (int i = 0; i < n; i++)
	{
		a[i] = b[i];
	}
	
}
//====================================================




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

int NPhrases(FILE *fin)
{
    int n = 0;
    char chaine[TAILLE_MAX] = "";
    fseek(fin, 0, SEEK_SET);
    while (fgets(chaine, TAILLE_MAX, fin) != NULL) // On lit le fichier tant qu'on ne reçoit pas d'erreur (NULL)
    {
       n = n + 1; // On affiche la chaîne qu'on vient de lire
       
    }

    return n ;

}

void MotsParPhrase(FILE *fin, PHRASE *phrase){

    char word[MAX_STRING];
    int np = NPhrases(fin);
    int p = 0;
    fseek(fin, 0, SEEK_SET); 
    int n = 0;
    fseek(fin, 0, SEEK_SET);
   
       while (1) {
            if (feof(fin) ) break;
            ReadWord(word, fin);
            //printf("\n%s\n", word);
            phrase[n].nm  = p = p + 1 ;
             if (strcmp(word, "</s>") == 0 && p!=0)
             //if (fgetc(fin)=='\n' && p!=0)
            {
                //printf("\n------------\n");
                p = 0;
                n = n + 1;
            }
        }

	for (int i = 0; i < np; i++)
    {
        
        phrase[i].nm = phrase[i].nm - 1;

    } 
    /* for (int i = 0; i < np; i++)
    {
        
        printf("%d \n",temp[i]-1);

    } */

    
}

void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

void alloc_phrase(PHRASE *phrase, int *mpp, int layer1_size, int np){
		printf("\n-----------0----------\n");

	for (int i = 0; i < np; i++)
	{
		printf("\n-----------1----------\n");
		//phrase[i].w2vec = malloc(sizeof(double *)*mpp[i]);
		printf("\n-----------2----------\n");

		
		for (int j = 0; j < mpp[i]; j++)
		{
			//phrase[i].w2vec[j] = malloc(sizeof(double)*layer1_size); 
		}  
		
	}
	
}


int load_target(int *target)
{
	int count = 0;
  	printf("Loading Dataset from dataset/theData.csv ...\n\n");
	FILE* stream = fopen("./targuet.txt", "r");
  	if (stream == NULL) {
    	fprintf(stderr, "Error reading file\n");
    	return 1;
  	}
  	while (fscanf(stream, "%d", &target[count]) == 1) {
      count = count+1;
  	}
	return 0;
    // Uncomment to display loaded data
    // for (int i = 0; i < (int)total_samples; i++) {
    //   printf(" x[%d]:%lf , y[%d]:%lf\n", i,x[i], i,y[i]);
    // }
}