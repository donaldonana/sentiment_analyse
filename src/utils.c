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
void forward(RNN *rnn, double **x, int t_p)
{
	//self.last_inputs = inputs
	//double **hs = allocate_dynamic_float_matrix(t_p+1, rnn->hidden_size);
	double *h = malloc(sizeof(double)*rnn->hidden_size);
	initialize_vect_zero(h, rnn->hidden_size);
	copy_vect(rnn->last_hs[0], h, rnn->hidden_size);
	rnn->last_intput = x;
	double *temp1, *temp2, *temp3, *temp4, *temp5, *temp6;
	temp3 = malloc(sizeof(double)*rnn->hidden_size);
	temp6 = malloc(sizeof(double)*rnn->output_size);
	temp2 = malloc(sizeof(double)*rnn->hidden_size);
	temp4 = malloc(sizeof(double)*rnn->hidden_size);
	temp5 = malloc(sizeof(double)*rnn->output_size);

	temp1 = malloc(sizeof(double)*rnn->hidden_size);



	for (int i = 0; i < t_p; i++)
	{
		//h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
		//printf("\n\n");
		//for (int p = 0; p < rnn->input_size; p++)
		//{
			//printf("%lf \n", x[i][p]);
		//}
		
		mat_mul(temp1 , x[i], rnn->Wxh, rnn->input_size, rnn->hidden_size);
		mat_mul(temp2 , rnn->last_hs[i], rnn->Whh, rnn->hidden_size, rnn->hidden_size);
		add_vect(temp3 ,temp1, temp2, rnn->hidden_size);
		add_vect(temp4, temp3, rnn->bh, rnn->hidden_size);
		tan_h(rnn->last_hs[i+1], rnn->hidden_size, temp4);
		tan_h(h, rnn->hidden_size, temp4);
		
	}

	mat_mul(temp5 , h, rnn->Wyh, rnn->hidden_size, rnn->output_size);
	add_vect(temp6 , temp5, rnn->by, rnn->output_size);
	
	softmax(rnn->y , rnn->output_size, temp6);
	

	//free(h);
	//free(temp1);
	//free(temp2);
	//free(temp3);
	//free(temp6);

	//free(temp4);
	//free(temp5);
	
	



	
}

void backforward(RNN *rnn, double *d_y, int t_p)
{

	double *temp1;
	double *temp2;
	double *temp ;
	double **donald1 = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->hidden_size);
	double **donald2 = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->input_size);
	// Calculate dL/dWhy and dL/dby.
    double *d_by = malloc(sizeof(double)*rnn->output_size);
	copy_vect(d_by, d_y, rnn->output_size);
	double **d_Why = allocate_dynamic_float_matrix(rnn->output_size, rnn->hidden_size);
	vect_mult(d_Why, d_y, rnn->last_hs[t_p], rnn->output_size, rnn->hidden_size);

	// Initialize dL/dWhh, dL/dWxh, and dL/dbh to zero.
	double **d_Whh = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->hidden_size);
	initialize_mat_zero(d_Whh, rnn->hidden_size, rnn->hidden_size);
	double **d_Wxh = allocate_dynamic_float_matrix( rnn->hidden_size, rnn->input_size);
	initialize_mat_zero(d_Wxh, rnn->hidden_size, rnn->input_size);
	double *d_bh = malloc(sizeof(double)*rnn->hidden_size);
	initialize_vect_zero(d_bh, rnn->hidden_size);

	//Calculate dL/dh for the last h.
	double *d_h = malloc(sizeof(double)*rnn->hidden_size);
	double **whyT = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->output_size);
	trans_mat(whyT, rnn->Wyh, rnn->output_size, rnn->hidden_size);
	mat_mul(d_h , d_y, whyT,  rnn->output_size, rnn->hidden_size);
    //printf("\n -----Bonjour---- \n");




	for (int i = t_p-1; i >= 0; i--)
	{
		temp1 = vect_pow_2(rnn->last_hs[i+1], rnn->hidden_size);
		temp2 = one_minus_vect(temp1, rnn->hidden_size);
		temp = hadamar_vect(d_h, temp2, rnn->hidden_size);


		// dL/db = dL/dh * (1 - h^2)
		add_vect(d_bh, d_bh, temp, rnn->hidden_size);

		//dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
		vect_mult(donald1, temp, rnn->last_hs[i], rnn->hidden_size, rnn->hidden_size);
		add_matrix(d_Whh , d_Whh, donald1 , rnn->hidden_size, rnn->hidden_size);

		//dL/dWxh = dL/dh * (1 - h^2) * x
		/*printf("\n\n");
		for (int p = 0; p < rnn->input_size; p++)
		{
			printf("%lf \n", rnn->last_intput[i][p]);
		}*/
		vect_mult(donald2, temp, rnn->last_intput[i], rnn->hidden_size, rnn->input_size );
		add_matrix(d_Wxh , d_Wxh, donald2, rnn->hidden_size, rnn->input_size);

		//Next dL/dh = dL/dh * (1 - h^2) * Whh
		mat_mul(d_h , temp, rnn->Whh,  rnn->hidden_size, rnn->hidden_size);
		
	}

	//Update weights and biases using gradient descent.


	
	//scal_mult_mat(d_Whh, d_Whh, 0.02, rnn->hidden_size, rnn->hidden_size);

	minus_matrix(rnn->Whh ,rnn->Whh, d_Whh, rnn->hidden_size, rnn->hidden_size);

	//free(donald); donald = NULL;
		

	//scal_mult_mat(d_Wxh, d_Wxh, 0.02, rnn->hidden_size, rnn->input_size);
	minus_matrix(rnn->Wxh ,rnn->Wxh, d_Wxh,rnn->hidden_size, rnn->input_size);
	//free(donald); donald = NULL;


	//scal_mult_mat(d_Why, d_Why, 0.02, rnn->output_size, rnn->hidden_size);

	minus_matrix(rnn->Wyh ,rnn->Wyh, d_Why , rnn->output_size, rnn->hidden_size);
	
	//free(donald); donald = NULL;


	scal_mult_vect(d_bh, d_bh, 0.02, rnn->hidden_size);
	minus_vect(rnn->bh ,rnn->bh, d_bh , rnn->hidden_size);
	//printf("\n---------------Bonjour----------\n");

	scal_mult_vect(d_by, d_by, 0.02, rnn->output_size);
	minus_vect(rnn->by, rnn->by, d_by, rnn->output_size);

	deallocate_dynamic_float_matrix(d_Whh, rnn->hidden_size);
	deallocate_dynamic_float_matrix(donald1, rnn->hidden_size);
	deallocate_dynamic_float_matrix(d_Wxh, rnn->hidden_size);
	deallocate_dynamic_float_matrix(donald2, rnn->hidden_size);
	deallocate_dynamic_float_matrix(d_Why, rnn->output_size);
	deallocate_dynamic_float_matrix(whyT, rnn->hidden_size);

	//deallocate_dynamic_float_matrix(last_h, t_p);

	free(temp); temp = NULL;
	free(d_bh);
	free(d_by);
	free(temp2);
	free(temp1);

	//free(d_by);
	free(d_h); d_h = NULL;
	//free(donald);


	//printf("\n ----tout est ok back----- \n");
	


}


void mat_mul(double *r, double* a, double** b, int n, int p) {
    // matrix a of size n x 1 (array)
    // matrix b of size p x n
    // matrix result of size p x 1 -- ( (p,n) x (n,1) )
    // result = b * a
	
    int j, k;
    for (j = 0; j < p; j++) {
        r[j] = 0.0;
        for (k = 0; k < n; k++)
            r[j] += (a[k] * b[j][k]);
    }

	
}

void vect_mult(double **r, double *a , double *b, int n , int m)
{
	// matrix a of size n x 1 (array)
    // matrix b of size 1 x m
    // matrix result of size n x m (array)
    // result = a * b

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			r[i][j] = a[i]*b[j]; 
		}
		
	}

	
	
}

void add_vect(double *r , double *a, double *b, int n)
{
	for (int i = 0; i < n; i++)
	{
		r[i] = a[i] + b[i] ;
	}

	
}

void minus_vect(double *r, double *a, double *b, int n)
{
	
	for (int i = 0; i < n; i++)
	{
		r[i] = a[i] - b[i] ;
	}

	
}

void scal_mult_vect(double *r, double *a, double scal, int n)
{
	for (int i = 0; i < n; i++)
	{
		r[i] = a[i] * scal ;
	}
}


void initialize_rnn(RNN *rnn, int input_size, int hidden_size, int output_size)
{

	rnn->input_size = input_size;
	rnn->hidden_size = hidden_size;
	rnn->output_size = output_size;
	rnn->Wxh = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->input_size);
	randomly_initalialize_mat(rnn->Wxh, rnn->hidden_size, rnn->input_size);

	rnn->Whh = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->hidden_size);
	randomly_initalialize_mat(rnn->Whh, rnn->hidden_size, rnn->hidden_size);
	
	rnn->Wyh = allocate_dynamic_float_matrix(rnn->output_size, rnn->hidden_size);
	randomly_initalialize_mat(rnn->Wyh, rnn->output_size, rnn->hidden_size);

	rnn->bh = malloc(sizeof(double)*rnn->hidden_size);
	initialize_vect_zero(rnn->bh, rnn->hidden_size);
	rnn->by = malloc(sizeof(double)*rnn->output_size);
	initialize_vect_zero(rnn->by, rnn->output_size);
	rnn->y = malloc(sizeof(double)*rnn->output_size);

	rnn->last_hs = allocate_dynamic_float_matrix(100, rnn->hidden_size);


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

void scal_mult_mat(double **r, double **a, double scal, int row, int col)
{
	
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{

			r[i][j] = scal*a[i][j];

		}
		
	}
	
}

void add_matrix(double **r, double **a , double **b, int row, int col)
{

	

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			r[i][j] = a[i][j] + b[i][j];
		}
		
	}

	
}

void minus_matrix(double **r, double **a , double **b, int row, int col)
{


	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			r[i][j] = a[i][j] - b[i][j];

		}
		
	}

}

void trans_mat(double **r, double **a, int row , int col)
{
	
	for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            r[j][i] = a[i][j];
        }
    }

	
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


void tan_h(double *r , int n, double* input) {
    //output[0] = 1; // Bias term


    int i;
    for (i = 0; i < n; i++) 
	{
        r[i] = tanh(input[i]); // tanh function

	}

	
}


void softmax(double *r, int n, double* input) {

    //output[0] = 1; // Bias term

    int i;
    double sum = 0.0;
    for (i = 0; i < n; i++)
	{
        sum += exp(input[i]);

	}

    for (i = 0; i < n; i++) 
	{
        r[i] = exp(input[i]) / sum; // Softmax function

	}

}

void copy_vect(double *a, double *b , int n)
{
	for (int i = 0; i < n; i++)
	{
		a[i] = b[i];
	}
	
}

PHRASE **BuildBacht(PHRASE *phrase, int np, int nthreads)
{
	int bacht_size = np / nthreads;
	PHRASE **finals = malloc(sizeof(PHRASE *)*bacht_size);
	for (int i = 0; i < nthreads; i++)
	{
		finals[i] = malloc(sizeof(phrase)*bacht_size);
	}
	

	for (int i = 0; i < nthreads; i++)
	{
		for (int j = 0; j < bacht_size; j++)
		{
			finals[i][j] = phrase[j];
		}
		
	}

	return finals;
	
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

void deallocate_dynamic_float_matrix(double **matrix, int row)
{
    int i;

    for (i = 0; i < row; ++i)
    {
        free(matrix[i]);
		matrix[i] = NULL;
    }
    //free(matrix);
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
	FILE* stream = fopen("./target.txt", "r");
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


void randomize(int *array, int n) {
    int i;
    for(i = n-1; i > 0; i--) {
        int j = rand() % (i+1);
        int t = array[j];
        array[j] = array[i];
        array[i] = t;
    }
}

 void ToEyeMatrix(double **A, int row, int col) {

for(int i=0;i<row;i++)                                                           
  {                                                                             
    for(int j=0;j<col;j++)                                                      
    {                                                                           
      if(i==j)                                                                  
      {                                                                         
        A[i][j] = 1;                                              
      }                                                                         
      else                                                                      
      {                                                                         
       A[i][j] = 0;                                               
      }                                                                         
    }                                                                           
  } 
}                 


void MatrixMult(double **c, double **a, double **b , int n){

	for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
		  c[i][j] = 0;
         for (int k = 0; k < n; ++k) {
            c[i][j] += a[i][k] * b[k][j];
         }
      }
   }
}
