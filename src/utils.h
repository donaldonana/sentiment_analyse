#ifndef DEF_UTILS
#define DEF_UTILS

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define DATA_SIZE 100
#define BACHT_SIZE 1
#define EPOCHS 450
#define LEARNING_RATE 0.005
#define NUM_COMMANDS 2


typedef struct RNN RNN;
struct RNN
{
	//vecteur de poid couche d'entré et couche cachée (inputs X neurons)
	double **Wxh;
	//vecteur de poid couche de contexte (neurons X neurons)
	double **Whh;
	//vecteur de poid couche cachée et couche de sortie (neurons X outputs)
	double **Wyh;
	double *bh;
	double *by;
	double *y ;
	int input_size;
	int hidden_size;
	int output_size;
};



void initialize_rnn(RNN *rnn, int input_size, int hidden_size, int output_size);

void randomly_initalialize_mat(double **a, int row, int col);

void initialize_vect_zero(double *a, int n);

double MSE(double *y_pred , double *y, double n) ;

double **allocate_dynamic_float_matrix(int row, int col);

void deallocate_dynamic_float_matrix(float **matrix, int row);

void display_matrix(double **a, int row, int col);

void plot_error_iter(double *e) ;

void randomize(int *array, int n) ;








#endif
