#ifndef DEF_UTILS
#define DEF_UTILS

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define DATA_SIZE 100
#define BACHT_SIZE 1
#define EPOCHS 450
#define LEARNING_RATE 0.005
#define NUM_COMMANDS 2



typedef struct PHRASE PHRASE;
struct PHRASE
{
	int nm;
	double **w2vec;
};

typedef struct RNN RNN;
struct RNN
{
	//vecteur de poid couche d'entré et couche cachée (inputs X neurons)
	double **Wxh;
	//vecteur de poid couche de contexte (neurons X neurons)
	double **Whh;
	//vecteur de poid couche cachée et couche de sortie (neurons X outputs)
	double **Wyh;
    double **last_intput;
	double *bh;
	double *by;
	double *y ;
	int input_size;
	int hidden_size;
	int output_size;
};




void initialize_rnn(RNN *rnn, int input_size, int hidden_size, int output_size);

void randomly_initalialize_mat(double **a, int row, int col);

double **forward(RNN *rnn, double **x, int t_p);

void backforward(RNN *rnn, double *d_y, double **last_h, int t_p);

double **vect_mult(double *a , double *b, int n , int m);

void initialize_mat_zero(double **a, int row, int col);

double **scal_mult_mat(double **a, double scal, int row, int col);

double *scal_mult_vect(double *a, double scal, int n);

double *minus_vect(double *a, double *b, int n);

double **trans_mat(double **a, int row , int col);

double *mat_mul(double* a, double** b, int n, int p);

double *vect_pow_2(double *a, int n);

double *one_minus_vect(double *a , int n);

double *hadamar_vect(double *a, double *b, int n);

double *softmax(int n, double* input);

double *add_vect(double *a, double *b, int n);

double *tan_h(int n, double* input) ;

void initialize_vect_zero(double *a, int n);

void copy_vect(double *a, double *b , int n);

double **add_matrix(double **a , double **b, int row, int col);

double **minus_matrix(double **a , double **b, int row, int col);

double MSE(double *y_pred , double *y, double n) ;

double **allocate_dynamic_float_matrix(int row, int col);

void deallocate_dynamic_float_matrix(float **matrix, int row);

void display_matrix(double **a, int row, int col);

void plot_error_iter(double *e) ;

void randomize(int *array, int n) ;


void MotsParPhrase(FILE *fin, PHRASE *phrase);

int NPhrases(FILE *fin);


void ReadWord(char *word, FILE *fin) ;

void alloc_phrase(PHRASE *phrase, int *mpp, int layer1_size, int np);

int load_target(int *target);


#endif
