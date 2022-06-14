

#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

int main()
{

   
	double **t = allocate_dynamic_float_matrix(2,3);

	double t2[3] = {2,2,3};
	
	/*t[0][0] = 2;
	t[0][1] = 1;
	t[0][2] = 1;
	t[1][0] = 3;
	t[1][1] = 0;
	t[1][2] = 1;

	double *r = malloc(sizeof(double)*3);

	mat_mul(r, t2, t, 4, 2);
    add_vect(r,t2, t2, 3);
	softmax(r, 3, t2);
	for (int i = 0; i < 3; i++)
	{
		printf(" %lf \n", r[i]);
	}
	

	free(r);
	deallocate_dynamic_float_matrix(t, 2);*/
	double r[] = {0,0};
	double *R;
	R = vect_pow_2(t2, 3);
	R = one_minus_vect(t2, 3);

	//double t[2] = {738.134414 , 734.719249};

	

//	softmax(r, 2, t);

	for (int i = 0; i < 3; i++)
	{
		printf(" %lf \n", R[i]);
	}



	//PHRASE **finals ;
  	//finals =  BuildBacht(phrases, np, MAX_THREADS);

	  //struct thread_param *mes_param ;
  //mes_param = (struct thread_param *) thread_data;




	return 0 ;

}
