#ifndef STO_IHT_H
#define STO_IHT_H

#include <armadillo>
#include <string>
#include "Functions.h"
using namespace arma;


/* Description: Parallel Stochastic Iterative Hard Thresholding (StoIHT) algorithm 
 to approximate the vector x from measurements u = A*x.*/

vec parallel_Sto_IHT(const mat A, const vec y, const int sparsity, const vec prob_vec,
		const unsigned int max_iter, const double gamma, const double tol, 
		unsigned int &num_iters, const simulation_parameters simulation_params);

uvec Sto_IHT_async_iteration(vec &x_hat,const vec &tally,const mat &A,const vec &y,
 			const int sparsity, const vec prob_vec, const double gamma);

uvec Faulty_Sto_IHT_async_iteration(vec &x_hat,	const int sparsity);

/* Description: Parallel Stochastic Iterative Hard Thresholding (StoIHT) algorithm with tally score to approximate the vector x from measurements u = A*x.*/

void update_tally(vec &tally,const uvec est_supp_local,const uvec prev_est_supp,const unsigned int iter_local);

void majority_voting(vec &tally,const uvec est_supp_local,const uvec prev_est_supp,const unsigned int iter_local);

vec tally_Sto_IHT(const mat &A, const vec &y, const int sparsity, const vec prob_vec,
		const unsigned int max_iter, const double gamma,const double tol, 
		unsigned int &num_iters, const simulation_parameters simulation_params,
		 string voting_type); 

#endif /* STO_IHT_H */ 
