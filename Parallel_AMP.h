#ifndef Parallel_AMP
#define Parallel_AMP

#include <armadillo>
#include <string>
#include "Functions.h"
using namespace arma;

vec eta_deriv(const vec &v, const double tau);

vec eta(const vec &v, const double tau);

vec eta_deriv(const vec &v, const vec tau);

vec eta(const vec &v, const vec tau);

vec AMP(const mat &A, const vec &y, const int sparsity, const unsigned int max_iter,
	 const double tol, unsigned int &num_iters, const simulation_parameters simulation_params);

vec R_MP_AMP(const mat &A, const vec &y, const int sparsity, const unsigned int max_iter,
	const double tol, unsigned int &num_iters, const simulation_parameters simulation_params);

vec Async_MP_AMP(const mat &A, const vec &y, const int sparsity, const unsigned int max_iter,
	const double tol, unsigned int &num_iters, const simulation_parameters simulation_params,
	unsigned int num_block);

#endif /* Parallel_AMP */
