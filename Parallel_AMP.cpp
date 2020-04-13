#include <unistd.h>	// sleep
#include <armadillo>
#include <vector>
#include <deque>
#include <algorithm>    // std::min

#include "Functions.h"

using namespace arma;
using namespace std;


vec eta(const vec &v, const double tau){	
	vec out(size(v),fill::zeros);

	uvec ind = find(abs(v) > tau);
	out(ind) = sign(v(ind))%(abs(v(ind)) - tau );
	return out;
}

vec eta_deriv(const vec &v, const double tau){	
	vec out(size(v),fill::zeros);

	uvec ind = find(abs(v) > tau);
	out(ind).ones();
	return out;
}


vec eta(const vec &v, const vec tau){	
	vec out(size(v),fill::zeros);

	uvec ind = find(abs(v) > tau);
	out(ind) = sign(v(ind))%(abs(v(ind)) - tau(ind) );
	return out;
}

vec eta_deriv(const vec &v, const vec tau){	
	vec out(size(v),fill::zeros);

	uvec ind = find(abs(v) > tau);
	out(ind).ones();
	return out;
}

vec AMP(const mat &A, const vec &y, const int sparsity, const unsigned int max_iter,
	const double tol, unsigned int &num_iters, const simulation_parameters simulation_params){

	const unsigned int N = A.n_cols;   	// signal dimension
	const unsigned int M = y.n_elem;	// number of measurements
	//const double delta = double(num_measurements)/sig_dim;
	unsigned int i = 0;
	vec x_t(N,fill::zeros);
	vec z_t = y;
	double tau = .1;
	bool done = false;
	vec pseudo_data (N,fill::zeros);

	while(!done){
		i++;		
		z_t = y - A*x_t + z_t * sum (eta_deriv( pseudo_data,tau) ) / M;
		pseudo_data = A.t() * z_t + x_t;
		tau = tau * sum(eta_deriv(pseudo_data,tau)) / M;
		x_t = eta(pseudo_data,tau) ;

		if (norm (y - A*x_t) < tol || i >= max_iter){
			done = true;
		}
	}

	num_iters = i;
	return x_t;
}


// J. Zhu, R. Pilgrim and D. Baron, "An overview of multi-processor approximate message passing,"
// http://ieeexplore.ieee.org/document/7926166/
vec R_MP_AMP(const mat &A, const vec &y, const int sparsity, const unsigned int max_iter,
	const double tol, unsigned int &num_iters, const simulation_parameters simulation_params){
	uvec slow_cores;
	set_slow_cores(slow_cores, simulation_params);


	const unsigned int N = A.n_cols;
	const unsigned int M = y.n_elem;
	const unsigned int P = simulation_params.num_cores;
	unsigned int i = 0;
	bool done = false;
	vec x_t(N,fill::zeros);
	double g_t = M;
	double tau = .1;

	vector <vec> pseudo_data (P,vec(N,fill::zeros));
	// parallel section of the code starts here
	#pragma omp parallel num_threads(simulation_params.num_cores) 
	{
	// initializing variables in local memory
	const int p = omp_get_thread_num();
	mat A_p = A.rows(M*p/P , M*(p+1)/P -1 ); 
	vec y_p = y.subvec( M*p/P  , M*(p+1)/P -1 ); 
	
	vec z_t_p = y_p;
	// R_MP_AMP itearations
	while(!done){
		//AT processor p:
		z_t_p = y_p - A_p*x_t + z_t_p * g_t / M;
		pseudo_data[p] = A_p.t() * z_t_p + x_t/P;
		
		//slow cores sleep for  simulation_params.sleep_slow_cores microseconds
		if (any( slow_cores == omp_get_thread_num()) ){
			usleep(simulation_params.sleep_slow_cores);
		}
		

		#pragma omp barrier
		//AT fusion center:
		#pragma omp single
		{	
			i++;
			vec pseudo_data_total(N,fill::zeros);
			for (unsigned int j = 0; j < P; j++){
				pseudo_data_total = pseudo_data_total + pseudo_data[j];
			}
		
			tau = tau * sum(eta_deriv(pseudo_data_total,tau)) / M;
			g_t = sum(eta_deriv(pseudo_data_total,tau));
			x_t = eta(pseudo_data_total,tau);
			if (norm (y - A*x_t) < tol || i >= max_iter){
				done = true;
			}
		}
		#pragma omp barrier
	}
	// parallel section of the code ends here

	}

	num_iters = i;
	return x_t;
}



