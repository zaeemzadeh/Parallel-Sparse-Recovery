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
	double tau = 1;
	bool done = false;
	vec pseudo_data (N,fill::zeros);

	while(!done){
		i++;
	//	cout << eta_deriv( A.t()*z + x_t_1)  << endl;
		
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

	const unsigned int N = A.n_cols;
	const unsigned int M = y.n_elem;
	const unsigned int P = simulation_params.num_cores-1;
	unsigned int i = 0;
	vec x_t(N,fill::zeros);
	double g_t = M;
	vector <vec> pseudo_data (P,vec(N,fill::zeros));
	bool done = false;
	double tau = 1;
	// parallel section of the code starts here
	#pragma omp parallel num_threads(simulation_params.num_cores)
	{
	// initializing variables in local memory
	const int p = omp_get_thread_num();
	 mat A_p; 
	 vec y_p; 
	if (p > 0){
		A_p = A.rows((M*(p-1))/P , (M*p)/P -1 );
		y_p = y.subvec( (M*(p-1))/P  , (M*p)/P -1 );
	}
	vec z_t_p = y_p;
	// R_MP_AMP itearations
	while(!done){

		//AT processor p:
		if (p > 0){ 
			i++;
			z_t_p = y_p - A_p*x_t + z_t_p * g_t / M;
			pseudo_data[p-1] = A_p.t() * z_t_p + x_t/P;
		}

		#pragma omp barrier
		//AT fusion center:
		if (p == 0)
		{
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


vec Async_MP_AMP(const mat &A, const vec &y, const int sparsity, const unsigned int max_iter,
	const double tol, unsigned int &num_iters, const simulation_parameters simulation_params){

	omp_lock_t deque_lock;	
	omp_lock_t data_lock;
	omp_init_lock(&deque_lock);
	omp_init_lock(&data_lock);
	const unsigned int N = A.n_cols;
	const unsigned int M = y.n_elem;
	const unsigned int P = simulation_params.num_cores-1;
	const unsigned int num_blocks = simulation_params.num_cores-1;	
	unsigned int i = 0;
	vec x_t(N,fill::zeros);
	bool new_data_ready = true;
	bool done = false;
	double g_t = M;
	double tau = 1;
	vector <mat> A_block (num_blocks); 
	vector <vec> y_block (num_blocks); 
	vector <vec> pseudo_data_block (num_blocks);
	vector <vec> z_t_block (num_blocks);

	for (unsigned int b = 0; b < num_blocks; b++){
		A_block[b]  = A.rows(M*b/num_blocks , M*(b+1)/num_blocks -1 );
		y_block[b]  = y.subvec( M*b/num_blocks  , M*(b+1)/num_blocks -1 );
		z_t_block[b]= y_block[b];
	}
	cout << "keep the light in" << endl << flush;

	unsigned int num_processed_blocks = 0;
	// parallel section of the code starts here
	#pragma omp parallel num_threads(simulation_params.num_cores)
	{
	// initializing variables in local memory
	const int p = omp_get_thread_num();
	// R_MP_AMP itearations
	while(!done){

		//AT processor p:
		if (p > 0){ 
			i++;
			int block = p-1;
			z_t_block[block] = y_block[block] - A_block[block]*x_t + z_t_block[block] * g_t / M;
			pseudo_data_block[block] =  A_block[block].t() * z_t_block[block];
			num_processed_blocks++;
			new_data_ready = false;
		}

		//AT fusion center:
		#pragma omp barrier
		//omp_set_lock(&deque_lock);

		if (p == 0 && (num_processed_blocks >= num_blocks))
		{
			vec pseudo_data_total(N,fill::zeros);

			for (unsigned int b = 0; b < num_blocks; b++){
				pseudo_data_total = pseudo_data_total + pseudo_data_block[b];
			}
			pseudo_data_total = pseudo_data_total + x_t;
			tau = tau * sum(eta_deriv(pseudo_data_total,tau)) / M;
			g_t = sum(eta_deriv(pseudo_data_total,tau));
			x_t = eta(pseudo_data_total,tau);
		
			if (norm (y - A*x_t) < tol || i >= max_iter){
				done = true;
			}
			new_data_ready = true;
		}
		
		//}
		#pragma omp barrier
	}
	// parallel section of the code ends here

	}
	omp_destroy_lock(&deque_lock);
	omp_destroy_lock(&data_lock);

	num_iters = i;
	return x_t;
}

/*
vec Async_MP_AMP(const mat &A, const vec &y, const int sparsity, const unsigned int max_iter,
	const double tol, unsigned int &num_iters, const simulation_parameters simulation_params){

	const unsigned int N = A.n_cols;
	const unsigned int M = y.n_elem;
	const unsigned int P = simulation_params.num_cores-1;
	unsigned int i = 0;
	vector <vec> deriv_shared (P,vec(N,fill::zeros));
	vec deriv_total (N,fill::zeros);
	vec x_total (N,fill::zeros);
	bool done = false;
	vec tau (N,fill::ones);
	double g_t = M;
	// parallel section of the code starts here
	#pragma omp parallel num_threads(simulation_params.num_cores)
	{
	// initializing variables in local memory
	const int p = omp_get_thread_num();
	mat A_p; 
	vec y_p; 
	vec pseudo_data_p (N,fill::zeros);
	vec x_t_p (N,fill::zeros);
	vec deriv_local (N,fill::zeros);
	if (p > 0){
		A_p = A.rows((M*(p-1))/P , (M*p)/P -1 );
		y_p = y.subvec( (M*(p-1))/P  , (M*p)/P -1 );
	}
	 if (p == 0){
		A_p = A;
		y_p = y;
	}
	vec z_t_p = y_p;
	while(!done){
		i++;

		//AT processor p:
		if (p == 1){ 
			g_t = sum(eta_deriv(pseudo_data_p,tau));
			x_t_p = eta(pseudo_data_p,tau);
			z_t_p = y_p - A_p*x_t_p + z_t_p * g_t / M;
			pseudo_data_p = A_p.t() * z_t_p + x_t_p;

			deriv_local = eta_deriv(pseudo_data_p,tau);

			tau = tau * sum(deriv_local) / M;			
			//uvec change = find (deriv_local != deriv_shared[p-1]);
			//deriv_shared[p-1](change) = deriv_local(change);
			//deriv_shared[p-1] = deriv_local;

			if (norm (y_p - A_p*x_t_p) < tol || i >= max_iter){
				done = true;
			}
		}

		#pragma omp barrier
		//AT fusion center:
		if (p == 0)
		{	
			vec deriv_total(N,fill::zeros);
			for (unsigned int j = 0; j < P; j++){
				deriv_total = deriv_total + deriv_shared[j];
			}
			deriv_total = deriv_total/P;
			//tau = tau * sum(deriv_total) / M;
			
			z_t_p = y_p - A_p*x_t_p + z_t_p * sum (eta_deriv( pseudo_data_p,tau) ) / M;
			pseudo_data_p = A_p.t() * z_t_p + x_t_p;
			x_t_p = eta(pseudo_data_p,tau);

		}
		#pragma omp barrier

	}
	// parallel section of the code ends here

	}


	num_iters = i;
	return x_total;
}
*/
