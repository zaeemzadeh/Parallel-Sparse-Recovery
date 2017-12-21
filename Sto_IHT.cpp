#include <unistd.h>	// sleep
#include <armadillo>
#include "Functions.h"

using namespace arma;
using namespace std;
/* Description: Parallel Stochastic Iterative Hard Thresholding (StoIHT) algorithm 
 to approximate the vector x from measurements u = A*x.
publication:  Linear Convergence of Stochastic Iterative Greedy Algorithms with Sparse Constraints
https://arxiv.org/abs/1407.0088
*/

vec parallel_Sto_IHT(const mat A, const vec y, const int sparsity, const vec prob_vec,
		const unsigned int max_iter, const double gamma, const double tol, 
		unsigned int &num_iters, const simulation_parameters simulation_params){
	// signal parameters
	const unsigned int sig_dim = A.n_cols;
	const unsigned int meas_num = A.n_rows;
	const unsigned int num_block = prob_vec.n_elem;
	const unsigned int block_size = meas_num/num_block;

	// initialization of variables that are shared among cores
	vec x_hat(sig_dim,fill::zeros);	// estimation of the signal
	bool done = false;		// flag to check the convergence criteria
	unsigned int i = 0;		// total number of iterations

	// parallel section of the code starts here
	#pragma omp parallel num_threads(simulation_params.num_cores)
	{
	//#pragma omp single // a single core executes the following line
	//cout << "Number of threads: " << omp_get_num_threads() << endl;
	
	// initializaiotn of variables that are local to each core
	vec x_hat_local(sig_dim,fill::zeros);
	unsigned int selected_block,first_ind_block,last_ind_block;
	mat A_block;			// submatrix of A
	vec y_block,gradient,b;	
	uvec sorted_ind,est_supp;

	// iterations to find the solutions
	while(!done){
		// master thread uses the tally vector to check the convergence criteria
		if (omp_get_thread_num() == 0 ){
			// check exit criteria
			if (norm (y - A*x_hat) < tol || i >= max_iter){
				done = true;	// the flag 'done' is shared among the cores
			}
			if (omp_get_num_threads()  > 1){
				continue;
			}
		}

		i++;
		// randomize
		selected_block = floor(randu()*num_block);
		first_ind_block = block_size*selected_block;
		last_ind_block = block_size*(selected_block+1)-1;
		//cout << first_ind_block << ":" << last_ind_block << endl;

		// Proxy
		A_block = A.rows(first_ind_block,last_ind_block);
		y_block = y.subvec(first_ind_block,last_ind_block);
		#pragma omp critical
		{x_hat_local = x_hat;	}	// read the global estimate while memory is locked

		gradient = -2 * A_block.t() *	(y_block - A_block*x_hat_local);
		b = x_hat_local - gradient * gamma/(num_block*prob_vec(selected_block));
		
		// Identify
		sorted_ind = sort_index(abs(b),"descend");
		est_supp = sorted_ind(span(0,sparsity - 1));

		// Estimate
		x_hat_local.zeros();
		x_hat_local(est_supp) = b(est_supp);
		
		#pragma omp critical
		{x_hat = x_hat_local;}	//write the global estimate while memory is locked
		

		
	}
	}
	// parallel section of the code ends here
	//cout << "#iterations = " << i << endl;
	num_iters = i;
	return x_hat;
}

/* Asynchronous StoIHT Iteration
Algorithm 2 in An Asynchronous Parallel Approach to Sparse Recovery
https://arxiv.org/abs/1701.03458*/

uvec Sto_IHT_async_iteration(vec &x_hat,const vec &tally,const mat &A,const vec &y,
 			const int sparsity, const vec prob_vec, const double gamma){
	// randomize
	const unsigned int num_block = prob_vec.n_elem;
	const unsigned int block_size = y.n_elem / num_block;
	const unsigned int selected_block = floor(randu()*num_block);
	const unsigned int first_ind_block = block_size*selected_block;
	const unsigned int last_ind_block = block_size*(selected_block+1)-1;

	// Proxy
	const mat A_block = A.rows(first_ind_block,last_ind_block);
	const vec y_block = y.subvec(first_ind_block,last_ind_block);
	const vec gradient = -2 * A_block.t() *	(y_block - A_block*x_hat);
	const vec b = x_hat - gradient * gamma/(num_block*prob_vec(selected_block));
	
	x_hat.zeros();  // this variable is local to each core. NOTE: passed by reference
	// Identify using b (local)
	uvec sorted_ind = sort_index(abs(b),"descend"); 
	const uvec est_supp_local = sorted_ind(span(0,sparsity - 1)); 
	// Estimate using b (local)		
	x_hat(est_supp_local) = b(est_supp_local);

	// Identify using tally (collective)
	sorted_ind = sort_index(tally,"descend"); 
	const uvec est_supp_collective = sorted_ind(span(0,sparsity - 1));  
	// Estimate using tally (collective)
	x_hat(est_supp_collective) = b(est_supp_collective);	

	return 	est_supp_local;
}

uvec Faulty_Sto_IHT_async_iteration(vec &x_hat,	const int sparsity){
	x_hat.randn();
	uvec sorted_ind = sort_index(abs(x_hat),"descend");	
	return sorted_ind(span(0,sparsity - 1)); 
}

void update_tally(vec &tally,const uvec est_supp_local,const uvec prev_est_supp,const unsigned int iter_local){
	/* update the tally score according the rules in:
	An Asynchronous Parallel Approach to Sparse Recovery
	https://arxiv.org/abs/1701.03458*/
	tally(est_supp_local) += iter_local; 
	if (iter_local >= 2){
		tally(prev_est_supp) -= iter_local-1;
	}
	return;
}


void majority_voting(vec &tally,const uvec est_supp_local,const uvec prev_est_supp,const unsigned int iter_local){
	tally(est_supp_local) += 1;
	if (iter_local >= 2){
		tally(prev_est_supp) -= 1;
	}
	return;
}
/* Description: Parallel Stochastic Iterative Hard Thresholding (StoIHT) algorithm with tally score to approximate the vector x from measurements u = A*x.
Publication: An Asynchronous Parallel Approach to Sparse Recovery
https://arxiv.org/abs/1701.03458*/

vec tally_Sto_IHT(const mat &A, const vec &y, const int sparsity, const vec prob_vec,
		const unsigned int max_iter, const double gamma,const double tol, 
		unsigned int &num_iters, const simulation_parameters simulation_params, 	 			string voting_type){
	uvec faulty_cores;
	uvec slow_cores;
	faulty_n_slow_cores(faulty_cores, slow_cores, simulation_params);

	const unsigned int sig_dim = A.n_cols;

	// initialization of variables that are shared among cores
	vec tally(sig_dim,fill::zeros);		// vector of tally scores
	vec x_hat_total(sig_dim,fill::zeros);	// estimation of the signal
	bool done = false;			// flag to check the convergence criteria
	unsigned int i = 0;			// total number of iterations

	// parallel section of the code starts here
	#pragma omp parallel num_threads(simulation_params.num_cores)
	{
	//#pragma omp single // a single core executes the following line
	//cout << "Number of threads: " << omp_get_num_threads() << endl;

	// initializaiotn of variables that are local to each core
	uvec prev_est_supp;			// estimated support in previous iteration
	vec x_hat_local(sig_dim,fill::zeros);  	// this is local to each core
	unsigned int iter_local = 0;		// number of iteration for this core

	// iterations to find the solutions
	while(!done){
		// master thread uses the tally vector to check the convergence criteria
		if (omp_get_thread_num() == 0){	
			const uvec sorted_ind = sort_index(abs(tally),"descend");	
			const uvec est_supp = sorted_ind(span(0,sparsity - 1));
			const mat A_supp = A.cols(est_supp);
			x_hat_local.zeros();
			x_hat_local(est_supp) = solve(A_supp,y);
			if (norm (y - A*x_hat_local) < tol || i >= max_iter){
				x_hat_total = x_hat_local;
				done = true;
			}
			if (omp_get_num_threads()  > 1){
				continue;
			}
		}

		//slow cores sleep for  simulation_params.sleep_slow_cores microseconds
		if (any( slow_cores == omp_get_thread_num()) ){
			usleep(simulation_params.sleep_slow_cores);
		}

		i++;
		iter_local++;

		// update the local estimate of the support
		uvec est_supp_local;
		if (any( faulty_cores == omp_get_thread_num())  ){
			est_supp_local = Faulty_Sto_IHT_async_iteration(x_hat_local, sparsity);
		}else{
			est_supp_local = Sto_IHT_async_iteration(x_hat_local, tally, A, y, 	
				sparsity, prob_vec, gamma);
		}
		// Update Tally
		if (voting_type == "iteration number"){
			update_tally(tally,est_supp_local,prev_est_supp,iter_local);
		}
		if (voting_type == "majority voting"){
			majority_voting(tally,est_supp_local,prev_est_supp,iter_local);
		}

		prev_est_supp = est_supp_local;		
	}
	}
	// parallel section of the code ends here
	
	//cout << "#iterations = " << i << endl;
	num_iters = i;
	//omp_destroy_lock(&lock);
	return x_hat_total;
}

