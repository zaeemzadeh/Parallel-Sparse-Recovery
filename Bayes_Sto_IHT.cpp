#include <stdlib.h>     // srand, rand
#include <unistd.h>	// sleep
#include <armadillo>
#include <boost/math/special_functions/digamma.hpp>

#include "Sto_IHT.h"
#include "Functions.h"

using namespace arma;
using namespace std;

vec expected_ln_beta_dist(const vec pos_count, const vec neg_count){
// The logarithm of the geometric mean GX of a distribution with random variable X is the arithmetic mean of ln(X)
//https://en.wikipedia.org/wiki/Beta_distribution#Geometric_mean
vec ln_GX(pos_count.n_elem,fill::zeros);
for (unsigned int n = 0; n < ln_GX.n_elem; n++){
	ln_GX(n) = boost::math::digamma(pos_count(n)) - boost::math::digamma(pos_count(n) + neg_count(n));

}
return ln_GX;  
}

double expected_ln_beta_dist(const double pos_count, const double neg_count){
// The logarithm of the geometric mean GX of a distribution with random variable X is the arithmetic mean of ln(X)
//https://en.wikipedia.org/wiki/Beta_distribution#Geometric_mean
double ln_GX = boost::math::digamma(pos_count) - boost::math::digamma(pos_count + neg_count);

return ln_GX;  
}

ivec generate_support(const uvec &curr_supp,const uvec &prev_supp,const int sig_dim){
	ivec supp_data(sig_dim,fill::ones);
	supp_data *= -1;

	uvec mask_old(sig_dim,fill::zeros);	// mask for the previous support
	mask_old(prev_supp) = ones<uvec>(prev_supp.n_elem);
	
	uvec mask_new(sig_dim,fill::zeros);	// mask for the current support
	mask_new(curr_supp) = ones<uvec>(curr_supp.n_elem);

	const uvec zero_indices = find(mask_old%(1 - mask_new));
	const uvec one_indices  = find(mask_new%(1 - mask_old)) ;
	supp_data.elem( zero_indices ).zeros(); 
	supp_data.elem( one_indices ).ones();

	return supp_data;
}


void update_tally_bayesian(vec &pos_count, vec &neg_count, double &reliability_pos, 
	double &reliability_neg, vec &expected_u, const ivec &support_data, 
	const ivec &prev_support_data, const double P_rand, const unsigned int global_iters, 
	const unsigned int local_iters){
	// initialization
	const unsigned int sig_dim = pos_count.n_elem;

	uvec old_zero_indices = find(prev_support_data ==0);	
	uvec old_one_indices = find(prev_support_data ==1);	
	vec  old_expected_u  = expected_u;

	//const double conv_err = 1e-7;
	//const unsigned int max_iter = 1;
	uvec zero_indices = find(support_data ==0);	
	uvec one_indices = find(support_data ==1);	
	const uvec obs_indices = find(support_data !=-1);


	// #### Update U  #####
	mat Ln_Q(sig_dim,2,fill::zeros);
	uvec U(1);
	// evaluate update rules for U = 0
	U.zeros();  
	//prior term
	Ln_Q(obs_indices,U).fill(expected_ln_beta_dist(reliability_neg,reliability_pos)) ; 

	//liklihood term 
	//Ln_Q(zero_indices,U) += log(1 - P_rand); 	
	//Ln_Q(one_indices,U)  += log(P_rand) ; 	
	Ln_Q(zero_indices,U) += expected_ln_beta_dist(pos_count(zero_indices),neg_count(zero_indices)); 	
	Ln_Q(one_indices,U)  += expected_ln_beta_dist(neg_count(one_indices),pos_count(one_indices));  

 
	// evaluate update rules for U = 1
	U.ones();
	// prior term
	Ln_Q(obs_indices,U).fill( expected_ln_beta_dist(reliability_pos,reliability_neg) );
	// liklihood term
	Ln_Q(zero_indices,U) += expected_ln_beta_dist(neg_count(zero_indices),pos_count(zero_indices)); 
	Ln_Q(one_indices,U) += expected_ln_beta_dist(pos_count(one_indices),neg_count(one_indices)); 

	mat Q = exp(Ln_Q);		// posterior mass function (not normalized)
	expected_u(obs_indices) = Q(obs_indices,U)/sum(Q.rows(obs_indices),1);
	//expected_u(obs_indices) = ( expected_u(obs_indices)  - min(expected_u(obs_indices))) / max(expected_u(obs_indices)  - min(expected_u(obs_indices)));
	//cout << omp_get_thread_num() << " u " << expected_u(obs_indices).t() << endl;
	//cout << omp_get_thread_num() << " q1 " << min(Q.col(1)) << " - " <<  max(Q.col(1))  << endl;
	//cout << omp_get_thread_num() << " q0 " << min(Q.col(0)) << " - " <<  max(Q.col(0))  << endl;

	// #### Update R  #####
	// using coefficient reliability
	// prior term (XXX: comment tu use the prevoius value as prior)
	reliability_pos = 1;
	reliability_neg = 1; 
	// liklihood term (coefficient reliability)
	reliability_pos += sum(expected_u(obs_indices));	
	reliability_neg += sum(1 - expected_u(obs_indices));
	//cout << omp_get_thread_num() << " r " << reliability_pos << endl;
	// liklihood term (number of iterations)
	//reliability_pos += local_iters;
	//reliability_neg += global_iters - local_iters;

	// #### Update Phi  #####	
	#pragma omp critical
	{
	pos_count(one_indices) += expected_u(one_indices);
	neg_count(zero_indices) += expected_u(zero_indices);
	if (any(prev_support_data!=-1)){
		pos_count(old_one_indices) -= old_expected_u(old_one_indices);
		neg_count(old_zero_indices) -= old_expected_u(old_zero_indices);
	}
	}
	

	return;
}




vec bayesian_Sto_IHT(const mat &A, const vec &y, const int sparsity, const vec prob_vec,
		const unsigned int max_iter, const double gamma,const double tol, 
		unsigned int &num_iters, const simulation_parameters simulation_params){
	uvec faulty_cores;
	uvec slow_cores;
	faulty_n_slow_cores(faulty_cores, slow_cores, simulation_params);
	//cout << " slow " << slow_cores.t() << endl;
	//cout <<" faulty " << faulty_cores.t() << endl;

	const unsigned int sig_dim = A.n_cols;
	// initialization of variables that are SHARED among cores
	uvec updated_indices;
	vec pos_count(sig_dim,fill::ones);	// positive count for tally score
	vec neg_count(sig_dim,fill::ones);	// negative count for tally score
	vec x_hat_total(sig_dim,fill::zeros);	// estimation of the signal
	bool done = false;			// flag to check the convergence criteria
	unsigned int i = 0;			// total number of iterations
	const double P_rand = sparsity/sig_dim;	//  probability of unreliable '1' measurement
	// parallel section of the code starts here
	#pragma omp parallel num_threads(simulation_params.num_cores)
	{
	//#pragma omp single // a single core executes the following line
	//{cout << "Number of threads: " << omp_get_num_threads() << endl;}

	// initializaiotn of variables that are LOCAL to each core
	uvec prev_est_supp;			// estimated support in previous iteration
	prev_est_supp.reset();
	uvec updated_indices;
	vec x_hat_local(sig_dim,fill::zeros);  	// this is local to each core
	unsigned int iter_local = 0;		// number of iteration for this core
	ivec support_data(sig_dim,fill::zeros);	// data on estimated support
	ivec prev_support_data(sig_dim);	// previous estimated support
	prev_support_data.fill(-1);
	double reliability_pos = 1;		// positive count for core reliability
	double reliability_neg = 1;		// negative count for core reliability
	vec expected_u(sig_dim);	// expected coeffcient reliability (vote)
	expected_u.fill(1);

	// iterations to find the solutions
	while(!done){
		vec tally= pos_count/ (pos_count + neg_count);
		// master thread uses the tally vector to check the convergence criteria
		if (omp_get_thread_num() == 0){
			const uvec sorted_ind = sort_index(abs(tally),"descend");
			const uvec est_supp = sorted_ind(span(0,sparsity - 1));
			const mat A_supp = A.cols(est_supp);
			x_hat_local.zeros();
			x_hat_local(est_supp) = solve(A_supp,y);
			//cout << i << ':' << norm (y - A*x_hat_local) << endl;
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
			sleep(simulation_params.sleep_slow_cores);
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

		// generate support data
		support_data.fill(-1);support_data.elem(est_supp_local).ones();
		//support_data = generate_support(est_supp_local,prev_est_supp,sig_dim);
	
		update_tally_bayesian(pos_count, neg_count, reliability_pos, reliability_neg, expected_u, support_data, prev_support_data, P_rand, i , iter_local );		

	//	tally(est_supp_local) += 1;
		//if (iter_local > 1)
		//	tally(prev_est_supp) -= 1;
		//tally(updated_indices) = exp(expected_ln_beta_dist(pos_count(updated_indices),neg_count(updated_indices)));

		prev_support_data = support_data;

		prev_est_supp = est_supp_local;
	}
	//#pragma omp critical
	//{
	//cout << omp_get_thread_num() << " r " << reliability_pos << '\t' << reliability_neg << endl;
	//cout << "tally: " << sort_index(abs(tally),"descend").t() << endl;
	//cout << omp_get_thread_num() << " p " << pos_count.t() << endl;
	//cout << omp_get_thread_num() << " n " << neg_count.t() << endl;
	//cout << omp_get_thread_num() << " : " << tally.t() << endl << flush;
	//}
	}
	// parallel section of the code ends here

	num_iters = i;
	//cout << "#iterations = " << i << endl;
	return x_hat_total;
}

