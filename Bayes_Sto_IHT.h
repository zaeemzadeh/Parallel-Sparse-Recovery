#ifndef BAYES_STO_IHT_H
#define BAYES_STO_IHT_H

/* Description: Parallel Stochastic Iterative Hard Thresholding (StoIHT) algorithm 
 to approximate the vector x from measurements u = A*x.
 A tally score is calculate to represent the probability of each coefficient
being in the support.
The tally score is calculated using Bayesian update rules

*/

vec expected_ln_beta_dist(const vec pos_count, const vec neg_count);

double expected_ln_beta_dist(const double pos_count, const double neg_count);

ivec generate_support(const uvec &curr_supp,const uvec &prev_supp,const int sig_dim);

void update_tally_bayesian(vec &pos_count, vec &neg_count, double &reliability_pos, 
	double &reliability_neg, vec &expected_u, const ivec &support_data, 
	const ivec &prev_support_data, const double P_rand, const unsigned int global_iters, 
	const unsigned int local_iters);

vec bayesian_Sto_IHT(const mat &A, const vec &y, const int sparsity, const vec prob_vec,
		const unsigned int max_iter, const double gamma,const double tol, 
		unsigned int &num_iters, const simulation_parameters simulation_params);
#endif /* BAYES_STO_IHT_H */
