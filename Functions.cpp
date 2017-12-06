#include <armadillo>
#include <vector>
#include "Sto_IHT.h"
#include "Functions.h"
#include "Bayes_Sto_IHT.h"

using namespace std;
using namespace arma;



performance_metrics calculate_metrics(const vector<trial_info> MC_runs, const unsigned int max_iter){
	performance_metrics metrics;
	// calculate means
	metrics.time_avg = 0;
	metrics.iter_avg = 0;
	metrics.success_avg = 0;
	for (unsigned int mc = 0; mc < MC_runs.size(); mc++){
		metrics.time_avg += MC_runs[mc].time;
		metrics.iter_avg += MC_runs[mc].iterations;		
		metrics.success_avg += double(MC_runs[mc].iterations < max_iter);
	}
	metrics.time_avg    = metrics.time_avg/MC_runs.size();
	metrics.iter_avg    = metrics.iter_avg/MC_runs.size();
	metrics.success_avg = metrics.success_avg/MC_runs.size();

	// calculate standard deviations	
	metrics.time_std = 0;
	metrics.iter_std = 0;
	metrics.success_std = 0;
	for (unsigned int mc = 0; mc < MC_runs.size(); mc++){
		metrics.time_std += pow(MC_runs[mc].time,2);
		metrics.iter_std += pow(MC_runs[mc].iterations,2) ;		
		metrics.success_std += pow(double(MC_runs[mc].iterations < max_iter),2);
	}
	metrics.time_std    = metrics.time_std/MC_runs.size();
	metrics.iter_std    = metrics.iter_std/MC_runs.size();
	metrics.success_std = metrics.success_std/MC_runs.size();

	metrics.time_std    -= pow(metrics.time_avg,2);
	metrics.iter_std    -= pow(metrics.iter_avg,2);
	metrics.success_std -= pow(metrics.success_avg,2);

	metrics.time_std = sqrt(metrics.time_std);
	metrics.iter_std = sqrt(metrics.iter_std);
	metrics.success_std = sqrt(metrics.success_std);
	return metrics;
}

vector<performance_metrics> run_mc_trials(const unsigned int sig_dim, const unsigned int sparsity, const unsigned int meas_num, const unsigned int max_iter, const double gamma, const double tol , const unsigned int block_size, const unsigned int num_block, const vec prob_vec, simulation_parameters simulation_params, const int num_mc_runs, const int SEED){

	vector<performance_metrics> MC_metrics;	
	
	// performance metrics for different methods in MC trials
	vector <trial_info> Sto_IHT_MC  	(num_mc_runs);				
	vector <trial_info> Parallel_Sto_IHT_MC (num_mc_runs);		
	vector <trial_info> Tally_Sto_IHT_MC 	(num_mc_runs);			
	vector <trial_info> Bayes_Sto_IHT_MC 	(num_mc_runs);			
	vector <trial_info> Majority_Sto_IHT_MC	(num_mc_runs);	
	for (int mc = 0; mc < num_mc_runs; mc++){
		if (SEED == -1){
			arma_rng::set_seed_random();
			srand(time(NULL));
		}else{
			arma_rng::set_seed(SEED);
			srand(SEED);
		}
		
		// Generate sparse signal
		vec support(sig_dim,fill::zeros);
		support(span(0,sparsity-1)).ones();
		random_shuffle(support.begin(),support.end());

		const vec x = support;				// Bernoulli signal
		//vec x = randn(sig_dim,1) % support * std;	// Gaussian

		// Generate Measurement matrix
		const mat A = randn(meas_num,sig_dim) / sqrt(meas_num);

		// compute measurements
		const vec y = A*x;
	
		double time;
		unsigned int num_iters;


		// Solve in Parallel with tally score with Bayesian update rules
		time = omp_get_wtime();
		const vec x_hat_bayesian = bayesian_Sto_IHT(A, y, sparsity, prob_vec, max_iter, gamma, tol, num_iters, simulation_params);
		Bayes_Sto_IHT_MC[mc].time = omp_get_wtime() - time;
		Bayes_Sto_IHT_MC[mc].iterations = num_iters;
		//cout << "Bayesian Tally Sto_IHT Error Norm: " << norm(x - x_hat_bayesian)/norm(x) << " in " << time << " s." << endl;
	
		// Solve in Parallel with tally score
		time = omp_get_wtime();
		const vec x_hat_tally = tally_Sto_IHT(A, y, sparsity, prob_vec, max_iter, gamma, tol, num_iters, simulation_params, "iteration number");
		Tally_Sto_IHT_MC[mc].time = omp_get_wtime() - time;
		Tally_Sto_IHT_MC[mc].iterations = num_iters;
		//cout << "Tally Sto_IHT Error Norm: " << norm(x - x_hat_tally)/norm(x) << " in " << time << " s." << endl<< endl;

		// Solve in Parallel with majority voting
		time = omp_get_wtime();
		const vec x_hat_majority = tally_Sto_IHT(A, y, sparsity, prob_vec, max_iter, gamma, tol, num_iters, simulation_params, "majority voting");
		Majority_Sto_IHT_MC[mc].time = omp_get_wtime() - time;
		Majority_Sto_IHT_MC[mc].iterations = num_iters;
		//cout << "Tally Sto_IHT Error Norm: " << norm(x - x_hat_tally)/norm(x) << " in " << time << " s." << endl<< endl;

		// Solve in Parallel
		/*time = omp_get_wtime();
		const vec x_hat_parallel = parallel_Sto_IHT(A, y, sparsity, prob_vec, max_iter, gamma, tol, num_iters, simulation_params);
		Parallel_Sto_IHT_MC[mc].time = omp_get_wtime() - time;
		Parallel_Sto_IHT_MC[mc].iterations = num_iters;
		//cout << "Parallel Sto_IHT Error Norm: " << norm(x - x_hat_parallel)/norm(x) << " in " << time << " s." << endl<< endl;



		// solve with non-parallel StoIHT
		simulation_parameters simulation_params_non_parallel = simulation_params;
		simulation_params_non_parallel.num_cores = 1;
		time = omp_get_wtime();
		//cout << endl << " stoiht ... " << flush;
		const vec x_hat = parallel_Sto_IHT(A, y, sparsity, prob_vec, max_iter, gamma, tol, num_iters, simulation_params_non_parallel);
		//cout << " Done!" << endl;
		Sto_IHT_MC[mc].time = omp_get_wtime() - time;
		Sto_IHT_MC[mc].iterations = num_iters;*/
		//cout << "Sto_IHT Error Norm: " << norm(x - x_hat)/norm(x) << " in " << time << " s."<< endl<< endl;
		
		//cout << "True Support: " << endl << find(support!=0).t() << endl << flush;
	}
	MC_metrics.push_back(calculate_metrics(Sto_IHT_MC,max_iter));

	MC_metrics.push_back(calculate_metrics(Parallel_Sto_IHT_MC,max_iter));

	MC_metrics.push_back(calculate_metrics(Tally_Sto_IHT_MC,max_iter));

	MC_metrics.push_back(calculate_metrics(Bayes_Sto_IHT_MC,max_iter));

	MC_metrics.push_back(calculate_metrics(Majority_Sto_IHT_MC,max_iter));
			
	return MC_metrics;	
}


void faulty_n_slow_cores(uvec &faulty_cores, uvec &slow_cores, const simulation_parameters simulation_params){
	vec cores(simulation_params.num_cores-1,fill::zeros);

	// generate faulty core indices
	faulty_cores.reset();
	if (simulation_params.num_faulty_cores > 0){
		cores(span(0,simulation_params.num_faulty_cores-1)).ones();
		random_shuffle(cores.begin(),cores.end());
		faulty_cores = find(cores!=0) + 1;
	}
	// generate slow core indices
	slow_cores.reset();
	if (simulation_params.num_slow_cores > 0){
		cores.zeros();
		cores(span(0,simulation_params.num_slow_cores-1)).ones();
		random_shuffle(cores.begin(),cores.end());
		slow_cores = find(cores!=0) + 1;
	}
	return;
}

