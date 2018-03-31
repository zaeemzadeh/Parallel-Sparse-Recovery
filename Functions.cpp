#include <iostream>
#include <iomanip>   // setpercision
#include <armadillo>
#include <vector>
#include <sys/stat.h>	//mkdir
#include <sys/types.h>	//mkdir
#include <string>
#include "Sto_IHT.h"
#include "Functions.h"
#include "Bayes_Sto_IHT.h"
#include "Parallel_AMP.h"

using namespace std;
using namespace arma;


// ---------------------------------------------------
// ---------------------------------------------------
// ----------- function: calculate metrics -----------
// ---------------------------------------------------
performance_metrics calculate_metrics(const vector<trial_info> MC_runs, const unsigned int max_iter){
	performance_metrics metrics;
	if (MC_runs.size() ==0 ){
		return metrics;
	}
		
	// calculate means
	metrics.time_avg = 0;
	metrics.iter_avg = 0;
	metrics.time_per_iter_avg = 0;
	metrics.success_avg = 0;
	for (unsigned int mc = 0; mc < MC_runs.size(); mc++){
		if (MC_runs[mc].time == -1){
			continue;	
		}
		if (MC_runs[mc].iterations < max_iter){
			metrics.time_avg += MC_runs[mc].time;
			metrics.iter_avg += MC_runs[mc].iterations;	
			metrics.time_per_iter_avg += MC_runs[mc].time/MC_runs[mc].iterations;	
		}
		metrics.success_avg += double(MC_runs[mc].iterations < max_iter);
	}
	metrics.time_avg    = metrics.time_avg/MC_runs.size();
	metrics.iter_avg    = metrics.iter_avg/MC_runs.size();
	metrics.success_avg = metrics.success_avg/MC_runs.size();
	metrics.time_per_iter_avg    = metrics.time_per_iter_avg/MC_runs.size();

	// calculate standard deviations	
	metrics.time_std = 0;
	metrics.iter_std = 0;
	metrics.success_std = 0;
	metrics.time_per_iter_std = 0;
	for (unsigned int mc = 0; mc < MC_runs.size(); mc++){
		if (MC_runs[mc].time == -1){
			continue;	
		}
		if (MC_runs[mc].iterations < max_iter){
			metrics.time_std += pow(MC_runs[mc].time,2);
			metrics.iter_std += pow(MC_runs[mc].iterations,2) ;
			metrics.time_per_iter_std += pow(MC_runs[mc].time/MC_runs[mc].iterations,2) ;
		}		
		metrics.success_std += pow(double(MC_runs[mc].iterations < max_iter),2);
	}
	metrics.time_std    = metrics.time_std/MC_runs.size();
	metrics.iter_std    = metrics.iter_std/MC_runs.size();
	metrics.success_std = metrics.success_std/MC_runs.size();
	metrics.time_per_iter_std    = metrics.time_per_iter_std/MC_runs.size();

	metrics.time_std    -= pow(metrics.time_avg,2);
	metrics.iter_std    -= pow(metrics.iter_avg,2);
	metrics.success_std -= pow(metrics.success_avg,2);
	metrics.time_per_iter_std    -= pow(metrics.time_per_iter_avg,2);

	metrics.time_std = sqrt(metrics.time_std);
	metrics.iter_std = sqrt(metrics.iter_std);
	metrics.time_per_iter_std = sqrt(metrics.time_per_iter_std);
	metrics.success_std = sqrt(metrics.success_std);
	return metrics;
}

// ---------------------------------------------------
// ---------------------------------------------------
// ----------- function: run_mc_trials ---------------
// ---------------------------------------------------
vector<performance_metrics> run_mc_trials(const unsigned int sig_dim, const unsigned int sparsity, const unsigned int meas_num, const unsigned int max_iter, const double gamma, const double tol , const unsigned int block_size, const unsigned int num_block, const vec prob_vec, simulation_parameters simulation_params, const int num_mc_runs, const int SEED){

	vector<performance_metrics> MC_metrics;	

	simulation_parameters simulation_params_non_parallel = simulation_params;
	simulation_params_non_parallel.num_cores = 1;
	
	// performance metrics for different methods in MC trials
	vector <trial_info> Sto_IHT_MC  	;				
	vector <trial_info> Parallel_Sto_IHT_MC ;		
	vector <trial_info> Tally_Sto_IHT_MC 	;			
	vector <trial_info> Bayes_Sto_IHT_MC 	;			
	vector <trial_info> Majority_Sto_IHT_MC	;			
	vector <trial_info> AMP_MC		;			
	vector <trial_info> R_MP_AMP_MC		;			
	vector <trial_info> Async_MP_AMP_MC	;	
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
		const mat A =normalise( randn(meas_num,sig_dim) );

		// compute measurements
		const vec y = A*x;
	
		double time;
		unsigned int num_iters;

		// Solve with Async_MP_AMP
		/*time = omp_get_wtime();
		const vec x_hat_Async_MP_AMP = Async_MP_AMP(A, y, sparsity, max_iter, tol, num_iters,simulation_params, num_block);
		Async_MP_AMP_MC.push_back(trial_info(omp_get_wtime() - time,num_iters ));*/


		// Solve in Parallel with tally score
		time = omp_get_wtime();
		const vec x_hat_tally = tally_Sto_IHT(A, y, sparsity, prob_vec, max_iter, gamma, tol, num_iters, simulation_params, "iteration number");
		Tally_Sto_IHT_MC.push_back(trial_info(omp_get_wtime() - time,num_iters ));
		

		// Solve in Parallel with majority voting
		time = omp_get_wtime();
		const vec x_hat_majority = tally_Sto_IHT(A, y, sparsity, prob_vec, max_iter, gamma, tol, num_iters, simulation_params, "majority voting");
		Majority_Sto_IHT_MC.push_back(trial_info(omp_get_wtime() - time,num_iters ));


		// Solve in Parallel with tally score with Bayesian update rules
		time = omp_get_wtime();
		const vec x_hat_bayesian = bayesian_Sto_IHT(A, y, sparsity, prob_vec, max_iter, gamma, tol, num_iters, simulation_params);
		Bayes_Sto_IHT_MC.push_back(trial_info(omp_get_wtime() - time,num_iters ));

		// Solve with R_MP_AMP
		time = omp_get_wtime();
		const vec x_hat_R_MP_AMP = R_MP_AMP(A, y, sparsity, max_iter, tol, num_iters,simulation_params);
		R_MP_AMP_MC.push_back(trial_info(omp_get_wtime() - time,num_iters ));

		// Solve with AMP
		time = omp_get_wtime();
		const vec x_hat_AMP = AMP(A, y, sparsity, max_iter, tol, num_iters,simulation_params);
		AMP_MC.push_back(trial_info(omp_get_wtime() - time,num_iters ));
/*

		// Solve with Parallel StoIHT
		time = omp_get_wtime();
		const vec x_hat_parallel = parallel_Sto_IHT(A, y, sparsity, prob_vec, max_iter, gamma, tol, num_iters, simulation_params);
		Parallel_Sto_IHT_MC.push_back(trial_info(omp_get_wtime() - time,num_iters ));
*/

		// solve with non-parallel StoIHT
		time = omp_get_wtime();
		const vec x_hat = parallel_Sto_IHT(A, y, sparsity, prob_vec, max_iter, gamma, tol, num_iters, simulation_params_non_parallel);
		Sto_IHT_MC.push_back(trial_info(omp_get_wtime() - time,num_iters ));
		
	}


	MC_metrics.push_back(calculate_metrics(Sto_IHT_MC,max_iter));

	MC_metrics.push_back(calculate_metrics(Parallel_Sto_IHT_MC,max_iter));

	MC_metrics.push_back(calculate_metrics(Tally_Sto_IHT_MC,max_iter));

	MC_metrics.push_back(calculate_metrics(Bayes_Sto_IHT_MC,max_iter));

	MC_metrics.push_back(calculate_metrics(Majority_Sto_IHT_MC,max_iter));

	MC_metrics.push_back(calculate_metrics(AMP_MC,max_iter));
			
	MC_metrics.push_back(calculate_metrics(R_MP_AMP_MC,max_iter));

	MC_metrics.push_back(calculate_metrics(Async_MP_AMP_MC,max_iter));
	return MC_metrics;	
}


// ---------------------------------------------------
// ---------------------------------------------------
// ----------- function: faulty_n_slow_cores ---------
// ---------------------------------------------------
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


// ---------------------------------------------------
// ---------------------------------------------------
// ----------- function: save_results ----------------
// ---------------------------------------------------
void save_results(const vector<string> alg_names, const vector <vector<performance_metrics>> sweep_metrics,
	const string parameter_to_sweep, const vec parameter_to_sweep_values,
	const unsigned int sig_dim, const unsigned int sparsity, const unsigned int meas_num, 
	const unsigned int max_iter, const unsigned int num_block, simulation_parameters simulation_params, const int num_mc_runs){
	// create directory
	time_t rawtime;
	struct tm * timeinfo;
	time ( &rawtime );
	timeinfo = localtime ( &rawtime );

	string dir = "Results/" + parameter_to_sweep + " " + string(asctime (timeinfo));
	mkdir("Results/", 0777);
	mkdir(&dir[0],0777);


	ofstream ofs;
	ofs.open (dir + "/simulation_env.txt", std::ofstream::out);
	ofs << "# of Processors Available: \t" << omp_get_num_procs() << endl << endl;
	ofs << "# of MC trials: \t" << num_mc_runs << endl << endl;
	ofs << "Signal Dimension: \t" << sig_dim << endl<<endl;
	ofs << "Sparsity: \t\t" << sparsity << endl<<endl;
	ofs << "# of measurements: \t" << meas_num << endl<<endl;
	ofs << "Max. # of Iters: \t" << max_iter << endl<<endl;
	ofs << "# of blocks: \t" << num_block << endl<<endl;
	ofs << "# of Faulty Cores: \t" << simulation_params.num_faulty_cores << endl<<endl;
	ofs << "# of Slow Cores: \t" << simulation_params.num_slow_cores;
	ofs << " (each sleeping for " << simulation_params.sleep_slow_cores/1000 << " ms )" << endl<<endl;
	ofs << "Parameter to Sweep: " << parameter_to_sweep << ':' << endl;
	ofs << parameter_to_sweep_values.t() << endl;

	ofs.close();

	char delimiter = '\t';
	for (unsigned int i = 0; i < alg_names.size(); i++){
		ofs.open (dir + "/" + alg_names[i] + ".csv", std::ofstream::out);
		ofs << " time_avg;time_std;success_avg;success_std;time_per_iter_avg;time_per_iter_std;iter_avg;iter_std" << endl;
		for (unsigned int param = 0; param <parameter_to_sweep_values.n_elem; param++ ){
			ofs 	<< 1000* sweep_metrics[param][i].time_avg 	<< delimiter 
				<< 1000* sweep_metrics[param][i].time_std 	<< delimiter
				<< sweep_metrics[param][i].success_avg 		<< delimiter
				<< sweep_metrics[param][i].success_std 		<< delimiter 
				<< 1000* sweep_metrics[param][i].time_per_iter_avg  << delimiter
				<< 1000* sweep_metrics[param][i].time_per_iter_std  << delimiter
				<< sweep_metrics[param][i].iter_avg  		<< delimiter 
				<< sweep_metrics[param][i].iter_std 		<< delimiter
				<< endl ;

		}
		ofs.close();
		
	}
}



// ---------------------------------------------------
// ---------------------------------------------------
// ----------- function: print_results----------------
// ---------------------------------------------------
void print_results(const vector<string> alg_names, const vector <vector<performance_metrics>> sweep_metrics,
	const string parameter_to_sweep, const vec parameter_to_sweep_values,
	const unsigned int sig_dim, const unsigned int sparsity, const unsigned int meas_num, 
	const unsigned int max_iter, const unsigned int num_block, simulation_parameters simulation_params, const int num_mc_runs){
	cout << setprecision(5);
	string delimiter = "\t\t";
	cout << endl;
	for (unsigned int i = 0; i < alg_names.size(); i++){
		cout << alg_names[i] << ":" << endl;
		cout << "time_avg" << '\t' << "success_avg" << '\t' << "time/iter_avg" 
			<< '\t' << "iter_avg" << endl;
		for (unsigned int param = 0; param <parameter_to_sweep_values.n_elem; param++ ){
			cout 	<< 1000* sweep_metrics[param][i].time_avg 	<< delimiter 
				<< sweep_metrics[param][i].success_avg 		<< delimiter
				<< 1000* sweep_metrics[param][i].time_per_iter_avg  << delimiter
				<< sweep_metrics[param][i].iter_avg  		<< delimiter 
				<< endl ;

		}
		cout << endl;
		
	}
	cout << endl;
	cout << "NOTE: negative values (if any) mean the corresponding algorithm was not executed" << endl;
	cout << "(uncomment them in Functions.cpp)" << endl;
	cout << "NOTE: Zero (if any) means the success rate of the corresponding algorithm is zero." << endl;
	cout << "(increase maxIter, decrease sparsity, increase MC trials, ....)" << endl;
}


// ---------------------------------------------------
// ---------------------------------------------------
// ----------- function: run_experiments--------------
// ---------------------------------------------------
void run_experiments(const vector<experiment> experiments, const vector <string> alg_names, unsigned int sig_dim, 
	unsigned int sparsity, unsigned int meas_num, const unsigned int max_iter, const double gamma, const double tol ,
	const unsigned int block_size, unsigned int num_block, const vec prob_vec, simulation_parameters simulation_params,
	 const int num_mc_runs, const int SEED){
	


	cout << setprecision(3);
	for (unsigned int exp = 0; exp < experiments.size(); exp++){	
		string parameter_to_sweep = experiments[exp].parameter_to_sweep;
		vec parameter_to_sweep_values = experiments[exp].parameter_to_sweep_values;	
		vector <vector<performance_metrics>> sweep_metrics(parameter_to_sweep_values.n_elem);
		cout << "Experiment #" << exp+1 << '/' << experiments.size();
		cout << " Parameter to Sweep: " << parameter_to_sweep << ':' << endl;

		for (unsigned int param = 0; param <parameter_to_sweep_values.n_elem; param++ ){
			if (parameter_to_sweep == "Cores"){
				simulation_params.num_cores = int(parameter_to_sweep_values[param]);
				num_block = simulation_params.num_cores; //FIXME: not necessarily
				simulation_params.num_slow_cores = (2 * simulation_params.num_cores) / 10; // FIXME: This should be set in main, not here
				cout << simulation_params.num_cores << '\t' << flush;
			}else if (parameter_to_sweep == "Sparsity"){
				sparsity = parameter_to_sweep_values[param]*sig_dim/100.0;
				cout << sparsity << '\t' << flush;
			}else if (parameter_to_sweep == "Measurements"){
				meas_num = parameter_to_sweep_values[param]*sig_dim/100.0;
				cout << meas_num << '\t' << flush;
			}else if (parameter_to_sweep == "Slow Cores"){
				simulation_params.num_slow_cores = parameter_to_sweep_values[param];
				cout << simulation_params.num_slow_cores << '\t' << flush;
			}else if (parameter_to_sweep == "Sleep Time"){
				simulation_params.sleep_slow_cores = parameter_to_sweep_values[param];
				cout << simulation_params.sleep_slow_cores << '\t' << flush;
			}else if (parameter_to_sweep == "Blocks"){
				num_block = parameter_to_sweep_values[param];
				cout << num_block << '\t' << flush;
			}else if (parameter_to_sweep == "Signal Dimension"){
				sig_dim = parameter_to_sweep_values[param];
				sparsity = 2*sig_dim/100;
				meas_num = 30*sig_dim/100;
				cout << sig_dim << '\t' << flush;
			}	
			sweep_metrics[param] =  run_mc_trials( sig_dim, sparsity, meas_num, max_iter, gamma, tol , block_size, num_block, prob_vec, simulation_params,  num_mc_runs,  SEED);
		}
		cout << endl;
		// Print/Save
		print_results(alg_names, sweep_metrics, parameter_to_sweep, parameter_to_sweep_values, sig_dim, sparsity,  meas_num, max_iter, num_block, simulation_params, num_mc_runs);
		save_results(alg_names, sweep_metrics, parameter_to_sweep, parameter_to_sweep_values, sig_dim, sparsity,  meas_num, max_iter, num_block, simulation_params, num_mc_runs);
		
	}

	return;
}

