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
vector<performance_metrics> run_mc_trials(const vector <string> alg_names, const unsigned int sig_dim, const unsigned int sparsity, const unsigned int meas_num, const unsigned int max_iter, const double gamma, const double tol , const vec prob_vec, simulation_parameters simulation_params, const int num_mc_runs, const int SEED){

	vector<performance_metrics> MC_metrics;	

	simulation_parameters simulation_params_non_parallel = simulation_params;
	simulation_params_non_parallel.num_cores = 1;
	
	// performance metrics for different methods in MC trials	
    vector <vector<trial_info>> alg_infos(alg_names.size());	
    
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

		// Generate Measurement matrix
		const mat A =normalise( randn(meas_num,sig_dim) );

		// compute measurements
		const vec y = A*x;
	
		double time;
		unsigned int num_iters;
        
        for (unsigned int alg = 0; alg < alg_names.size(); alg++) {
            if(alg_names[alg].compare("Bayesian Sto_IHT") == 0){
                // Solve in Parallel with Bayesian update rules
                time = omp_get_wtime();
                const vec x_hat_bayesian = bayesian_Sto_IHT(A, y, sparsity, prob_vec, max_iter, gamma, tol, num_iters, simulation_params);
                alg_infos[alg].push_back(trial_info(omp_get_wtime() - time,num_iters ));
            }
            else if(alg_names[alg].compare("Sto_IHT") == 0){
                // solve with non-parallel StoIHT
                time = omp_get_wtime();
                const vec x_hat = parallel_Sto_IHT(A, y, sparsity, prob_vec, max_iter, gamma, tol, num_iters, simulation_params_non_parallel);
                alg_infos[alg].push_back(trial_info(omp_get_wtime() - time,num_iters ));     
            }
            else if(alg_names[alg].compare("Parallel Sto_IHT") == 0){                    
                // Solve with Parallel StoIHT
                time = omp_get_wtime();
                const vec x_hat_parallel = parallel_Sto_IHT(A, y, sparsity, prob_vec, max_iter, gamma, tol, num_iters, simulation_params);
                alg_infos[alg].push_back(trial_info(omp_get_wtime() - time,num_iters ));
            }
            else if(alg_names[alg].compare("Tally Sto_IHT") == 0){
                // Solve in Parallel with tally score
                time = omp_get_wtime();
                const vec x_hat_tally = tally_Sto_IHT(A, y, sparsity, prob_vec, max_iter, gamma, tol, num_iters, simulation_params);
                alg_infos[alg].push_back(trial_info(omp_get_wtime() - time,num_iters ));     
            }
            else if(alg_names[alg].compare("AMP") == 0){
                // Solve with AMP
                time = omp_get_wtime();
                const vec x_hat_AMP = AMP(A, y, sparsity, max_iter, tol, num_iters,simulation_params);
                alg_infos[alg].push_back(trial_info(omp_get_wtime() - time,num_iters ));     
            }
            else if(alg_names[alg].compare("Parallel AMP") == 0){
                // Solve with R_MP_AMP
                time = omp_get_wtime();
                const vec x_hat_R_MP_AMP = R_MP_AMP(A, y, sparsity, max_iter, tol, num_iters,simulation_params);
                alg_infos[alg].push_back(trial_info(omp_get_wtime() - time,num_iters ));  
            }
            else{
                cout << "Unknown Algorithm name: " << alg_names[alg] << endl << endl;
                throw;
            }
            
        }
	}

    for (unsigned int alg = 0; alg < alg_names.size(); alg++) {
        MC_metrics.push_back(calculate_metrics(alg_infos[alg],max_iter));
    }
	return MC_metrics;	
}


// ---------------------------------------------------
// ---------------------------------------------------
// ----------- function: set_slow_cores ---------
// ---------------------------------------------------
// NOTE: Thread #0 is never selected to be slow
// Because it is already performing extra calculation to check the convergence criteria
// This way it is easier to interpret the results
void set_slow_cores(uvec &slow_cores, const simulation_parameters simulation_params){
	vec cores(simulation_params.num_cores-1,fill::zeros);
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
	const unsigned int max_iter, simulation_parameters simulation_params, const int num_mc_runs){
	// create directory
	string dir = "../results";
	mkdir(dir.c_str(),0777);


	ofstream ofs;
	ofs.open (dir + "/simulation_env.txt", std::ofstream::out);
	ofs << "# of Processors Available: \t" << omp_get_num_procs() << endl << endl;
	ofs << "# of MC trials: \t" << num_mc_runs << endl << endl;
	ofs << "Signal Dimension: \t" << sig_dim << endl<<endl;
	ofs << "Sparsity: \t\t" << sparsity << endl<<endl;
	ofs << "# of measurements: \t" << meas_num << endl<<endl;
	ofs << "Max. # of Iters: \t" << max_iter << endl<<endl;
	ofs << "Percentage of Slow Cores: \t" << simulation_params.slow_cores_ratio * 100;
	ofs << " (each sleeping for " << simulation_params.sleep_slow_cores/1000 << " ms )" << endl<<endl;
	ofs << "Parameter to Sweep: " << parameter_to_sweep << ':' << endl;
	ofs << parameter_to_sweep_values.t() << endl;

	ofs.close();

	char delimiter = '\t';
	for (unsigned int i = 0; i < alg_names.size(); i++){        
		if (sweep_metrics[0][i].time_avg < 0){
			continue;
		}
		ofs.open (dir + "/" + alg_names[i] + ".txt", std::ofstream::out);
		ofs << parameter_to_sweep << "\t time_avg\ttime_std\tsuccess_avg\tsuccess_std\ttime_per_iter_avg\ttime_per_iter_std\titer_avg\titer_std" << endl;
		for (unsigned int param = 0; param <parameter_to_sweep_values.n_elem; param++ ){
			ofs << parameter_to_sweep_values[param]         << delimiter
                << 1000* sweep_metrics[param][i].time_avg 	<< delimiter 
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
	const unsigned int max_iter, simulation_parameters simulation_params, const int num_mc_runs){
	cout << setprecision(5);
	string delimiter = "\t\t";
	cout << endl;
    cout << parameter_to_sweep << delimiter << "time_avg" << delimiter << "success_avg" 
        << delimiter << "time/iter_avg"  << delimiter<< "iter_avg" << endl;
	for (unsigned int i = 0; i < alg_names.size(); i++){
		if (sweep_metrics[0][i].time_avg < 0){
			continue;
		}
		cout << alg_names[i] << ":" << endl;
		for (unsigned int param = 0; param <parameter_to_sweep_values.n_elem; param++ ){
			cout<< parameter_to_sweep_values[param]         << delimiter
                << 1000* sweep_metrics[param][i].time_avg 	<< delimiter 
				<< sweep_metrics[param][i].success_avg 		<< delimiter
				<< 1000* sweep_metrics[param][i].time_per_iter_avg  << delimiter
				<< sweep_metrics[param][i].iter_avg  		<< delimiter 
				<< endl ;

		}
		cout << endl;
		
	}
	cout << endl;
	cout << "NOTE: Zero (if any) means the success rate of the corresponding algorithm is zero." << endl;
	cout << "(increase maxIter, decrease sparsity, increase MC trials, ....)" << endl;
}


// ---------------------------------------------------
// ---------------------------------------------------
// ----------- function: run_experiments--------------
// ---------------------------------------------------
void run_experiments(const vector <string> alg_names, unsigned int sig_dim, unsigned int sparsity, unsigned int meas_num, const unsigned int max_iter, const double gamma, const double tol , const vec prob_vec, simulation_parameters simulation_params,
	 const int num_mc_runs, const int SEED){
	cout << setprecision(3);
    string parameter_to_sweep = "Cores";
    cout << " Parameter to Sweep: " << parameter_to_sweep << ':' << endl;
    
    vec parameter_to_sweep_values = floor(linspace(1,10,5));	
    
    vector <vector<performance_metrics>> sweep_metrics(parameter_to_sweep_values.n_elem);

    for (unsigned int param = 0; param <parameter_to_sweep_values.n_elem; param++ ){
        simulation_params.num_cores = int(parameter_to_sweep_values[param]);
        simulation_params.num_slow_cores = int(simulation_params.slow_cores_ratio * simulation_params.num_cores); 
        cout << simulation_params.num_cores << '\t' << flush;
        sweep_metrics[param] =  run_mc_trials(alg_names, sig_dim, sparsity, meas_num, max_iter, gamma, tol , prob_vec, simulation_params,  num_mc_runs,  SEED);
    }
    cout << endl;
    // Print/Save
    print_results(alg_names, sweep_metrics, parameter_to_sweep, parameter_to_sweep_values, sig_dim, sparsity,  meas_num, max_iter, simulation_params, num_mc_runs);
    save_results(alg_names, sweep_metrics, parameter_to_sweep, parameter_to_sweep_values, sig_dim, sparsity,  meas_num, max_iter, simulation_params, num_mc_runs);

	return;
}

