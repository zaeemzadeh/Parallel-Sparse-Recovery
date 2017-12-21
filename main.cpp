#include <iostream>
#include <string>	// for stoi (argv)
#include <armadillo>
#include <algorithm> 	// shuffle
#include <stdlib.h>     // srand, rand
#include <time.h>     	// time
#include <vector>
#include <boost/math/special_functions/digamma.hpp>
//#include <math.h>       /* pow sqrt*/

#include "Functions.h"

using namespace arma;
using namespace std;

int main(int argc, char* argv[]){
	// argv[1]  : seed for random generator  (set to -1 for random seed)
	// argv[2]  : number of threads requested
	// argv[3]  : maximum number of MC trials
	// initializing random number generator
	simulation_parameters simulation_params;
	simulation_params.num_cores = stoi(argv[2],nullptr,10);
	cout << "Number of Processors Available: \t" << omp_get_num_procs() << endl << endl;

	const int num_mc_runs = stoi(argv[3],nullptr,10);// number of MC trials
	cout << "Number of MC trials: \t" << num_mc_runs << endl << endl;
	const int SEED = stoi(argv[1],nullptr,10);		
	
	// signal Parameters
	const unsigned int sig_dim 	= 3e3;			// signal dimension
	const unsigned int sparsity	= 5*sig_dim/100;	// sparsity level of signal
	const unsigned int meas_num	= 30*sig_dim/100;	// number of measurements
	cout << "Signal Dimension: \t" << sig_dim << endl<<endl;

	// algorithm parameters
	const unsigned int max_iter = 1e4;
	const double gamma = 1e0;
	const double tol = 1e-7;
	const int unsigned block_size = fmin(meas_num,sparsity);	
	const int unsigned num_block = meas_num / block_size;	// number of blocks
	// set probabilities of selecting each block
	const vec prob_vec = ones(num_block)/num_block;
	simulation_params.num_faulty_cores = 0;
	cout << "Number of Faulty Cores: \t" << simulation_params.num_faulty_cores << endl<<endl;
	simulation_params.num_slow_cores = 3;
	simulation_params.sleep_slow_cores = 2e3;   // microseconds to sleep
	cout << "Number of Slow Cores: \t" << simulation_params.num_slow_cores << endl<<endl;

	vector<string> alg_names;
	alg_names.push_back("Sto_IHT");
	alg_names.push_back("Parallel Sto_IHT");
	alg_names.push_back("Tally Sto_IHT");
	alg_names.push_back("Bayes Sto_IHT");
	alg_names.push_back("Majority Voting Sto_IHT");
	alg_names.push_back("AMP");
	alg_names.push_back("R_MP_AMP");
	alg_names.push_back("Async_MP_AMP");

	string parameter_to_sweep = "Cores";
	vec parameter_to_sweep_values = linspace(12,12,1);	
	vector <vector<performance_metrics>> sweep_metrics(parameter_to_sweep_values.n_elem);

	cout << setprecision(3);
	cout << "Parameter to Sweep: " << parameter_to_sweep << ':' << endl;

	for (unsigned int param = 0; param <parameter_to_sweep_values.n_elem; param++ ){
		if (parameter_to_sweep == "Cores"){
			simulation_params.num_cores = int(parameter_to_sweep_values[param]);
			cout << simulation_params.num_cores << '\t' << flush;
		}
		
		sweep_metrics[param] =  run_mc_trials( sig_dim, sparsity, meas_num, max_iter, gamma, tol , block_size, num_block, prob_vec, simulation_params,  num_mc_runs,  SEED);

		/*for (unsigned int i = 0; i < alg_names.size(); i++){
			cout << alg_names[i] << " Sto_IHT Metrics: " << endl;
			cout << "Success: \t" << sweep_metrics[param][i].success_avg* 100 ; 
			cout << "\t +/- " << sweep_metrics[param][i].success_std* 100  << " % " << endl;
			cout << "Time: \t\t" << sweep_metrics[param][i].time_avg; 
			cout << "\t +/- " << sweep_metrics[param][i].time_std << " s" << endl; 
			cout << "Iterations: \t" << sweep_metrics[param][i].iter_avg ; 
			cout << "\t +/- " << sweep_metrics[param][i].iter_std << endl<<endl; 
		}*/
	}
	// Print
	cout << endl;
	for (unsigned int i = 0; i < alg_names.size(); i++){
		cout << alg_names[i] << " Metrics: " << endl;
		for (unsigned int param = 0; param <parameter_to_sweep_values.n_elem; param++ ){
			cout << int(sweep_metrics[param][i].time_avg * 1000)
			<< '(' << sweep_metrics[param][i].success_avg << ',' 
			<< sweep_metrics[param][i].iter_avg  << ')' << '\t' ;

		}
		cout << endl;
	}

	return 1;
}


