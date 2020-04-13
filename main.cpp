#include <iostream>
#include <string>	// for stoi (argv)
#include <armadillo>
#include <algorithm> 	// shuffle
#include <stdlib.h>     // srand, rand
#include <time.h>     	// time
#include <vector>

#include "Functions.h"

using namespace arma;
using namespace std;

int main(int argc, char* argv[]){
	// argv[1]  : seed for random generator  (set to -1 for random seed)
	// argv[2]  : maximum number of MC trials

	// disable the dynamic adjustment of the number of threads within a team. 
	omp_set_dynamic(false);		
	// disable nested parallel regions, i.e., whether team members are allowed to create new teams.
	omp_set_nested(false);
	omp_set_num_threads(1);

	cout << endl;
	// initializing random number generator
	simulation_parameters simulation_params;
	cout << "# of Processors Available: \t" << omp_get_num_procs() << endl << endl;

	const int num_mc_runs = stoi(argv[2],nullptr,10);// number of MC trials
	cout << "# of MC trials: \t" << num_mc_runs << endl << endl;
	const int SEED = stoi(argv[1],nullptr,10);		
	
	// signal Parameters
	unsigned int sig_dim 	= 1e4;			// signal dimension
	cout << "Signal Dimension: \t" << sig_dim << endl<<endl;
	unsigned int sparsity	= 2*sig_dim/100;	// sparsity level of signal
	cout << "Sparsity: \t\t" << sparsity << endl<<endl;
	unsigned int meas_num	= 30*sig_dim/100;	// number of measurements
	cout << "# of Measurements: \t" << meas_num << endl<<endl;

	// algorithm parameters
	const unsigned int max_iter = 1.5e3;
	const double gamma = 1e0;
	const double tol = 1e-7;
	const int unsigned block_size = fmin(meas_num,sparsity);	
	const vec prob_vec = normalise(ones(meas_num / block_size),1);		// set probabilities of selecting each block

    simulation_params.slow_cores_ratio = 0.2;
	simulation_params.sleep_slow_cores = 15e5;   // microseconds to sleep
	cout << "Percentage of Slow Cores: \t" << simulation_params.slow_cores_ratio * 100;
	cout << " (each sleeping for " << simulation_params.sleep_slow_cores/1000 << " ms )" << endl<<endl;

	vector<string> alg_names;
    // Algorithms to test, comment out each line to skip running its corresponding algorithm 
	alg_names.push_back("Bayesian Sto_IHT");
	alg_names.push_back("Sto_IHT");
	alg_names.push_back("Parallel Sto_IHT");
	alg_names.push_back("Tally Sto_IHT");
	alg_names.push_back("AMP");
	alg_names.push_back("Parallel AMP");	
	
	run_experiments(alg_names, sig_dim, sparsity, meas_num, max_iter, gamma, tol ,
		prob_vec, simulation_params, num_mc_runs, SEED);

	return 1;
}


