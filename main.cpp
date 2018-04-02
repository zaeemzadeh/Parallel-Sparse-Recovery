#include <iostream>
#include <string>	// for stoi (argv)
#include <armadillo>
#include <algorithm> 	// shuffle
#include <stdlib.h>     // srand, rand
#include <time.h>     	// time
#include <vector>
//#include <math.h>       /* pow sqrt*/

#include "Functions.h"

using namespace arma;
using namespace std;



int main(int argc, char* argv[]){
	// argv[1]  : seed for random generator  (set to -1 for random seed)
	// argv[2]  : number of threads requested
	// argv[3]  : maximum number of MC trials

	// disable the dynamic adjustment of the number of threads within a team. 
	omp_set_dynamic(false);		
	// disable nested parallel regions, i.e., whether team members are allowed to create new teams.
	omp_set_nested(false);
	omp_set_num_threads(1);

	cout << endl;
	// initializing random number generator
	simulation_parameters simulation_params;
	simulation_params.num_cores = stoi(argv[2],nullptr,10);
	cout << "# of Processors Available: \t" << omp_get_num_procs() << endl << endl;

	const int num_mc_runs = stoi(argv[3],nullptr,10);// number of MC trials
	cout << "# of MC trials: \t" << num_mc_runs << endl << endl;
	const int SEED = stoi(argv[1],nullptr,10);		
	
	// signal Parameters
	unsigned int sig_dim 	= 1e4;			// signal dimension
	cout << "Signal Dimension: \t" << sig_dim << endl<<endl;
	unsigned int sparsity	= 2*sig_dim/100;	// sparsity level of signal
	cout << "Sparsity: \t\t" << sparsity << endl<<endl;
	unsigned int meas_num	= 30*sig_dim/100;	// number of measurements
	cout << "# of measurements: \t" << meas_num << endl<<endl;

	// algorithm parameters
	const unsigned int max_iter = 1.5e3;
	const double gamma = 1e0;
	const double tol = 1e-7;
	const int unsigned block_size = fmin(meas_num,sparsity);	
	const vec prob_vec = normalise(ones(meas_num / block_size),1);		// set probabilities of selecting each block

	unsigned int num_block = simulation_params.num_cores;		// number of blocks
	cout << "# of Cores Requested: \t" << simulation_params.num_cores << endl<<endl;

	simulation_params.num_faulty_cores = 0;
	cout << "# of Faulty Cores: \t" << simulation_params.num_faulty_cores << endl<<endl;
	simulation_params.num_slow_cores = (20 * simulation_params.num_cores) / 100;
	simulation_params.sleep_slow_cores = 15e3;   // microseconds to sleep
	cout << "# of Slow Cores: \t" << simulation_params.num_slow_cores;
	cout << " (each sleeping for " << simulation_params.sleep_slow_cores/1000 << " ms )" << endl<<endl;

	vector<string> alg_names;
	alg_names.push_back("Sto_IHT");
	alg_names.push_back("Parallel Sto_IHT");
	alg_names.push_back("Tally Sto_IHT");
	alg_names.push_back("Bayes Sto_IHT");
	alg_names.push_back("Majority Voting Sto_IHT");
	alg_names.push_back("AMP");
	alg_names.push_back("R_MP_AMP");
	alg_names.push_back("Async_MP_AMP");

	// define experiments
	vector <experiment> experiments;
	//experiments.push_back(experiment("Sparsity",linspace(1,5,3)));
	//experiments.push_back(experiment("Signal Dimension",logspace(3,3,1)));
	//experiments.push_back(experiment("Slow Cores",linspace(0,6,7)));
	experiments.push_back(experiment("Sleep Time",linspace(0,100e3,4)));
	//experiments.push_back(experiment("Cores",linspace(1,18,6)));
	//experiments.push_back(experiment("Blocks",simulation_params.num_cores*linspace(1,10,5)));
	
	run_experiments(experiments, alg_names, sig_dim, sparsity, meas_num, max_iter, gamma, tol ,
		block_size,  num_block,  prob_vec, simulation_params, num_mc_runs, SEED);


	return 1;
}


