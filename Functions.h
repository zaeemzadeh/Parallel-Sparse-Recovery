#ifndef FUNCTIONS
#define FUNCTIONS

#include <armadillo>
#include <vector>

using namespace arma;
using namespace std;

class simulation_parameters{
	public:
	unsigned int 	num_cores;
	unsigned int 	num_slow_cores;
	double 		    sleep_slow_cores;// microseconds to sleep
    double          slow_cores_ratio;
	simulation_parameters() : num_cores(1),num_slow_cores(0),
			sleep_slow_cores(0),slow_cores_ratio(0){};
};


class trial_info{
	public:
	double 	time;
	unsigned int 	iterations;
	trial_info() : time(-1),iterations(0){};
	trial_info(double t, unsigned int i) : time(t),iterations(i){};
};


class performance_metrics{
	public:
	double 	time_avg;
	double 	time_std;
	double 	iter_avg;
	double 	iter_std;
	double 	time_per_iter_avg;
	double 	time_per_iter_std;
	double 	success_avg;
	double  success_std;
	performance_metrics() : time_avg(-1), time_std(-1), iter_avg(-1), iter_std(-1), time_per_iter_avg(-1), time_per_iter_std(-1), success_avg(-1), success_std(-1) {};
};


performance_metrics calculate_metrics(const vector<trial_info> MC_runs, const unsigned int max_iter);


vector<performance_metrics> run_mc_trials(const vector <string> alg_names, const unsigned int sig_dim, const unsigned int sparsity, const unsigned int meas_num, const unsigned int max_iter, const double gamma, const double tol ,   const vec prob_vec,  simulation_parameters simulation_params, const int num_mc_runs, const int SEED);


void set_slow_cores(uvec &slow_cores, const simulation_parameters simulation_params);


void save_results(const vector<string> alg_names, const vector <vector<performance_metrics>> sweep_metrics,
	const string parameter_to_sweep, const vec parameter_to_sweep_values,
	const unsigned int sig_dim, const unsigned int sparsity, const unsigned int meas_num, 
	const unsigned int max_iter,  simulation_parameters simulation_params, const int num_mc_runs);


void print_results(const vector<string> alg_names, const vector <vector<performance_metrics>> sweep_metrics,
	const string parameter_to_sweep, const vec parameter_to_sweep_values,
	const unsigned int sig_dim, const unsigned int sparsity, const unsigned int meas_num, 
	const unsigned int max_iter, simulation_parameters simulation_params, const int num_mc_runs);


void run_experiments(const vector <string> alg_names, unsigned int sig_dim, 
	unsigned int sparsity, unsigned int meas_num, const unsigned int max_iter, const double gamma, const double tol , const vec prob_vec, simulation_parameters simulation_params,
	 const int num_mc_runs, const int SEED);


#endif /* FUNCTIONS */
