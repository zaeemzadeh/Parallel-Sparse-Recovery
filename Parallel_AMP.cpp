#include <unistd.h>	// sleep
#include <armadillo>
#include <vector>
#include <deque>
#include <algorithm>    // std::min

#include "Functions.h"

using namespace arma;
using namespace std;


vec eta(const vec &v, const double tau){	
	vec out(size(v),fill::zeros);

	uvec ind = find(abs(v) > tau);
	out(ind) = sign(v(ind))%(abs(v(ind)) - tau );
	return out;
}

vec eta_deriv(const vec &v, const double tau){	
	vec out(size(v),fill::zeros);

	uvec ind = find(abs(v) > tau);
	out(ind).ones();
	return out;
}


vec eta(const vec &v, const vec tau){	
	vec out(size(v),fill::zeros);

	uvec ind = find(abs(v) > tau);
	out(ind) = sign(v(ind))%(abs(v(ind)) - tau(ind) );
	return out;
}

vec eta_deriv(const vec &v, const vec tau){	
	vec out(size(v),fill::zeros);

	uvec ind = find(abs(v) > tau);
	out(ind).ones();
	return out;
}

vec AMP(const mat &A, const vec &y, const int sparsity, const unsigned int max_iter,
	const double tol, unsigned int &num_iters, const simulation_parameters simulation_params){

	const unsigned int N = A.n_cols;   	// signal dimension
	const unsigned int M = y.n_elem;	// number of measurements
	//const double delta = double(num_measurements)/sig_dim;
	unsigned int i = 0;
	vec x_t(N,fill::zeros);
	vec z_t = y;
	double tau = .1;
	bool done = false;
	vec pseudo_data (N,fill::zeros);

	while(!done){
		i++;		
		z_t = y - A*x_t + z_t * sum (eta_deriv( pseudo_data,tau) ) / M;
		pseudo_data = A.t() * z_t + x_t;
		tau = tau * sum(eta_deriv(pseudo_data,tau)) / M;
		x_t = eta(pseudo_data,tau) ;

		if (norm (y - A*x_t) < tol || i >= max_iter){
			done = true;
		}
	}

	num_iters = i;
	return x_t;
}


// J. Zhu, R. Pilgrim and D. Baron, "An overview of multi-processor approximate message passing,"
// http://ieeexplore.ieee.org/document/7926166/
vec R_MP_AMP(const mat &A, const vec &y, const int sparsity, const unsigned int max_iter,
	const double tol, unsigned int &num_iters, const simulation_parameters simulation_params){
	uvec faulty_cores;
	uvec slow_cores;
	faulty_n_slow_cores(faulty_cores, slow_cores, simulation_params);


	const unsigned int N = A.n_cols;
	const unsigned int M = y.n_elem;
	const unsigned int P = simulation_params.num_cores;
	unsigned int i = 0;
	bool done = false;
	vec x_t(N,fill::zeros);
	double g_t = M;
	double tau = .1;

	vector <vec> pseudo_data (P,vec(N,fill::zeros));
	// parallel section of the code starts here
	#pragma omp parallel num_threads(simulation_params.num_cores)
	{
	// initializing variables in local memory
	const int p = omp_get_thread_num();
	mat A_p; 
	vec y_p; 
	
	A_p = A.rows(M*p/P , M*(p+1)/P -1 );
	y_p = y.subvec( M*p/P  , M*(p+1)/P -1 );
	
	vec z_t_p = y_p;
	// R_MP_AMP itearations
	while(!done){
		//AT processor p:
		z_t_p = y_p - A_p*x_t + z_t_p * g_t / M;
		pseudo_data[p] = A_p.t() * z_t_p + x_t/P;
		
		//slow cores sleep for  simulation_params.sleep_slow_cores microseconds
		if (any( slow_cores == omp_get_thread_num()) ){
			usleep(simulation_params.sleep_slow_cores);
		}
		

		#pragma omp barrier
		//AT fusion center:
		#pragma omp single
		{	
			i++;
			vec pseudo_data_total(N,fill::zeros);
			for (unsigned int j = 0; j < P; j++){
				pseudo_data_total = pseudo_data_total + pseudo_data[j];
			}
		
			tau = tau * sum(eta_deriv(pseudo_data_total,tau)) / M;
			g_t = sum(eta_deriv(pseudo_data_total,tau));
			x_t = eta(pseudo_data_total,tau);
			if (norm (y - A*x_t) < tol || i >= max_iter){
				done = true;
			}
		}
		#pragma omp barrier
	}
	// parallel section of the code ends here

	}

	num_iters = i;
	return x_t;
}





typedef vector<vec*> msg_board;

vec Async_MP_AMP(const mat &A, const vec &y, const int sparsity, const unsigned int max_iter,
	const double tol, unsigned int &num_iters, const simulation_parameters simulation_params,
	unsigned int num_blocks){

	uvec faulty_cores;
	uvec slow_cores;
	faulty_n_slow_cores(faulty_cores, slow_cores, simulation_params);

	const unsigned int N = A.n_cols;
	const unsigned int M = y.n_elem;
	const unsigned int P = 4;
	unsigned int i = 0;
	bool done = false;

	
	msg_board message_board;
	vector <omp_lock_t> message_board_locks (P);
	for (unsigned int j = 0; j < P; j++){
		message_board.push_back(NULL);
		omp_init_lock( &(message_board_locks[j]) );
	}
	// parallel section of the code starts here
	#pragma omp parallel num_threads(P)
	{
	// initializing variables in local memory
	const int p = omp_get_thread_num();

	vec x_t(N,fill::zeros);
	vec pseudo_data (N,fill::zeros);
	vec shared_data (N,fill::zeros);
	vec agg_data (N,fill::zeros);
	#pragma omp critical
	{message_board[p] = &shared_data;}
	
	double tau = .1;

	mat A_p; 
	vec y_p;
	vec norm_A_p; 
	unsigned int M_p;

	A_p = A.rows(M*p/P , M*(p+1)/P -1 );
	y_p = y.subvec( M*p/P  , M*(p+1)/P -1 );

	norm_A_p = sqrt( sum(pow(A_p,2))).t();M_p = y_p.n_rows; 
	//norm_A_p = ones(N,1);M_p = M; 
	A_p   = A_p*diagmat(1/norm_A_p);


	//cout << y_p.n_rows << endl;
	
	vec z_t_p = y_p;
	// R_MP_AMP itearations
	while(!done){
		#pragma omp single nowait
		{i++;}
		if (norm (y - A*diagmat(1/norm_A_p)*x_t) < tol || i >= max_iter){
			done = true;
		}

		//slow cores sleep for  simulation_params.sleep_slow_cores microseconds
		if (any( slow_cores == omp_get_thread_num()) ){
			usleep(simulation_params.sleep_slow_cores);
		}

		z_t_p = y_p - A_p*x_t + z_t_p * sum(eta_deriv(pseudo_data,tau)) / M_p;
		pseudo_data = A_p.t() * z_t_p;//
	
		#pragma omp critical
		{
		//cout << cnt << endl << flush;
		//cout << p << ' ' << tau << endl << flush;
		if (randu() < 1 ){
			shared_data = diagmat(1/norm_A_p)*x_t;
		}	
		}	

		//omp_unset_lock( &(message_board_locks[p]) );
		#pragma omp barrier
		agg_data.zeros(); 
		int cnt = 0;
		for (unsigned int j = 0; j < P; j++){
			if ((j != p) && (randu() < 0.8)){
				continue;
			}
			//omp_set_lock( &(message_board_locks[j]) );
			#pragma omp critical
			{
			//cout << (j - p)%P << endl << flush;
			if (message_board[j] != NULL){
				//cout << (*(message_board[j])).t() << endl << flush;
				//cout << shared_data.t() << endl << flush;
				agg_data += *(message_board[j]);
				cnt++;
			}
			}
			//omp_unset_lock( &(message_board_locks[j]) );
		}
		agg_data  /=  cnt; 
		agg_data  =  diagmat(norm_A_p)*agg_data; /**/
		
		pseudo_data +=  agg_data;//
		
		tau = tau * sum(eta_deriv(pseudo_data,tau)) / M_p;
		x_t = eta(pseudo_data,tau);	
		#pragma omp barrier
	}
	// parallel section of the code ends here

	}
	for (unsigned int j = 0; j < P; j++){
		omp_destroy_lock( &(message_board_locks[j]) );
	}
	cout << "Async#iterations: " << i << endl;
	num_iters = i;
	return zeros(N,1);   //TODO: return x_t
}


/*
vec Async_MP_AMP(const mat &A, const vec &y, const int sparsity, const unsigned int max_iter,
	const double tol, unsigned int &num_iters, const simulation_parameters simulation_params,
	unsigned int num_blocks){

	const unsigned int N = A.n_cols;
	const unsigned int M = y.n_elem;
	//const unsigned int P = simulation_params.num_cores;
	unsigned int i = 0;

	num_blocks = 2;//std::min(num_blocks,M);
	vector <mat> A_block (num_blocks); 
	vector <vec> y_block (num_blocks); 
	vector <vec> pseudo_data_block (num_blocks);
	vector <vec> z_t_block (num_blocks);
	//vector <vec> norms_block (num_blocks);
	vector <vec> x_t (num_blocks);
	bool done (false);

	// parallel section of the code starts here
	#pragma omp parallel num_threads(num_blocks)
	{
	double tau = 1;

	#pragma omp for 
	for (unsigned int b = 0; b < num_blocks; b++){
		A_block[b]   = A.rows(M*b/num_blocks , M*(b+1)/num_blocks -1 );//A;//
		//norms_block[b] = sqrt( sum(pow(A_block[b],2))).t();
		//A_block[b] = normalise(A_block[b]);
		//A_block[b]   = A_block[b]*diagmat(1/norms_block[b]);
		y_block[b]   = y.subvec( M*b/num_blocks  , M*(b+1)/num_blocks -1 );//y; //
		z_t_block[b] = y_block[b];
		x_t [b] = zeros(N,1);
		pseudo_data_block[b] = zeros(N,1);
		//cout << A_block[b].n_rows << ' ' << endl;
	}

	// Async_MP_AMP itearations
	unsigned int b = omp_get_thread_num();
	
	while(!done){
		//AT processor p:
		#pragma omp single
		{i++;	}	

		//unsigned int M_block = A_block[b].n_rows;		
		z_t_block[b] = y_block[b] - A_block[b]*x_t[b] + z_t_block[b] * sum (eta_deriv( pseudo_data_block[b],tau) ) / M;
		pseudo_data_block[b] = A_block[b].t() * z_t_block[b] + x_t[b]/num_blocks;
		#pragma omp barrier
		vec mean = zeros(N,1);
		#pragma omp single
		{
		for (unsigned int j = 0; j < num_blocks; j++){
			//if (j == b){continue;}
			//vec data = diagmat(1/norms_block[j])*pseudo_data_block[j];
			vec data = pseudo_data_block[j];
			mean = mean + data; //x_t[b]
		}
		//mean = mean/(num_blocks);
		}
		#pragma omp barrier
		pseudo_data_block[b] = mean;
		//pseudo_data_block[b] = pseudo_data_block[b] + x_t[b];

		//cout << mean.t() << endl << endl << flush;
		tau = tau * sum(eta_deriv(pseudo_data_block[b],tau)) / M;
		x_t[b] = eta(pseudo_data_block[b],tau) ;

		//x_t[b] = diagmat(1/norms_block[b])*x_t[b] ;
		if (norm (y_block[b] - A_block[b]*x_t[b]) < tol){	
			//cout << b << " converged!" << endl;
			done = true;
		}

		if (i >= max_iter){	
			//cout << b << " did not converge!" << endl;
			done = true;
		}
		#pragma omp barrier
	}
	
	}

	cout << "Async#iterations: " << i << endl;
	num_iters = i;
	return zeros(N,1);   //TODO: return x_t
}
*/
/*
vec Async_MP_AMP(const mat &A, const vec &y, const int sparsity, const unsigned int max_iter,
	const double tol, unsigned int &num_iters, const simulation_parameters simulation_params,
	unsigned int num_blocks){
	uvec faulty_cores;
	uvec slow_cores;
	faulty_n_slow_cores(faulty_cores, slow_cores, simulation_params);

	const unsigned int N = A.n_cols;
	const unsigned int M = y.n_elem;
	//const unsigned int P = simulation_params.num_cores;
	unsigned int i = 0;
	bool done = false;
	vec pseudo_data_total(N,fill::zeros);

	num_blocks = std::min(num_blocks,M);
	vector <mat> A_block (num_blocks); 
	vector <vec> y_block (num_blocks); 
	vector <vec> pseudo_data_block (num_blocks);
	vector <vec> z_t_block (num_blocks);
	unsigned int num_processed_blocks = 0;


	// parallel section of the code starts here
	#pragma omp parallel num_threads(simulation_params.num_cores)
	{
	vec x_t(N,fill::zeros);
	double g_t = M;
	double tau = 1;
	unsigned int b;	

	#pragma omp for schedule(dynamic,1)
	for (unsigned int b = 0; b < num_blocks; b++){
		A_block[b]   = A.rows(M*b/num_blocks , M*(b+1)/num_blocks -1 );
		y_block[b]   = y.subvec( M*b/num_blocks  , M*(b+1)/num_blocks -1 );
		z_t_block[b] = y_block[b];
	}

	// initializing variables in local memory
	// Async_MP_AMP itearations
	while(!done){
		//AT processor p:
		
		while (num_processed_blocks < num_blocks){		
			#pragma omp critical
			{
			b = num_processed_blocks;
			num_processed_blocks++;
			}
			//slow cores sleep for  simulation_params.sleep_slow_cores microseconds
			if (any( slow_cores == omp_get_thread_num()) ){
				//cout << omp_get_thread_num() << " sleeping!" << endl << flush;
				usleep(simulation_params.sleep_slow_cores);
				//cout << omp_get_thread_num() << " back!" << endl << flush;
			}
			if (b < num_blocks){
				z_t_block[b] = y_block[b] - A_block[b]*x_t + z_t_block[b] * g_t / M;	
				pseudo_data_block[b] =  A_block[b].t() * z_t_block[b];
			}
			//cout << omp_get_thread_num() << 'J' << endl << flush;
		}
		//#pragma omp single
		//{cout << " waiting for slow cores!" << endl << flush;}
		#pragma omp barrier
		//AT fusion center:
		#pragma omp single
		{
			
			//cout << omp_get_thread_num() << 'F' << endl << flush;
			i++;	
			pseudo_data_total.zeros();		
			for (unsigned int b = 0; b < num_blocks; b++){
				pseudo_data_total = pseudo_data_total + pseudo_data_block[b];
			}
			pseudo_data_total = pseudo_data_total + x_t;
		
			if (norm (y - A*x_t) < tol || i >= max_iter){
				done = true;
			}	
			num_processed_blocks = 0;
		}

		tau = tau * sum(eta_deriv(pseudo_data_total,tau)) / M;
		g_t = sum(eta_deriv(pseudo_data_total,tau));
		x_t = eta(pseudo_data_total,tau);
	}
	// parallel section of the code ends here

	}

	num_iters = i;
	return pseudo_data_total;  //TODO: return x_t
}
*/

