# Parallel-Sparse-Recovery

To run this program you need to install armadillo and boost packages:

https://github.com/conradsnicta/armadillo-code/

https://stackoverflow.com/questions/12578499/how-to-install-boost-on-ubuntu

- To compile, just type "make" in your terminal
- To execute, type ./final argv[1] argv[2] argv[3]

	argv[1]  : seed for random generator  (set to -1 for random seed)

	argv[2]  : number of threads requested

	argv[3]  : maximum number of MC trials

	e.g. ./final -1 10 2

Results will be saved in "/Results" subdriectory.
