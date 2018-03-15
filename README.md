# Asynchronous Parallel Sparse Recovery

To run this program you need to install armadillo, BLAS, LAPACK, and boost packages:

**IMPORTANT NOTE**: Do not install/use OpenBLAS package. OpenBLAS is optimized version of BLAS which uses multi thread computation to improve performance. This makes the comparison of parallel approaches and non-parallel approaches meaningless, because OpenBLAS perfroms the non-parallel versions in a multi-thread manner. So to have a meaningful comparison do not use OpenBLAS.


- To compile, just type "make" in your terminal
- To execute, type ./final argv[1] argv[2] argv[3]

	argv[1]  : seed for random generator  (set to -1 for random seed)

	argv[2]  : number of threads requested

	argv[3]  : maximum number of MC trials

	e.g. ./final -1 10 50

Results will be saved in "/Results" subdriectory.
