# Asynchronous Parallel Sparse Recovery
implementation of the asynchronous parallel sparse recovery algorithm proposed in: 

A. Zaeemzadeh, J. Haddock, D. Needell, and N. Rahnavard, "A Bayesian Approach for Asynchronous Parallel Sparse Recovery," in Asilomar Conference on Signals, Systems, and Computers., 2018.
[link](https://ieeexplore.ieee.org/abstract/document/8645176)

## Citing this work
Please use the following BibTeX entries.
```
@inproceedings{Zaeemzadeh2018,
author = {Zaeemzadeh, Alireza and Haddock, Jamie and Rahnavard, Nazanin and Needell, Deanna},
booktitle = {2018 52nd Asilomar Conference on Signals, Systems, and Computers},
doi = {10.1109/ACSSC.2018.8645176},
isbn = {978-1-5386-9218-9},
month = {oct},
pages = {1980--1984},
publisher = {IEEE},
title = {{A Bayesian Approach for Asynchronous Parallel Sparse Recovery}},
url = {https://ieeexplore.ieee.org/document/8645176/},
year = {2018}
}

@inproceedings{Haddock2019,
author = {Haddock, Jamie and Needell, Deanna and Zaeemzadeh, Alireza and Rahnavard, Nazanin},
booktitle = {2019 53rd Asilomar Conference on Signals, Systems, and Computers},
doi = {10.1109/IEEECONF44664.2019.9048787},
isbn = {978-1-7281-4300-2},
month = {nov},
pages = {276--279},
publisher = {IEEE},
title = {{Convergence of Iterative Hard Thresholding Variants with Application to Asynchronous Parallel Methods for Sparse Recovery}},
url = {https://ieeexplore.ieee.org/document/9048787/},
year = {2019}
}
```

## Running the Code
To run this program you need to install armadillo, BLAS, LAPACK, and boost packages.

**Recommended Packages:** cmake, libarmadillo-dev, libarpack-dev, libarpack2-dev, libblas-dev, libboost-dev, liblapack-dev, libomp-dev, libsuperlu-dev. You can install these libraries using apt-get.   


**IMPORTANT NOTE**: Do not install/use OpenBLAS package. OpenBLAS is optimized version of BLAS which uses multi thread computation to improve performance. This makes the comparison of parallel approaches and non-parallel approaches meaningless, because OpenBLAS perfroms the non-parallel versions in a multi-thread manner. So to have a meaningful comparison do not use OpenBLAS.


- To compile, just type "make" in your terminal
- To execute, type ./final argv[1] argv[2] argv[3]

	argv[1]  : seed for random generator  (set to -1 for random seed)

	argv[2]  : number of threads requested

	argv[3]  : maximum number of MC trials

	e.g. ./final -1 10 50

Results will be saved in "/Results" subdriectory.
