matrixzq
========

Functions to carry out operations on matrices over Zq (Z/qZ) in pure Python without using Numpy or Scipy.

All matrix elements are expected to be nonnegative integers in the range [0, q-1]. 
All computations are carried out modulo q, where q is an integer greater than 1.

The modulus q is stored as a global variable `__Q`. 
It must be set using the set_modulus() function before calling functions that set or carry out arithmetic operations 
(add, multiply, etc.) on the matrix elements.

If the modulus q is not a prime then any operation that involves division (invert, solve) is undefined.

## Documentation

[Read the documentation](https://www.di-mgt.com.au/matrixzqdoc/html/index.html)

We have provided basic Python functions rather than a more-complicated class so you can easily extract and work with the underlying list of lists.


## History

* v1.1.1 (2024-10-11)
    - Fixed error in determinant method when adding prior total to a 2 x 2 matrix.

* v1.1.0 (2024-02-13)
    - Added new functions `sprint_matrix` and `sprint_vector` to print directly to strings.
	- Added `roundfrac2int` to round a fraction to an integer without using float operations.
	- Change name of test module from `matrixzq_t.py` to `test_matrixzq.py` to work better with pytest.


## Acknowledgements

This code was inspired by and some parts are derived from `LinearAlgebraPurePython.py` by Thom Ives  
<https://github.com/ThomIves/BasicLinearAlgebraToolsPurePy>  
<https://integratedmlai.com/basic-linear-algebra-tools-in-pure-python-without-numpy-or-scipy/>  

-------------------------
David Ireland  
<https://www.di-mgt.com.au/contact/>  
This document last updated 2024-10-11  
