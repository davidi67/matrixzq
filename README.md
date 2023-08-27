matrixzq
========

Functions to carry out operations on matrices over Zq (Z/qZ) in pure Python without using Numpy or Scipy.

All matrix elements are expected to be nonnegative integers in the range [0, q-1]. 
All computations are carried out modulo q, where q is an integer greater than 1.

The modulus q is stored as a global variable `__Q`. 
It must be set using the set_modulus() function before calling functions that set or carry out arithmetic operations 
(add, multiply, etc.) on the matrix elements.

If the modulus q is not a prime then any operation that involves division (invert, solve) will fail.

We have provided basic Python functions rather than a more-complicated class so you can easily extract and work with the underlying list of lists.

This code was inspired by and some parts are derived from `LinearAlgebraPurePython.py` by Thom Ives  
<https://github.com/ThomIves/BasicLinearAlgebraToolsPurePy>  
<https://integratedmlai.com/basic-linear-algebra-tools-in-pure-python-without-numpy-or-scipy/>  

-------------------------
David Ireland  
<https://www.di-mgt.com.au/contact/>  
This document last updated 2023-08-27  
