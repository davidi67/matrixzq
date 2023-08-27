matrixzq
========

Functions to carry out operations on matrices over Zq (Z/qZ) where q is a positive integer greater than 1.

All matrix elements are expected to be nonnegative integers in the range [0, q-1]. 
All computations are carried out modulo q.

The modulus q is stored as a global variable `__Q`. 
It must be set using the set_modulus() function before calling functions that set or carry out arithmetic operations 
(add, multiply, etc.) on the matrix elements.

If the modulus q is not a prime then any operation that involves division (invert, solve) will fail.

-------------------------
David Ireland  
<https://www.di-mgt.com.au/contact/>  
This document last updated 2023-08-27  
