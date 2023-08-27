# $Id: matrixzq.py $

# $Date: 2023-08-07 08:23Z $

# ****************************** LICENSE ***********************************
# Copyright (C) 2023 David Ireland, DI Management Services Pty Limited.
# All rights reserved. <www.di-mgt.com.au> <www.cryptosys.net>
# The code in this module is licensed under the terms of the MIT license.
# @license MIT
# For a copy, see <http://opensource.org/licenses/MIT>
# **************************************************************************

"""Functions to carry out operations on matrices over Zq (Z/qZ) where q is a positive integer greater than 1.

All matrix elements are expected to be nonnegative integers in the range [0, q-1].
All computations are carried out modulo q.

A *Matrix* of n rows and m columns is stored as an n x m list of lists.

A *Vector* type is stored as an n x 1 column matrix and can be used in any *Matrix* function.
There are some specific *Vector* functions which, for convenience, make their underlying n x 1 matrix
look like a simple vector of length n, for example :py:func:`print_vector`.

The modulus q is stored as a global variable ``__Q``. It must be set using the
:py:func:`set_modulus` function before calling functions that set or carry out arithmetic operations
(add, multiply, etc.) on the matrix elements.

If the modulus q is not a prime then any operation that involves division (invert, solve) will fail.
"""

# This code was inspired by and some parts are derived from LinearAlgebraPurePython.py by Thom Ives
# https://github.com/ThomIves/BasicLinearAlgebraToolsPurePy
# https://integratedmlai.com/basic-linear-algebra-tools-in-pure-python-without-numpy-or-scipy/

import random

__version__ = "1.0.0"

# Debugging stuff
DEBUG = False  # Set to True to show debugging output
DPRINT = print if DEBUG else lambda *a, **k: None


# Global variables
__Q = 0


def set_modulus(q):
    """Set the global modulus value ``q``.

    Args:
        q (int): modulus value, an integer greater than one

    Returns:
        int: Modulus value as set.
    """
    global __Q
    __Q = int(q)
    # Check __Q is valid
    # Must be a positive integer greater than one
    if __Q <= 1:
        __Q = 0
        raise ValueError("Invalid modulus")

    return __Q


def get_modulus():
    """ Return the global modulus value ``q`` value set by a previous call to :py:func:`set_modulus`."""
    return __Q


def new_matrix(M):
    """Create a new matrix given a list of lists.

    Args:
        M: list of lists.

    Returns:
        New matrix.

    Example:
        >>> set_modulus(11)
        >>> NM = new_matrix([[0,1,2,3],[4,5,6,8],[7,8,9,10]])
        >>> print_matrix(NM)
        [0, 1, 2, 3]
        [4, 5, 6, 8]
        [7, 8, 9, 10]
        >>> print("matrix_size =", matrix_size(NM))
        matrix_size = (3, 4)
    """
    # Expecting a list of lists
    # - each element is reduced modulo __Q
    if __Q == 0:
        raise RuntimeError("__Q is not set")
    if not all(isinstance(i, list) for i in M):
        raise TypeError("Expecting a list of lists")
    rows = len(M)
    cols = len(M[0])
    MC = zeros_matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            MC[i][j] = M[i][j] % __Q

    return MC


def new_vector(v):
    """Create a new vector.

    A Vector is stored as an ``n x 1`` column matrix and
    can be used as a Matrix in all matrix computations.

    Args:
        v (:obj:`list`): A single list ``[v1,v2,...,vn]``

    Returns:
        The vector as an ``n x 1`` matrix.

    Example:
        >>> v = new_vector([1,2,3,4,5])
        >>> print("Vector v:"); print_vector(v)
        Vector v:
        [1, 2, 3, 4, 5]
        >>> print("Vector as Matrix:"); print_matrix(v)
        Vector as Matrix:
        [1]
        [2]
        [3]
        [4]
        [5]
    """
    # Expecting a single list
    if __Q == 0:
        raise RuntimeError("__Q is not set")
    if not isinstance(v, list):
        raise TypeError("Expecting a single list for a vector")
    # Convert to a n x 1 column matrix
    rows = len(v)
    MV = zeros_matrix(rows, 1)
    for i in range(rows):
        MV[i][0] = v[i]

    return MV


def _isavector(v):
    return len(v[0]) == 1


def _issquare(A):
    return len(A) == len(A[0])


def set_element(M, row, col, value):
    """Set element (row,col) in matrix M.

    Args:
        M: Input matrix
        row (int): Index of row (zero-based)
        col (int): Index of column (zero-based)
        value (int): Value to replace existing item

    Returns:
        New matrix with element at (row,col) changed.

    Raises:
        IndexError: If (row,col) is out of range.
    """
    MC = copy(M)
    # This will fail with IndexError if out of range
    MC[row][col] = value
    return MC


def set_vector_elem(v, pos, value):
    """Set element in a vector.

    Args:
        v: vector to be changed
        pos (int): position of element (starts at zero)
        value (int): Value to replace existing item

    Returns:
        New vector with element changed.

    Raises:
        IndexError: If pos is out of range.
    """
    if not _isavector(v):
        raise TypeError("Not a vector")
    return set_element(v, pos, 0, value)


def get_element(M, row, col):
    """Get element at (row,col) in matrix M.

    Args:
        M: Input matrix
        row (int): Index of row (zero-based)
        col (int): Index of column (zero-based)

    Returns:
        (int) Value of element at (row,col).

    Raises:
        IndexError: If (row,col) is out of range.
    """
    return M[row][col]


def zeros_matrix(rows, cols):
    """Creates a matrix filled with zeros.

    Args:
        rows (int): Number of rows required in the matrix
        cols (int): Number of columns required in the matrix

    Returns:
        New all-zero matrix of size ``rows x cols``.
    """
    M = []
    while len(M) < rows:
        M.append([])
        while len(M[-1]) < cols:
            M[-1].append(0)

    return M


def zeros_vector(n):
    """Create an all-zeros vector.

    Args:
        n (int): Required length of vector.

    Returns:
        Vector as an all-zeros ``n x 1`` matrix.
    """
    return zeros_matrix(n, 1)


def identity_matrix(n):
    """Create and return an identity matrix.

    Args:
        n (int): the square size of the matrix

    Returns:
        A square identity matrix.
    """
    I = zeros_matrix(n, n)
    for i in range(n):
        I[i][i] = 1

    return I


def copy(M):
    """Creates and returns a copy of a matrix.

    Args:
        M: Input matrix to be copied

    Returns:
        Copy of matrix.
    """
    rows = len(M)
    cols = len(M[0])

    MC = zeros_matrix(rows, cols)

    for i in range(rows):
        for j in range(cols):
            MC[i][j] = M[i][j]

    return MC


def add(M, N):
    """Adds matrices M and N.

    Args:
        M: first matrix to be added to
        N: second matrix to be added, must be the same size as M

    Returns:
        Sum of M and N.

    Raises:
        ValueError: If matrices are not the same size
    """
    rows = len(M)
    cols = len(M[0])
    if rows != len(N) or cols != len(N[0]):
        raise ValueError("Matrices must be same size to add.")
    if __Q == 0:
        raise RuntimeError("__Q is not set")

    MC = zeros_matrix(rows, cols)

    for i in range(rows):
        for j in range(cols):
            MC[i][j] = (M[i][j] + N[i][j]) % __Q

    return MC


def equality(A, B):
    """Returns True if matrices are equal.

    Args:
        A: First matrix
        B: Second matrix

    Returns:
        True if matrices are equal, otherwise False.
    """
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        return False

    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] != B[i][j]:
                return False
    return True


def transpose(M):
    """Returns the transpose of a matrix.

    Args:
        M: Matrix to be transposed

    Returns:
        Transpose of given matrix.
    """
    rows = len(M)
    cols = len(M[0])

    MT = zeros_matrix(cols, rows)

    for i in range(rows):
        for j in range(cols):
            MT[j][i] = M[i][j]

    return MT


def augment_matrix(A, B):
    """Create an augmented matrix [A | B].

    Args:
        A: First matrix
        B: Second matrix, must have the same number of rows as A

    Returns:
        The augmented matrix ``[A | B]``.

    Raises:
        ValueError: if number of rows are not equal.

    Example:
        >>> set_modulus(11)
        >>> M = new_matrix([[2,3,7],[4,5,10],[9,0,7]])
        >>> N = new_matrix([[7,8,9,10],[1,2,3,4],[2,3,4,5]])
        >>> MN = augment_matrix(M, N)
        >>> print("[M|N]="); print_matrix(MN)
        [M|N]=
        [2, 3, 7, 7, 8, 9, 10]
        [4, 5, 10, 1, 2, 3, 4]
        [9, 0, 7, 2, 3, 4, 5]
    """
    # Must have same numbers of rows
    rowsA = len(A)
    colsA = len(A[0])
    rowsB = len(B)
    colsB = len(B[0])
    if (rowsA != rowsB):
        raise ValueError('Number of rows must be equal.')
    C = zeros_matrix(rowsA, colsA + colsB)
    for i in range(rowsA):
        C[i] = A[i] + B[i]
    return C


def slice_matrix(M, startcol, numcols=0):
    """Slice matrix vertically (opposite of :py:func:`augment_matrix`).

    Args:
        M: Input matrix to be split of size n x m
        startcol (int): Start column to slice (zero-based); if negative count backwards from end
        numcols (int): Number of columns to copy (default to end of row)

    Returns:
        Matrix slice of size ``n x numcols``.

    Examples:
        >>> print("[M|N]="); print_matrix(MN)
        [M|N]=
        [2, 3, 7, 7, 8, 9, 10]
        [4, 5, 10, 1, 2, 3, 4]
        [9, 0, 7, 2, 3, 4, 5]
        >>> MS = slice_matrix(MN, 3)
        >>> print("matrix_slice(3)="); print_matrix(MS)
        matrix_slice(3)=
        [7, 8, 9, 10]
        [1, 2, 3, 4]
        [2, 3, 4, 5]
        >>> MS = slice_matrix(MN, -1)
        >>> print("matrix_slice(-1)="); print_matrix(MS)
        matrix_slice(-1)=
        [10]
        [4]
        [5]
        >>> MS = slice_matrix(MN, -6, 3)
        >>> print("matrix_slice(-6, 3)="); print_matrix(MS)
        matrix_slice(-6, 3)=
        [3, 7, 7]
        [5, 10, 1]
        [0, 7, 2]
    """
    rows = len(M)
    cols = len(M[0])
    if (startcol < 0):
        startcol = cols + startcol
    if startcol < 0 or startcol >= cols:
        raise IndexError("Out of range")
    width = numcols if 0 < numcols <= cols else cols - startcol
    MS = zeros_matrix(rows, width)
    for i in range(rows):
        for j in range(width):
            MS[i][j] = M[i][j + startcol]
    return MS


def vector_concat(u, v):
    """Concatenate vectors u and v.

    Args:
        u: First vector ``(u1,...,uM)``
        v: Second vector ``(v1,...,vN)``

    Returns:
        Concatenation of vectors u and v
        ``(u1,...,uM,v1,...vN)``
    """
    rows1, cols1 = matrix_size(u)
    rows2, cols2 = matrix_size(v)
    if cols1 != 1 or cols2 != 1:
        raise TypeError("Not a vector.")
    uv = new_vector([x[0] for x in u] + [y[0] for y in v])
    return uv


def multiply(A, B):
    """Compute the product of the matrices A and B.

    Args:
        A: First matrix of size n x m
        B: Second matrix, must have m rows

    Returns:
        Matrix product A * B.

    Raises:
        ValueError: if number of columns in A is not equal to number of rows in B
    """
    rowsA = len(A)
    colsA = len(A[0])

    rowsB = len(B)
    colsB = len(B[0])

    if __Q == 0:
        raise RuntimeError("__Q is not set")
    if colsA != rowsB:
        raise ValueError("Number of A columns %d must equal number of B rows %d." % (colsA, rowsB))

    C = zeros_matrix(rowsA, colsB)

    for i in range(rowsA):
        for j in range(colsB):
            total = 0
            for ii in range(colsA):
                total += (A[i][ii] * B[ii][j])
            C[i][j] = total % __Q

    return C


def row_swap(M, x, y):
    """Elementary row operation: Interchange (swap) rows x and y in matrix M.

    ``R_x <--> R_y``

    Args:
        M: Input matrix
        x (int): Index of first row to be swapped (zero-based)
        y (int): Index of second row

    Returns:
        Matrix with rows swapped.
    """
    rows, cols = matrix_size(M)
    if x < 0 or y < 0 or x >= rows or y >= rows:
        raise IndexError("Index out of range")
    MC = copy(M)
    # Swap rows using ``a, b = b, a``
    MC[x], MC[y] = MC[y], MC[x]
    return MC


def row_scale(M, i, k):
    """Elementary row operation: Scale row i by a multiple of itself.

    ``Ri --> k * Ri``

    Args:
        M: Input matrix
        i (int): index of row to be scaled (zero-based)
        k (int): scalar value

    Returns:
        Matrix with row scaled.
    """
    rows, cols = matrix_size(M)
    if i < 0 or i >= rows:
        raise IndexError("Index out of range")
    MC = copy(M)
    # Scale row i by value
    for j in range(cols):
        MC[i][j] = zp_mult(MC[i][j], k)
    return MC


def row_addition(M, x, y, k):
    """Elementary row operation: Add a multiple k of row y to row x.

    ``R_x --> R_x + k * R_y``

    Args:
        M: Input matrix
        x (int): Index of row to be added to (zero-based)
        y (int): Index of row to be added
        k (int): factor

    Returns:
        New matrix.
    """
    rows, cols = matrix_size(M)
    if x < 0 or y < 0 or x >= rows or y >= rows:
        raise IndexError("Index out of range")
    MC = copy(M)
    # Add row y times k to row x
    for j in range(cols):
        MC[x][j] = zp_add(MC[x][j], zp_mult(MC[y][j], k))

    return MC


#################################
# Arithmetic in Zq: NB global __Q
#################################

def zp_add(a, b):
    return (a + b) % __Q


def zp_subtract(a, b):
    x = int(a) - b
    if x < 0:
        x += __Q
    return x % __Q


def zp_mult(a, b):
    return (a * b) % __Q


def zp_negate(a):
    # Return minus a modulo q.
    return (__Q - a) % __Q


def zp_inverse(a):
    a %= __Q
    if (a == 0):
        raise ValueError("Zero has no inverse!")
    inv = zp_modinv(a, __Q)
    if (inv == 0):
        raise RuntimeError("Failed to compute inverse of " + a)

    return inv


def _egcd(a, b):
    """Extended GCD algorithm"""
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = _egcd(b % a, a)
        return (g, x - (b // a) * y, y)


def zp_modinv(a, m):
    # Compute modular inverse of a mod m, if it exists.
    g, x, y = _egcd(a, m)
    if g != 1:
        raise RuntimeError('Modular inverse does not exist')
    else:
        return x % m


def round2int(x):
    """Compute x rounded to the nearest integer with ties being rounded up.

    Args:
        x (float): Real value to be rounded

    Returns:
        (int) Rounded integer value (*not* modulo q)

    Examples:
        >>> round2int(42.4999)
        42
        >>> round2int(42.5)
        43
    """
    return int(x + 0.5)


def determinant(A, total=0):
    """Compute determinant of matrix.

    Args:
        A: Input matrix
        total (int): Optional previous total to be added to output.

    Returns:
        (int) Determinant of matrix modulo q (plus any existing total).
    """
    indices = list(range(len(A)))

    if __Q == 0:
        raise RuntimeError("__Q is not set")

    if len(A) == 2 and len(A[0]) == 2:
        # Simple solution for 2 x 2 matrix
        val = zp_subtract(zp_mult(A[0][0], A[1][1]), zp_mult(A[1][0], A[0][1]))
        return val % __Q

    for fc in indices:
        As = copy(A)
        As = As[1:]
        height = len(As)

        for i in range(height):
            As[i] = As[i][0:fc] + As[i][fc + 1:]

        sign = (-1) ** (fc % 2)
        sub_det = determinant(As)
        if sign < 0:
            total += zp_negate(zp_mult(A[0][fc], sub_det))
        else:
            total += zp_mult(A[0][fc], sub_det)

    return total % __Q


def invert(A):
    """Invert a matrix.

    Args:
        A: Input matrix, must be square and non-singular

    Returns:
        Inverted matrix.

    Note:
        The modulus ``q`` must be a prime.
    """
    if __Q == 0:
        raise RuntimeError("__Q is not set")
    # Section 1: Make sure A can be inverted.
    # check_squareness(A)
    if not _issquare(A):
        raise ArithmeticError("Matrix must be square to invert.")
    # check_non_singular(A)
    if determinant(A) == 0:
        raise ArithmeticError("Singular Matrix!")

    # Section 2: Make copies of A & I, AM & IM, to use for row operations
    n = len(A)
    AM = copy(A)
    I = identity_matrix(n)
    IM = copy(I)

    # Section 3: Perform row operations
    indices = list(range(n))  # to allow flexible row referencing ***
    for fd in range(n):  # fd stands for focus diagonal
        fdScaler = zp_inverse(AM[fd][fd])
        # fdScaler = 1.0 / AM[fd][fd]
        # FIRST: scale fd row with fd inverse.
        for j in range(n):  # Use j to indicate column looping.
            AM[fd][j] = zp_mult(AM[fd][j], fdScaler)
            IM[fd][j] = zp_mult(IM[fd][j], fdScaler)
        # SECOND: operate on all rows except fd row as follows:
        for i in indices[0:fd] + indices[fd + 1:]:  # *** skip row with fd in it.
            crScaler = AM[i][fd]  # cr stands for "current row".
            for j in range(n):  # cr - crScaler * fdRow, but one element at a time.
                AM[i][j] = zp_subtract(AM[i][j], zp_mult(crScaler, AM[fd][j]))
                IM[i][j] = zp_subtract(IM[i][j], zp_mult(crScaler, IM[fd][j]))

    return IM


def rref(M):
    """Compute the reduced row echelon form (RREF) of a matrix.

    Args:
        M: Input matrix

    Returns:
        Matrix in RREF (Row canonical form).
    """

    rows, cols = matrix_size(M)
    A = copy(M)
    j = 0  # current column of interest
    for i in range(rows):
        # Element A[i,j] is the pivot
        # If it is zero then find an element *below* that is nonzero
        if j >= cols:  # Catch overrun for DPRINT statement
            break
        DPRINT(f"A[{i}][{j}]={A[i][j]} is pivot ")
        while (i < rows and j < cols and A[i][j] == 0):
            DPRINT(f"Checking [{i}][{j}]={A[i][j]}")
            if i == rows - 1:  # We are on the bottom row
                DPRINT(f"At bottom row with j = {j}")
                # Move right and loop
                j += 1
                continue
            found_pivot = False
            for k in range(i + 1, rows):
                if A[k][j] != 0:
                    # Swap rows k and i then exit while loop
                    DPRINT(f"Swopping rows {i} and {k}")
                    A = row_swap(A, i, k)
                    found_pivot = True
                    break
            if not found_pivot:
                # Column is all zeros below zero pivot so increment j <-- j+1
                DPRINT(f"Column {j} is all zeros")
                j += 1
        if j >= cols:
            # We have reached RHS so stop, we are done
            DPRINT("Reached RHS, so stop)")
            break
        # We have a nonzero pivot at A[i,j]
        # Scale row i by 1/A[i,j], this sets pivot to one
        inv = zp_inverse(A[i][j])
        DPRINT(f"Scaling row {i} by {inv}")
        A = row_scale(A, i, inv)
        # Set every element in column j equal to zero except pivot A[i,j]
        # Rk -> Rk - A[k,j]*Rk
        for k in range(rows):
            if k == i:
                continue
            DPRINT(f"Multiplying row {k} by minus {A[k][j]}")
            A = row_addition(A, k, i, zp_negate(A[k][j]))
        if DEBUG: print_matrix(A)
        # Increment j and loop for next row
        j += 1

    return A


def solve(A, b):
    """Solve the matrix equation ``Ax = b``.

    Args:
        A: Input matrix, n x n square, non-singular
        b: Vector of length n

    Returns:
        Vector solution for ``x`` of length n.

    Note:
        The modulus ``q`` must be a prime.
    """
    if not _isavector(b):
        raise TypeError("b must be a vector")
    AI = invert(A)
    x = multiply(AI, b)
    return x


def scalar_mult(M, k):
    """Multiply matrix M by scalar.

    Args:
        M: Input matrix
        k (int): scalar value, may be negative, e.g. -1

    Returns:
        Matrix multiplied by scalar ``k[M]`` (modulo q).

    Examples:
        >>> set_modulus(7)
        >> M = new_matrix([[1,2,3],[4,5,6]])
        >>> print("M:"); print_matrix(M)
        M:
        [1, 2, 3]
        [4, 5, 6]
        >>> k = 3
        >>> kM = scalar_mult(M, k)
        >>> print(f"kM (k={k}):"); print_matrix(kM)
        kM (k=3):
        [3, 6, 2]
        [5, 1, 4]
        >>> minusM = scalar_mult(M, -1)
        >>> print("-M:"); print_matrix(minusM)
        -M:
        [6, 5, 4]
        [3, 2, 1]
    """
    if __Q == 0:
        raise RuntimeError("__Q is not set")
    KM = [[x * k % __Q for x in y] for y in M]
    return KM


def dotproduct(a, b):
    """Compute dot product of two vectors.

    Args:
        a: first vector
        b: second vector of same length as first

    Returns:
        (int) A scalar equal to a dot b modulo q.
    """
    # Vector is an n x 1 matrix
    if len(a[0]) != 1 or len(b[0]) != 1:
        raise TypeError("Both arguments must be vectors.")
    if len(a) != len(b):
        raise IndexError("Must be two vectors of equal length.")
    dp = multiply(transpose(a), b)
    # dot product should be a 1 x 1 matrix, so get the scalar value
    return dp[0][0]


def trace(A):
    """Compute the trace of a matrix.

    Args:
        A: Input matrix; must be square

    Returns:
        Scalar value of trace (=sum of diagonals modulo q)
    """
    if __Q == 0:
        raise RuntimeError("__Q is not set")
    if not _issquare(A):
        raise TypeError("Matrix must be square")
    tr = 0
    for i in range(len(A)):
        tr = (tr + A[i][i]) % __Q
    return tr


def matrix_size(M):
    """Return size (rows, cols) of matrix M."""
    rows = len(M)
    cols = len(M[0])
    return rows, cols


def print_matrix(M):
    """Print a matrix.

    Args:
        M: Matrix to be printed.
    """
    for row in M:
        print([x for x in row])


def print_vector(v):
    """Print a vector.

    Args:
        v: Vector to be printed.
    """
    rows, cols = matrix_size(v)
    if cols != 1:
        raise TypeError("Not a vector.")
    # Print column vector horizontally
    print_matrix(transpose(v))


def print_matrix_latex(M, delim='b'):
    """Print matrix in LaTeX markup.

    Copy and paste the output into your LaTeX document which uses the amsmath package::

        \\usepackage{amsmath}
        % ...
        \\[
        \\begin{bmatrix}
        1 & 2 & 3 \\\\
        4 & 5 & 6
        \\end{bmatrix}
        \\]

    Args:
        M: Matrix to be printed.
        delim: delimiter in ['', 'p', 'b', 'B', 'v', 'V']; default 'b' for "bmatrix"

    Examples:
        >>> M = new_matrix([[1,2,3],[4,5,6]])
        >>> print_matrix_latex(M)
        \\begin{bmatrix}
        1 & 2 & 3 \\\\
        4 & 5 & 6
        \\end{bmatrix}
        >>> print_matrix_latex(M, delim='')
        \\begin{matrix}
        1 & 2 & 3 \\\\
        4 & 5 & 6
        \\end{matrix}
    """
    oklist = ['', 'p', 'b', 'B', 'v', 'V']
    if delim not in oklist:
        raise RuntimeError("Invalid delim character: expecting one of " + str(oklist))
    s = " \\\\\n".join(' & '.join(str(x) for x in row) for row in M)
    arg = delim + "matrix"  # default = "bmatrix"
    print(r'\begin{' + arg + "}\n" + s + "\n"  + r'\end{' + arg + '}')


# RANDOM FEATURES
# Requires ``import random``
def random_element():
    """Return a random element in the range ``[0, q-1]``."""
    return random.randint(0, __Q - 1)


def main():
    Z = zeros_matrix(4, 5)
    print_matrix(Z)
    I = identity_matrix(3)
    print_matrix(I)
    # Exception if __Q not set
    # M = new_matrix([[1,2,3],[4,5,6],[7,8,9]])

    set_modulus(11)
    NM = new_matrix([[0, 1, 2, 3], [4, 5, 6, 8], [7, 8, 9, 10]])
    print_matrix(NM)
    print("matrix_size =", matrix_size(NM))

    set_modulus(11)
    print("__Q =", get_modulus())
    A = copy(I)
    print("Copy I:")
    print_matrix(A)
    M = new_matrix([[1,2,3],[4,5,6],[7,8,9],[10,11,12,13]])
    print("M:")
    print_matrix(M)
    print("M^T:")
    print_matrix(transpose(M))
    v = new_matrix([[1,2,3, 4, 25]])
    print("vector:")
    print_matrix(v)
    k = 3
    kM = scalar_mult(M, k)
    print(f"kM (k={k}):")
    print_matrix(kM)
    minusM = scalar_mult(M, -1)
    print("-M:")
    print_matrix(minusM)
    A = add(M, minusM)
    print("M-M:")
    print_matrix(A)
    print(equality(A, zeros_matrix(matrix_size(A)[0], matrix_size(A)[1])))
    A = new_matrix([[5,4,3,2,1],[4,3,2,1,5],[3,2,9,5,4],[2,1,5,4,3],[1,2,3,4,5]])
    AA = multiply(A, A)
    print("A*A:")
    print_matrix(AA)

    set_modulus(7)
    M = new_matrix([[1,2,3],[4,5,6]])
    print("M:"); print_matrix(M)
    k = 3
    kM = scalar_mult(M, k)
    print(f"kM (k={k}):")
    print_matrix(kM)
    minusM = scalar_mult(M, -1)
    print("-M:")
    print_matrix(minusM)

    set_modulus(31)
    print("__Q =", get_modulus())
    B = new_matrix([[18,1,25,13],[16,5,29,29],[10,4,20,25],[30,30,19,25]])
    print("B:"); print_matrix(B)
    R = new_matrix([[0,1,0,1]])
    print("R:"); print_matrix(R)
    BR = multiply(B, transpose(R))
    print("B*R:")
    print_matrix(transpose(BR))

    set_modulus(11)
    print("__Q =", get_modulus())
    M = new_matrix([[2,3],[4,5]])
    print("M:"); print_matrix(M)
    det = determinant(M)
    print("det(M) =", det)
    print("-7 mod 11 =", zp_negate(7))
    print("-1 mod 11 =", zp_negate(1))
    M = new_matrix([[2,3,7],[4,5,10],[9,0,7]])
    print("M:"); print_matrix(M)
    det = determinant(M)
    print("det(M) =", det)

    set_modulus(11)
    v = new_vector([1,2,3,4,5])
    print("Vector v:", print_vector(v))
    print("Vector as Matrix:"); print_matrix(v)

    v = new_vector([1,2,3,4,5])
    print("v:"); print_vector(v)
    w = new_vector([10,6,7,8,0])
    print("w:"); print_vector(w)
    print("v dot w =", dotproduct(v, w))

    u = new_vector([1,2])
    v = new_vector([3,4,5,6])
    w = vector_concat(u,v)
    print("(u,w)=",end=''); print_vector(w)

    M = new_matrix([[2,3,7],[4,5,10],[9,0,7]])
    N = new_matrix([[7,8,9,10],[1,2,3,4],[2,3,4,5]])
    MN = augment_matrix(M, N)
    print("[M|N]=")
    print_matrix(MN)
    MS = slice_matrix(MN, 3)
    print("matrix_slice(3)="); print_matrix(MS)
    MS = slice_matrix(MN, 2, 8)
    print("matrix_slice(2, 8)="); print_matrix(MS)
    MS = slice_matrix(MN, -1)
    print("matrix_slice(-1)="); print_matrix(MS)
    # This should be a vector
    print("vec=",end=''); print_vector(MS)
    MS = slice_matrix(MN, -7)
    print("matrix_slice(-7)="); print_matrix(MS)
    MS = slice_matrix(MN, -6, 3)
    print("matrix_slice(-6, 3)="); print_matrix(MS)

    M = new_matrix([[2,3,7],[4,5,10],[9,0,7]])
    print("M ="); print_matrix(M)
    IM = invert(M)
    print("M^{-1}=")
    print_matrix(IM)
    print_matrix(multiply(IM, M))
    I = identity_matrix(len(M))
    print(equality(I, multiply(IM, M)))

    set_modulus(5)
    A = new_matrix([[1,2],[3,4]])
    print("A="); print_matrix(A)
    IA = invert(A)
    print("inv(A)="); print_matrix(IA)
    print_matrix(multiply(IA, A))
    I = identity_matrix(len(A))
    print(equality(I, multiply(A, IA)))

    set_modulus(31)
    P = new_matrix([[18,16,10,30], [1,5,29,29],[25,29,20,19],[13,29,25,25]])
    print("P="); print_matrix(P)
    print("P^T="); print_matrix(transpose(P))
    PI = invert(P)
    print("inv(P)="); print_matrix(PI)
    print_matrix(multiply(PI, P))
    I = identity_matrix(len(P))
    print(equality(I, multiply(P, PI)))

    set_modulus(11)
    # Ref: https://www.di-mgt.com.au/cgi-bin/matrix_stdform.cgi#solveeqn
    A = new_matrix([[1,1,1,1],[2,4,6,7],[4,5,3,5],[8,9,7,2]])
    b = new_vector([6,0,4,5])
    print("A=");print_matrix(A)
    print("b=",end='');print_vector(b)
    x = solve(A, b)
    print("x=", end=''); print_vector(x)
    print("tr(A)=", trace(A))
    AW = row_swap(A, 1, 3)
    print("A.row_swap(1,3)=");print_matrix(AW)
    AW = row_swap(A, 0, 1)
    print("A.row_swap(0,1)=");print_matrix(AW)

    # RREF
    G = augment_matrix(A, b)
    print("G=[A|b]"); print_matrix(G)

    GR = rref(G)
    print("G.rref="); print_matrix(GR)
    x = slice_matrix(GR, -1)
    print(equality(x, new_vector([10, 2, 8, 8])))
    # Check RREF of RREF is same
    GRR = rref(GR)
    print("G'="); print_matrix(GRR)
    assert equality(GRR, GR)

    set_modulus(5)
    A = new_matrix([[0,0,0,0],[0,0,0,1],[2,4,1,4],[4,0,3,2]])
    print("A=");print_matrix(A)
    AR = rref(A)
    print("A.rref=");print_matrix(AR)
    # Check RREF of RREF is same
    print("A'="); print_matrix(rref(AR))

    A = new_matrix([[0,0,0,0,3,0],[0,0,0,4,0,0]])
    print("A=");print_matrix(A)
    AR = rref(A)
    print("A.rref=");print_matrix(AR)
    print("AR="); print_matrix(rref(AR))
    # Check RREF of RREF is same
    assert equality(AR, rref(AR))

    # Test latex markup output
    print_matrix_latex(AR)
    print_matrix_latex(AR, 'B')
    print_matrix_latex(transpose(AR), '')


if __name__ == "__main__":
    main()
