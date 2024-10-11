# $Id: test_matrixzq.py $
# $Date: 2024-02-13 09:29Z $
# $Revision: 1.1.0 $

"""Tests for matrixzq."""

# ****************************** LICENSE ***********************************
# Copyright (C) 2023-24 David Ireland, DI Management Services Pty Limited.
# All rights reserved. <www.di-mgt.com.au> <www.cryptosys.net>
# The code in this module is licensed under the terms of the MIT license.
# @license MIT
# For a copy, see <http://opensource.org/licenses/MIT>
# **************************************************************************

import matrixzq as mzq


def test_all():
    """Mostly the same tests at the end of ``matrixzq.py`` but using the prefix ``mzq``."""
    print("Doing tests in", __file__, "...")
    Z = mzq.zeros_matrix(4, 5)
    mzq.print_matrix(Z)
    I = mzq.identity_matrix(3)
    mzq.print_matrix(I)
    # Exception if __Q not set
    try:
        M = mzq.new_matrix([[1,2,3],[4,5,6],[7,8,9]])
    except Exception as e:
        print("ERROR (expected):", e)

    mzq.set_modulus(11)
    NM = mzq.new_matrix([[0, 1, 2, 3], [4, 5, 6, 8], [7, 8, 9, 10]])
    mzq.print_matrix(NM)
    print("matrix_size =", mzq.matrix_size(NM))

    mzq.set_modulus(11)
    print("__Q =", mzq.get_modulus())
    A = mzq.copy(I)
    print("Copy I:")
    mzq.print_matrix(A)
    M = mzq.new_matrix([[1,2,3],[4,5,6],[7,8,9],[10,11,12,13]])
    print("M:")
    mzq.print_matrix(M)
    print("M^T:")
    mzq.print_matrix(mzq.transpose(M))
    v = mzq.new_matrix([[1,2,3, 4, 25]])
    print("vector:")
    mzq.print_matrix(v)
    k = 3
    kM = mzq.scalar_mult(M, k)
    print(f"kM (k={k}):")
    mzq.print_matrix(kM)
    minusM = mzq.scalar_mult(M, -1)
    print("-M:")
    mzq.print_matrix(minusM)
    A = mzq.add(M, minusM)
    print("M-M:")
    mzq.print_matrix(A)
    print(mzq.equality(A, mzq.zeros_matrix(mzq.matrix_size(A)[0], mzq.matrix_size(A)[1])))
    A = mzq.new_matrix([[5,4,3,2,1],[4,3,2,1,5],[3,2,9,5,4],[2,1,5,4,3],[1,2,3,4,5]])
    AA = mzq.multiply(A, A)
    print("A*A:")
    mzq.print_matrix(AA)

    mzq.set_modulus(7)
    M = mzq.new_matrix([[1,2,3],[4,5,6]])
    print("M:"); mzq.print_matrix(M)
    k = 3
    kM = mzq.scalar_mult(M, k)
    print(f"kM (k={k}):")
    mzq.print_matrix(kM)
    minusM = mzq.scalar_mult(M, -1)
    print("-M:")
    mzq.print_matrix(minusM)

    mzq.set_modulus(31)
    print("__Q =", mzq.get_modulus())
    B = mzq.new_matrix([[18,1,25,13],[16,5,29,29],[10,4,20,25],[30,30,19,25]])
    print("B:"); mzq.print_matrix(B)
    R = mzq.new_matrix([[0,1,0,1]])
    print("R:"); mzq.print_matrix(R)
    BR = mzq.multiply(B, mzq.transpose(R))
    print("B*R:")
    mzq.print_matrix(mzq.transpose(BR))

    mzq.set_modulus(11)
    print("__Q =", mzq.get_modulus())
    M = mzq.new_matrix([[2,3],[4,5]])
    print("M:"); mzq.print_matrix(M)
    det = mzq.determinant(M)
    print("det(M) =", det)
    # Add in a previous total
    det = mzq.determinant(M, 3)
    print("3 + det(M) =", det)
    print("-7 mod 11 =", mzq.zp_negate(7))
    print("-1 mod 11 =", mzq.zp_negate(1))
    M = mzq.new_matrix([[2,3,7],[4,5,10],[9,0,7]])
    print("M:"); mzq.print_matrix(M)
    det = mzq.determinant(M)
    print("det(M) =", det)

    mzq.set_modulus(11)
    v = mzq.new_vector([1,2,3,4,5])
    print("Vector v:"); mzq.print_vector(v)
    print("Vector as Matrix:"); mzq.print_matrix(v)

    v = mzq.new_vector([1,2,3,4,5])
    print("v:"); mzq.print_vector(v)
    w = mzq.new_vector([10,6,7,8,0])
    print("w:"); mzq.print_vector(w)
    print("v dot w =", mzq.dotproduct(v, w))

    M = mzq.new_matrix([[2,3,7],[4,5,10],[9,0,7]])
    N = mzq.new_matrix([[7,8,9,10],[1,2,3,4],[2,3,4,5]])
    MN = mzq.augment_matrix(M, N)
    print("[M|N]=")
    mzq.print_matrix(MN)
    MS = mzq.slice_matrix(MN, 3)
    print("matrix_slice(3)="); mzq.print_matrix(MS)
    MS = mzq.slice_matrix(MN, 2, 8)
    print("matrix_slice(2, 8)="); mzq.print_matrix(MS)
    MS = mzq.slice_matrix(MN, -1)
    print("matrix_slice(-1)="); mzq.print_matrix(MS)
    # This should be a vector
    print("vec=",end=''); mzq.print_vector(MS)
    MS = mzq.slice_matrix(MN, -7)
    print("matrix_slice(-7)="); mzq.print_matrix(MS)
    MS = mzq.slice_matrix(MN, -6, 3)
    print("matrix_slice(-6, 3)="); mzq.print_matrix(MS)

    M = mzq.new_matrix([[2,3,7],[4,5,10],[9,0,7]])
    print("M ="); mzq.print_matrix(M)
    IM = mzq.invert(M)
    print("M^{-1}=")
    mzq.print_matrix(IM)
    mzq.print_matrix(mzq.multiply(IM, M))
    I = mzq.identity_matrix(len(M))
    print(mzq.equality(I, mzq.multiply(IM, M)))

    mzq.set_modulus(5)
    A = mzq.new_matrix([[1,2],[3,4]])
    print("A="); mzq.print_matrix(A)
    IA = mzq.invert(A)
    print("inv(A)="); mzq.print_matrix(IA)
    mzq.print_matrix(mzq.multiply(IA, A))
    I = mzq.identity_matrix(len(A))
    print(mzq.equality(I, mzq.multiply(A, IA)))

    mzq.set_modulus(31)
    P = mzq.new_matrix([[18,16,10,30], [1,5,29,29],[25,29,20,19],[13,29,25,25]])
    print("P="); mzq.print_matrix(P)
    print("P^T="); mzq.print_matrix(mzq.transpose(P))
    PI = mzq.invert(P)
    print("inv(P)="); mzq.print_matrix(PI)
    mzq.print_matrix(mzq.multiply(PI, P))
    I = mzq.identity_matrix(len(P))
    print(mzq.equality(I, mzq.multiply(P, PI)))

    mzq.set_modulus(11)
    # Ref: https://www.di-mgt.com.au/cgi-bin/matrix_stdform.cgi#solveeqn
    A = mzq.new_matrix([[1,1,1,1],[2,4,6,7],[4,5,3,5],[8,9,7,2]])
    b = mzq.new_vector([6,0,4,5])
    print("A=");mzq.print_matrix(A)
    print("b=",end='');mzq.print_vector(b)
    x = mzq.solve(A, b)
    print("x=", end=''); mzq.print_vector(x)
    print("tr(A)=", mzq.trace(A))
    AW = mzq.row_swap(A, 1, 3)
    print("A.row_swap(1,3)=");mzq.print_matrix(AW)
    AW = mzq.row_swap(A, 0, 1)
    print("A.row_swap(0,1)=");mzq.print_matrix(AW)

    # RREF
    G = mzq.augment_matrix(A, b)
    print("G=[A|b]"); mzq.print_matrix(G)

    GR = mzq.rref(G)
    print("G.rref="); mzq.print_matrix(GR)
    x = mzq.slice_matrix(GR, -1)
    print(mzq.equality(x, mzq.new_vector([10, 2, 8, 8])))
    # Check RREF of RREF is same
    GRR = mzq.rref(GR)
    print("G'="); mzq.print_matrix(GRR)
    assert(mzq.equality(GRR, GR))

    mzq.set_modulus(5)
    A = mzq.new_matrix([[0,0,0,0],[0,0,0,1],[2,4,1,4],[4,0,3,2]])
    print("A=");mzq.print_matrix(A)
    AR = mzq.rref(A)
    print("A.rref=");mzq.print_matrix(AR)
    # Check RREF of RREF is same
    print("A'="); mzq.print_matrix(mzq.rref(AR))

    A = mzq.new_matrix([[0,0,0,0,3,0],[0,0,0,4,0,0]])
    print("A=");mzq.print_matrix(A)
    AR = mzq.rref(A)
    print("A.rref=");mzq.print_matrix(AR)
    # Check RREF of RREF is same
    print("A'="); mzq.print_matrix(mzq.rref(AR))

    print("matrixzq version =", mzq.__version__)

    print("\nALL DONE.")


if __name__ == "__main__":
    test_all()
