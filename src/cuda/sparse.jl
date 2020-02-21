"""
    Multiplies dense A * sparse B using CUSPARSE
"""
A::CuMatrix * B::CuArrays.CUSPARSE.CuSparseMatrixCSC =
    permutedims(CuArrays.CUSPARSE.mm('T', B, permutedims(A, (2,1)), 'O'), (2,1))
#TODO: Possible inefficiencies due to transposes

#TODO: Gradients w.r.t. sparse matrix multiplication are not considered
Zygote.@adjoint A::CuMatrix * B::CuArrays.CUSPARSE.CuSparseMatrixCSC =
    A * B, Δ -> (nothing, nothing)
