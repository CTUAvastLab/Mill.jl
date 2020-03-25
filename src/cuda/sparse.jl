using LinearAlgebra
using MacroTools

# algorithm based on scipy/sparse/sparsetools/csr.h (taken from https://github.com/JuliaSparse/SparseMatrices.jl/blob/master/src/csr.jl)
function csr2csc(indptr::Vector{Ti}, indval::Vector{Ti},
                        nzval::Vector{Tv}, m::Int, n::Int) where {Tv,Ti}
    # allocate
    Bptr = zeros(Ti, n+1)
    Bind = similar(indval)
    Bval = similar(nzval)

    numnz = indptr[m+1] - 1

    # get colptr by accumulating hits on colval and doing cumsum
    @inbounds Bptr[1] = 1
    @inbounds for i=1:numnz Bptr[indval[i]+1] += 1; end
    Bptr = cumsum(Bptr)

    @inbounds for row=1:m
        for j=indptr[row]:(indptr[row+1]-1)
            col = indval[j]
            dest = Bptr[col]

            Bind[dest] = row
            Bval[dest] = nzval[j]
            Bptr[col] += 1
        end
    end

    # fix up Bptr
    last = 1
    @inbounds for col=1:n
        temp = Bptr[col]
        Bptr[col] = last
        last = temp
    end
    Bptr, Bind, Bval
end

struct CuSparseMatrix <: AbstractArray{Float32,2}
    csc::CuArrays.CUSPARSE.CuSparseMatrixCSC{Float32}
    csr::CuArrays.CUSPARSE.CuSparseMatrixCSR{Float32}
end
function CuSparseMatrix(csc::SparseArrays.SparseMatrixCSC)
    Bptr, Bind, Bval = csr2csc(csc.colptr, csc.rowval, csc.nzval, csc.n, csc.m)
    cucsr = CuArrays.CUSPARSE.CuSparseMatrixCSR(CuArray{Int32}(Bptr), CuArray{Int32}(Bind), CuArray{Float32}(Bval), (csc.m, csc.n))
    CuSparseMatrix(
        CuArrays.CUSPARSE.CuSparseMatrixCSC(SparseArrays.SparseMatrixCSC(csc)),
        cucsr)
end
MacroTools.@forward CuSparseMatrix.csc Base.size, Base.getindex, Base.axes

"""
    Multiplies dense A * sparse B using CUSPARSE
"""
# A::CuMatrix * B::CuArrays.CUSPARSE.CuSparseMatrixCSC =
#     permutedims(CuArrays.CUSPARSE.mm('T', B, permutedims(A, (2,1)), 'O'), (2,1))
# #TODO: Possible inefficiencies due to transposes
#
# A::CuMatrix * B::Adjoint{T,CuArrays.CUSPARSE.CuSparseMatrixCSC{T}} where T =
#     permutedims(CuArrays.CUSPARSE.mm('N', B', permutedims(A, (2,1)), 'O'), (2,1))

A::CuMatrix * B::CuSparseMatrix =
    permutedims(CuArrays.CUSPARSE.mm('T', B.csc, permutedims(A, (2,1)), 'O'), (2,1))
#TODO: Possible inefficiencies due to transposes

A::CuMatrix * B::Adjoint{Float32,CuSparseMatrix} where T =
    permutedims(CuArrays.CUSPARSE.mm('N', (B').csr, permutedims(A, (2,1)), 'O'), (2,1))


# #TODO: Gradients w.r.t. sparse matrices are not considered
# Zygote.@adjoint A::CuMatrix * B::CuArrays.CUSPARSE.CuSparseMatrixCSC =
#     A * B, Δ -> (Δ * B', nothing)

Zygote.@adjoint A::CuMatrix * B::CuSparseMatrix =
    A * B, Δ -> (Δ * B', nothing)
