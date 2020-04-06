"""
		sparsify(x,nnzrate)

		replace matrices with at most `nnzrate` fraction of non-zeros with SparseMatrixCSC

```juliadoctest
julia> x = ProductNode((
				ProductNode((
					MatrixNode(randn(5,5)),
					MatrixNode(zeros(5,5))
						)),
				MatrixNode(zeros(5,5))
				))
julia> mapdata(i -> sparsify(i,0.05),x)

```
"""
sparsify(x,nnzrate) = x
sparsify(x::Matrix,nnzrate) = (mean(x .!= 0) <nnzrate) ? sparse(x) : x

function Base.:*(A::AbstractMatrix, B::Adjoint{Bool,<: Flux.OneHotMatrix})
    m = size(A,1)
    Y = similar(A, m, size(B,2))
    Y .= 0
    BT = B'
    for (j,ohv) in enumerate(BT.data)
        ix = ohv.ix
        for i in 1:m
            @inbounds Y[i,ix] += A[i,j]
        end
    end
    Y
end
