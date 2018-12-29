"""
		sparsify(x,nnzrate)

		replace matrices with at most `nnzrate` fraction of non-zeros with SparseMatrixCSC

```juliadoctest
julia> x = TreeNode((
				TreeNode((
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

