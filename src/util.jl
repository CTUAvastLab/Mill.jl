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

function length2bags(ls::Vector{Int})
	ls = vcat([0],cumsum(ls))
	bags = map(i -> i[1]+1:i[2],zip(ls[1:end-1],ls[2:end]))
	map(b -> isempty(b) ? (0:-1) : b,bags)
end

function catbags(oldbags...)
	offset = 0
	newbags = Bags()
	for b in oldbags
		append!(newbags,b .+ offset)
		offset += max(0,mapreduce(i -> i.stop,max,b))
	end
	mask = length.(newbags) .== 0
	if sum(mask) > 0
		newbags[mask] = fill(0:-1,sum(mask))
	end
	newbags
end

"""
		function bag(k::Vector)

		create vector of unit ranges from keys k, assuming they are continuous

```juliadoctest
julia> Mill.bag([2, 2, 2, 1, 1, 3])
3-element Array{UnitRange{Int64},1}:
 1:3
 4:5
 6:6
```

this will throw error ```Mill.bag([2, 2, 2, 1, 1, 3, 1])```
"""
bag(b::Bags) = b
function bag(k::Vector)
	b = Bags(length(unique(k)))
	i = 1
	bi = 1
	for j in 2:length(k)
		if k[j] != k[i]
			b[bi] = i:j-1
			bi, i = bi+1,j
		end
	end
	b[bi] = i:length(k)
	bi != length(b) && error("The number of bags should correspond to the number of unique items in k")
	b
end


"""
		function remapbag(b::Bags,indices::Vector{Int})

		bags corresponding to indices with collected indices

```juliadoctest
julia> Mill.remapbag([1:1,2:3,4:5],[2,3])
(UnitRange{Int64}[2:3, 2:3], [2, 3, 4, 5])

```

```juliadoctest
julia> Mill.remapbag([1:1,2:3,4:5],[1])
(UnitRange{Int64}[1:1], [1])
```

"""
function remapbag(b::Bags,indices::VecOrRange)
	rb = Bags(length(indices))
	offset = 1
	for (i,j) in enumerate(indices)
		rb[i] = (b[j] == 0:-1) ? b[j] : b[j] - b[j].start + offset
		offset += length(b[j])
	end
	rb, vcat(map(i -> collect(b[i]),indices)...)
end
