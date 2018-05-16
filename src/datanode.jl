using LearnBase
using DataFrames

"""
	DataNode(data,bags,metadata)

	Core structure of the package. It allows to represent multi-instance problems with columns of data being 
	instances and bags identifying samples. metadata for every instance can be stored in metadata.

	If bags are void, then normal matrix with samples is stored. Again it is assumed that each column indicate one sample.

```juliadoctest 
julia> DataNode(randn(3,4),[1:2,3:4])
```

creates dataset with two samples, each having two instances. 

The same can be achieved by 
```juliadoctest 
julia> DataNode(randn(3,4),[1,1,2,2])
```

Note that following construction DataNode(randn(3,4),[1,2,1,2]) leads to incorrect results, since ids of data should be sorted.

If one wishes to add metadata, it can be added as 

```juliadoctest 
julia> DataNode(randn(3,4),[1,1,2,2],[1,2,3,4])
```

The specialty of DataNode is that datasets can be nested and the data can be tuple of datasets. For example

```juliadoctest 
julia> DataNode((rand(3,2),rand(3,1),DataNode(randn(3,2))))
```
contains a tuple with Array, Vector, and nested simple DataNode Array

or 
```juliadoctest 
julia> DataNode(DataNode(rand(3,10),[1:2,3:3,0:-1,4:5,6:6,7:10]),[1:2,3:3,4:5,6:6])
```
containes two nested multiple-instance containers
"""
mutable struct DataNode{A,B<:Union{Void,Bags},C}
	data::A 
	bags::B
	metadata::C
end

function Base.show(io::IO, d::DataNode,offset::Int=0)
	paddedprint(io,"DataNode",offset)
	d.bags !=nothing && print(io," with $(length(d.bags)) bag(s)")
	print(io,"\n")
	customprint(io,data(d),offset+2)
end

Base.show(io::IO, a::A,offset) where {A<:AbstractArray} = customprint(io,a,offset)
customprint(io,a::A,offset) where {A<:Tuple} = foreach(a -> Base.show(io,a,offset),data(a))
customprint(io,a::A,offset) where {A<:DataNode} =  show(io,a,offset)
customprint(io,a::A,offset) where {A<:AbstractArray} = paddedprint(io,"array of size $(size(a))\n",offset)
customprint(io,a::A,offset) where {A<:Void} = paddedprint(io,"empty\n",offset)

DataNode(data) = DataNode(data,nothing,nothing)
DataNode(data,bags::B) where {B<:Union{Void,Bags}} = DataNode(data,bags,nothing)
DataNode(data,i::Vector,metadata = nothing) = DataNode(data,bag(i),nothing)

mapdata(f,x::DataNode{A,B,C}) where {A<:DataNode,B,C} = DataNode(mapdata(f,data(x)),x.bags,x.metadata)
mapdata(f,x::DataNode{A,B,C}) where {A<:Tuple,B,C} = DataNode(tuple(map(i -> mapdata(f,i),data(x))...),x.bags,x.metadata)
mapdata(f,x::DataNode{A,B,C}) where {A<:Void,B,C} = DataNode(nothing,x.bags,x.metadata)
mapdata(f,x::DataNode{A,B,C}) where {A,B,C} = DataNode(f(data(x)),x.bags,x.metadata)
mapdata(f,x) = f(x)

LearnBase.nobs(a::DataNode{A,<:Bags,C}) where {A,C} = length(a.bags)
LearnBase.nobs(a::DataNode{A,<:Void,C}) where {A,C} = nobs(a.data,ObsDim.Last)
LearnBase.nobs(a::DataNode{A,<:Bags,C},::Type{ObsDim.Last}) where {A,C} = length(a.bags)
LearnBase.nobs(a::DataNode{A,<:Void,C},::Type{ObsDim.Last}) where {A,C} = nobs(a.data,ObsDim.Last)
LearnBase.nobs(a::T,::Type{ObsDim.Last}) where {T<:AbstractMatrix} = size(a,2)
LearnBase.nobs(a::Vector,::Type{ObsDim.Last}) = length(a)

data(x) = x
data(x::DataNode) = x.data

function LearnBase.nobs(a::Tuple,::Type{ObsDim.Last})
	n = nobs(a[1],ObsDim.Last)
	for i in 2:length(a)
		assert(nobs(a[i],ObsDim.Last) == n)
	end
	n
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
function remapbag(b::Bags,indices::V) where {V<:VecOrRange}
	rb = Bags(length(indices))
	offset = 1
	for (i,j) in enumerate(indices)
		rb[i] = (b[j] == 0:-1)? b[j] : b[j] - b[j].start + offset
		offset += length(b[j])
	end
	rb, vcat(map(i -> collect(b[i]),indices)...)
end

"""
		function Base.cat(a,b,c...) where {T<:DataNode} 

		concatenates datasets a,b,c

```juliadoctest
a = DataNode(rand(3,4),[1:4])
b = DataNode(rand(3,4),[1:2,3:4])
cat(a,b)
```

	Internally, the functions calls package-specific function lastcat to enforce concatenations
	assuming that last dimension are observations. If you want to use DataNode with special datastores, you should extend it
"""
function Base.cat(a::DataNode...)
	data = lastcat(collect(Iterators.filter(i -> i!= nothing,map(d -> d.data,a)))...)
	metadata = lastcat(Iterators.filter(i -> i!= nothing,map(d -> d.metadata,a))...)
	bags = all(map(d -> d.bags == nothing,a)) ? nothing : catbags(map(d -> (d.bags == nothing) ? [0:-1] : d.bags ,a)...)
	return(DataNode(data, bags, metadata))
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

lastcat(a::T...) where {T<:AbstractMatrix{V} where V} = hcat(a...)
lastcat(a::Vector...) = vcat(a...)
lastcat(a::DataFrame...) = vcat(a...)
lastcat(a::DataNode...) = cat(a...)
lastcat(a::T...) where {T<:Void} = nothing
lastcat(a::Tuple...) = tuple([lastcat([a[j][i] for j in 1:length(a)]...) for i in 1:length(a[1])]...)
lastcat() = nothing


function Base.getindex(x::DataNode{A,B},i::I) where {A,B<:Bags,I<:VecOrRange}
	nb, ii = remapbag(x.bags,i)
	DataNode(subset(x.data,ii),nb,subset(x.metadata,ii))
end

Base.getindex(x::DataNode{A,<:Void,C},i::I) where {A,C,I<:VecOrRange} = DataNode(subset(x.data,i),nothing,subset(x.metadata,i))
Base.getindex(x::DataNode,i::Int) = x[i:i]
MLDataPattern.getobs(x::DataNode,i) = x[i]
MLDataPattern.getobs(x::DataNode,i,::LearnBase.ObsDim.Undefined) = x[i]
MLDataPattern.getobs(x::DataNode,i,::LearnBase.ObsDim.Last) = x[i]

subset(x::T,i) where {T<: AbstractMatrix} = x[:,i]
subset(x::Vector,i) = x[i]
subset(x::DataNode,i) = x[i]
subset(x::DataFrame,i) = x[i,:]
subset(x::T,i) where {T<:Void} = nothing
subset(xs::Tuple,i) = tuple(map(x -> x[i],xs)...)

"""
		sparsify(x,nnzrate)

		replace matrices with at most `nnzrate` fraction of non-zeros with SparseMatrixCSC

```juliadoctest 
julia> x = DataNode((DataNode((randn(5,5),zeros(5,5))),zeros(5,5)))	
julia> mapdata(i -> sparsify(i,0.05),x)

```
"""
sparsify(x,nnzrate) = x
sparsify(x::Matrix,nnzrate) = (mean(x .!= 0) <nnzrate) ? sparse(x) : x