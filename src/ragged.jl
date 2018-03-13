using LearnBase

mutable struct Ragged{A,B<:Union{Void,Bags}}
	data::A 
	bags::B
end

Ragged(data) = Ragged(data,nothing)
LearnBase.nobs(a::Ragged{A,<:Bags}) where A = length(a.bags)
LearnBase.nobs(a::Ragged{A,<:Void}) where A = nobs(a.data,ObsDim.Last)
LearnBase.nobs(a::Ragged{A,<:Bags},::Type{ObsDim.Last}) where A = length(a.bags)
LearnBase.nobs(a::Ragged{A,<:Void},::Type{ObsDim.Last}) where A = nobs(a.data,ObsDim.Last)
LearnBase.nobs(a::Matrix,::Type{ObsDim.Last}) = size(a,2)
LearnBase.nobs(a::Vector,::Type{ObsDim.Last}) = length(a)
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
julia> NestedMill.bag([2, 2, 2, 1, 1, 3])
3-element Array{UnitRange{Int64},1}:
 1:3
 4:5
 6:6
```

this will throw error ```NestedMill.bag([2, 2, 2, 1, 1, 3, 1])```
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
julia> NestedMill.remapbag([1:1,2:3,4:5],[2,3])
(UnitRange{Int64}[2:3, 2:3], [2, 3, 4, 5])

```

```juliadoctest
julia> NestedMill.remapbag([1:1,2:3,4:5],[1])
(UnitRange{Int64}[1:1], [1])
```

"""
function remapbag(b::Bags,indices::I) where {I<:Union{UnitRange{Int},Vector{Int}}}
	rb = Bags(length(indices))
	offset = 1
	for (i,j) in enumerate(indices)
		rb[i] = (b[j] == 0:-1)? b[j] : b[j] - b[j].start + offset
		offset += length(b[j])
	end
	rb, vcat(map(i -> collect(b[i]),indices)...)
end

"""
		function Base.cat(a,b,c...) where {T<:Ragged} 

		concatenates datasets a,b,c

```juliadoctest
a = Ragged(rand(3,4),[1:4])
b = Ragged(rand(3,4),[1:2,3:4])
cat(a,b)
```

	Internally, the functions calls package-specific function lastcat to enforce concatenations
	assuming that last dimension are observations.
"""
function Base.cat(a::Ragged{A,<:Bags}...) where {A} 
	data = lastcat(map(d -> d.data,a)...)
	bags, offset = Bags(), 0;
	for i in 1:length(a)
		append!(bags,a[i].bags .+ offset)
		offset += nobs(a[i].data,ObsDim.Last)
	end
	mask = length.(bags) .== 0
	if sum(mask) > 0
		bags[mask] = fill(0:-1,sum(mask))
	end
	Ragged(data,bags)
end

function Base.cat(a::Ragged{A,<:Void}...) where {A} 
	data = lastcat(map(d -> d.data,a)...)
	Ragged(data,nothing)
end

lastcat(a::Matrix...) = hcat(a...)
lastcat(a::Vector...) = vcat(a...)
lastcat(a::Ragged...) = cat(a...)
lastcat(a::Tuple...) = tuple([lastcat([a[j][i] for j in 1:length(a)]...) for i in 1:length(a[1])]...)


function Base.getindex(x::Ragged{A,B},i::Union{UnitRange{Int},Vector{Int}}) where {A,B<:Bags}
	nb, ii = remapbag(x.bags,i)
	Ragged(subset(x.data,ii),nb)
end

Base.getindex(x::Ragged{A,B},i::Union{UnitRange{Int},Vector{Int}}) where {B<:Void,A} = Ragged(subset(x.data,ii),nothing)
Base.getindex(x::Ragged,i::Int) = x[i:i]

subset(x::Matrix,i) = x[:,i]
subset(x::Vector,i) = x[i]
subset(x::Ragged,i) = x[i]
subset(xs::Tuple,i) = tuple(map(x -> x[i],xs))


LearnBase.nobs(a::Ragged{A,B},i) where {A,B<:Void} = nobs(a.data,i)
LearnBase.nobs(a::Ragged{A,B},i) where {A,B} = length(a.bags)