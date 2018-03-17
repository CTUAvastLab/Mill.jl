
"""
	struct Scalar{T}
		c::T
		s::T
		T::DataType
	end

	extract a scalar value and center it with c and s
"""
struct Scalar{T}
	T::DataType
	c::T
	s::T
end

Scalar() = Scalar(Float64,0.0,1.0)
dimension(s::Scalar) = 1
(s::Scalar)(v) = s.s*(s.T(v) - s.c)
(s::Scalar)(v::String) = s(parse(s.T,v))

"""
	struct Categorical{T}
		items::T
	end

	convert value to one-hot encoded array
"""

struct Categorical{I}
	T::DataType
	items::I
end

Categorical(items) = Categorical(Float64,items)
dimension(s::Categorical)  = length(s.items)
function (s::Categorical)(v) 
	x = zeros(s.T,length(s.items))
	i = findfirst(s.items,v)
	if i > 0
		x[i] = 1
	end
	x
end

"""
	struct ArrayOf{T}
		item::T
	end

	convert array of values to one Bag of values converted with item 

```juliadoctest
julia> sc = ArrayOf(Categorical(Float64,2:4))
julia> sc([2,3,1,4]).data
3×4 Array{Float64,2}:
 1.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0
 0.0  0.0  0.0  1.0

```

```juliadoctest
julia> sc = ArrayOf(Scalar())
julia> sc([2,3,4]).data
 2.0  3.0  4.0
```
"""

struct ArrayOf{T}
	item::T
end

dimension(s::ArrayOf)  = dimension(s.item)
(s::ArrayOf)(v) = DataNode(hcat(s.item.(v)...),[1:length(v)])


"""
	struct Branch
		T::DataType
		vec::Dict{String,Any}
		other::Dict{String,Any}
		fnum::Int
	end

	Extracts DataNode, where data part is an array of vector (extractors stored in vec) and set of datanodes
	stored in other

"""
struct Branch
	T::DataType
	vec::Dict{String,Any}
	other::Dict{String,Any}
	fnum::Int
end

Branch(T,vec,other) = Branch(T,vec,other,mapreduce(dimension,+,values(vec)))

function (s::Branch)(v::Dict)
	x = vcat(map(k -> haskey(v,k) ? s.vec[k](v[k]) : zeros(s.T,dimension(s.vec[k])), keys(s.vec))...)
	x = reshape(x,:,1)
	o = map(k -> haskey(v,k) ? s.other[k](v[k]) : voidnode(), keys(s.other))	
	DataNode(Array{Any}([x,o...]),nothing,nothing)
end