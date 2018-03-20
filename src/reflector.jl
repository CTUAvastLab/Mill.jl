abstract type AbstractReflector end;

"""
	struct Scalar{T}
		c::T
		s::T
		T::Type{T}
	end

	extract a scalar value and center it with c and s
"""
struct Scalar{T,V} <: AbstractReflector
	datatype::Type{T}
	c::V
	s::V
end

Scalar(::Type{T}) where {T<:Number} = Scalar(T,T(0),T(1))
Scalar(::Type{T}) where {T<:AbstractString} = Scalar(T,nothing,nothing)
Scalar(d::Dict{String,Any}) = Scalar(getdatatype(Float64,d),get(d,"center",0),get(d,"scale",1))
tojson(s::Scalar) = "{\"type\": \"Scalar\", \"datatype\": \"$(s.datatype)\", \"center\": $(s.c), \"scale\": $(s.s)}"
dimension(s::Scalar) = 1
(s::Scalar{T,V})(v) where {T<:Number,V}						= s.s*(s.datatype(v) - s.c)
(s::Scalar{T,V})(v::S) where {T<:Number,V,S<:Void}= 0
(s::Scalar{T,V} where {V,T<:Number})(v::String)   = s(parse(s.datatype,v))
(s::Scalar{T,V} where {V,T<:AbstractString})(v)   = v
(s::Scalar{T,V})(v::S) where {T<:Number,V,S<:AbstractString} = ""

"""
	struct Categorical{T}
		items::T
	end

	convert value to one-hot encoded array
"""

struct Categorical{T,I} <: AbstractReflector
	datatype::Type{T}
	items::I
end

Categorical(items) = Categorical(Float64,items)
Categorical(d::Dict{String,Any}) = Categorical(getdatatype(Float64,d),d["items"])
tojson(s::Categorical) = "{\"type\": \"Categorical\", \"datatype\": \"$(s.datatype)\", items: "*JSON.json(s.items)*"}"
dimension(s::Categorical)  = length(s.items)
function (s::Categorical)(v) 
	x = zeros(s.datatype,length(s.items))
	i = findfirst(s.items,v)
	if i > 0
		x[i] = 1
	end
	x
end

(s::Categorical)(v::V) where {V<:Void} =  zeros(s.datatype,length(s.items))

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

struct ArrayOf{T} <: AbstractReflector
	item::T
end

ArrayOf(d::Dict{String,Any}) = ArrayOf(interpret(d["item"]))
tojson(s::ArrayOf) = "{\"type\": \"ArrayOf\", \"item\": "*tojson(s.item)*"}"
dimension(s::ArrayOf)  = dimension(s.item)
(s::ArrayOf)(v) = DataNode(hcat(s.item.(v)...),[1:length(v)])
(s::ArrayOf)(v::V) where {V<:Void} = DataNode(nothing,[0:-1])

"""
	struct Branch
		T::Type{T}
		vec::Dict{String,Any}
		other::Dict{String,Any}
		fnum::Int
	end

	Extracts DataNode, where data part is an array of vector (extractors stored in vec) and set of datanodes
	stored in other

"""
struct Branch{T,S,V} <: AbstractReflector
	datatype::Type{T}
	vec::S
	other::V
	fnum::Int
end

function Branch(T,vec,other)
	v = (vec ==  nothing || isempty(vec)) ? nothing : vec
	fnum = vec ==  nothing ? 0 : mapreduce(dimension,+,values(vec));
	o = (other ==  nothing || isempty(other)) ? nothing : other
	Branch(T,v,o,fnum)
end

Branch(d::Dict{String,Any}) = Branch(getdatatype(Float64,d),interpret(get(d,"vec","{}")),interpret(get(d,"other","{}")))
function tojson(s::Branch,indent=0)
	o = "{\"type\": \"Branch\", \"datatype\": \"$(s.datatype)\",\n"
	o *= "\"vec\": {"
	if s.vec != nothing && !isempty(s.vec)
		o *= "\n"
		o *=join(map(k -> @sprintf("\"%s\": %s",k,tojson(s.vec[k])),keys(s.vec)),",\n")
		o *= "\n"
	end
  o *= "},\n \"other\": {"
  if s.other != nothing && !isempty(s.other)
  	o *= "\n"
		o *=join(map(k -> @sprintf("\"%s\": %s",k,tojson(s.other[k])),keys(s.other)),",\n")
		o *= "\n"
	end
	o *= "}\n"
	o *= "}\n"
end

(s::Branch)(v::V) where {V<:Void} = s(Dict{String,Any}())
function (s::Branch{T,S,V})(v::Dict) where {T,S<:Dict,V<:Dict}
	x = vcat(map(k -> s.vec[k](get(v,k,nothing)),keys(s.vec))...)
	x = reshape(x,:,1)
	o = map(k -> s.other[k](get(v,k,nothing)), keys(s.other))
	data = [x,o...]
	DataNode(Array{Any}([x,o...]),nothing,nothing)
end

function (s::Branch{T,S,V})(v::Dict) where {T,S<:Dict,V<:Void}
	x = vcat(map(k -> s.vec[k](get(v,k,nothing)),keys(s.vec))...)
	x = reshape(x,:,1)
	DataNode(x,nothing,nothing)
end

function (s::Branch{T,S,V})(v::Dict) where {T,S<:Void,V<:Dict}
	x = map(k -> s.other[k](get(v,k,nothing)), keys(s.other))
	x = (length(x) == 1) ? x[1] : x
	DataNode(x,nothing,nothing)
end

extractormap = Dict(:Branch => Branch, :ArrayOf => ArrayOf, :Scalar => Scalar, :Categorical => Categorical)
interpret(s::String) = interpret(JSON.parse(s))
function interpret(d::Dict{String,Any})
	if !haskey(d,"type")
		return(Dict{String,Any}(map(k -> (k,interpret(d[k])),keys(d))))
	else 
		t = Symbol(d["type"])
		!haskey(extractormap,t) && error("unknown extractor $(t)")
		return(extractormap[t](d))
	end
end

typemap = Dict(:Float64 => Float64, :Float32 => Float32, :String => String )
getdatatype(T,d::Dict) = haskey(d,"datatype") ? getdatatype(d["datatype"]) : T
function getdatatype(s::String)
	t = Symbol(s)
	!haskey(typemap,t) && error("unknown type $(t)")
	return(typemap[t])
end
