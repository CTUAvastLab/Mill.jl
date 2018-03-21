abstract type AbstractReflector end;

"""
	struct ExtractScalar{T}
		c::T
		s::T
		T::Type{T}
	end

	extract a scalar value and center it with c and s
"""
struct ExtractScalar{T,V} <: AbstractReflector
	datatype::Type{T}
	c::V
	s::V
end

ExtractScalar(::Type{T}) where {T<:Number} = ExtractScalar(T,T(0),T(1))
ExtractScalar(::Type{T}) where {T<:AbstractString} = ExtractScalar(T,nothing,nothing)
ExtractScalar(d::Dict{String,Any}) = ExtractScalar(getdatatype(Float32,d),get(d,"center",0),get(d,"scale",1))
tojson(s::ExtractScalar) = "{\"type\": \"ExtractScalar\", \"datatype\": \"$(s.datatype)\", \"center\": $(s.c), \"scale\": $(s.s)}"
dimension(s::ExtractScalar) = 1
(s::ExtractScalar{T,V})(v) where {T<:Number,V}						= s.s*(s.datatype(v) - s.c)
(s::ExtractScalar{T,V})(v::S) where {T<:Number,V,S<:Void}= 0
(s::ExtractScalar{T,V} where {V,T<:Number})(v::String)   = s(parse(s.datatype,v))
(s::ExtractScalar{T,V} where {V,T<:AbstractString})(v)   = v
(s::ExtractScalar{T,V})(v::S) where {T<:Number,V,S<:AbstractString} = ""

"""
	struct ExtractCategorical{T}
		items::T
	end

	convert value to one-hot encoded array
"""

struct ExtractCategorical{T,I<:Vector} <: AbstractReflector
	datatype::Type{T}
	items::I
end

ExtractCategorical(items) = ExtractCategorical(Float32,items)
ExtractCategorical(d::Dict{String,Any}) = ExtractCategorical(getdatatype(Float32,d),d["items"])
ExtractCategorical(T,s::Entry) = ExtractCategorical(T,sort(collect(keys(s.counts))))
tojson(s::ExtractCategorical) = "{\"type\": \"ExtractCategorical\", \"datatype\": \"$(s.datatype)\", items: "*JSON.json(s.items)*"}"
dimension(s::ExtractCategorical)  = length(s.items)
function (s::ExtractCategorical)(v) 
	x = zeros(s.datatype,length(s.items))
	i = findfirst(s.items,v)
	if i > 0
		x[i] = 1
	end
	x
end

(s::ExtractCategorical)(v::V) where {V<:Void} =  zeros(s.datatype,length(s.items))

"""
	struct ExtractArray{T}
		item::T
	end

	convert array of values to one Bag of values converted with item 

```juliadoctest
julia> sc = ExtractArray(ExtractCategorical(Float32,2:4))
julia> sc([2,3,1,4]).data
3×4 Array{Float32,2}:
 1.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0
 0.0  0.0  0.0  1.0

```

```juliadoctest
julia> sc = ExtractArray(ExtractScalar())
julia> sc([2,3,4]).data
 2.0  3.0  4.0
```
"""

struct ExtractArray{T} <: AbstractReflector
	item::T
end

ExtractArray(d::Dict{String,Any}) = ExtractArray(interpret(d["item"]))
tojson(s::ExtractArray) = "{\"type\": \"ExtractArray\", \"item\": "*tojson(s.item)*"}"
dimension(s::ExtractArray)  = dimension(s.item)
(s::ExtractArray)(v) = DataNode(hcat(s.item.(v)...),[1:length(v)])
(s::ExtractArray)(v::V) where {V<:Void} = DataNode(nothing,[0:-1])

"""
	struct ExtractBranch
		T::Type{T}
		vec::Dict{String,Any}
		other::Dict{String,Any}
		fnum::Int
	end

	Extracts DataNode, where data part is an array of vector (extractors stored in vec) and set of datanodes
	stored in other

"""
struct ExtractBranch{T,S,V} <: AbstractReflector
	datatype::Type{T}
	vec::S
	other::V
	fnum::Int
end

function ExtractBranch(T,vec,other)
	v = (vec ==  nothing || isempty(vec)) ? nothing : vec
	fnum = vec ==  nothing ? 0 : mapreduce(dimension,+,values(vec));
	o = (other ==  nothing || isempty(other)) ? nothing : other
	ExtractBranch(T,v,o,fnum)
end

ExtractBranch(d::Dict) = ExtractBranch(getdatatype(Float32,d),interpret(get(d,"vec","{}")),interpret(get(d,"other","{}")))
function tojson(s::ExtractBranch,indent=0)
	o = "{\"type\": \"ExtractBranch\", \"datatype\": \"$(s.datatype)\",\n"
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

(s::ExtractBranch)(v::V) where {V<:Void} = s(Dict{String,Any}())
function (s::ExtractBranch{T,S,V})(v::Dict) where {T,S<:Dict,V<:Dict}
	x = vcat(map(k -> s.vec[k](get(v,k,nothing)),keys(s.vec))...)
	x = reshape(x,:,1)
	o = map(k -> s.other[k](get(v,k,nothing)), keys(s.other))
	data = [x,o...]
	DataNode(Array{Any}([x,o...]),nothing,nothing)
end

function (s::ExtractBranch{T,S,V})(v::Dict) where {T,S<:Dict,V<:Void}
	x = vcat(map(k -> s.vec[k](get(v,k,nothing)),keys(s.vec))...)
	x = reshape(x,:,1)
	DataNode(x,nothing,nothing)
end

function (s::ExtractBranch{T,S,V})(v::Dict) where {T,S<:Void,V<:Dict}
	x = map(k -> s.other[k](get(v,k,nothing)), keys(s.other))
	x = (length(x) == 1) ? x[1] : x
	DataNode(x,nothing,nothing)
end

extractormap = Dict(:ExtractBranch => ExtractBranch, :ExtractArray => ExtractArray, :ExtractScalar => ExtractScalar, :ExtractCategorical => ExtractCategorical)
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

typemap = Dict(:Float32 => Float32, :Float32 => Float32, :String => String )
getdatatype(T,d::Dict) = haskey(d,"datatype") ? getdatatype(d["datatype"]) : T
function getdatatype(s::String)
	t = Symbol(s)
	!haskey(typemap,t) && error("unknown type $(t)")
	return(typemap[t])
end
