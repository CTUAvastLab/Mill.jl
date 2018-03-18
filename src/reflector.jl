"""
	struct Scalar{T}
		c::T
		s::T
		T::DataType
	end

	extract a scalar value and center it with c and s
"""
struct Scalar{T}
	datatype::DataType
	c::T
	s::T
end

Scalar() = Scalar(Float64,0.0,1.0)
Scalar(d::Dict{String,Any}) = Scalar(getdatatype(Float64,d),get(d,"center",0),get(d,"scale",1))
tojson(s::Scalar) = "{\"type\": \"Scalar\", \"datatype\": \"$(s.datatype)\", \"center\": $(s.c), \"scale\": $(s.s)}"
dimension(s::Scalar) = 1
(s::Scalar)(v) = s.s*(s.datatype(v) - s.c)
(s::Scalar)(v::String) = s(parse(s.datatype,v))

"""
	struct Categorical{T}
		items::T
	end

	convert value to one-hot encoded array
"""

struct Categorical{I}
	datatype::DataType
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

ArrayOf(d::Dict{String,Any}) = ArrayOf(interpret(d["item"]))
tojson(s::ArrayOf) = "{\"type\": \"ArrayOf\", \"item\": "*tojson(s.item)*"}"
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
	datatype::DataType
	vec::Dict{String,Any}
	other::Dict{String,Any}
	fnum::Int
end

Branch(T,vec,other) = Branch(T,vec,other,mapreduce(dimension,+,values(vec)))
Branch(d::Dict{String,Any}) = Branch(getdatatype(Float64,d),interpret(get(d,"vec","{}")),interpret(get(d,"other","{}")))

function tojson(s::Branch,indent=0)
	o = "{\"type\": \"Branch\", \"datatype\": \"$(s.datatype)\",\n"
	o *= "\"vec\": {"
	if !isempty(s.vec)
		o *= "\n"
		o *=join(map(k -> @sprintf("\"%s\": %s",k,tojson(s.vec[k])),keys(s.vec)),",\n")
		o *= "\n"
	end
  o *= "},\n \"other\": {"
  if !isempty(s.other)
  	o *= "\n"
		o *=join(map(k -> @sprintf("\"%s\": %s",k,tojson(s.other[k])),keys(s.other)),",\n")
		o *= "\n"
	end
	o *= "}\n"
	o *= "}\n"
end

function (s::Branch)(v::Dict)
	x = vcat(map(k -> haskey(v,k) ? s.vec[k](v[k]) : zeros(s.datatype,dimension(s.vec[k])), keys(s.vec))...)
	x = reshape(x,:,1)
	o = map(k -> haskey(v,k) ? s.other[k](v[k]) : voidnode(), keys(s.other))	
	DataNode(Array{Any}([x,o...]),nothing,nothing)
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
