using JSON

abstract type JSONEntry end;
StringOrNumber = Union{String,Number};
const max_keys = 1000

mutable struct Entry <: JSONEntry
	counts::Dict{Any,Int}
	called::Int
end

Entry() = Entry(Dict{Any,Int}(),0);
types(e::Entry) = unique(typeof.(collect(keys(e.counts))))
Base.show(io::IO, e::Entry,offset::Int=0) = paddedprint(io, @sprintf("[Scalar - %s], %d unique values, called = %d",join(types(e)),length(keys(e.counts)),e.called),0)


function accomodate!(a::Entry,v)
	if length(keys(a.counts)) < max_keys
		a.counts[v] = get(a.counts,v,0) + 1
	end
	a.called +=1
end


mutable struct VectorEntry{A<:JSONEntry} <: JSONEntry
	items::A
	l::Dict{Int,Int}
	called::Int
end

VectorEntry(items) = VectorEntry(items,Dict{Int,Int}(),0)

function Base.show(io::IO, e::VectorEntry,offset::Int=0) 
	paddedprint(io, "[Vector of\n",0);
	show(io,e.items,offset+2)
	paddedprint(io, @sprintf(" ], called = %d ",e.called),offset)
end

function accomodate!(a::VectorEntry,b::Vector)
	n = length(b)
	a.l[n] = get(a.l,n,0) + 1
	foreach(v -> accomodate!(a.items,v),b)
	a.called +=1
end


mutable struct DictEntry <: JSONEntry
	childs::Dict{String,Any}
	called::Int
end

DictEntry() = DictEntry(Dict{String,Any}(),0)
Base.getindex(s::DictEntry,k) = s.childs[k]

function Base.show(io::IO, e::DictEntry,offset::Int=0)
	paddedprint(io,"Dict\n",offset)
	for k in keys(e.childs)
			paddedprint(io,@sprintf("%s: ",k),offset+2);
	  	Base.show(io,e.childs[k],offset+4)
	  	print(io,"\n")
  end
end

function accomodate!(s::DictEntry,d::Dict)
	s.called +=1
	for (k,v) in d
		i = get(s.childs,k,newitem(v))
		accomodate!(i,v)
		s.childs[k] = i
	end
end




newitem(v::Dict) = DictEntry()
newitem(v::A) where {A<:StringOrNumber} = Entry()
newitem(v::Vector) = VectorEntry(newitem(v[1]))

# conversion to data extractor
called(s::T) where {T<:JSONEntry} = s.called
recommendscheme(T,e::Entry,mincount) = ExtractScalar(eltype(map(identity,keys(e.counts))))
recommendscheme(T,e::VectorEntry,mincount) = ExtractArray(recommendscheme(T,e.items,mincount))
function recommendscheme(T,e::DictEntry, mincount::Int = typemax(Int))
	ks = Iterators.filter(k -> called(e.childs[k]) > mincount, keys(e.childs))
	if isempty(ks)
		return(ExtractBranch(T,Dict{String,Any}(),Dict{String,Any}()))
	end
	c = map(k -> (k,recommendscheme(T, e.childs[k], mincount)),ks)
	mask = map(i -> typeof(i[2])<:NestedMill.ExtractScalar,c)
	mask = mask .| map(i -> typeof(i[2])<:NestedMill.ExtractCategorical,c)
	ExtractBranch(T,Dict(c[mask]),Dict(c[.!mask]))
end
