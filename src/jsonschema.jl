using JSON

abstract type JSONEntry end;

mutable struct Entry{A} <: JSONEntry
	counts::Dict{A,Int}
	called::Int
end

Base.show(io::IO, e::Entry,offset::Int=0) = print(io, @sprintf("[ExtractScalar], %d unique values, called = %d",length(keys(e.counts)),e.called))

mutable struct VectorEntry{A} <: JSONEntry
	counts::Dict{A,Int}
	l::Dict{A,Int}
	called::Int
end

Base.show(io::IO, e::VectorEntry,offset::Int=0) = print(io, @sprintf("[Vector], %d unique values, called = %d ",length(keys(e.counts)),e.called))

function accomodate!(a::A,b) where {A<:Union{VectorEntry,Entry}}
	if length(keys(a.counts)) < 1000
		a.counts[b] = get(a.counts,b,0) + 1
	end 
	a.called +=1
end

function accomodate!(a::VectorEntry,b::Vector)
	a.called +=1
	n = length(b)
	a.l[n] = get(a.l,n,0) + 1
	for v in b
		if length(keys(a.counts)) < 1000
			a.counts[b] = get(a.counts,b,0) + 1
		end 
	end		
end

mutable struct DictEntry <: JSONEntry
	counts::Dict{String,Int}
	childs::Dict{String,Any}
	called
end

DictEntry() = DictEntry(Dict{String,Int}(),Dict{String,Any}(),0)
Base.getindex(s::DictEntry,k) = s.childs[k]

function Base.show(io::IO, e::DictEntry,offset::Int=0)
	print(io,"\n")
	for k in keys(e.childs)
			padview(io,offset)
			print(io,@sprintf("%s: ",k));
	  	Base.show(io,e.childs[k],offset+2)
	  	print(io,"\n")
  end
end

function accomodate!(s::DictEntry,v::Dict)
	s.called +=1
	for k in keys(v)
		s.counts[k] = get(s.counts,k,0) +1 
	end
	accomodate!(s.childs,v)
end




# newitem(v::Dict) = Dict{String,Any}()
newitem(v::Dict) = DictEntry()
newitem(v::A) where {A} = Entry(Dict{A,Int}(),0)

function accomodate!(s::Dict{String,Any},d)
	for (k,v) in d
		i = get(s,k,newitem(v))
		accomodate!(i,v)
		s[k] = i
	end
end


# conversion to data extractor
called(s::T) where {T<:JSONEntry} = sum(s.called)
recommendscheme(S,e::Entry{T},mincount) where {T} = ExtractScalar(T)
recommendscheme(S,e::VectorEntry{T},mincount)  where {T} = ExtractArray(ExtractScalar(T))
function recommendscheme(T,e::DictEntry, mincount::Int = typemax(Int))
	ks = filter(k -> called(e.childs[k]) > mincount, keys(e.childs))
	if isempty(ks)
		return(ExtractBranch(T,Dict{String,Any}(),Dict{String,Any}()))
	end
	c = map(k -> (k,recommendscheme(T, e.childs[k], mincount)),ks)
	mask = map(i -> typeof(i[2])<:NestedMill.ExtractScalar,c)
	mask = mask .| map(i -> typeof(i[2])<:NestedMill.ExtractCategorical,c)
	ExtractBranch(T,Dict(c[mask]),Dict(c[.!mask]))
end
