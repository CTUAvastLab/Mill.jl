using JSON

mutable struct Entry{A}
	counts::Dict{A,Int}
end

mutable struct VectorEntry{A}
	counts::Dict{A,Int}
	l::Dict{A,Int}
end

mutable struct DictEntry
	counts::Dict{String,Int}
	childs::Dict{String,Any}
end

Base.show(io::IO, e::Entry,offset::Int=0) = print(io, @sprintf("[Scalar], %d unique values",length(keys(e.counts))))
Base.show(io::IO, e::VectorEntry,offset::Int=0) = print(io, @sprintf("[Vector], %d unique values",length(keys(e.counts))))

function Base.show(io::IO, e::DictEntry,offset::Int=0)
	print(io,"\n")
	for k in keys(e.childs)
			padview(io,offset)
			print(io,@sprintf("%s: ",k));
	  	Base.show(io,e.childs[k],offset+2)
	  	print(io,"\n")
  end
end


DictEntry() = DictEntry(Dict{String,Int}(),Dict{String,Any}())
Base.getindex(s::DictEntry,k) = s.childs[k]

function accomodate!(s::DictEntry,v::Dict)
	for k in keys(v)
		s.counts[k] = get(s.counts,k,0) +1 
	end
	accomodate!(s.childs,v)
end

function accomodate!(a::A,b) where {A<:Union{VectorEntry,Entry}}
	if length(keys(a.counts)) < 1000
		a.counts[b] = get(a.counts,b,0) + 1
	end 
end

function accomodate!(a::VectorEntry,b::Vector)
	n = length(b)
	a.l[n] = get(a.l,n,0) + 1
	foreach(v -> accomodate!(a,v),b)
end


# newitem(v::Dict) = Dict{String,Any}()
newitem(v::Dict) = DictEntry()
newitem(v::A) where {A} = Entry(Dict{A,Int}())

function accomodate!(s::Dict{String,Any},d)
	for (k,v) in d
		i = get(s,k,newitem(v))
		accomodate!(i,v)
		s[k] = i
	end
end