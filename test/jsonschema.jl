using JSON

mutable struct Entry{A}
	counts::Dict{A,Int}
end

mutable struct VectorEntry{A}
	counts::Dict{A,Int}
	l::Dict{A,Int}
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


newitem(v::Dict) = Dict{String,Any}()
newitem(v::A) where {A} = Entry(Dict{A,Int}())

function accomodate!(s::Dict{String,Any},d)
	for (k,v) in d
		i = get(s,k,newitem(v))
		accomodate!(i,v)
		s[k] = i
	end
end

schema = Dict{String,Any}()
open("prescription.jsonl") do fid
	foreach(f -> accomodate!(schema,JSON.parse(f)),readlines(fid))
end

