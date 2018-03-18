__precompile__()
module NestedMill
using JSON 

Bags = Vector{UnitRange{Int64}}

function padview(io::IO,n) 
	for i in 1:n
		print(io," ")
	end 
end
include("datanode.jl")
# include("jsonschema.jl")
include("reflector.jl")


export DataNode
end
