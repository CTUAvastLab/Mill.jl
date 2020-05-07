module Mill

using Flux
using MLDataPattern
using SparseArrays
using Statistics
using Combinatorics
using Zygote
using HierarchicalUtils
using Zygote: @adjoint
using LinearAlgebra
import Base.reduce

import HierarchicalUtils: NodeType, childrenfields, children, InnerNode, SingletonNode, LeafNode, printtree, noderepr

MLDataPattern.nobs(::Missing) = nothing

const VecOrRange = Union{UnitRange{Int},AbstractVector{Int}}

"""
	catobs(xs...)

	concatenates all observations from all xs together
"""
function catobs end;

include("bags.jl")
export AlignedBags, ScatteredBags, length2bags

include("util.jl")
include("threadfuns.jl")

include("datanodes/datanode.jl")
export AbstractNode, AbstractProductNode, AbstractBagNode
export ArrayNode, BagNode, WeightedBagNode, ProductNode, LazyNode
export NGramMatrix, NGramIterator
export catobs, removeinstances

include("aggregations/aggregation.jl")
# agg. types exported in aggregation.jl
export AggregationFunction, Aggregation

include("modelnodes/modelnode.jl")
export AbstractMillModel, ArrayModel, BagModel, ProductModel, LazyModel
export reflectinmodel

include("conv.jl")
export bagconv, BagConv

include("bagchain.jl")
export BagChain

include("replacein.jl")
export replacein

include("hierarchical_utils.jl")

Base.show(io::IO, ::T) where T <: Union{AbstractNode, AbstractMillModel, AggregationFunction} = show(io, Base.typename(T))
Base.show(io::IO, ::MIME"text/plain", n::Union{AbstractNode, AbstractMillModel}) = HierarchicalUtils.printtree(io, n; trunc_level=2)
Base.getindex(n::Union{AbstractNode, AbstractMillModel}, i::AbstractString) = HierarchicalUtils.walk(n, i)

include("partialeval.jl")
const _emptyismissing = Ref(false)

function emptyismissing(a)
    _emptyismissing[] = a
end

const _terseprint = Ref(true)

function terseprint(a)
    _terseprint[] = a
end

function Base.show(io::IO, x::Type{T}) where {T<:Union{AbstractNode,AbstractMillModel}}
	if _terseprint[]
		if !hasproperty(x, :name) && hasproperty(x, :body)
			print(io, "$(x.body.name){…}")
			return
		else
			print(io, "$(x.name){…}")
			return
		end
	# basically copied from the Julia sourcecode, seems it's one of most robust fixes to Pevňákoviny
    elseif x isa DataType
        Base.show_datatype(io, x)
        return
    elseif x isa Union
        print(io, "Union")
        Base.show_delim_array(io, Base.uniontypes(x), '{', ',', '}', false)
        return
    end
    x::UnionAll

    if Base.print_without_params(x)
        return show(io, Base.unwrap_unionall(x).name)
    end

    if x.var.name === :_ || Base.io_has_tvar_name(io, x.var.name, x)
        counter = 1
        while true
            newname = Symbol(x.var.name, counter)
            if !Base.io_has_tvar_name(io, newname, x)
                newtv = TypeVar(newname, x.var.lb, x.var.ub)
                x = UnionAll(newtv, x{newtv})
                break
            end
            counter += 1
        end
    end

    show(IOContext(io, :unionall_env => x.var), x.body)
    print(io, " where ")
    show(io, x.var)
end

export printtree

end
