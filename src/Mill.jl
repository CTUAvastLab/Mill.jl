__precompile__()
module Mill
using JSON, Flux, MLDataPattern, SparseArrays, Statistics
import Base.reduce
const COLORS = [:blue, :red, :green, :yellow, :cyan, :magenta]

MLDataPattern.nobs(::Nothing) = nothing

function paddedprint(io, s...; color=:default, pad=[])
    for (c, p) in pad
        printstyled(io, p, color=c)
    end
    printstyled(io, s..., color=color)
end

function powerset(x::Vector{T}) where T
    result = Vector{T}[[]]
    for elem in x, j in eachindex(result)
        push!(result, [result[j] ; elem])
    end
    result
end

const Bags = Vector{UnitRange{Int64}}
const VecOrRange = Union{UnitRange{Int},AbstractVector{Int}}
const MillFunction = Union{Flux.Dense, Flux.Chain, Function}

"""
	catobs(xs...)

	concatenates all observations from all xs together
"""
function catobs end;

include("util.jl")
include("threadfuns.jl")

include("datanodes/datanodes.jl")
export AbstractNode, AbstractTreeNode, AbstractBagNode
export ArrayNode, BagNode, WeightedBagNode, TreeNode
export catobs, removeinstances

include("aggregation.jl")
# agg. types exported in aggregation.jl
export Aggregation

include("modelnode.jl")
export MillModel, ArrayModel, BagModel, ProductModel

include("conv.jl")
export bagconv, BagConv

include("bagchain.jl")
export BagChain

include("replacein.jl")
export replacein

include("explainer.jl")
export explain

end
