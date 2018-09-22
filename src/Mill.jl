__precompile__()
module Mill
using JSON, Flux, Adapt, MLDataPattern, SparseArrays, Statistics

const COLORS = [:blue, :red, :green, :yellow, :cyan, :magenta]

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

include("util.jl")

include("datanode.jl")
export AbstractNode, AbstractTreeNode, AbstractBagNode
export ArrayNode, BagNode, WeightedBagNode, TreeNode

include("aggregation.jl")
export PNorm, LSE, Aggregation
export SegmentedMean, SegmentedMax, SegmentedPNorm, SegmentedMeanMax, SegmentedPNormMeanMax

include("modelnode.jl")
export MillModel, ArrayModel, BagModel, ProductModel

end
