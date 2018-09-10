__precompile__(false)
module Mill
using JSON, Flux, Adapt, MLDataPattern, SparseArrays, Statistics

const COLORS = [:blue, :cyan, :green, :magenta, :yellow, :red]

function paddedprint(io, s...; color=:default, pad=[])
    for (c, p) in pad
        print_styled(io, p, color=c)
    end
    print_with_color(io, s..., color=color)
end

const Bags = Vector{UnitRange{Int64}}
const VecOrRange = Union{UnitRange{Int},AbstractVector{Int}}
const MillFunction = Union{Flux.Dense, Flux.Chain, Function}

include("util.jl")
include("datanode.jl")
include("modelnode.jl")
include("aggregation/aggregation.jl")

export AbstractNode, AbstractTreeNode, AbstractBagNode
export ArrayNode, BagNode, WeightedBagNode, TreeNode
export MillModel, ArrayModel, BagModel, ProductModel

end
