module Mill

using ChainRulesCore
using Combinatorics
using DataFrames
using FiniteDifferences
using Flux
using HierarchicalUtils
using LinearAlgebra
using MLUtils
using MacroTools
using OneHotArrays
using Preferences
using Setfield
using SparseArrays
using Statistics

using Base: CodeUnits, nameof
using ChainRulesCore: NotImplemented, NotImplementedException
using HierarchicalUtils: encode, stringify
using Setfield: IdentityLens, PropertyLens, IndexLens, ComposedLens

import Base: *, ==

# COMMON ALIASES
using Base: AbstractVecOrMat, AbstractVecOrTuple
const VecOrRange{T} = Union{UnitRange{T}, AbstractVector{T}}
const VecOrTupOrNTup{T} = Union{Vector{<:T}, Tuple{Vararg{T}}, NamedTuple{K, <:Tuple{Vararg{T}}} where K}
const Maybe{T} = Union{T, Missing}
const Optional{T} = Union{T, Nothing}

const DOCTEST_FILTER = r"\s*-?[0-9]+\.[0-9]+[\.]*\s*"

_promote_types(x) = typeof(x)
_promote_types(x, y...) = promote_type(typeof(x), _promote_types(y...))

promote_maybe_type(::Type{<:Maybe{T}}, ::Type{<:Maybe{U}}) where {T, U} = promote_type(T, U)
promote_maybe_type(t1, t2) = promote_type(t1, t2)

include("switches.jl")

include("bags.jl")
export AbstractBags, AlignedBags, ScatteredBags, length2bags, remapbags, bags, adjustbags

include("datanodes/datanode.jl")
export AbstractMillNode, AbstractProductNode, AbstractBagNode
export ArrayNode, BagNode, WeightedBagNode, ProductNode, LazyNode
export numobs, getobs, catobs, removeinstances, dropmeta

include("special_arrays/special_arrays.jl")
export MaybeHotVector, MaybeHotMatrix, maybehot, maybehotbatch, maybecold
export NGramMatrix, NGramIterator, ngrams, ngrams!, countngrams, countngrams!
export ImputingMatrix, PreImputingMatrix, PostImputingMatrix
export ImputingDense, PreImputingDense, PostImputingDense
export preimputing_dense, postimputing_dense, identity_dense

include("aggregations/aggregations.jl")
export AbstractAggregation, AggregationStack
export BagCount
# combined agg. types exported in aggregations.jl
export SegmentedMean, SegmentedMax, SegmentedSum, SegmentedLSE, SegmentedPNorm

include("modelnodes/modelnode.jl")
export AbstractMillModel, ArrayModel, BagModel, ProductModel, LazyModel
export reflectinmodel

const AbstractMillStruct = Union{AbstractMillModel, AbstractMillNode}

include("conv.jl")
export bagconv, BagConv

include("bagchain.jl")
export BagChain

# include("mill_string.jl")
# export MillString, @mill_str

include("util.jl")
export pred_lens, list_lens, find_lens, findnonempty_lens
export replacein, code2lens, lens2code, model_lens, data_lens

include("gradients.jl")

Base.getindex(n::AbstractMillStruct, i::AbstractString) = HierarchicalUtils.walk(n, i)

include("show.jl")
export datasummary, modelsummary

include("hierarchical_utils.jl")
export printtree

end
