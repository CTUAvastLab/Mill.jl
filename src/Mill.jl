module Mill

using ChainRulesCore
using Combinatorics
using DataFrames
using Flux
using HierarchicalUtils
using LearnBase
using LinearAlgebra
using MLDataPattern
using Setfield
using SparseArrays
using Statistics
using StatsBase
using Zygote

using Base: CodeUnits, nameof
using Setfield: IdentityLens, PropertyLens, IndexLens, ComposedLens

import Base, Flux, ChainRulesCore
import Base: *, ==, reduce

# COMMON ALIASES
const VecOrRange{T} = Union{UnitRange{T}, AbstractVector{T}}
using Base: AbstractVecOrMat, AbstractVecOrTuple
const Maybe{T} = Union{T, Missing}
const Optional{T} = Union{T, Nothing}

_promote_types(x) = typeof(x)
_promote_types(x, y...) = promote_type(typeof(x), _promote_types(y...))

include("globals.jl")

include("bags.jl")
export AbstractBags, AlignedBags, ScatteredBags, length2bags, remapbags, bags, adjustbags

include("datanodes/datanode.jl")
export AbstractNode, AbstractProductNode, AbstractBagNode
export ArrayNode, BagNode, WeightedBagNode, ProductNode, LazyNode
export catobs, removeinstances

include("special_arrays/special_arrays.jl")
export MaybeHotVector, MaybeHotMatrix, maybehot, maybehotbatch
export NGramMatrix, NGramIterator, ngrams, ngrams!, countngrams, countngrams!
export ImputingMatrix, PreImputingMatrix, PostImputingMatrix
export ImputingDense, PreImputingDense, PostImputingDense
export preimputing_dense, postimputing_dense, identity_dense

(::Flux.LayerNorm)(x::Mill.NGramMatrix) = x

include("aggregations/aggregation.jl")
# agg. types exported in aggregation.jl
export Aggregation

include("modelnodes/modelnode.jl")
export AbstractMillModel, ArrayModel, BagModel, ProductModel, LazyModel
export IdentityModel, identity_model
export HiddenLayerModel
export mapactivations, reflectinmodel

const MillStruct = Union{AbstractMillModel, AbstractNode}

include("conv.jl")
export bagconv, BagConv

include("bagchain.jl")
export BagChain

include("hierarchical_utils.jl")
export printtree

include("partialeval.jl")
export partialeval

include("mill_string.jl")
export MillString, @mill_str

include("util.jl")
export sparsify, pred_lens, list_lens, find_lens, findnonempty_lens
export replacein, code2lens, lens2code, model_lens, data_lens

Base.show(io::IO, ::MIME"text/plain", @nospecialize(n::MillStruct)) =
    HierarchicalUtils.printtree(io, n; htrunc=3, vtrunc=3)

_show(io, x) = _show_fields(io, x)

function _show_fields(io, x::T; context=:compact=>true) where T
    print(io, nameof(T), "(", join(["$f = $(repr(getfield(x, f); context))" for f in fieldnames(T)],", "), ")")
end

Base.getindex(n::MillStruct, i::AbstractString) = HierarchicalUtils.walk(n, i)

end
