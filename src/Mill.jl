module Mill

using Combinatorics
using Flux
using HierarchicalUtils
using LearnBase
using LinearAlgebra
using MLDataPattern
using SparseArrays
using Statistics
using Zygote
using ChainRulesCore

using Base: CodeUnits, nameof

import Base: *, ==, hash, show, cat, vcat, hcat, _cat
import Base: size, length, first, last, firstindex, lastindex, getindex, setindex!
import Base: reduce, eltype, print_array

import Flux: Params, params!, IdSet, onehot, onehotbatch

import ChainRulesCore: rrule

# GLOBAL SWITCHES
const _emptyismissing = Ref(false)
const _bagcount = Ref(true)

emptyismissing!(a) = _emptyismissing[] = a
bagcount!(a) = _bagcount[] = a

emptyismissing() = _emptyismissing[]
bagcount() = _bagcount[]

# COMMON ALIASES
const VecOrRange{T} = Union{UnitRange{T}, AbstractVector{T}}
using Base: AbstractVecOrMat
const Maybe{T} = Union{T, Missing}
const Optional{T} = Union{T, Nothing}

MLDataPattern.nobs(::Missing) = nothing

"""
	catobs(xs...)

	concatenates all observations from all xs together
"""
function catobs end

include("bags.jl")
export AlignedBags, ScatteredBags, length2bags, remapbag, bags

include("util.jl")
include("threadfuns.jl")

include("matrices/matrix.jl")
export MaybeHotVector, MaybeHotMatrix, maybehot, maybehotbatch
export NGramMatrix, NGramIterator
export ImputingMatrix, RowImputingMatrix, ColImputingMatrix
export ImputingDense, RowImputingDense, ColImputingDense

(::Flux.LayerNorm)(x::Mill.NGramMatrix) = x

include("datanodes/datanode.jl")
export AbstractNode, AbstractProductNode, AbstractBagNode
export ArrayNode, BagNode, WeightedBagNode, ProductNode, LazyNode
export catobs, removeinstances

include("aggregations/aggregation.jl")
# agg. types exported in aggregation.jl
export AggregationOperator, Aggregation

include("modelnodes/modelnode.jl")
export AbstractMillModel, ArrayModel, BagModel, ProductModel, LazyModel, IdentityModel, identity_model
export HiddenLayerModel
export mapactivations, reflectinmodel

include("conv.jl")
export bagconv, BagConv

include("bagchain.jl")
export BagChain

include("replacein.jl")
export replacein, findin

include("hierarchical_utils.jl")
export printtree

include("partialeval.jl")
export partialeval

Base.show(io::IO, ::MIME"text/plain", @nospecialize n::T) where T <: Union{AbstractNode, AbstractMillModel} = 
    HierarchicalUtils.printtree(io, n; htrunc=3)

_show(io, x) = _show_fields(io, x)

function _show_fields(io, x::T;context=:compact=>true) where T
    print(io, nameof(T), "(", join(["$f = $(repr(getfield(x, f); context))" for f in fieldnames(T)],", "), ")")
end

Base.getindex(n::Union{AbstractNode, AbstractMillModel}, i::AbstractString) = HierarchicalUtils.walk(n, i)

end
