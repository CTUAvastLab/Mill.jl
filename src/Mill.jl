module Mill

using ChainRulesCore
using Combinatorics
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

import Base: *, ==, isequal, hash, show, cat, vcat, hcat, _cat
import Base: size, length, first, last, firstindex, lastindex, eachindex, getindex, setindex!
import Base: reduce, eltype, print_array
import Base: isascii, codeunits, ncodeunits, codeunit

import Flux: Dense, Chain, Params, params!, IdSet, onehot, onehotbatch

import ChainRulesCore: rrule

# GLOBAL SWITCHES
const _emptyismissing =     Ref(false)
const _bagcount =           Ref(true)
const _wildcard_code =      Ref(UInt8(0)) # NUL in ascii
const _string_start_code =  Ref(UInt8(2)) # STX in ascii
const _string_end_code =    Ref(UInt8(3)) # ETX in ascii

for s in Symbol.(["emptyismissing", "bagcount", "wildcard_code", "string_start_code", "string_end_code"])
    ex = Symbol(s, '!')
    us = Symbol('_', s)
    @eval @inline $ex(a) = $us[] = a
    @eval @inline $s() = $us[]
end

# COMMON ALIASES
const VecOrRange{T} = Union{UnitRange{T}, AbstractVector{T}}
using Base: AbstractVecOrMat
const Maybe{T} = Union{T, Missing}
const Optional{T} = Union{T, Nothing}

StatsBase.nobs(::Missing) = nothing

include("bags.jl")
export AlignedBags, ScatteredBags, length2bags, remapbag, bags

include("datanodes/datanode.jl")
export AbstractNode, AbstractProductNode, AbstractBagNode
export ArrayNode, BagNode, WeightedBagNode, ProductNode, LazyNode
export catobs, removeinstances

include("matrices/matrix.jl")
export MaybeHotVector, MaybeHotMatrix, maybehot, maybehotbatch
export NGramMatrix, NGramIterator
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
export sparsify, findnonempty, ModelLens, replacein, findin

Base.show(io::IO, ::MIME"text/plain", @nospecialize(n::Union{AbstractNode, AbstractMillModel})) =
    HierarchicalUtils.printtree(io, n; htrunc=3)

_show(io, x) = _show_fields(io, x)

function _show_fields(io, x::T; context=:compact=>true) where T
    print(io, nameof(T), "(", join(["$f = $(repr(getfield(x, f); context))" for f in fieldnames(T)],", "), ")")
end

Base.getindex(n::Union{AbstractNode, AbstractMillModel}, i::AbstractString) = HierarchicalUtils.walk(n, i)

end
