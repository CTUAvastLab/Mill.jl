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

import Base: *, ==, hash, show, cat, vcat, hcat, _cat
import Base: size, length, first, last, firstindex, lastindex, getindex, setindex!
import Base: reduce, eltype, print_array

import Flux: Params, params!, IdSet

import ChainRulesCore: rrule

# GLOBAL SWITCHES
const _emptyismissing = Ref(false)
const _terseprint = Ref(true)
const _bagcount = Ref(true)
emptyismissing(a) = _emptyismissing[] = a
terseprint(a) = _terseprint[] = a
bagcount(a) = _bagcount[] = a

# COMMON ALIASES
const VecOrRange{T} = Union{UnitRange{T},AbstractVector{T}}
using Base: AbstractVecOrMat
const MissingElement{T} = Union{T, Missing}
const MaybeAbstractMatrix{T} = Union{Missing, AbstractMatrix{T}}
const AggregationWeights{T} = Union{Nothing, AbstractVecOrMat{<:T}}

MLDataPattern.nobs(::Missing) = nothing

"""
	catobs(xs...)

	concatenates all observations from all xs together
"""
function catobs end

include("bags.jl")
export AlignedBags, ScatteredBags, length2bags, remapbag

include("util.jl")
include("threadfuns.jl")

include("matrices/maybehot.jl")
export MaybeHotVector, MaybeHotMatrix

include("matrices/ngram_matrix.jl")
export NGramMatrix, NGramIterator

include("matrices/imputing_matrix.jl")
export ImputingMatrix, ImputingDense

include("datanodes/datanode.jl")
export AbstractNode, AbstractProductNode, AbstractBagNode
export ArrayNode, BagNode, WeightedBagNode, ProductNode, LazyNode
export catobs, removeinstances

include("aggregations/aggregation.jl")
# agg. types exported in aggregation.jl
export AggregationFunction, Aggregation

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

Base.show(io::IO, ::T) where T <: Union{AbstractNode, AbstractMillModel, AggregationFunction} = show(io, Base.typename(T))
Base.show(io::IO, ::MIME"text/plain", n::Union{AbstractNode, AbstractMillModel}) = HierarchicalUtils.printtree(io, n; htrunc=3)
Base.getindex(n::Union{AbstractNode, AbstractMillModel}, i::AbstractString) = HierarchicalUtils.walk(n, i)

include("partialeval.jl")
export partialeval

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
