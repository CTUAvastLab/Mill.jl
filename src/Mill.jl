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
export ArrayNode, BagNode, WeightedBagNode, ProductNode, LazyNode, IdentityModel
export NGramMatrix, NGramIterator
export catobs, removeinstances

include("aggregations/aggregation.jl")
# agg. types exported in aggregation.jl
export AggregationFunction, Aggregation

include("modelnodes/modelnode.jl")
export AbstractMillModel, ArrayModel, BagModel, ProductModel, LazyModel, IdentityModel, identity_model
export reflectinmodel

include("conv.jl")
export bagconv, BagConv

include("bagchain.jl")
export BagChain

include("replacein.jl")
export replacein, findin

include("hierarchical_utils.jl")
export printtree

Base.show(io::IO, ::T) where T <: Union{AbstractNode, AbstractMillModel, AggregationFunction} = show(io, Base.typename(T))
Base.show(io::IO, ::MIME"text/plain", n::Union{AbstractNode, AbstractMillModel}) = HierarchicalUtils.printtree(io, n; htrunc=3)
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
# function Base_show(io::IO, x::Type{T}) where {T<:Union{AbstractNode,AbstractMillModel}}
    # print(io, "\ntvoje máma: calling base show, typeof: $(typeof(x))\n")
    # print(io, "\ntvoje máma: calling base show, <: Type: $(x <: Type)\n")
    # print(io, "\ntvoje máma: calling base show, isa Type: $(x isa Type)\n")
    if _terseprint[]
        if !hasproperty(x, :name) && hasproperty(x, :body)
            print(io, "$(x.body.name){…}")
            return
        else
            print(io, "$(x.name){…}")
            return
        end
    else
        # basically copied from the Julia sourcecode, seems it's one of most robust fixes to Pevňákoviny
        # specifically function show(io::IO, @nospecialize(x::Type))
        if x isa DataType
            # print(io, "\ntvoje máma: elseif x isa DataType\n")
            Base.show_datatype(io, x)
            return
        elseif x isa Union
            # print(io, "\ntvoje máma: elseif x isa Union\n")
            if x.a isa DataType && Core.Compiler.typename(x.a) === Core.Compiler.typename(DenseArray)
                T2, N = x.a.parameters
                if x == StridedArray{T2,N}
                    print(io, "StridedArray")
                    Base.show_delim_array(io, (T2,N), '{', ',', '}', false)
                    return
                elseif x == StridedVecOrMat{T2}
                    print(io, "StridedVecOrMat")
                    Base.show_delim_array(io, (T2,), '{', ',', '}', false)
                    return
                elseif StridedArray{T2,N} <: x
                    print(io, "Union")
                    Base.show_delim_array(io, vcat(StridedArray{T2,N}, Base.uniontypes(Core.Compiler.typesubtract(x, StridedArray{T2,N}))), '{', ',', '}', false)
                    return
                end
            end
            print(io, "Union")
            Base.show_delim_array(io, Base.uniontypes(x), '{', ',', '}', false)
            return
        end

        # print(io, "tvoje máma typeof(x): \n$(typeof(x))\n")
        # print(io, "tvoje máma fieldnames(x): \n$(fieldnames(x))\n")
        # print(io, "\ntvoje máma <: DataType: $(x <: DataType)\n")
        # print(io, "\ntvoje máma isa DataType: $(x isa DataType)\n")
        # print(io, "\ntvoje máma :data in fieldnames(x)?: $(:data in fieldnames(x)) \n")
        # print(io, "\ntvoje máma zkouší x.var\n")
        # x.var

        # this type assert is behaving obscurely. When in Mill, it does not assert that LazyNode{T<:Symbol,D} where D is UnionAll, but in debugging using Debugger, it does
        # x::UnionAll
        if Base.print_without_params(x)
            # print(io, "\ntvoje máma: if Base.print_without_params(x)\n")
            return show(io, Base.unwrap_unionall(x).name)
        end

        if x.var.name === :_ || Base.io_has_tvar_name(io, x.var.name, x)
            # print(io, "\ntvoje máma: if x.var.name === :_ || Base.io_has_tvar_name(io, x.var.name, x)\n")
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

        # print(io, "\ntvoje máma před: show(IOContext(io, :unionall_env => x.var), x.body)\n")
        # print(io, "tvoje máma typeof(x): \n$(typeof(x))\n")
        # print(io, "tvoje máma fieldnames(x): \n$(fieldnames(x))\n")
        show(IOContext(io, :unionall_env => x.var), x.body)
        print(io, " where ")
        show(io, x.var)
    end
end

end
